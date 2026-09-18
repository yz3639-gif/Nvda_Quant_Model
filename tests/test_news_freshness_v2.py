import json

import pandas as pd
import pytest

from nvda_quant_model.news_sentiment import (NewsArticle, NewsFetchResult, assess_news_freshness,
    fetch_live_news, get_live_news_overlay, parse_published_at)

NOW = pd.Timestamp('2026-09-18T16:00:00Z')


def cache_payload(age_hours=1):
    stamp = (NOW - pd.Timedelta(hours=age_hours)).isoformat()
    return dict(article_count=3, signal=-1, sentiment_score=-.8, confidence_adjustment=.08,
                risk_flags=['regulatory_risk'], top_articles=[{'title': 'historical headline'}],
                fetched_at=stamp, generated_at=stamp, available_at=stamp)


def write_cache(tmp_path, payload):
    path = tmp_path / 'live_news_overlay.json'
    path.write_text(json.dumps(payload))
    return path


def fail(**kwargs):
    raise RuntimeError('offline failure')


def test_fresh_cache_keeps_signal_and_propagates_fetch_error(tmp_path):
    write_cache(tmp_path, cache_payload())
    overlay = get_live_news_overlay(tmp_path, as_of=NOW, fetcher=fail)
    assert overlay['signal'] == -1 and overlay['eligible_for_signal']
    assert overlay['source'] == 'cache'
    assert overlay['freshness'] == 'fresh'
    assert 'offline failure' in overlay['fetch_error']


@pytest.mark.parametrize('failure', [fail, lambda **kwargs: []])
def test_expired_cache_is_context_only_and_file_is_not_overwritten(tmp_path, failure):
    path = write_cache(tmp_path, cache_payload(25))
    before = path.read_bytes()
    overlay = get_live_news_overlay(tmp_path, as_of=NOW, fetcher=failure)
    assert overlay['signal'] == overlay['sentiment_score'] == overlay['confidence_adjustment'] == 0
    assert overlay['risk_flags'] == [] and not overlay['eligible_for_signal']
    assert overlay['historical_context']['signal'] == -1
    assert overlay['freshness'] == 'expired' and overlay['source'] == 'cache'
    assert path.read_bytes() == before
    assert overlay['fetch_error']


@pytest.mark.parametrize('mutate,reason', [
    (lambda x: x.pop('fetched_at'), 'missing_timestamp'),
    (lambda x: x.update(available_at='invalid'), 'missing_timestamp'),
    (lambda x: x.update(generated_at=(NOW + pd.Timedelta(hours=1)).isoformat()), 'future_timestamp'),
])
def test_unknown_or_future_timestamp_cache_is_ineligible(tmp_path, mutate, reason):
    payload = cache_payload()
    mutate(payload)
    write_cache(tmp_path, payload)
    overlay = get_live_news_overlay(tmp_path, as_of=NOW, fetcher=fail)
    assert overlay['signal'] == 0 and not overlay['eligible_for_signal']
    assert overlay['freshness'] == reason


@pytest.mark.parametrize('raw', ['{bad json', '[]', 'null', '{"article_count":"bad"}'])
def test_corrupt_cache_degrades_to_neutral_with_both_errors(tmp_path, raw):
    (tmp_path / 'live_news_overlay.json').write_text(raw)
    overlay = get_live_news_overlay(tmp_path, as_of=NOW, fetcher=fail)
    assert overlay['source'] == 'neutral' and overlay['signal'] == 0
    assert overlay['cache_error'] and 'offline failure' in overlay['fetch_error']


def test_empty_fetch_without_cache_is_neutral(tmp_path):
    overlay = get_live_news_overlay(tmp_path, as_of=NOW, fetcher=lambda **kwargs: [])
    assert overlay['source'] == 'neutral' and not overlay['eligible_for_signal']
    assert overlay['fetch_error'] == 'live_news_fetch_returned_zero_articles'


def article(**changes):
    fields = dict(id='1', title='Nvidia ban', source='Example', link='https://example.com',
        published_at=(NOW-pd.Timedelta(hours=2)).isoformat(), summary='', query='NVDA',
        sentiment_score=-.8, event_scores={'regulatory': 1},
        fetched_at=(NOW-pd.Timedelta(minutes=1)).isoformat(), available_at=(NOW-pd.Timedelta(minutes=1)).isoformat())
    fields.update(changes)
    return NewsArticle(**fields)


def test_fresh_live_success_records_all_timestamps_and_partial_errors(tmp_path):
    fetched = NOW.isoformat()
    batch = NewsFetchResult([article()], fetched_at=fetched, errors=['one source failed'])
    overlay = get_live_news_overlay(tmp_path, as_of=NOW, fetcher=lambda **kwargs: batch)
    assert overlay['source'] == 'live' and overlay['eligible_for_signal'] and overlay['signal'] == -1
    assert overlay['fetched_at'] == fetched
    assert overlay['generated_at'] == NOW.isoformat()
    assert overlay['available_at']
    assert overlay['fetch_error'] == 'one source failed'
    persisted = json.loads((tmp_path / 'live_news_overlay.json').read_text())
    assert persisted['generated_at'] == NOW.isoformat()


@pytest.mark.parametrize('changes', [dict(available_at=None, fetched_at=None), dict(published_at='NaT'),
    dict(published_at=(NOW+pd.Timedelta(days=1)).isoformat()),
    dict(published_at=(NOW-pd.Timedelta(days=8)).isoformat())])
def test_unverifiable_future_or_old_live_article_does_not_affect_signal(tmp_path, changes):
    overlay = get_live_news_overlay(tmp_path, as_of=NOW, fetcher=lambda **kwargs: [article(**changes)])
    assert overlay['signal'] == 0 and not overlay['eligible_for_signal']


def test_fetch_batch_preserves_partial_errors(monkeypatch):
    def fetch(query, **kwargs):
        if query == 'bad':
            raise ValueError('source unavailable')
        return [article()]
    monkeypatch.setattr('nvda_quant_model.news_sentiment.fetch_google_news', fetch)
    batch = fetch_live_news(['good', 'bad'], sleep_seconds=0)
    assert len(batch) == 1 and batch.fetched_at
    assert batch.errors == ['bad: ValueError: source unavailable']


def test_invalid_publication_time_is_never_replaced_with_current_time():
    assert pd.isna(parse_published_at(None))
    assert pd.isna(parse_published_at('invalid'))


def test_custom_ttl_is_respected():
    assert assess_news_freshness(cache_payload(30), source='cache', as_of=NOW, ttl_hours=48)['signal'] == -1


def test_daily_features_use_exchange_close_and_do_not_backdate_after_hours_news():
    from nvda_quant_model.news_sentiment import build_daily_news_features
    dates = pd.DatetimeIndex(['2026-03-06', '2026-03-09', '2026-03-10'])
    rows = pd.DataFrame([
        dict(id='fri_before', published_at='2026-03-06T20:00:00Z', available_at='2026-03-06T20:00:00Z', sentiment_score=1),
        dict(id='fri_after', published_at='2026-03-06T22:00:00Z', available_at='2026-03-06T22:00:00Z', sentiment_score=1),
        dict(id='weekend', published_at='2026-03-07T12:00:00Z', available_at='2026-03-07T12:00:00Z', sentiment_score=1),
        dict(id='mon_after_dst', published_at='2026-03-09T20:30:00Z', available_at='2026-03-09T20:30:00Z', sentiment_score=1),
    ])
    result = build_daily_news_features(rows, dates)
    assert result.news_count_1d.tolist() == [1, 2, 1]
    assert result.attrs['news_time_contract']['point_in_time_verified']


def test_daily_features_tag_missing_availability_assumption():
    from nvda_quant_model.news_sentiment import build_daily_news_features
    rows = pd.DataFrame([dict(id='1', published_at='2026-09-18T12:00:00Z', sentiment_score=1)])
    result = build_daily_news_features(rows, pd.DatetimeIndex(['2026-09-18']))
    assert not result.attrs['news_time_contract']['point_in_time_verified']


def test_offline_import_preserves_availability_and_normalizes_offset_without_inventing_time(tmp_path):
    from nvda_quant_model.news_sentiment import _articles_from_input_json
    path = tmp_path / 'input.json'
    path.write_text(json.dumps([
        dict(title='News', link='https://example.com/1', published_at='2026-09-18T12:00:00-04:00',
             available_at=NOW.isoformat(), fetched_at=NOW.isoformat()),
        dict(title='Missing time', link='https://example.com/2', published_at='broken')]))
    imported = _articles_from_input_json(path)
    good = next(a for a in imported if a.title == 'News')
    bad = next(a for a in imported if a.title == 'Missing time')
    assert good.published_at == NOW.isoformat()
    assert good.available_at == good.fetched_at == NOW.isoformat()
    assert bad.published_at == 'NaT'
