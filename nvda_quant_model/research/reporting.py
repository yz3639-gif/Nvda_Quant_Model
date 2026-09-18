"""Rebuild research figures and prose from saved evidence; never invent a score."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator, LogLocator, ScalarFormatter, NullFormatter, FuncFormatter
import numpy as np
import pandas as pd

NAVY = '#0b1525'
PANEL = '#111f33'
INK = '#e7edf5'
MUTED = '#9fb0c7'
GRID = '#263951'
TEAL = '#4dd5c1'
ORANGE = '#f4b46b'
BLUE = '#82adff'
PURPLE = '#c2a1ff'
COLORS = [TEAL, ORANGE, BLUE, PURPLE, '#dce6f3', '#ef819b']
CORE = ['rule_calibrated', 'logistic_calibrated', 'boosting_calibrated', 'trend', 'buy_hold']
LABELS = {'rule': 'Volume / momentum', 'logistic': 'Logistic', 'boosting': 'Shallow boosting',
          'rule_calibrated': 'Rule + calibration', 'logistic_calibrated': 'Logistic + calibration',
          'boosting_calibrated': 'Boosting + calibration', 'trend': 'Trend control', 'buy_hold': 'NVDA buy & hold'}
STYLE = {'figure.facecolor': NAVY, 'axes.facecolor': PANEL, 'savefig.facecolor': NAVY,
    'text.color': INK, 'axes.labelcolor': MUTED, 'axes.edgecolor': GRID,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'grid.color': GRID,
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'axes.titlepad': 13, 'legend.fontsize': 8,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _label(name) -> str:
    name = str(name)
    return LABELS.get(name, name.replace('_', ' '))


def _num(value, digits=3):
    return f'{float(value):.{digits}f}' if pd.notna(value) and np.isfinite(float(value)) else 'unavailable'


def _pct(value):
    return f'{float(value):.2%}' if pd.notna(value) and np.isfinite(float(value)) else 'unavailable'


def _date(value):
    try:
        return pd.Timestamp(value).strftime('%Y-%m-%d') if value is not None else 'unavailable'
    except (TypeError, ValueError):
        return 'unavailable'


def _empty(ax, text='No recorded result'):
    ax.text(.5, .5, text, transform=ax.transAxes, ha='center', va='center', color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])


def _tidy(ax):
    ax.grid(axis='y', alpha=.65, linewidth=.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=7)
    if ax.get_yscale() == 'log':
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1,2,5)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(MaxNLocator(5))


def _legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, labelcolor=INK, **kwargs)


def _title(fig, title, subtitle, synthetic, halt_note=None):
    fig.suptitle(title, x=.065, y=.965, ha='left', fontsize=23, fontweight='bold')
    fig.text(.065, .917, subtitle, color=MUTED, fontsize=10)
    if synthetic:
        fig.text(.945, .962, 'SYNTHETIC EXAMPLE', ha='right', va='top', fontsize=10,
                 color=ORANGE, fontweight='bold')
        fig.text(.51, .49, 'SYNTHETIC DATA', ha='center', va='center', fontsize=48,
                 color=INK, alpha=.07, rotation=20, zorder=20)
    if halt_note:
        fig.text(.065, .055, halt_note, color=ORANGE, fontsize=8)
    fig.text(.065, .025, 'Development evaluation  /  No automatic promotion  /  Source: saved run artifacts',
             color=MUTED, fontsize=8)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(path.with_suffix('.'+suffix), dpi=180, bbox_inches='tight', pad_inches=.2)
    plt.close(fig)


def _strategy_names(frame):
    names = [name for name in CORE if name in frame.columns]
    return names or [c for c in frame.columns if c != 'Date'][:5]


def _equity(ax, returns, drawdown=False):
    ax.set_title('Drawdown from prior peak' if drawdown else 'Net wealth · initial = 100 · log scale', loc='left')
    if returns.empty or 'Date' not in returns:
        return _empty(ax)
    dates = pd.to_datetime(returns.Date)
    for color, name in zip(COLORS, _strategy_names(returns)):
        series = pd.to_numeric(returns[name], errors='coerce')
        if series.isna().any():
            continue  # Do not replace unknown returns with zero.
        equity = (1+series).cumprod()
        if not drawdown and (equity <= 0).any():
            raise ValueError(f'Log-wealth chart requires positive saved wealth: {name}')
        # Include initial capital when finding the running peak, so entry costs count.
        values = equity / equity.cummax().clip(lower=1)-1 if drawdown else 100*equity
        ax.plot(dates, values, label=_label(name), color=color, linewidth=1.7,
                linestyle='--' if name == 'buy_hold' else '-')
    if drawdown:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    else:
        ax.set_yscale('log')
        ax.axhline(100, color=MUTED, lw=.6, alpha=.5)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    _legend(ax, loc='best', ncol=1)
    _tidy(ax)


def _brier(ax, scores):
    ax.set_title('Probability error · Brier, lower is better', loc='left')
    needed = {'model', 'mapping', 'brier'}
    if scores.empty or not needed.issubset(scores):
        return _empty(ax)
    models = sorted(scores.model.unique())
    x = np.arange(len(models))
    for offset, mapping, color in [(-.24, 'raw', BLUE), (0, 'calibrated', TEAL), (.24, 'base_rate', ORANGE)]:
        rows = scores[scores.mapping == mapping].drop_duplicates('model').set_index('model')
        values = rows.brier.reindex(models)
        ax.bar(x+offset, values, width=.22, color=color, label={'raw':'Raw', 'calibrated':'Calibrated', 'base_rate':'Training base rate'}[mapping])
    ax.set_xticks(x, [_label(m) for m in models], fontsize=8)
    upper = pd.to_numeric(scores.brier, errors='coerce').max()
    ax.set_ylim(0, upper*1.32 if np.isfinite(upper) and upper>0 else 1)
    _legend(ax, loc='upper right', ncol=1)
    _tidy(ax)


def _reliability(ax, data):
    ax.set_title('Reliability · calibrated probabilities', loc='left')
    if data.empty or not {'model','mapping','predicted','observed','n'}.issubset(data):
        return _empty(ax)
    ax.plot([0,1], [0,1], '--', color=MUTED, lw=1, label='Perfect calibration')
    for color, (name, group) in zip(COLORS, data[data.mapping == 'calibrated'].groupby('model')):
        color = {'rule':TEAL,'logistic':ORANGE,'boosting':BLUE}.get(name,color)
        group = group.sort_values('predicted')
        ax.plot(group.predicted, group.observed, color=color, lw=1.3, label=_label(name))
        ax.scatter(group.predicted, group.observed, s=18+np.sqrt(group.n)*5, color=color, alpha=.9, zorder=3)
    ax.set(xlim=(0,1), ylim=(0,1), xlabel='Mean predicted probability', ylabel='Observed up frequency')
    ax.text(.98,.04,'Marker size reflects bin count',ha='right',transform=ax.transAxes,color=MUTED,fontsize=7)
    _legend(ax, loc='upper left')
    _tidy(ax)


def _dist_selection(data):
    if data.empty or not {'model','design','horizon','level','coverage'}.issubset(data):
        return pd.DataFrame()
    selected = data[data.design == 'nonoverlap']
    if selected.empty:
        return pd.DataFrame()  # Never silently substitute overlapping outcomes.
    horizon = 21 if 21 in selected.horizon.values else int(selected.horizon.max())
    level = .95 if np.isclose(selected.level, .95).any() else float(selected.level.max())
    return selected[(selected.horizon == horizon) & np.isclose(selected.level, level)].sort_values('model')


def _coverage(ax, data):
    rows = _dist_selection(data)
    ax.set_title('Terminal-return interval coverage', loc='left')
    if rows.empty:
        return _empty(ax, 'No nonoverlapping distribution windows')
    x = np.arange(len(rows))
    uncertainty = pd.to_numeric(rows.get('coverage_hac_se', pd.Series(np.nan,index=rows.index)), errors='coerce')*1.96
    ax.axhline(rows.level.iloc[0], linestyle='--', color=ORANGE, lw=1.2,
               label=f'{rows.level.iloc[0]:.0%} nominal')
    ax.errorbar(x, rows.coverage, yerr=uncertainty, fmt='o', color=TEAL, ecolor=TEAL, capsize=4, ms=6,
                label='Observed ± 1.96 HAC SE')
    ax.set_xticks(x, [_label(m).replace(' ', '\n', 1) for m in rows.model], fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    low = min(float((rows.coverage-uncertainty.fillna(0)).min())-.03, .75)
    ax.set_ylim(max(-.05,low), max(1.03,float((rows.coverage+uncertainty.fillna(0)).max())+.03))
    n = rows.n
    count = str(int(n.min())) if n.min()==n.max() else f'{int(n.min())}–{int(n.max())}'
    ax.set_xlabel(f'{int(rows.horizon.iloc[0])}-session horizon · nonoverlapping · n = {count}', fontsize=8)
    _legend(ax, loc='lower left')
    _tidy(ax)


def _interval_score(ax, data):
    rows = _dist_selection(data)
    ax.set_title('Interval score · lower is better', loc='left')
    if rows.empty or 'interval_score' not in rows:
        return _empty(ax)
    y = np.arange(len(rows))
    ax.barh(y, rows.interval_score, color=[TEAL if str(name).endswith('ewma') else BLUE for name in rows.model], height=.65)
    ax.set_yticks(y, [_label(m) for m in rows.model], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f'{rows.level.iloc[0]:.0%} interval · {int(rows.horizon.iloc[0])} sessions · return units',fontsize=8)
    ax.grid(axis='x', alpha=.5)
    ax.set_axisbelow(True)


def _cost(ax, data):
    ax.set_title('Cost sensitivity · change versus 1×', loc='left')
    if data.empty or not {'model','cost_multiplier','total_return'}.issubset(data):
        return _empty(ax)
    names = [m for m in CORE if m in data.model.values]
    for color, name in zip(COLORS,names):
        rows = data[data.model == name].sort_values('cost_multiplier')
        baseline = rows.loc[rows.cost_multiplier == 1, 'total_return']
        if len(baseline) != 1:
            continue
        change = (rows.total_return - baseline.iloc[0])*100
        ax.plot(rows.cost_multiplier, change, color=color, marker='o', markersize=4, label=_label(name))
    ax.axhline(0,color=MUTED,lw=.7)
    ax.set_xticks(sorted(data.cost_multiplier.unique()), [f'{x:g}×' for x in sorted(data.cost_multiplier.unique())])
    ax.set_xlabel('Commission + slippage multiplier', fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g} pp'))
    _legend(ax,loc='best')
    _tidy(ax)


def _neighbor(ax, data, model):
    ax.set_title(_label(model), loc='left')
    rows = data[data.model == model]
    grid = rows.pivot(index='stop_multiplier', columns='take_multiplier',values='total_return').sort_index(ascending=False)
    values = grid.to_numpy()
    bound = max(float(pd.to_numeric(data.total_return,errors='coerce').abs().max()),.001)
    image = ax.imshow(values,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(len(grid.columns)), [f'{x:g}×' for x in grid.columns])
    ax.set_yticks(range(len(grid.index)), [f'{x:g}×' for x in grid.index])
    ax.set(xlabel='Take-profit multiplier',ylabel='Stop-loss multiplier')
    for i in range(len(grid)):
        for j in range(len(grid.columns)):
            if np.isfinite(values[i,j]):
                ax.text(j,i,f'{values[i,j]:.1%}',ha='center',va='center',color='#07101d' if abs(values[i,j])<bound*.5 else 'white',fontsize=11,fontweight='bold')
    return image


def _replay_comparison(ax, replay):
    ax.set_title('Frozen strict signals · execution sensitivity', loc='left')
    if replay.empty or not {'family','variant','lookback_months','annualized_return'}.issubset(replay):
        return _empty(ax)
    rows = replay[replay.family == 'strict_baseline']
    periods = sorted(rows.lookback_months.unique())
    if not periods:
        return _empty(ax)
    labels = [('legacy_close_daily_reset','Legacy close / daily reset',BLUE),
              ('corrected_close_daily_reset','Corrected cash + gap fills',TEAL),
              ('corrected_next_open_entry','Next open + entry stop',ORANGE)]
    x=np.arange(len(periods))
    for offset,(variant,label,color) in zip([-.25,0,.25],labels):
        values=rows[rows.variant == variant].set_index('lookback_months').annualized_return.reindex(periods)
        ax.bar(x+offset,values,width=.23,color=color,label=label)
    ax.set_xticks(x,[f'{int(m)} months' for m in periods])
    ax.axhline(0,color=MUTED,lw=.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    low,high=rows.annualized_return.min(),rows.annualized_return.max()
    span=max(high-low,.1)
    ax.set_ylim(min(0,low)-span*.15,max(0,high)+span*.55)
    ax.set_xlabel('Annualized return · next-open case also changes stop semantics',fontsize=8)
    _legend(ax,loc='upper right')
    _tidy(ax)


def _markdown_table(data, columns, limit=30):
    if data.empty or not set(columns).issubset(data):
        return '_No recorded rows._'
    names = list(columns.values())
    lines = ['| '+' | '.join(names)+' |', '| '+' | '.join(['---']*len(names))+' |']
    for _,row in data.head(limit).iterrows():
        values = []
        for col in columns:
            value = row[col]
            if pd.isna(value): value = 'unavailable'
            elif col in {'total_return','annualized_return','max_drawdown','mean_abs_exposure','coverage','level'}: value = _pct(value)
            elif col in {'n','horizon','lookback_months'}: value = str(int(value))
            elif isinstance(value,(int,float,np.number)): value = _num(value)
            else: value = str(value)
            values.append(str(value).replace('|','/'))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def _write_report(output, tables, manifest, subtitle, synthetic):
    summary = tables['strategy_summary']
    scores = tables['probability_scores']
    distribution = _dist_selection(tables['distribution_summary'])
    news = _json(output/'news_eligibility.json')
    config = manifest.get('config',{})
    lines = ['# NVDA daily research: evidence and limits', '',
        '**Synthetic correctness example. These prices are simulated and do not demonstrate investment performance.**' if synthetic else '**Historical development evaluation. Previously researched data are not an untouched holdout.**', '',
        subtitle+'.', '',
        'This run compares a volume/momentum rule, logistic regression and shallow boosting under one daily research design. '
        'Probability mappings use chronological inner blocks; net outcomes use the saved execution and cost configuration. '
        'The charts show prespecified model identities, not the best-performing subset. Full variants, including losses and controls, remain in the CSV tables.', '',
        f"Input dates: {_date(manifest.get('data_first'))} to {_date(manifest.get('data_last'))}; rows: {manifest.get('data_rows','unavailable')}. "
        f"Evaluation status: `{manifest.get('holdout_status',config.get('evaluation_status','unavailable'))}`. "
        f"Prediction target: `{manifest.get('prediction_target','unavailable')}`.", '',
        '## Net strategy outcomes', '',
        _markdown_table(summary, {'model':'Strategy','total_return':'Net total return','annualized_return':'Annualized return','sharpe_ratio':'Sharpe','max_drawdown':'Max drawdown','mean_abs_exposure':'Mean exposure'}), '']
    if not summary.empty and 'halted' in summary:
        halted = summary[summary.halted.astype(str).str.lower() == 'true'].model.tolist()
        if halted:
            lines += ['The drawdown guard halted these variants during evaluation: '+', '.join(f'`{m}`' for m in halted)+'. Their later cash exposure is part of the reported outcome.', '']
    if not summary.empty and {'total_return','model'}.issubset(summary):
        positive = int((summary.total_return>0).sum())
        lines += [f'{positive} of {len(summary)} saved strategy variants have positive net total return at baseline costs. '
                  'The variants share data and are not independent replications.', '']
    stress=tables['cost_stress']
    if not stress.empty and {'cost_multiplier','total_return'}.issubset(stress):
        largest=stress.cost_multiplier.max()
        stressed=stress[stress.cost_multiplier == largest]
        lines += [f'At {largest:g}× recorded costs, {int((stressed.total_return>0).sum())} of {len(stressed)} variants remain net positive. '
                  'Cost sensitivity is evaluated on the same frozen signals; it does not include an order-book capacity model.', '']
    uncertainty = tables['block_uncertainty']
    if not uncertainty.empty and {'low','high'}.issubset(uncertainty):
        valid = uncertainty.dropna(subset=['low','high'])
        crossing = int(((valid.low <= 0)&(valid.high >= 0)).sum())
        lines += [f'Of {len(valid)} recorded return-difference intervals versus buy-and-hold, {crossing} include zero. '
                  'These are descriptive circular-block intervals for mean daily net-return differences, not selection-adjusted tests. '
                  'Annualization and a short favorable window do not establish a durable edge.', '']
    lines += ['The exposure controls use training-estimated exposure; they do not guarantee identical realized risk. '
              'Use `cost_stress.csv`, `parameter_neighborhood.csv` and `stratified_results.csv` to inspect costs, nearby risk settings and annual/volatility-state variation.', '',
              '## Probability evidence', '',
              _markdown_table(scores, {'model':'Model','mapping':'Mapping','brier':'Brier','log_loss':'Log loss','n':'Predictions'}), '']
    if not scores.empty and {'model','mapping','brier'}.issubset(scores):
        for model, group in scores.groupby('model'):
            group = group.drop_duplicates('mapping').set_index('mapping')
            if {'calibrated','raw','base_rate'}.issubset(group.index):
                calibrated, raw, base = (group.loc[m,'brier'] for m in ['calibrated','raw','base_rate'])
                if all(pd.notna(x) for x in [calibrated,raw,base]):
                    lines.append(f'- {_label(model)}: selected calibration Brier {_num(calibrated,4)}; raw {_num(raw,4)}; training base rate {_num(base,4)}. '
                                 f'Calibration {"reduced" if calibrated < raw else "did not reduce"} observed error versus raw; '
                                 f'it {"beat" if calibrated < base else "did not beat"} the base-rate score in this run.')
        lines += ['', 'Reliability markers show bin averages and counts. Sparse bins and the forward calibration/refit transfer limit inference. '
                  'The uncertainty table is descriptive; calibrated labels alone do not justify greater exposure.', '']
    probability_uncertainty=tables['probability_uncertainty']
    if not probability_uncertainty.empty and {'mapping','low','high'}.issubset(probability_uncertainty):
        intervals=probability_uncertainty[probability_uncertainty.mapping=='calibrated'].dropna(subset=['low','high'])
        below=int((intervals.high<0).sum())
        lines += [f'{below} of {len(intervals)} recorded calibrated-minus-base-rate Brier intervals lie entirely below zero. '
                  'This is a descriptive block-bootstrap comparison; shared histories and model selection prevent treating these as independent confirmatory tests.', '']
    lines += ['## Distribution and tail evidence', '',
              _markdown_table(distribution, {'model':'Distribution','horizon':'Sessions','level':'Nominal level','coverage':'Coverage','interval_score':'Interval score','n':'Windows'}), '',
              'The primary panels use nonoverlapping terminal-return outcomes; overlapping windows are a supplement in the saved tables. '
              'Coverage whiskers use 1.96 times the recorded HAC standard error. They are descriptive and are not a proof of correct calibration. '
              'Interval score penalizes both wide intervals and missed outcomes; inspect width, left/right misses and PIT together. '
              'Terminal-return coverage is not a stop-loss-hit or maximum-drawdown forecast.', '',
              '## News eligibility', '',
              f"Recorded status: `{news.get('status','unavailable')}`; events: {news.get('events','unavailable')}. "
              'No news alpha claim is made. Eligibility is a minimum provenance/sample gate, not proof of statistical power.']
    reasons = news.get('reasons', [news.get('reason')] if news.get('reason') else [])
    lines += ['']
    lines += ['- '+str(reason) for reason in reasons]
    lines += ['', '## Reproducibility and decision', '',
              f"Input SHA-256: `{manifest.get('input_sha256','unavailable')}`. Source digest: `{manifest.get('source_sha256','unavailable')}`. "
              'Where produced, the saved configuration, folds, predictions, trade/fill ledgers, account reconciliation and dependency records support this report.', '',
              f"Price basis: {manifest.get('price_adjustment','Adjustment provenance unavailable; verify before economic interpretation.')}. "
              f"Decision clock: {manifest.get('decision_clock','unavailable')}. Execution clock: {manifest.get('execution_clock','unavailable')}.", '',
              'The historical model search makes reuse of these periods development evidence. '
              'There is no automatic strategy promotion. Negative results remain part of the research record; '
              'any future deployment decision requires prospective predictions, stable economic evidence and a separate review.']
    if (output/'historical_replay').exists():
        files = sorted((output/'historical_replay').glob('*summary*.csv'))
        if (output/'historical_replay'/'historical_replay.csv').exists():
            files.append(output/'historical_replay'/'historical_replay.csv')
        lines += ['', 'Historical replay artifacts are retained in `historical_replay/`. '
                  'Legacy, corrected-close and next-open/entry-reference cases must retain separate identities; '
                  'execution and strategy-semantic changes must not be presented as a single pure bug fix.']
        for file in files:
            replay = _read(file)
            selected = {c:c.replace('_',' ').title() for c in ['family','lookback_months','variant','legacy_reference_status','total_return','annualized_return','max_drawdown'] if c in replay}
            if len(selected)>1:
                lines += ['', f'Saved comparison: `{file.relative_to(output)}`.', '', _markdown_table(replay,selected)]
    missing=[name for name,table in tables.items() if table.empty]
    if missing:
        lines += ['', 'At export, these result tables have no recorded rows: '+', '.join(f'`{name}.csv`' for name in missing)+'. Missing evidence is not a zero score.']
    (output/'research_report.md').write_text('\n'.join(lines)+'\n')


def build_report(output_dir: Path):
    """Export figures/report from a complete or partially written run directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _json(output/'manifest.json')
    names = ['strategy_summary','cost_stress','parameter_neighborhood','stratified_results','daily_returns',
             'block_uncertainty','probability_scores','reliability','probability_uncertainty','distribution_summary','distribution_windows']
    tables = {name:_read(output/(name+'.csv')) for name in names}
    synthetic = bool(manifest.get('synthetic',False))
    returns = tables['daily_returns']
    if not returns.empty and 'Date' in returns:
        dates = pd.to_datetime(returns.Date)
        date_text = f'{dates.min():%Y-%m-%d} — {dates.max():%Y-%m-%d}'
    else:
        date_text = f"{_date(manifest.get('data_first'))} — {_date(manifest.get('data_last'))}"
    status_text='Simulated prices · correctness demonstration' if synthetic else 'Frozen historical inputs · development evaluation'
    subtitle=f"Input: {_date(manifest.get('data_first'))} — {_date(manifest.get('data_last'))}  |  "+status_text
    trading_subtitle='Strategy evaluation: '+date_text+'  |  '+status_text
    distribution_subtitle=subtitle
    windows=tables['distribution_windows']
    selected_dist=_dist_selection(tables['distribution_summary'])
    if not windows.empty and not selected_dist.empty and {'as_of','label_end','design','horizon'}.issubset(windows):
        windows=windows[(windows.design=='nonoverlap') & (windows.horizon==selected_dist.horizon.iloc[0])]
        if not windows.empty:
            distribution_subtitle=f"Forecast outcomes: {_date(pd.to_datetime(windows.as_of).min())} — {_date(pd.to_datetime(windows.label_end).max())}  |  "+status_text
    replay=_read(output/'historical_replay'/'historical_replay.csv')
    composite_subtitle=subtitle.replace('Input:', 'Input dates:')+'  |  Strategy folds: '+date_text
    halt_note=None
    summary=tables['strategy_summary']
    if not summary.empty and {'model','halted'}.issubset(summary):
        displayed=summary[summary.model.isin(_strategy_names(returns))]
        halted=displayed.halted.astype(str).str.lower().eq('true')
        if halted.any():
            limit=manifest.get('config',{}).get('max_drawdown_limit')
            guard=f'{float(limit):.0%} drawdown guard' if limit is not None else 'configured drawdown guard'
            halt_note=f'{int(halted.sum())} of {len(displayed)} displayed variants reached the {guard}; later flat periods represent cash after a risk halt.'
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(2,3,figsize=(18,11))
        fig.subplots_adjust(left=.065,right=.97,top=.83,bottom=.11,wspace=.3,hspace=.5)
        _title(fig,'NVDA | daily research audit',composite_subtitle,synthetic,halt_note)
        _equity(axes[0,0],returns)
        _equity(axes[0,1],returns,True)
        _cost(axes[0,2],tables['cost_stress'])
        _brier(axes[1,0],tables['probability_scores'])
        _reliability(axes[1,1],tables['reliability'])
        _coverage(axes[1,2],tables['distribution_summary'])
        _save(fig,output/'overview')

        for name,plot,title in [('equity',lambda ax:_equity(ax,returns),'Net wealth'),
            ('drawdown',lambda ax:_equity(ax,returns,True),'Drawdown'),
            ('cost_sensitivity',lambda ax:_cost(ax,tables['cost_stress']),'Cost sensitivity')]:
            fig,ax=plt.subplots(figsize=(12,7))
            fig.subplots_adjust(left=.1,right=.95,top=.8,bottom=.15)
            _title(fig,'NVDA | '+title,trading_subtitle,synthetic,halt_note)
            plot(ax)
            _save(fig,output/'figures'/name)

        fig,axes=plt.subplots(1,2,figsize=(15,7))
        fig.subplots_adjust(left=.08,right=.96,top=.8,bottom=.15,wspace=.3)
        _title(fig,'NVDA | probability evidence',subtitle,synthetic)
        _reliability(axes[0],tables['reliability'])
        _brier(axes[1],tables['probability_scores'])
        _save(fig,output/'figures'/'probability_reliability')

        fig,axes=plt.subplots(1,2,figsize=(15,7))
        fig.subplots_adjust(left=.08,right=.96,top=.8,bottom=.2,wspace=.4)
        _title(fig,'NVDA | distribution evidence',distribution_subtitle,synthetic)
        _coverage(axes[0],tables['distribution_summary'])
        _interval_score(axes[1],tables['distribution_summary'])
        _save(fig,output/'figures'/'distribution_coverage_score')

        neighbors=tables['parameter_neighborhood']
        models=sorted(neighbors.model.unique())[:4] if not neighbors.empty and 'model' in neighbors else []
        fig,axes=plt.subplots(2,2,figsize=(13,10))
        fig.subplots_adjust(left=.1,right=.94,top=.8,bottom=.12,wspace=.3,hspace=.6)
        _title(fig,'NVDA | local risk-parameter sensitivity',subtitle+' · cells = net total return',synthetic)
        for i,ax in enumerate(axes.flat):
            if i<len(models): _neighbor(ax,neighbors,models[i])
            else: _empty(ax)
        _save(fig,output/'figures'/'parameter_neighborhood')

        fig,axes=plt.subplots(2,2,figsize=(16,10))
        fig.subplots_adjust(left=.075,right=.96,top=.81,bottom=.12,wspace=.33,hspace=.5)
        _title(fig,'NVDA | Auditable quantitative research',composite_subtitle,synthetic,halt_note)
        _equity(axes[0,0],returns)
        _brier(axes[0,1],tables['probability_scores'])
        _cost(axes[1,0],tables['cost_stress'])
        if not replay.empty and not synthetic:
            _replay_comparison(axes[1,1],replay)
        else:
            _interval_score(axes[1,1],tables['distribution_summary'])
        _save(fig,output/'linkedin_main')
        if not replay.empty:
            fig,ax=plt.subplots(figsize=(13,7))
            fig.subplots_adjust(left=.1,right=.95,top=.8,bottom=.17)
            replay_start=_date(pd.to_datetime(replay.date_start).min()) if 'date_start' in replay else 'unavailable'
            replay_end=_date(pd.to_datetime(replay.date_end).max()) if 'date_end' in replay else 'unavailable'
            _title(fig,'NVDA | historical replay',f'{replay_start} — {replay_end}  |  Frozen signals, not model refitting',False)
            _replay_comparison(ax,replay)
            _save(fig,output/'figures'/'historical_replay')
    _write_report(output,tables,manifest,subtitle,synthetic)
    return {'overview':str(output/'overview.png'),'linkedin_main':str(output/'linkedin_main.png'),
            'report':str(output/'research_report.md')}
