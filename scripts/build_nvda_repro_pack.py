"""Build a small offline research package without personal caches or credentials."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path):
    files = set()
    for directory in ['nvda_quant_model', 'methods', 'utils', 'data']:
        files.update(p for p in (ROOT/directory).rglob('*.py') if '__pycache__' not in p.parts and 'cache' not in p.parts and 'outputs' not in p.parts)
    files.update(ROOT/p for p in ['README.md', 'conftest.py', 'pytest.ini', 'requirements-research.txt'])
    for directory in ['research/configs', 'research/sample']:
        files.update(p for p in (ROOT/directory).rglob('*') if p.is_file())
    files.update((ROOT/'research').glob('*.md'))
    files.add(ROOT/'research/environment.lock.txt')
    files.add(ROOT/'research/REPRODUCTION_CHECK.json')
    files.add(ROOT/'research/LINKEDIN_DESCRIPTION.txt')
    files.add(ROOT/'docs/LEGACY_REPOSITORY_README.md')
    files.update((ROOT/'tests').glob('test_*_v2.py'))
    files.add(Path(__file__).resolve())
    # Market outputs are research evidence, not a redistributable vendor feed.
    # Include compact result tables and figures; omit raw OHLCV and original caches.
    result = ROOT/'research/results/frozen_20260918'
    for pattern in ['*.csv', '*.md', 'manifest.json', 'news_eligibility.json', 'linkedin_main.png', 'overview.png']:
        files.update(result.glob(pattern))
    files.update((result/'historical_replay').glob('historical_replay.csv'))
    records = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for p in sorted(files):
            archive.write(p, 'nvda-research-v2/'+str(p.relative_to(ROOT)))
        archive.writestr('nvda-research-v2/PACKAGE_MANIFEST.json', json.dumps({'files': records, 'market_cache_included': False,
            'purpose': 'Synthetic offline replication plus compact market research evidence. Full market run manifests also reference artifacts retained in the source repository.'}, indent=2)+'\n')
    return {'path': str(destination.resolve()), 'files': len(records), 'bytes': destination.stat().st_size,
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
