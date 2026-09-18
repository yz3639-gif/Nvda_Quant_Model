from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_json(value):
    """Strict JSON: unsupported/NaN results are null, never fabricated zeros."""
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(x) for x in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    return value


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(safe_json(data), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def code_state(root: Path) -> dict:
    def git(*args):
        p = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True)
        return p.stdout.strip() if p.returncode == 0 else None
    files = {}
    for directory in ['nvda_quant_model', 'methods', 'utils']:
        for path in sorted((root / directory).rglob('*.py')):
            files[str(path.relative_to(root))] = sha256(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'commit': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
            'dirty_status': git('status', '--short'), 'source_sha256': digest, 'files': files}


def dependencies() -> dict:
    result = {'python': platform.python_version()}
    result.update({d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')})
    return result


def directory_hash(path: Path | None, pattern='*') -> str | None:
    if path is None:
        return None
    files = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob(pattern)) if p.is_file()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def block_mean_interval(values, repetitions=1000, block=21, seed=42) -> dict:
    """Circular block bootstrap of a mean; descriptive, not search-adjusted."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return {'mean': float(x.mean()) if len(x) else None, 'low': None, 'high': None, 'n': len(x)}
    if len(x) <= block:
        return {'mean': float(x.mean()), 'low': None, 'high': None, 'n': len(x),
                'block': block, 'inference': 'insufficient_effective_blocks'}
    block = max(1, min(int(block), len(x)))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), (repetitions, int(np.ceil(len(x) / block))))
    indices = (starts[..., None] + np.arange(block)) % len(x)
    samples = x[indices.reshape(repetitions, -1)[:, :len(x)]].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return {'mean': float(x.mean()), 'low': float(low), 'high': float(high), 'n': len(x),
            'block': block, 'repetitions': repetitions, 'inference': 'descriptive_not_selection_adjusted'}
