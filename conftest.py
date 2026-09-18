"""Pytest path bootstrap for the local multi-tool workspace."""

from __future__ import annotations

import sys
import os
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def offline_test_network_guard(monkeypatch):
    """CI test suite cannot accidentally refresh prices or contact a live API."""
    if os.environ.get('NVDA_OFFLINE_TESTS') != '1':
        return
    import socket
    def denied(*args, **kwargs):
        raise RuntimeError('Network is disabled in offline research tests')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
