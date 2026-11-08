from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def settings():
    from policylens.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def index(settings):
    """A built index, shared across the session.

    Building costs a few seconds and every retrieval test needs the same one, so it
    is built once rather than per test.
    """
    from policylens.indexer import load_or_build_index

    return load_or_build_index(settings)


@pytest.fixture(scope="session")
def agent(index, settings):
    from policylens.agent.graph import build_agent

    return build_agent(index, settings)
