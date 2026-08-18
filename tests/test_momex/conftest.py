"""Shared fixtures for the Momex tests.

Keeps unit tests independent of the machine they run on.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_from_ambient_dotenv(tmp_path_factory, monkeypatch):
    """Run each test from a directory with no .env in it.

    Momex resolves .env relative to the working directory, so a developer with
    real credentials in the repo root would otherwise have them leak into any
    test that builds configuration from the environment -- making results
    depend on whether the machine happens to be set up for online use. Tests
    that want a .env create one and chdir to it themselves.
    """
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
