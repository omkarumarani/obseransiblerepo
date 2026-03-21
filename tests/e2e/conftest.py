# tests/e2e/conftest.py
# Minimal pytest configuration for e2e tests.
# Marks all tests in this directory with the 'e2e' marker so you can
# exclude them from unit test runs with:  pytest -m "not e2e"
import pytest

def pytest_collection_modifyitems(items):
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
