"""Pytest configuration for knowledge base tests."""

def pytest_addoption(parser):
    parser.addoption("--fast", action="store_true", default=True, help="Use smaller model for faster tests")
