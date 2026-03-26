"""Basic tests to ensure the package is properly installed."""

import importlib


def test_package_import() -> None:
    """Test that the package can be imported."""
    package_name = "recipe_curator"
    module = importlib.import_module(package_name)
    assert module is not None


def test_version_exists() -> None:
    """Test that the package has a version attribute."""
    package_name = "recipe_curator"
    module = importlib.import_module(package_name)
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)
