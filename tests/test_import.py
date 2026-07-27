"""Package import tests."""

import qensfit


def test_package_import() -> None:
    """The package can be imported and exposes its version."""
    assert qensfit.__version__ == "0.1.0"
