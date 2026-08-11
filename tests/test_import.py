"""Package import tests."""

import ezqens


def test_package_import() -> None:
    """The package can be imported and exposes its version."""
    assert ezqens.__version__ == "0.1.0"
