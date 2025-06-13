import mammos_ai


def test_version():
    """Check that __version__ exists and is a string."""
    assert isinstance(mammos_ai.__version__, str)
