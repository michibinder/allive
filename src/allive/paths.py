import os
from pathlib import Path

DATA_ENV = "ALLIVE_DATA"
CHECKOUT = Path(__file__).resolve().parents[2]


def data_dir():
    """The data directory: $ALLIVE_DATA, else data/ of a source checkout, else ~/allive-data."""
    if os.environ.get(DATA_ENV):
        return Path(os.environ[DATA_ENV]).expanduser()
    if (CHECKOUT / "pyproject.toml").is_file():
        return CHECKOUT / "data"
    return Path.home() / "allive-data"


def figure_dir():
    """Directory of the figures."""
    return data_dir() / "figures"


def archive_dir():
    """Directory of the archived raw ALIMA files."""
    return data_dir() / "alima"


def ifs_dir():
    """Directory of the IFS downloads."""
    return data_dir() / "ifs"
