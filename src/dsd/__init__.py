from importlib.metadata import PackageNotFoundError, version

from loguru import logger

try:
    __version__ = version(__package__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

logger.disable(__package__)
