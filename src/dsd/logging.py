import datetime
from sys import stderr

from loguru import logger


def init_logger(verbose: int, save_log: bool = True, msg_format: str | None = None) -> None:
    logger.enable("dsd")
    timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo

    try:
        from IPython import get_ipython

        in_notebook = get_ipython() is not None
    except ImportError:
        in_notebook = False

    if msg_format is None:
        if in_notebook:
            msg_format = "<green>{name}</green>:<red>{function}</red>:<blue>{line}</blue>·-·<level>{message}</level>"
        else:
            msg_format = "{time:YYYY-MM-DD at HH:mm:ss} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>·-·<level>{message}</level>"

    match verbose:
        case 3:
            log_level = "DEBUG"  # catches DEBUG, INFO, WARNING, ERROR, and CRITICAL
        case 2:
            log_level = "INFO"  # catches INFO, WARNING, ERROR, and CRITICAL
        case 1:
            log_level = "WARNING"  # catches WARNING, ERROR, and CRITICAL
        case _:
            log_level = "ERROR"  # Catches ERROR and CRITICAL

    logger.add(
        sink=stderr,
        format=msg_format,
        level=log_level,
        filter="dsd",
        colorize=True
    )
    # config = {
    #     "handlers": [
    #         {"sink": stderr, "format": msg_format, "level": log_level, "filter": "dsd"},
    #     ]
    # }

    if save_log:
        # config["handlers"].append(
        #     {
        #         "sink": f"dsd_{datetime.datetime.now(tz=timezone).strftime('%Y-%d-%m--%H-%M-%S')}.log",
        #         "level": "DEBUG",
        #         "format": "{time:YYYY-MM-DD at HH:mm:ss} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>·-·<level>{message}</level>",
        #         "filter": "dsd",
        #         "backtrace": True,
        #         "diagnose": True,
        #     }
        # )
        logger.add(
            sink=f"dsd_{datetime.datetime.now(tz=timezone).strftime('%Y-%d-%m--%H-%M-%S')}.log",
            level=log_level,
            format="{time:YYYY-MM-DD at HH:mm:ss} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>·-·<level>{message}</level>",
            filter="dsd",
            backtrace=True,
            diagnose=True,
            colorize=True
        )

    # logger.configure(**config)
