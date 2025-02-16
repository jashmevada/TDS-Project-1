import logging
import logging.config

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
fileHandler = logging.FileHandler("fastAPI.log") 

# logger.addFilter(console_handler)

# logging.basicConfig(
#     level=logging.WARNING,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         fileHandler,
#         console_handler
#     ]
# )

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "[%(levelname)s] - %(asctime)s - %(name)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
            "filters": ["sensitive_data_filter"],
        },
        "file": {
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "filename": "my_log.log",
            "mode": "a",
        },
    },
    "loggers": {
        "my_customer_logger": {
            "handlers": ["file", "console"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}

# logging.config.dictConfig(LOGGING_CONFIG)