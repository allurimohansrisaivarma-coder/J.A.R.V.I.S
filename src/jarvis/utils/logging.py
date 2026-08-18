"""Logging configuration and utilities using structlog."""

import logging
import logging.handlers
import sys

import structlog

from jarvis import __version__
from jarvis.config.settings import LoggingSettings


def add_app_version(
    logger: structlog.types.BindableLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Add application version to log events."""
    event_dict["app_version"] = __version__
    return event_dict


def setup_logging(settings: LoggingSettings) -> None:
    """Set up the structlog configuration and stdlib handlers.
    
    Args:
        settings: Logging settings configuration instance.
    """
    log_dir = settings.file.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Standard logging configuration
    stdlib_level = getattr(logging, settings.level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_app_version,
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]

    # File Handler (JSON)
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.file,
        maxBytes=settings.max_size_mb * 1024 * 1024,
        backupCount=settings.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)

    # Console Handler (JSON or Console based on settings)
    console_renderer = (
        structlog.dev.ConsoleRenderer(colors=True)
        if settings.format != "json"
        else structlog.processors.JSONRenderer()
    )
    
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            console_renderer,
        ],
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(stdlib_level)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a bound structlog logger.
    
    Args:
        name: Name of the logger, typically __name__.
        
    Returns:
        A structlog bound logger instance.
    """
    return structlog.get_logger(name)
