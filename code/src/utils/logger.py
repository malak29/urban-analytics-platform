import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from ..config.settings import settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure production logging system
    
    Args:
        log_level: Override default log level
    """
    
    # Create logs directory
    settings.logging.LOG_FILE.parent.mkdir(exist_ok=True, parents=True)
    
    # Configure root logger
    level = getattr(logging, log_level or settings.logging.LOG_LEVEL)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # Create handlers
    handlers = []
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.logging.LOG_FILE,
        maxBytes=settings.logging.MAX_LOG_SIZE,
        backupCount=settings.logging.BACKUP_COUNT
    )
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)
    handlers.append(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(level)
    handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers,
        force=True
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}, file={settings.logging.LOG_FILE}")


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance for module
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    
    return logging.getLogger(name)