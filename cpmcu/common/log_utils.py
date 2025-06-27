import logging
import time
from contextlib import contextmanager
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
import typing

# Define custom log levels
SUCCESS = 25
STAGE = 26
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(STAGE, "STAGE")

class CpmcuLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        self._current_stage = None
        self._stage_start_time = None
        
    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, message, args, **kws)
    
    def stage(self, message, *args, **kws):
        """Log a new stage/phase"""
        if self.isEnabledFor(STAGE):
            self._log(STAGE, message, args, **kws)
        self._current_stage = message
        self._stage_start_time = time.time()
    
    def stage_complete(self, message=None):
        """Complete current stage with timing"""
        if self._current_stage and self._stage_start_time:
            elapsed = time.time() - self._stage_start_time
            msg = message or f"{self._current_stage} completed"
            plain_msg = msg.replace("[dim]", "").replace("[/dim]", "")
            self.success(f"{plain_msg} ({elapsed:.2f}s)")
            self._current_stage = None
            self._stage_start_time = None

# Set the logger class before getting the logger
logging.setLoggerClass(CpmcuLogger)

# Enhanced Rich Console and Theme
_console = Console(
    theme=Theme({
        "logging.level.success": "bold green",
        "logging.level.stage": "bold blue",
        "logging.level.info": "cyan",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
    })
)

_rich_handler = RichHandler(
    console=_console,
    show_time=True,
    show_path=False,
    rich_tracebacks=True,
    markup=True,
    keywords=[],
    log_time_format="[%H:%M:%S]",
)

# Plain handler for compatibility
_plain_handler = logging.StreamHandler()
_plain_formatter = logging.Formatter(
    fmt='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
_plain_handler.setFormatter(_plain_formatter)

def get_logger(name="cpmcu", use_plain=False):
    """Get a logger instance configured with appropriate handler."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return typing.cast(CpmcuLogger, logger)
    
    logger.setLevel(logging.INFO)
    
    logger.addHandler(_plain_handler if use_plain else _rich_handler)
    
    logger.propagate = False
    return typing.cast(CpmcuLogger, logger)

# Default Logger
logger = get_logger()

def update_logger_mode(use_plain=False):
    """Update the global logger to use plain or rich mode"""
    global logger
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(_plain_handler if use_plain else _rich_handler)

@contextmanager
def stage_context(stage_name):
    """Context manager for stage logging with automatic completion"""
    logger.stage(stage_name)
    start_time = time.time()
    try:
        yield
        elapsed = time.time() - start_time
        logger.success(f"{stage_name} ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"{stage_name} failed ({elapsed:.2f}s): {e}")
        raise

def configure_plain_mode(use_plain=False):
    """Configure both logging and display plain mode"""
    from .display import set_plain_mode
    set_plain_mode(use_plain)
    update_logger_mode(use_plain)
