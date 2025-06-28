import logging
import time
from contextlib import contextmanager
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Custom log levels
SUCCESS = 25
STAGE = 26
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(STAGE, "STAGE")


class Logger:
    """Unified logger with delayed initialization singleton"""
    
    _instance = None
    _mode_configured = False
    _use_plain_mode = False
    
    @classmethod
    def configure(cls, use_plain_mode=False):
        """Configure logger mode before first use"""
        if cls._instance is not None:
            raise RuntimeError("Logger already initialized, cannot configure")
        cls._use_plain_mode = use_plain_mode
        cls._mode_configured = True
    
    def __new__(cls):
        if cls._instance is None:
            if not cls._mode_configured:
                # Use default configuration if not explicitly configured
                cls._use_plain_mode = False
                cls._mode_configured = True
            cls._instance = super().__new__(cls)
            cls._instance._init_once()
        return cls._instance
    
    def _init_once(self):
        """Initialize logger instance once"""
        self.use_plain_mode = self._use_plain_mode
        self.logger = logging.getLogger("cpmcu")
        self.logger.setLevel(logging.INFO)
        self.current_stage = None
        self.stage_start_time = None
        
        # Use the same handler creation logic for consistency
        handler = self._create_handler()
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def _create_handler(self):
        """Create a handler identical to CPM.cu's internal handler"""
        if self.use_plain_mode:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                fmt="[%(asctime)s] %(levelname)-8s %(message)s", 
                datefmt="%H:%M:%S"
            ))
        else:
            console = Console(theme=Theme({
                "logging.level.success": "bold green",
                "logging.level.stage": "bold blue", 
                "logging.level.info": "cyan",
                "logging.level.warning": "yellow",
                "logging.level.error": "bold red",
            }))
            
            handler = RichHandler(
                console=console, show_time=True, show_path=False,
                rich_tracebacks=True, markup=True, keywords=[],
                log_time_format="[%H:%M:%S]"
            )
        return handler

    def configure_external_loggers(self, logger_names: list[str]) -> None:
        """Configure external loggers to use the same handler as CPM.cu for consistent colors"""
        
        # Create the same handler as CPM.cu uses
        handler = self._create_handler()
        
        # Configure each external logger
        for logger_name in logger_names:
            external_logger = logging.getLogger(logger_name)
            external_logger.handlers.clear()  # Remove existing handlers
            external_logger.addHandler(handler)
            external_logger.setLevel(logging.INFO)
            external_logger.propagate = False
    
    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)
        
    def success(self, message, *args, **kwargs):
        self.logger.log(SUCCESS, message, *args, **kwargs)
        
    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)
        
    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)
        
    def stage(self, message, *args, **kwargs):
        self.logger.log(STAGE, message, *args, **kwargs)
        self.current_stage = message
        self.stage_start_time = time.time()

    @contextmanager
    def stage_context(self, stage_name):
        """Context manager for stage logging with automatic completion tracking"""
        self.stage(stage_name)
        start_time = time.time()
        try:
            yield self
            self.success(f"{stage_name} ({time.time() - start_time:.2f}s)")
        except Exception as e:
            self.error(f"{stage_name} failed ({time.time() - start_time:.2f}s): {e}")
            raise


class LoggerProxy:
    """Proxy for delayed logger initialization"""
    
    def __init__(self):
        self._logger = None
    
    def _get_logger(self):
        if self._logger is None:
            self._logger = Logger()
        return self._logger
    
    def __getattr__(self, name):
        return getattr(self._get_logger(), name)


# Global logger instance (proxy for delayed creation)
logger = LoggerProxy()
