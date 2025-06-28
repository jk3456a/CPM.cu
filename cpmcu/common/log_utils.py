import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Custom log levels
SUCCESS = 25
STAGE = 26
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(STAGE, "STAGE")


class HandlerStrategy(ABC):
    """Abstract strategy for logger handlers"""
    
    @abstractmethod
    def create_handler(self):
        """Create and return logging handler"""
        pass


class RichHandlerStrategy(HandlerStrategy):
    """Strategy for Rich-formatted logging"""
    
    def create_handler(self):
        console = Console(theme=Theme({
            "logging.level.success": "bold green",
            "logging.level.stage": "bold blue", 
            "logging.level.info": "cyan",
            "logging.level.warning": "yellow",
            "logging.level.error": "bold red",
        }))
        
        return RichHandler(
            console=console, show_time=True, show_path=False,
            rich_tracebacks=True, markup=True, keywords=[],
            log_time_format="[%H:%M:%S]"
        )


class PlainHandlerStrategy(HandlerStrategy):
    """Strategy for plain text logging"""
    
    def create_handler(self):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s', 
            datefmt='%H:%M:%S'
        ))
        return handler


class Logger:
    """Unified logger with strategy-based handler management"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger("cpmcu")
        self.logger.setLevel(logging.INFO)
        self.current_stage = None
        self.stage_start_time = None
        self._strategy = RichHandlerStrategy()
        self._setup_handler()
        self._initialized = True
    
    def set_strategy(self, strategy: HandlerStrategy):
        """Change handler strategy and update logger"""
        self._strategy = strategy
        self._setup_handler()
    
    def switch_mode(self, use_plain_mode):
        """Switch logger mode between plain text and rich formatting"""
        if use_plain_mode:
            self.set_strategy(PlainHandlerStrategy())
        else:
            self.set_strategy(RichHandlerStrategy())
    
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
    
    def _setup_handler(self):
        """Setup handler using current strategy"""
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add new handler from strategy
        handler = self._strategy.create_handler()
        self.logger.addHandler(handler)
        self.logger.propagate = False


# Global logger instance (singleton)
logger = Logger()

@contextmanager
def stage_context(stage_name, logger_instance=None):
    """Context manager for stage logging with automatic completion tracking"""
    log = logger_instance or logger
    log.stage(stage_name)
    start_time = time.time()
    try:
        yield log
        log.success(f"{stage_name} ({time.time() - start_time:.2f}s)")
    except Exception as e:
        log.error(f"{stage_name} failed ({time.time() - start_time:.2f}s): {e}")
        raise
