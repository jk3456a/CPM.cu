import logging
import sys
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
import typing

# Define custom log levels
SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")

class CpmcuLogger(logging.Logger):
    def success(self, message, *args, **kws):
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, message, args, **kws)

# Set the logger class before getting the logger
logging.setLoggerClass(CpmcuLogger)

# --- Rich Console and Handler Setup ---
_console = Console(
    theme=Theme({
        "logging.level.success": "bold green",
    })
)

_handler = RichHandler(
    console=_console,
    show_time=True,
    show_path=False,
    rich_tracebacks=True,
    tracebacks_suppress=[],
    markup=True,
)

# --- Logger Configuration ---
def get_logger(name="cpmcu"):
    """
    Get a logger instance configured with RichHandler.
    """
    logger = logging.getLogger(name)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return typing.cast(CpmcuLogger, logger)

    logger.setLevel(logging.INFO)
    logger.addHandler(_handler)
    logger.propagate = False

    return typing.cast(CpmcuLogger, logger)

# --- Default Logger ---
logger = get_logger()

# --- Public API ---
__all__ = ["logger", "get_logger", "Console"]

if __name__ == "__main__":
    # --- Example Usage ---
    from rich.panel import Panel
    from rich.table import Table

    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.success("This is a success message!")
    
    logger.info("Rich markup is also supported, e.g. [bold magenta]this is bold magenta[/bold magenta].")

    try:
        1 / 0
    except Exception:
        logger.exception("This is an exception traceback.")

    # Example of using the console directly for more complex layouts
    console = Console()

    config_table = Table(title="Generation Configuration", show_header=False, box=None)
    config_table.add_row("Model Path:", "[cyan]openbmb/MiniCPM-Llama3-V-2_5[/cyan]")
    config_table.add_row("Data Type:", "[cyan]float16[/cyan]")
    config_table.add_row("CUDA Graph:", "[green]True[/green]")

    console.print(Panel(config_table, title="[bold yellow]Configuration[/bold yellow]", border_style="blue")) 