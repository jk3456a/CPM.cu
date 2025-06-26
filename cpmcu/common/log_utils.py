import logging
import sys
import time
from contextlib import contextmanager
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.status import Status
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
import typing
from rich.columns import Columns

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
            self.success(f"{msg} [dim]({elapsed:.2f}s)[/dim]")
            self._current_stage = None
            self._stage_start_time = None

# Set the logger class before getting the logger
logging.setLoggerClass(CpmcuLogger)

# --- Enhanced Rich Console and Theme ---
_console = Console(
    theme=Theme({
        "logging.level.success": "bold green",
        "logging.level.stage": "bold blue",
        "logging.level.info": "cyan",
        "logging.level.warning": "yellow",
        "logging.level.error": "bold red",
        "config.key": "bold white",
        "config.value": "cyan",
        "metric.label": "bold white",
        "metric.value": "green",
        "metric.unit": "dim white",
    })
)

_handler = RichHandler(
    console=_console,
    show_time=True,
    show_path=False,
    rich_tracebacks=True,
    tracebacks_suppress=[],
    markup=True,
    keywords=[],  # Disable keyword highlighting for cleaner output
)

# --- Logger Configuration ---
def get_logger(name="cpmcu"):
    """Get a logger instance configured with RichHandler."""
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

# --- Enhanced Display Components ---
def create_compact_config_display(config_data, title="Configuration"):
    """Create a compact, single-line configuration display"""
    config_items = []
    for key, value in config_data.items():
        if isinstance(value, bool):
            icon = "✓" if value else "✗"
            color = "green" if value else "red"
            config_items.append(f"{key}: [{color}]{icon}[/{color}]")
        else:
            config_items.append(f"{key}: [cyan]{value}[/cyan]")
    
    config_line = " • ".join(config_items)
    panel = Panel(
        config_line,
        title=f"[bold blue]{title}[/bold blue]",
        border_style="blue",
        padding=(0, 1)
    )
    return panel

def create_performance_summary(stats, title="Performance Summary"):
    """Create a beautiful performance summary panel"""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="metric.label", min_width=20)
    table.add_column(style="metric.value", justify="right")
    table.add_column(style="metric.unit")
    
    # Add metrics with proper formatting
    if 'prefill_length' in stats:
        table.add_row("Prefill Length", f"{stats['prefill_length']}", "tokens")
    if 'prefill_time' in stats and stats['prefill_time'] > 0:
        table.add_row("Prefill Time", f"{stats['prefill_time']:.2f}", "s")
        table.add_row("Prefill Speed", f"{stats['prefill_length'] / stats['prefill_time']:.1f}", "tokens/s")
    
    if 'accept_lengths' in stats and stats['accept_lengths']:
        mean_accept = sum(stats['accept_lengths']) / len(stats['accept_lengths'])
        table.add_row("Mean Accept Length", f"{mean_accept:.2f}", "tokens")
    
    if 'decode_length' in stats:
        table.add_row("Decode Length", f"{stats['decode_length']}", "tokens")
    if 'decode_time' in stats and stats['decode_time'] > 0:
        table.add_row("Decode Time", f"{stats['decode_time']:.2f}", "s")
        table.add_row("Decode Speed", f"{stats['decode_length'] / stats['decode_time']:.1f}", "tokens/s")
    
    return Panel(
        table,
        title=f"[bold green]{title}[/bold green]",
        border_style="green",
        padding=(1, 2)
    )

@contextmanager
def progress_context(description="Processing..."):
    """Context manager for showing progress with spinner"""
    with Status(f"[cyan]{description}[/cyan]", console=_console, spinner="dots"):
        yield

@contextmanager
def stage_context(stage_name):
    """Context manager for stage logging with automatic completion"""
    logger.stage(stage_name)
    start_time = time.time()
    try:
        yield
        elapsed = time.time() - start_time
        logger.success(f"{stage_name} [dim]({elapsed:.2f}s)[/dim]")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"{stage_name} failed [dim]({elapsed:.2f}s)[/dim]: {e}")
        raise

@contextmanager
def download_progress_context(description="Downloading"):
    """Context manager for download progress with progress bar"""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=_console
    )
    
    with progress:
        task = progress.add_task(description, total=None)
        yield progress, task

def log_model_info(model_path, model_type, features=None):
    """Log comprehensive model information"""
    model_name = model_path.split('/')[-1] if '/' in model_path else model_path
    
    info_items = [f"[bold cyan]{model_name}[/bold cyan]"]
    info_items.append(f"Type: [yellow]{model_type}[/yellow]")
    
    if features:
        feature_list = []
        if features.get('quantized'):
            feature_list.append("[yellow]Quantized[/yellow]")
        if features.get('speculative'):
            feature_list.append("[green]Speculative[/green]")
        if features.get('sparse'):
            feature_list.append("[magenta]Sparse[/magenta]")
        if features.get('yarn'):
            feature_list.append("[blue]YARN[/blue]")
        
        if feature_list:
            info_items.append(f"Features: {' + '.join(feature_list)}")
    
    info_text = " • ".join(info_items)
    panel = Panel(
        info_text,
        title="[bold blue]Model Information[/bold blue]",
        border_style="blue",
        padding=(0, 1)
    )
    _console.print(panel)

def log_system_info():
    """Log system information"""
    import torch
    import platform
    
    system_items = []
    system_items.append(f"OS: [cyan]{platform.system()} {platform.release()}[/cyan]")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        system_items.append(f"GPU: [green]{gpu_name} ({gpu_memory}GB)[/green]")
        system_items.append(f"CUDA: [yellow]{torch.version.cuda}[/yellow]")
    else:
        system_items.append("GPU: [red]Not Available[/red]")
    
    system_items.append(f"PyTorch: [cyan]{torch.__version__}[/cyan]")
    
    system_text = " • ".join(system_items)
    panel = Panel(
        system_text,
        title="[bold yellow]System Information[/bold yellow]",
        border_style="yellow",
        padding=(0, 1)
    )
    _console.print(panel)

def create_status_table(status_data):
    """Create a real-time status table"""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold white", min_width=15)
    table.add_column(style="cyan")
    
    for key, value in status_data.items():
        if isinstance(value, bool):
            status = "[green]✓[/green]" if value else "[red]✗[/red]"
            table.add_row(f"{key}:", status)
        elif isinstance(value, (int, float)):
            if key.lower().endswith('time'):
                table.add_row(f"{key}:", f"{value:.2f}s")
            elif key.lower().endswith('speed'):
                table.add_row(f"{key}:", f"{value:.1f} tokens/s")
            else:
                table.add_row(f"{key}:", str(value))
        else:
            table.add_row(f"{key}:", str(value))
    
    return table

def print_complete_config_summary(args, title="Complete Configuration"):
    """Print complete configuration summary with all parameters organized by groups"""
    # Helper function to create a section panel with consistent sizing
    def create_section_panel(section_title, items, color="cyan", min_width=40):
        if not items:
            return None
            
        table = Table(show_header=False, box=None, padding=(0, 1), min_width=min_width)
        table.add_column(style="bold white", min_width=18, max_width=18)
        table.add_column(style="cyan", min_width=20)
        
        for key, value in items:
            if value is not None:
                if isinstance(value, bool):
                    styled_value = f"[green]✓[/green]" if value else f"[red]✗[/red]"
                elif isinstance(value, (int, float)):
                    if key.lower().endswith('size') and value >= 1024:
                        styled_value = f"[cyan]{value//1024}K[/cyan]"
                    elif isinstance(value, float):
                        styled_value = f"[cyan]{value:.2f}[/cyan]"
                    else:
                        styled_value = f"[cyan]{value}[/cyan]"
                else:
                    # Truncate long values to fit nicely
                    str_value = str(value)
                    if len(str_value) > 25:
                        str_value = str_value[:22] + "..."
                    styled_value = f"[cyan]{str_value}[/cyan]"
                
                # Ensure key is properly truncated
                display_key = key[:16] + ":" if len(key) > 16 else f"{key}:"
                table.add_row(display_key, styled_value)
        
        return Panel(
            table,
            title=f"[{color}]{section_title}[/{color}]",
            border_style=color,
            padding=(0, 1),
            width=min_width + 8  # Add padding for borders
        )
    
    panels = []
    
    # System Information (always show first)
    import torch
    import platform
    
    system_items = [
        ("OS", f"{platform.system()} {platform.release()}"),
        ("Python", f"{platform.python_version()}"),
    ]
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Smart GPU name truncation
        if "NVIDIA GeForce" in gpu_name:
            gpu_name = gpu_name.replace("NVIDIA GeForce ", "")
        if len(gpu_name) > 20:
            gpu_name = gpu_name[:17] + "..."
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        system_items.extend([
            ("GPU", f"{gpu_name} ({gpu_memory}GB)"),
            ("CUDA", torch.version.cuda),
            ("PyTorch", torch.__version__.split('+')[0])  # Remove CUDA suffix
        ])
    else:
        system_items.append(("GPU", "Not Available"))
    
    system_panel = create_section_panel("System Information", system_items, "yellow", 48)
    if system_panel:
        panels.append(system_panel)
    
    # Model Configuration
    model_items = []
    if hasattr(args, 'model_path'):
        model_name = args.model_path.split('/')[-1] if '/' in args.model_path else args.model_path
        model_items.append(("Model Path", model_name))
    
    if hasattr(args, 'model_type'):
        model_items.append(("Model Type", args.model_type))
    
    if hasattr(args, 'dtype'):
        model_items.append(("Data Type", args.dtype))
    
    if hasattr(args, 'draft_model_path') and args.draft_model_path:
        draft_name = args.draft_model_path.split('/')[-1] if '/' in args.draft_model_path else args.draft_model_path
        model_items.append(("Draft Model", draft_name))
    
    if hasattr(args, 'frspec_path') and args.frspec_path:
        frspec_name = args.frspec_path.split('/')[-1] if '/' in args.frspec_path else args.frspec_path
        model_items.append(("FRSpec Path", frspec_name))
    
    if hasattr(args, 'minicpm4_yarn'):
        model_items.append(("MiniCPM4 YARN", args.minicpm4_yarn))
    
    model_panel = create_section_panel("Model Configuration", model_items, "cyan", 48)
    if model_panel:
        panels.append(model_panel)
    
    # Generation Configuration
    gen_items = []
    if hasattr(args, 'num_generate'):
        gen_items.append(("Max Tokens", args.num_generate))
    if hasattr(args, 'use_stream'):
        gen_items.append(("Use Streaming", args.use_stream))
    if hasattr(args, 'temperature'):
        gen_items.append(("Temperature", args.temperature))
    if hasattr(args, 'random_seed') and args.random_seed is not None:
        gen_items.append(("Random Seed", args.random_seed))
    if hasattr(args, 'ignore_eos'):
        gen_items.append(("Ignore EOS", args.ignore_eos))
    if hasattr(args, 'use_chat_template'):
        gen_items.append(("Chat Template", args.use_chat_template))
    
    if gen_items:
        gen_panel = create_section_panel("Generation Config", gen_items, "blue", 48)
        panels.append(gen_panel)
    
    # System Configuration
    system_config_items = []
    if hasattr(args, 'cuda_graph'):
        system_config_items.append(("CUDA Graph", args.cuda_graph))
    if hasattr(args, 'memory_limit'):
        system_config_items.append(("Memory Limit", args.memory_limit))
    if hasattr(args, 'chunk_length'):
        system_config_items.append(("Chunk Length", args.chunk_length))
    
    # Add prompt info to system config if available
    if hasattr(args, 'prompt_file'):
        system_config_items.append(("Prompt File", bool(args.prompt_file)))
    if hasattr(args, 'prompt_text'):
        system_config_items.append(("Prompt Text", bool(args.prompt_text)))
    
    # Add server info if available
    if hasattr(args, 'host') and hasattr(args, 'port'):
        system_config_items.append(("Server", f"{args.host}:{args.port}"))
    
    if system_config_items:
        system_config_panel = create_section_panel("System Config", system_config_items, "white", 48)
        panels.append(system_config_panel)
    
    # Speculative Decoding (if draft model exists)
    if hasattr(args, 'draft_model_path') and args.draft_model_path:
        spec_items = [
            ("Window Size", getattr(args, 'spec_window_size', 1024)),
            ("Iterations", getattr(args, 'spec_num_iter', 2)),
            ("Top-K per Iter", getattr(args, 'spec_topk_per_iter', 10)),
            ("Tree Size", getattr(args, 'spec_tree_size', 12)),
        ]
        if hasattr(args, 'frspec_vocab_size') and args.frspec_vocab_size > 0:
            spec_items.append(("FRSpec Vocab", args.frspec_vocab_size))
        
        spec_panel = create_section_panel("Speculative Decode", spec_items, "purple", 48)
        panels.append(spec_panel)
    
    # Sparse Attention (for MiniCPM4)
    if hasattr(args, 'model_type') and args.model_type == 'minicpm4':
        sparse_items = [
            ("Sink Window", getattr(args, 'sink_window_size', 1)),
            ("Block Window", getattr(args, 'block_window_size', 8)),
            ("Sparse Top-K", getattr(args, 'sparse_topk_k', 64)),
            ("Sparse Switch", getattr(args, 'sparse_switch', 0)),
            ("Compress LSE", getattr(args, 'use_compress_lse', True)),
        ]
        
        sparse_panel = create_section_panel("Sparse Attention", sparse_items, "red", 48)
        panels.append(sparse_panel)
    
    # Display all panels with consistent layout
    if panels:
        # Check terminal width to determine if we can display panels side by side
        terminal_width = _console.size.width
        panel_width = 48 + 8  # 56 chars per panel
        min_width_for_two_columns = panel_width * 2 + 8  # 120 chars needed for two panels + gap
        
        # Determine layout based on terminal width and number of panels
        use_two_columns = len(panels) >= 2 and terminal_width >= min_width_for_two_columns
        
        if use_two_columns:
            # Create main title panel with width matching two panels side by side
            title_width = panel_width * 2 + 8  # 120 chars
            
            title_panel = Panel(
                f"[bold blue]{title}[/bold blue]",
                border_style="blue",
                width=title_width,
                padding=(0, 0)
            )
            _console.print(title_panel)
            
            # Display panels in pairs
            for i in range(0, len(panels), 2):
                if i + 1 < len(panels):
                    _console.print(Columns([panels[i], panels[i + 1]], equal=True, expand=True))
                else:
                    # Center the last panel if it's alone
                    _console.print(Columns([panels[i]], equal=True, expand=True))
        else:
            # Single column layout - title width matches single panel
            title_width = panel_width  # 56 chars
            
            title_panel = Panel(
                f"[bold blue]{title}[/bold blue]",
                border_style="blue",
                width=title_width,
                padding=(0, 0)
            )
            _console.print(title_panel)
            
            # Display panels vertically
            for panel in panels:
                _console.print(panel)
    
    _console.print()  # Add spacing after configuration

def print_performance_summary(stats):
    """Print performance summary in a beautiful format"""
    panel = create_performance_summary(stats)
    _console.print(panel)

# --- Public API ---
__all__ = [
    "logger", "get_logger", "Console", 
    "progress_context", "stage_context",
    "print_complete_config_summary", "print_performance_summary",
    "create_compact_config_display", "create_performance_summary",
    "log_model_info", "log_system_info", "create_status_table",
]

if __name__ == "__main__":
    # --- Example Usage ---
    logger.info("Testing enhanced logging system...")
    
    # Test stage logging
    with stage_context("Model Loading"):
        time.sleep(1)  # Simulate work
    
    # Test complete config
    class MockArgs:
        model_path = "openbmb/MiniCPM4-8B"
        model_type = "minicpm4"
        cuda_graph = True
        use_stream = True
        num_generate = 1024
        draft_model_path = "draft-model"
        dtype = "float16"
        frspec_path = "frspec-path"
        minicpm4_yarn = True
        prompt_file = True
        prompt_text = True
        use_chat_template = True
        cuda_graph = True
        memory_limit = 16.0
        chunk_length = 1024
        temperature = 0.8
        random_seed = 42
        ignore_eos = False
        spec_window_size = 1024
        spec_num_iter = 2
        spec_topk_per_iter = 10
        spec_tree_size = 12
        frspec_vocab_size = 50272
        sink_window_size = 1
        block_window_size = 8
        use_compress_lse = True
    
    print_complete_config_summary(MockArgs())
    
    # Test performance summary
    test_stats = {
        'prefill_length': 13,
        'prefill_time': 0.19,
        'decode_length': 36,
        'decode_time': 0.13,
        'accept_lengths': [2, 3, 2, 3, 2]
    }
    print_performance_summary(test_stats)
    
    logger.success("Enhanced logging system ready!") 