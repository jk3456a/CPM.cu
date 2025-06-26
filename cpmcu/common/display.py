#!/usr/bin/env python3
"""
CPM.cu Display Utilities

Beautiful display and formatting utilities for configuration and performance summaries.
"""

import platform
from rich.console import Console, Group
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table


# Enhanced Rich Console for display
_display_console = Console(
    theme=Theme({
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "info": "cyan",
        "dim": "dim white",
    })
)


def print_config_summary(args, title="Configuration"):
    """Print configuration summary with beautiful formatting"""
    try:
        import torch
    except ImportError:
        torch = None
    
    # Configuration sections mapping
    CONFIG_SECTIONS = {
        "Model Configuration": {
            "color": "bright_blue",
            "fields": [
                ("model_path", "Model Path", None),
                ("model_type", "Model Type", None),
                ("dtype", "Data Type", None),
                ("draft_model_path", "Draft Model", None),
                ("frspec_path", "FRSpec Path", None),
                ("minicpm4_yarn", "MiniCPM4 YARN", None),
            ]
        },
        "Server Configuration": {
            "color": "bright_green",
            "fields": [("host", "Host", None), ("port", "Port", None)],
            "condition": lambda args: hasattr(args, 'host') or hasattr(args, 'port')
        },
        "Prompt Configuration": {
            "color": "bright_green", 
            "fields": [
                ("prompt_file", "Prompt File", lambda x: bool(x)),
                ("prompt_text", "Prompt Text", lambda x: bool(x)),
                ("use_chat_template", "Use Chat Template", None),
            ],
            "condition": lambda args: any(hasattr(args, f) for f in ['prompt_file', 'prompt_text', 'use_chat_template'])
        },
        "Generation Configuration": {
            "color": "bright_cyan",
            "fields": [
                ("num_generate", "Max Tokens", None),
                ("use_stream", "Use Stream", None),
                ("ignore_eos", "Ignore EOS", None),
                ("temperature", "Temperature", None),
                ("random_seed", "Random Seed", None),
            ],
            "condition": lambda args: any(hasattr(args, f) for f in ['num_generate', 'use_stream', 'ignore_eos', 'temperature', 'random_seed'])
        },
        "System Configuration": {
            "color": "bright_blue",
            "fields": [
                ("cuda_graph", "CUDA Graph", None),
                ("memory_limit", "Memory Limit", None),
                ("chunk_length", "Chunk Length", None),
            ]
        },
        "Speculative Decoding": {
            "color": "bright_magenta",
            "fields": [
                ("spec_window_size", "Window Size", None),
                ("spec_num_iter", "Iterations", None),
                ("spec_topk_per_iter", "Top-K per Iter", None),
                ("spec_tree_size", "Tree Size", None),
                ("frspec_vocab_size", "FRSpec Vocab Size", None),
            ],
            "condition": lambda args: hasattr(args, 'draft_model_path') and args.draft_model_path
        },
        "Sparse Attention": {
            "color": "purple",
            "fields": [
                ("sink_window_size", "Sink Window Size", None),
                ("block_window_size", "Block Window Size", None),
                ("sparse_topk_k", "Sparse Top-K", None),
                ("sparse_switch", "Sparse Switch", None),
                ("use_compress_lse", "Use Compress LSE", None),
            ],
            "condition": lambda args: hasattr(args, 'model_type') and args.model_type == 'minicpm4'
        }
    }
    
    def fmt_val(value):
        """Format value for display"""
        if isinstance(value, bool):
            return f"[green]✓[/green]" if value else f"[red]✗[/red]"
        elif isinstance(value, float):
            return f"[cyan]{value:.2f}[/cyan]"
        elif isinstance(value, int):
            return f"[cyan]{value}[/cyan]"
        else:
            s = str(value)
            return f"[cyan]{s[:45] + '...' if len(s) > 50 else s}[/cyan]"
    
    sections = []
    
    # System Information (always first)
    sys_info = [("OS", f"{platform.system()} {platform.release()}"), ("Python", platform.python_version())]
    
    if torch and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        sys_info.extend([
            ("GPU", f"{gpu} ({mem}GB)"), 
            ("CUDA", torch.version.cuda), 
            ("PyTorch", torch.__version__.split('+')[0])
        ])
    else:
        sys_info.append(("GPU", "Not Available"))
    
    sections.append(("System Information", sys_info, "bright_yellow"))
    
    # Process configuration sections
    for section_name, config in CONFIG_SECTIONS.items():
        if config.get("condition") and not config["condition"](args):
            continue
            
        items = []
        for field_name, display_name, transformer in config["fields"]:
            if hasattr(args, field_name):
                value = getattr(args, field_name)
                if transformer:
                    value = transformer(value)
                items.append((display_name, value))
        
        if items:
            sections.append((section_name, items, config["color"]))
    
    # Create all section panels
    section_panels = []
    for section_title, items, color in sections:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold white", min_width=18)
        table.add_column(style="cyan")
        
        for key, value in items:
            table.add_row(f"{key}:", fmt_val(value))
        
        section_panels.append(Panel(table, title=f"[{color}]{section_title}[/{color}]", border_style=color))
    
    # Create main configuration panel containing all sections
    content_group = Group(*section_panels)
    
    _display_console.print(Panel(content_group, title=f"[bold blue]{title}[/bold blue]", border_style="blue", padding=(1, 2)))
    
    _display_console.print()


def print_performance_summary(stats):
    """Print performance summary with beautiful formatting"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold white", min_width=18)
    table.add_column(style="green", justify="right")
    table.add_column(style="dim white")
    
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
    
    panel = Panel(table, title="[bold green]Performance Summary[/bold green]", border_style="green")
    _display_console.print(panel)

