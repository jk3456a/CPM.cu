#!/usr/bin/env python3
"""
CPM.cu Display Utilities

Beautiful display and formatting utilities for configuration and performance summaries.
"""

import platform
import torch
from rich.console import Console, Group
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table

# Global configuration for display mode
_use_plain_mode = False

def set_plain_mode(enabled=True):
    """Set whether to use plain text mode for all display functions"""
    global _use_plain_mode
    _use_plain_mode = enabled

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
            ("plain_log", "Plain Log", None),
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

def _format_value(value):
    """Format value for display"""
    if isinstance(value, bool):
        return f"[green]✓[/green]" if value else f"[red]✗[/red]" if not _use_plain_mode else ("✓" if value else "✗")
    elif isinstance(value, float):
        return f"[cyan]{value:.2f}[/cyan]" if not _use_plain_mode else f"{value:.2f}"
    elif isinstance(value, int):
        return f"[cyan]{value}[/cyan]" if not _use_plain_mode else str(value)
    else:
        s = str(value)
        truncated = s[:45] + '...' if len(s) > 50 else s
        return f"[cyan]{truncated}[/cyan]" if not _use_plain_mode else truncated

def _create_display_content(items, title, color="green", show_units=False):
    """Unified function to create display content for both plain and rich modes"""
    if _use_plain_mode:
        print(f"\n--- {title} ---")
        for item in items:
            if show_units and len(item) == 3:
                key, value, unit = item
                print(f"{key}: {value} {unit}")
            else:
                key, value = item[:2]
                print(f"{key}: {value}")
    else:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="bold white", min_width=18)
        table.add_column(style=color if show_units else "cyan", justify="right")
        if show_units:
            table.add_column(style="dim white")
        
        for item in items:
            if show_units and len(item) == 3:
                key, value, unit = item
                table.add_row(f"{key}:", str(value), unit or "")
            else:
                key, value = item[:2]
                formatted_value = _format_value(value) if not show_units else str(value)
                table.add_row(f"{key}:", formatted_value)
        
        panel = Panel(table, title=f"[bold {color}]{title}[/bold {color}]", border_style=color)
        if show_units:
            _display_console.print(panel)
            return None
        else:
            return panel

def _get_system_info():
    """Get system information for display"""
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
    
    return sys_info

def print_config_summary(args, title="Configuration"):
    """Print configuration summary with beautiful formatting"""
    sections = [("System Information", _get_system_info(), "bright_yellow")]
    
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
    
    # Display sections
    if _use_plain_mode:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        for section_title, items, _ in sections:
            _create_display_content(items, section_title)
        print(f"\n{'='*60}\n")
    else:
        section_panels = []
        for section_title, items, color in sections:
            panel = _create_display_content(items, section_title, color)
            if panel:
                section_panels.append(panel)
        
        content_group = Group(*section_panels)
        _display_console.print(Panel(content_group, title=f"[bold blue]{title}[/bold blue]", border_style="blue", padding=(1, 2)))
        _display_console.print()

def print_performance_summary(stats):
    """Print performance summary with beautiful formatting"""
    rows = []
    
    if 'prefill_length' in stats:
        rows.append(("Prefill Length", f"{stats['prefill_length']}", "tokens"))
    if 'prefill_time' in stats and stats['prefill_time'] > 0:
        rows.append(("Prefill Time", f"{stats['prefill_time']:.2f}", "s"))
        rows.append(("Prefill Speed", f"{stats['prefill_length'] / stats['prefill_time']:.1f}", "tokens/s"))
    
    if 'accept_lengths' in stats and stats['accept_lengths']:
        mean_accept = sum(stats['accept_lengths']) / len(stats['accept_lengths'])
        rows.append(("Mean Accept Length", f"{mean_accept:.2f}", "tokens"))
    
    if 'decode_length' in stats:
        rows.append(("Decode Length", f"{stats['decode_length']}", "tokens"))
    if 'decode_time' in stats and stats['decode_time'] > 0:
        rows.append(("Decode Time", f"{stats['decode_time']:.2f}", "s"))
        rows.append(("Decode Speed", f"{stats['decode_length'] / stats['decode_time']:.1f}", "tokens/s"))
    
    _create_display_content(rows, "Performance Summary", "green", show_units=True)
    
    if not _use_plain_mode:
        _display_console.print()

def display_line(text, style=None):
    """Display function for basic text output with rich/plain mode support"""
    if _use_plain_mode:
        print(text)
    else:
        _display_console.print(text, style=style)

class TextStreamer:
    """Context manager for streaming text display"""
    def __init__(self, title="Generated Response"):
        self.title = title
        self.current_text = ""
        self.started = False
        
    def __enter__(self):
        display_line(f"\n{self.title}", "bold green")
        display_line("-" * 50)
        self.started = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.started:
            display_line("\n" + "-" * 50 + "\n")
            
    def update(self, new_text):
        """Append new text and update display"""
        if new_text is not None and self.started:
            self.current_text += new_text
            print(new_text, end="", flush=True)
        
    def set_text(self, text):
        """Replace current text and update display"""
        if self.started:
            self.current_text = text if text is not None else ""
            print(self.current_text, end="", flush=True)

