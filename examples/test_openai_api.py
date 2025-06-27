#!/usr/bin/env python3
import requests
import json
import sys
import argparse
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cpmcu.common.log_utils import logger, stage_context
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize Rich console for beautiful output
console = Console()

def print_request_summary(data, url, mode):
    """Print request configuration with beautiful formatting"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold white", min_width=15)
    table.add_column(style="cyan")
    
    table.add_row("URL:", f"[yellow]{url}[/yellow]")
    table.add_row("Mode:", f"[green]{mode}[/green]")
    table.add_row("Model:", f"[cyan]{data['model']}[/cyan]")
    table.add_row("Max Tokens:", f"[cyan]{data['max_tokens']}[/cyan]")
    table.add_row("Temperature:", f"[cyan]{data['temperature']}[/cyan]")
    table.add_row("Stream:", f"[green]✓[/green]" if data['stream'] else f"[red]✗[/red]")
    
    # Show message content (truncated if too long)
    content = data['messages'][0]['content']
    display_content = content[:60] + "..." if len(content) > 60 else content
    table.add_row("Prompt:", f"[dim white]{display_content}[/dim white]")
    
    panel = Panel(table, title="[bold blue]Request Configuration[/bold blue]", border_style="blue")
    console.print(panel)

def test_chat_completion(host="localhost", port=8000, stream=True):
    """Test chat completion with streaming or non-streaming mode"""
    url = f"http://{host}:{port}/v1/chat/completions"
    
    data = {
        "model": "model",
        "messages": [
            {"role": "user", "content": "Please tell me how to write quick sort algorithm"}
        ],
        "stream": stream,
        "max_tokens": 1024,
        "temperature": 0.0
    }
    
    mode = "streaming" if stream else "non-streaming"
    
    with stage_context(f"Testing {mode} chat completion"):
        print_request_summary(data, url, mode.capitalize())
        
        try:
            with stage_context("Sending API request"):
                response = requests.post(url, json=data, stream=stream, timeout=60)
                response.raise_for_status()
            
            if stream:
                return _handle_streaming_response(response)
            else:
                return _handle_non_streaming_response(response)
                
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return False

def _handle_streaming_response(response):
    """Handle streaming response with enhanced display"""
    logger.info("Processing streaming response...")
    
    # Setup response display
    response_text = Text()
    total_chars = 0
    start_time = time.time()
    first_token_time = None
    
    with Live(Panel(response_text, title="[bold green]Generated Response[/bold green]", 
                   border_style="green"), refresh_per_second=10) as live:
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                data_str = line[6:]  # Remove 'data: ' prefix
                
                # Check for end of stream
                if data_str.strip() == '[DONE]':
                    break
                
                # Skip empty data
                if not data_str.strip():
                    continue
                    
                try:
                    chunk = json.loads(data_str)
                    
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        choice = chunk['choices'][0]
                        
                        # Handle content delta
                        if 'delta' in choice and 'content' in choice['delta']:
                            content = choice['delta']['content']
                            if content:  # Only add non-empty content
                                if first_token_time is None:
                                    first_token_time = time.time()
                                response_text.append(content)
                                total_chars += len(content)
                                live.update(Panel(response_text, 
                                                title="[bold green]Generated Response[/bold green]", 
                                                border_style="green"))
                        
                        # Handle finish reason
                        elif choice.get('finish_reason'):
                            logger.success(f"Generation finished: {choice['finish_reason']}")
                            break
                            
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON: {e}")
                    logger.error(f"Raw data: {repr(data_str)}")
                    continue
    
    # Print performance statistics
    end_time = time.time()
    total_time = end_time - start_time
    if first_token_time:
        time_to_first_token = first_token_time - start_time
        _print_streaming_stats(total_chars, total_time, time_to_first_token)
    
    return True

def _handle_non_streaming_response(response):
    """Handle non-streaming response with enhanced display"""
    try:
        with stage_context("Processing response"):
            result = response.json()
        
        # Extract and display the content
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                content = choice['message']['content']
                
                # Display generated content in a beautiful panel
                console.print(Panel(content, title="[bold green]Generated Content[/bold green]", 
                                  border_style="green", padding=(1, 2)))
                
                # Display usage statistics if available
                if 'usage' in result:
                    _print_usage_stats(result['usage'])
                    
                logger.success("Response processed successfully")
            else:
                logger.warning("No message content found in response")
        else:
            logger.warning("No choices found in response")
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing response JSON: {e}")
        return False

def _print_streaming_stats(total_chars, total_time, time_to_first_token):
    """Print streaming performance statistics"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold white", min_width=20)
    table.add_column(style="green", justify="right")
    table.add_column(style="dim white")
    
    table.add_row("Total Characters:", f"{total_chars}", "chars")
    table.add_row("Total Time:", f"{total_time:.2f}", "s")
    table.add_row("Time to First Token:", f"{time_to_first_token:.2f}", "s")
    if total_time > 0:
        table.add_row("Average Speed:", f"{total_chars / total_time:.1f}", "chars/s")
    
    panel = Panel(table, title="[bold cyan]Streaming Performance[/bold cyan]", border_style="cyan")
    console.print(panel)

def _print_usage_stats(usage):
    """Print token usage statistics"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold white", min_width=18)
    table.add_column(style="cyan", justify="right")
    table.add_column(style="dim white")
    
    table.add_row("Prompt Tokens:", f"{usage.get('prompt_tokens', 'N/A')}", "tokens")
    table.add_row("Completion Tokens:", f"{usage.get('completion_tokens', 'N/A')}", "tokens")  
    table.add_row("Total Tokens:", f"{usage.get('total_tokens', 'N/A')}", "tokens")
    
    panel = Panel(table, title="[bold magenta]Token Usage[/bold magenta]", border_style="magenta")
    console.print(panel)

def check_server_health(host="localhost", port=8000):
    """Check if server is running and model is loaded"""
    with stage_context("Checking server health"):
        try:
            health_response = requests.get(f"http://{host}:{port}/health")
            health_response.raise_for_status()
            health_data = health_response.json()
            
            # Display health status in a table
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="bold white", min_width=15)
            table.add_column(style="cyan")
            
            for key, value in health_data.items():
                if isinstance(value, bool):
                    formatted_value = f"[green]✓[/green]" if value else f"[red]✗[/red]"
                else:
                    formatted_value = f"[cyan]{value}[/cyan]"
                table.add_row(f"{key.replace('_', ' ').title()}:", formatted_value)
            
            panel = Panel(table, title="[bold yellow]Server Health Status[/bold yellow]", border_style="yellow")
            console.print(panel)
            
            if not health_data.get('model_loaded', False):
                logger.error("Model not loaded! Please start the server first")
                return False
                
            logger.success("Server health check passed")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            logger.error(f"Please make sure the server is running on {host}:{port}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPM.cu Server API Test")
    parser.add_argument('--no-stream', action='store_true', 
                       help='Use non-streaming mode instead of streaming (default: streaming)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Server port (default: 8000)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Server host (default: localhost)')
    args = parser.parse_args()
    
    use_stream = not args.no_stream
    mode = "Streaming" if use_stream else "Non-streaming"
    
    # Print beautiful header
    header_table = Table(show_header=False, box=None, padding=(0, 2))
    header_table.add_column(style="bold white", min_width=18)
    header_table.add_column(style="bright_blue")
    
    header_table.add_row("Test Mode:", f"[green]{mode}[/green]")
    header_table.add_row("Server:", f"[yellow]{args.host}:{args.port}[/yellow]")
    
    header_panel = Panel(header_table, title="[bold blue]CPM.cu OpenAI API Test[/bold blue]", 
                        border_style="blue", padding=(1, 2))
    console.print(header_panel)
    console.print()
    
    # Check server health first
    if not check_server_health(host=args.host, port=args.port):
        sys.exit(1)
    
    console.print()
    
    # Run test
    success = test_chat_completion(host=args.host, port=args.port, stream=use_stream)
    
    console.print()
    if success:
        logger.success(f"{mode} test completed successfully!")
    else:
        logger.error(f"{mode} test failed!")
        sys.exit(1)
