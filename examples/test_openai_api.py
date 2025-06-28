#!/usr/bin/env python3
import requests
import json
import sys
import argparse
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cpmcu.common.logging import logger
from cpmcu.common.display import display
from cpmcu.common.args import str2bool

def render_config_table(data, title, color="blue"):
    """Render configuration data as table"""
    display._render_summary_table(data, title, color)

def handle_response(response, stream=True):
    """Handle both streaming and non-streaming responses"""
    if stream:
        logger.info("Processing streaming response...")
        finish_reason = None
        with display.create_stream("Generated Response") as stream_display:
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data: '):
                    data_str = line[6:].strip()
                    if data_str == '[DONE]' or not data_str:
                        continue
                    
                    try:
                        chunk = json.loads(data_str)
                        if 'choices' in chunk and chunk['choices']:
                            choice = chunk['choices'][0]
                            if 'delta' in choice and 'content' in choice['delta']:
                                content = choice['delta']['content']
                                if content:
                                    stream_display.append(content)
                            elif choice.get('finish_reason'):
                                finish_reason = choice['finish_reason']
                                break
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON: {e}")
        
        # Log completion after stream context ends
        if finish_reason:
            logger.success(f"Generation finished: {finish_reason}")
    else:
        with logger.stage_context("Processing response"):
            result = response.json()
            if 'choices' in result and result['choices']:
                choice = result['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content']
                    with display.create_stream("Generated Response") as stream_display:
                        stream_display.replace(content)
                    logger.success("Response processed successfully")
                    return True
                else:
                    logger.warning("No message content found in response")
            else:
                logger.warning("No choices found in response")
    return True

def test_chat_completion(host, port, stream):
    """Test chat completion"""
    url = f"http://{host}:{port}/v1/chat/completions"
    data = {
        "model": "model",
        "messages": [{"role": "user", "content": "Please tell me how to write quick sort algorithm"}],
        "stream": stream,
        "max_tokens": 1024,
        "temperature": 0.0
    }
    
    mode = "streaming" if stream else "non-streaming"
    
    with logger.stage_context(f"Testing {mode} chat completion"):
        # Show request config
        content = data['messages'][0]['content']
        config_data = [
            ("URL", url),
            ("Mode", mode.capitalize()),
            ("Model", data['model']),
            ("Max Tokens", data['max_tokens']),
            ("Temperature", data['temperature']),
            ("Stream", data['stream']),
            ("Prompt", content[:60] + "..." if len(content) > 60 else content),
        ]
        render_config_table(config_data, "Request Configuration", "blue")
        
        try:
            with logger.stage_context("Sending API request"):
                response = requests.post(url, json=data, stream=stream, timeout=60)
                response.raise_for_status()
            return handle_response(response, stream)
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return False

def check_server_health(host, port):
    """Check server health"""
    with logger.stage_context("Checking server health"):
        try:
            health_response = requests.get(f"http://{host}:{port}/health")
            health_response.raise_for_status()
            health_data = health_response.json()
            
            health_items = [(key.replace('_', ' ').title(), value) for key, value in health_data.items()]
            render_config_table(health_items, "Server Health Status", "yellow")
            
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
    
    # Server Configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument('--host', type=str, default='localhost', 
                             help='Server host (default: localhost)')
    server_group.add_argument('--port', type=int, default=8000, 
                             help='Server port (default: 8000)')
    
    # Test Configuration
    test_group = parser.add_argument_group('Test Configuration')
    test_group.add_argument('--use-stream', '--use_stream', default=True,
                           type=str2bool, nargs='?', const=True,
                           help='Use streaming mode (default: True)')
    test_group.add_argument('--plain-output', '--plain_output', default=False,
                           type=str2bool, nargs='?', const=True,
                           help='Use plain text output instead of rich formatting (default: False)')
    
    args = parser.parse_args()
    
    # Configure display and logger mode
    from cpmcu.common.display import Display
    from cpmcu.common.logging import Logger
    Display.configure(use_plain_mode=args.plain_output)
    Logger.configure(use_plain_mode=args.plain_output)
    
    # Show test header
    mode = "Streaming" if args.use_stream else "Non-streaming"
    header_data = [("Test Mode", mode), ("Server", f"{args.host}:{args.port}")]
    render_config_table(header_data, "CPM.cu OpenAI API Test", "blue")
    
    # Run tests
    if check_server_health(args.host, args.port):
        success = test_chat_completion(args.host, args.port, args.use_stream)
        result_msg = f"{mode} test {'completed successfully' if success else 'failed'}!"
        (logger.success if success else logger.error)(result_msg)
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)
