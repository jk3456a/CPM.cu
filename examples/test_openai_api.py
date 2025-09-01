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
    def _iter_sse_events(resp):
        """Yield parsed JSON objects from an SSE stream.

        Robustly handles multi-line data fields and line-wrapped JSON by buffering
        until an empty line (event boundary). If a single JSON spans multiple
        data lines, they will be concatenated with newlines per SSE spec.
        """
        event_data_lines = []
        pending_json_fragment = ""
        decoder = json.JSONDecoder()

        for raw_line in resp.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.rstrip("\r")

            # Empty line indicates end of an SSE event
            if line == "":
                if not event_data_lines:
                    continue
                data_str = "\n".join(event_data_lines).strip()
                event_data_lines.clear()

                if data_str == "[DONE]":
                    yield {"_done": True}
                    continue

                # Attempt to parse JSON; if incomplete, buffer and try again later
                candidate = pending_json_fragment + data_str
                try:
                    obj = json.loads(candidate)
                    pending_json_fragment = ""
                    yield obj
                except json.JSONDecodeError:
                    # Keep buffering; a subsequent event may contain the rest
                    pending_json_fragment = candidate + "\n"
                continue

            # Normal data line
            if line.startswith("data:"):
                part = line[5:].lstrip()
                # Per SSE, multiple data lines compose one event payload
                event_data_lines.append(part)
            else:
                # Line-wrapped continuation without "data:" prefix; append as-is
                if event_data_lines:
                    event_data_lines[-1] += "\n" + line

        # Flush any remaining buffered event on stream end
        if event_data_lines:
            data_str = "\n".join(event_data_lines).strip()
            if data_str and data_str != "[DONE]":
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    pass

    if stream:
        logger.info("Processing streaming response...")
        finish_reason = None
        with display.create_stream("Generated Response") as stream_display:
            for chunk in _iter_sse_events(response):
                if isinstance(chunk, dict) and chunk.get("_done"):
                    break

                try:
                    if isinstance(chunk, dict):
                        choices = chunk.get('choices')
                        if isinstance(choices, list) and len(choices) > 0:
                            choice = choices[0]
                            if isinstance(choice, dict):
                                delta = choice.get('delta') or {}
                                if isinstance(delta, dict):
                                    content = delta.get('content')
                                    if isinstance(content, str) and content:
                                        stream_display.append(content)
                                finish = choice.get('finish_reason')
                                if isinstance(finish, str) and finish:
                                    finish_reason = finish
                                    break
                except Exception as e:
                    logger.error(f"Error handling chunk: {e}")

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

def test_chat_completion(host, port, stream, temperature):
    """Test chat completion"""
    url = f"http://{host}:{port}/v1/chat/completions"
    data = {
        "model": "model",
        "messages": [{"role": "user", "content": "帮我写一个名为<旧书>的小说"}],
        "stream": stream,
        "max_tokens": 4096,
        "temperature": temperature
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
    test_group.add_argument('--temperature', '--temp', type=float, default=0.0,
                           help='Sampling temperature (default: 0.0)')
    
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
        success = test_chat_completion(args.host, args.port, args.use_stream, args.temperature)
        result_msg = f"{mode} test {'completed successfully' if success else 'failed'}!"
        (logger.success if success else logger.error)(result_msg)
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)
