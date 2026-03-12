"""
Alibaba Qwen processor for TEI XML generation.

Uses llama-server (llama.cpp) for local inference via an OpenAI-compatible
HTTP API.  The server is started automatically on first use and stopped when
the script exits.

TROUBLESHOOTING
===============
- "llama-server binary not found":
    Download a pre-built binary from
    https://github.com/ggml-org/llama.cpp/releases
    and either place it on your PATH or set LLAMA_SERVER_PATH in config.py.

- "Port already in use":
    Change LLAMA_SERVER_PORT in config.py (default: 8080).
    Find the conflicting process with:
      Windows: netstat -ano | findstr :8080
      Linux:   ss -tlnp | grep 8080

- "Server did not become ready in time":
    Increase LLAMA_SERVER_STARTUP_TIMEOUT in config.py.
    Large models need more time to load.

- "Server died during inference":
    Usually an out-of-memory error.  Try USE_GPU = False or a smaller model.
"""

# Standard library imports
import atexit
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional, Tuple

# Third-party imports
from openai import OpenAI

# Local imports
import config
from utils.utils import (
    create_segment_error_response,
    create_test_response,
    create_test_tei_response,
    create_text_encoding_error,
    create_text_encoding_result,
    extract_xml_from_response,
    get_gpu_usage,
    get_system_usage,
    parse_json_response,
    prepare_prompts_for_segment,
    validate_text_data
)

# Global server state
_server_process: Optional[subprocess.Popen] = None  # running llama-server subprocess
_server_client:  Optional[OpenAI] = None             # OpenAI client pointed at the server


def count_tokens_qwen(text: str) -> int:
    """Approximate token count (words × 1.3).  Used for test-mode estimates only."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


def _format_qwen_prompt(system_message: str, user_message: str) -> str:
    """
    Format system and user messages into Qwen chat template.

    Args:
        system_message: System prompt message.
        user_message: User prompt message.

    Returns:
        Formatted prompt string for Qwen model. Includes /no_think tag
        if thinking mode is disabled.
    """
    if config.QWEN_USE_THINKING:
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant"""
    else:
        return f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}
/no_think<|im_end|>
<|im_start|>assistant"""


def _run_qwen_inference(
    system_message: str,
    user_message: str,
    coordinator: Optional[Any] = None
) -> Tuple[str, int, int, int, Dict[str, Any]]:
    """
    Run Qwen model inference via the llama-server HTTP API.

    Token counts are taken directly from the server's usage metadata (exact).
    GPU usage is sampled via GPUtil/psutil as before.

    Args:
        system_message: System prompt message.
        user_message: User prompt message.
        coordinator: Unused — kept for API compatibility with other processors.

    Returns:
        Tuple containing:
            - raw_response: Raw response text from the model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics.
    """
    client = _ensure_server_running()

    # For Qwen3: /no_think in the user message disables the thinking step
    effective_user = user_message if config.QWEN_USE_THINKING else user_message + "\n/no_think"

    gpu_usage_before = get_gpu_usage()

    try:
        response = client.chat.completions.create(
            model="local",   # llama-server ignores this field
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": effective_user},
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
        )
    except Exception as e:
        if _server_process and _server_process.poll() is not None:
            raise RuntimeError(
                "\n[ERROR] llama-server died during inference.\n"
                "  The server process is no longer running.\n"
                "  Restart the script — the server will be started again.\n"
                "  Common cause: out-of-memory. Try USE_GPU = False or a smaller model.\n"
            ) from e
        raise RuntimeError(
            f"\n[ERROR] HTTP request to llama-server failed: {e}\n"
            f"  Is the server still running on port {config.LLAMA_SERVER_PORT}?\n"
        ) from e

    raw_response  = (response.choices[0].message.content or "").strip()
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens  = response.usage.total_tokens

    gpu_usage = {
        'before': gpu_usage_before,
        'after':  get_gpu_usage(),
        'system': get_system_usage(),
        'thinking_mode': config.QWEN_USE_THINKING,
    }

    return raw_response, total_tokens, input_tokens, output_tokens, gpu_usage


def _start_llama_server() -> None:
    """Start the llama-server subprocess and wait until it is ready to accept requests."""
    global _server_process, _server_client

    # --- Check: binary exists ------------------------------------------------
    binary = config.LLAMA_SERVER_PATH
    if not shutil.which(binary) and not os.path.isfile(binary):
        raise FileNotFoundError(
            f"\n[ERROR] llama-server binary not found: '{binary}'\n\n"
            "TROUBLESHOOTING\n"
            "  1. Download a pre-built binary for your OS and GPU from:\n"
            "       https://github.com/ggml-org/llama.cpp/releases\n"
            "     Pick the build that matches your system (CPU-only, CUDA 12.x, Vulkan, …).\n"
            "  2. Either place the binary on your PATH, or set LLAMA_SERVER_PATH\n"
            "     in config.py to the full path of the binary.\n"
            "  3. On Linux/macOS: make it executable first:\n"
            "       chmod +x llama-server\n"
        )

    # --- Check: model file exists --------------------------------------------
    model_path = config.MODEL_PATH_QWEN3
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n[ERROR] Model file not found: '{model_path}'\n"
            "  Check the MODEL_PATH_QWEN3 setting in config.py.\n"
        )

    # --- Build server command ------------------------------------------------
    n_threads = max(1, multiprocessing.cpu_count() - 1)
    cmd = [
        binary,
        "--model",        model_path,
        "--port",         str(config.LLAMA_SERVER_PORT),
        "--ctx-size",     "32768",
        "--batch-size",   "1024",
        "--n-gpu-layers", str(-1 if config.USE_GPU else 0),
        "--threads",      str(n_threads),
        "--host",         "127.0.0.1",  # only listen locally for security
    ]

    print(f"Starting llama-server on port {config.LLAMA_SERVER_PORT} ...")
    print(f"  Model : {model_path}")
    print(f"  GPU   : {'all layers offloaded' if config.USE_GPU else 'CPU only'}")

    try:
        kwargs: Dict[str, Any] = {}
        if sys.platform == "win32":
            # Prevent Ctrl+C from propagating to the server on Windows
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        _server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(
            f"\n[ERROR] Failed to launch llama-server: {e}\n"
            "  Verify that LLAMA_SERVER_PATH in config.py points to the correct binary.\n"
        ) from e

    # Register cleanup so the server stops when the script exits
    atexit.register(_stop_llama_server)

    # --- Wait for server to be ready -----------------------------------------
    health_url = f"http://127.0.0.1:{config.LLAMA_SERVER_PORT}/health"
    timeout    = config.LLAMA_SERVER_STARTUP_TIMEOUT
    deadline   = time.time() + timeout
    last_error = None

    print(f"Waiting for server to be ready (timeout: {timeout}s) ...", end="", flush=True)
    while time.time() < deadline:
        # Check whether the process already exited unexpectedly
        if _server_process.poll() is not None:
            stderr_output = _server_process.stderr.read().decode(errors="replace")
            _server_process = None
            raise RuntimeError(
                "\n[ERROR] llama-server exited unexpectedly during startup.\n\n"
                f"Server output:\n{stderr_output}\n\n"
                "TROUBLESHOOTING\n"
                "  - Port conflict: if 'address already in use' appears above, change\n"
                f"    LLAMA_SERVER_PORT in config.py (currently {config.LLAMA_SERVER_PORT}).\n"
                "  - Wrong binary: ensure the binary matches your OS and CUDA version.\n"
                "  - Unsupported model: ensure the .gguf file is compatible with your\n"
                "    llama.cpp version.\n"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if json.loads(resp.read()).get("status") == "ok":
                    print(" ready.")
                    break
        except Exception as e:
            last_error = e
            print(".", end="", flush=True)
            time.sleep(2)
    else:
        _server_process.terminate()
        _server_process = None
        raise RuntimeError(
            f"\n[ERROR] llama-server did not become ready within {timeout}s.\n\n"
            "TROUBLESHOOTING\n"
            f"  1. Increase LLAMA_SERVER_STARTUP_TIMEOUT in config.py (currently {timeout}s).\n"
            "  2. Large models take longer to load — increase the timeout.\n"
            f"  3. Check if port {config.LLAMA_SERVER_PORT} is already in use:\n"
            "       Windows: netstat -ano | findstr :8080\n"
            "       Linux:   ss -tlnp | grep 8080\n"
            "  4. If USE_GPU = True, check GPU drivers and available VRAM.\n"
            f"  Last HTTP error: {last_error}\n"
        )

    # --- Create the OpenAI client pointed at our local server ----------------
    _server_client = OpenAI(
        base_url=f"http://127.0.0.1:{config.LLAMA_SERVER_PORT}/v1",
        api_key="not-needed",  # llama-server does not require a real API key
    )


def _stop_llama_server() -> None:
    """Gracefully stop the llama-server subprocess."""
    global _server_process, _server_client
    if _server_process and _server_process.poll() is None:
        print("\nStopping llama-server ...")
        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
    _server_process = None
    _server_client  = None


def _ensure_server_running() -> OpenAI:
    """Return the OpenAI client, starting the server first if not yet running."""
    if _server_process is not None and _server_process.poll() is not None:
        # Server was started previously but died unexpectedly
        stderr_output = _server_process.stderr.read().decode(errors="replace")
        raise RuntimeError(
            "\n[ERROR] llama-server process died unexpectedly.\n\n"
            f"Server output:\n{stderr_output}\n\n"
            "TROUBLESHOOTING\n"
            "  - Out of memory: try USE_GPU = False or use a smaller model.\n"
            "  - Restart the script — the server will be started again.\n"
        )
    if _server_process is None:
        _start_llama_server()
    return _server_client


# ---------------------------------------------------------------------------
# Legacy alias — kept so any code that calls get_qwen_model() still works.
# It now returns the OpenAI client instead of a Llama object.
# ---------------------------------------------------------------------------
def get_qwen_model() -> Optional[Any]:
    """
    Compatibility shim: returns the OpenAI client for the local llama-server.
    Call _ensure_server_running() directly in new code.
    """
    if not config.ENABLE_API_CALLS:
        return None
    return _ensure_server_running()



def encode_text_segment_qwen(
    text_segment: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any], Dict[str, Any]]:
    """
    Qwen-specific implementation for encoding text segments extracted from JSON objects.

    Encodes text segments (values from specific keys in JSON) into TEI encodings.

    Args:
        text_segment: Dictionary containing text segment data with keys to extract.
            Must contain keys specified in config.JSON_CONTEXT_KEY and config.JSON_ITEMS_KEY.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary mapping extracted key values to their TEI encodings.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from Qwen model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics (before/after/system/thinking_mode).
            - parsing_metadata: Dictionary with parse_success and parse_method fields.
    """
    try:
        # Prepare prompts using centralized routing logic
        system_message, user_message, items_to_analyze = prepare_prompts_for_segment(text_segment, coordinator)

        # Handle test mode
        if not config.ENABLE_API_CALLS:
            # Create test response for each item
            results = create_test_response(items_to_analyze)
            test_response = f"TEST MODE RESPONSE\n{json.dumps(results, indent=2)}"

            # Use general prompt formatting helper
            test_prompt = _format_qwen_prompt(system_message, user_message)

            # For test mode, estimate token count and get system usage
            input_tokens = count_tokens_qwen(test_prompt)
            output_tokens = count_tokens_qwen(test_response)
            total_tokens = input_tokens + output_tokens
            gpu_usage = {'before': {}, 'after': {}, 'system': get_system_usage(), 'thinking_mode': config.QWEN_USE_THINKING}

            return results, (system_message, user_message), test_response, total_tokens, input_tokens, output_tokens, gpu_usage

        # Use general inference helper
        raw_response, total_tokens, input_tokens, output_tokens, gpu_usage = _run_qwen_inference(
            system_message, user_message, coordinator
        )

        # Parse response expecting the configured items
        results, parse_success, parse_method = parse_json_response(raw_response, items_to_analyze)
        
        # Add parsing metadata to results
        parsing_metadata = {
            'parse_success': parse_success,
            'parse_method': parse_method
        }

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage, parsing_metadata

    except Exception as e:
        # Create error response for all items
        items_key = config.JSON_ITEMS_KEY
        
        # Check if this is a model loading error - these should be fatal
        error_str = str(e).lower()
        if "failed to initialize" in error_str or "model file not found" in error_str or "llama-cpp-python" in error_str:
            # Fatal error - re-raise to stop execution
            raise RuntimeError(f"Model loading failed: {e}") from e
        
        # For other errors, return error response
        results = create_segment_error_response(
            text_segment.get(items_key, []), "Qwen", str(e)
        )
        error_response = f"ERROR: {str(e)}"
        parsing_metadata = {'parse_success': False, 'parse_method': 'error'}
        return results, ("", ""), error_response, 0, 0, 0, {}, parsing_metadata


def encode_text_qwen(
    text_data: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any]]:
    """
    Qwen-specific implementation for encoding a complete plaintext file into TEI XML.

    Args:
        text_data: Dictionary with 'filename' and 'content' keys containing plaintext file data.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary with 'tei_xml' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from Qwen model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics (before/after/system/thinking_mode).

    Raises:
        ValueError: If 'filename' or 'content' fields are missing from text_data.
    """
    try:
        # Validate and extract required fields
        filename, text_content = validate_text_data(text_data)

        # Create the prompt using the coordinator's methods
        if coordinator:
            system_message = coordinator.create_system_message()
            user_message = coordinator.create_user_message_for_text(text_content)
        else:
            # Fallback for backward compatibility
            system_message = "System message placeholder"
            user_message = f"Encode this text: {text_content}"

        # Handle test mode
        if not config.ENABLE_API_CALLS:
            # Test mode - return realistic fake TEI XML response
            raw_response = create_test_tei_response("Qwen", filename)

            # Create the combined prompt in the same format as API calls
            test_prompt = _format_qwen_prompt(system_message, user_message)

            # For test mode, estimate token count and get system usage
            input_tokens = count_tokens_qwen(test_prompt)
            output_tokens = count_tokens_qwen(raw_response)
            total_tokens = input_tokens + output_tokens
            gpu_usage = {'before': {}, 'after': {}, 'system': get_system_usage(), 'thinking_mode': config.QWEN_USE_THINKING}

            # Parse the response to extract TEI XML
            tei_xml = extract_xml_from_response(raw_response)

            results = create_text_encoding_result(tei_xml)

            return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage

        # Call the model via llama-server
        raw_response, total_tokens, input_tokens, output_tokens, gpu_usage = _run_qwen_inference(
            system_message, user_message, coordinator
        )

        tei_xml = extract_xml_from_response(raw_response)
        results  = create_text_encoding_result(tei_xml)

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage

    except Exception as e:
        results = create_text_encoding_error(str(e))
        error_response = f"ERROR: {str(e)}"
        return results, ("", ""), error_response, 0, 0, 0, {}


def process_json_object_qwen(
    json_object: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any]]:
    """
    Qwen-specific implementation for processing a complete JSON object.

    Processes the entire JSON object as a unit (object processing mode).

    Args:
        json_object: A single JSON object from the input array (any structure).
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary with 'output_content' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from Qwen model.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - gpu_usage: Dictionary with GPU usage metrics (before/after/system/thinking_mode).
    """
    try:
        # Create the prompt using the coordinator's methods
        if coordinator:
            system_message = coordinator.create_system_message()
            user_message = coordinator.create_user_message_for_json_object(json_object)
        else:
            # Fallback for backward compatibility
            system_message = "System message placeholder"
            user_message = f"Process this JSON object: {json.dumps(json_object, indent=2)}"

        # Handle test mode
        if not config.ENABLE_API_CALLS:
            # Test mode - return placeholder response
            object_id = json_object.get('id') or json_object.get('object_id') or 'unknown'
            raw_response = f"""<!-- QWEN TEST MODE OUTPUT for object: {object_id} -->
<rdf:RDF>
  <rdf:Description>
    <content>{json_object.get('content', 'N/A')}</content>
  </rdf:Description>
</rdf:RDF>"""

            # Create the combined prompt in the same format as API calls
            if config.QWEN_USE_THINKING:
                test_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant"""
            else:
                test_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}
/no_think<|im_end|>
<|im_start|>assistant"""

            # For test mode, estimate token count and get system usage
            input_tokens = count_tokens_qwen(test_prompt)
            output_tokens = count_tokens_qwen(raw_response)
            total_tokens = input_tokens + output_tokens
            gpu_usage = {'before': {}, 'after': {}, 'system': get_system_usage(), 'thinking_mode': config.QWEN_USE_THINKING}

            results = {
                'output_content': raw_response,
                'success': True
            }

            return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage

        # Call the model via llama-server
        raw_response, total_tokens, input_tokens, output_tokens, gpu_usage = _run_qwen_inference(
            system_message, user_message, coordinator
        )

        output_content = extract_output_from_response(raw_response)
        results = {
            'output_content': output_content,
            'success': bool(output_content and output_content.strip())
        }

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, gpu_usage

    except Exception as e:
        results = {
            'output_content': '',
            'success': False,
            'error': str(e)
        }
        error_response = f"ERROR: {str(e)}"
        return results, ("", ""), error_response, 0, 0, 0, {}


def extract_output_from_response(response: str) -> str:
    """
    Extract output content from Qwen's response.

    Handles thinking tags and markdown code blocks. This is Qwen-specific
    as it handles the /no_think tag and thinking mode responses.

    Args:
        response: Raw response string from Qwen model.

    Returns:
        Cleaned output content with thinking tags and markdown code blocks removed.
    """
    # Remove thinking process if present
    if '<think>' in response and '</think>' in response:
        think_end = response.find('</think>')
        response = response[think_end + len('</think>'):].strip()

    # Remove markdown code blocks if present
    response = response.strip()

    # Remove ```xml or ```rdf and ``` markers if present
    if response.startswith('```'):
        lines = response.split('\n')
        # Remove first line (```xml or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        response = '\n'.join(lines).strip()

    return response


if __name__ == "__main__":
    print("Please run: python llm_processing.py")