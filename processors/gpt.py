"""
OpenAI GPT processor for TEI XML generation.

This module provides GPT-specific implementations for encoding text segments
and complete plaintext files into TEI XML format using OpenAI's API.
"""

# Standard library imports
import json
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
    parse_json_response,
    prepare_prompts_for_segment,
    validate_text_data
)


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client with lazy-loaded API key.

    Returns:
        Initialized OpenAI client instance.

    Raises:
        ValueError: If API key is not configured when API calls are enabled.
    """
    return OpenAI(api_key=config.OPENAI_API_KEY)


def encode_text_segment_gpt(
    text_segment: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int, Dict[str, Any]]:
    """
    GPT-specific implementation for encoding text segments extracted from JSON objects.

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
            - raw_response: Raw response text from GPT API.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.
            - parsing_metadata: Dictionary with parse_success and parse_method fields.

    Raises:
        ValueError: If required fields are missing from text_segment.
    """
    try:
        # Prepare prompts using centralized routing logic
        system_message, user_message, items_to_analyze = prepare_prompts_for_segment(text_segment, coordinator)

        # Call the OpenAI API (if enabled in config)
        if config.ENABLE_API_CALLS:
            # Determine which parameters to use based on model
            api_params = {
                "model": config.MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            }

            # Reasoning models (o1, o3) use max_completion_tokens
            is_reasoning_model = any(model_name in config.MODEL_NAME.lower() for model_name in [
                'o1-', 'o3-'
            ])
            
            # GPT-5 models don't support token limit parameters in Chat Completions API
            is_gpt5 = 'gpt-5' in config.MODEL_NAME.lower()

            if is_reasoning_model:
                api_params['max_completion_tokens'] = config.MAX_TOKENS
            elif not is_gpt5:
                # Standard models use max_tokens
                api_params['max_tokens'] = config.MAX_TOKENS
                api_params['temperature'] = config.TEMPERATURE
            # For GPT-5, don't set any token limit parameter

            client = get_openai_client()
            response = client.chat.completions.create(**api_params)

            # Get the raw response content
            raw_response = response.choices[0].message.content.strip()

            # Get exact token usage from API response
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # Fallback: estimate token counts (rough approximation)
                input_tokens = len((system_message + user_message).split()) * 1.3  # Rough token estimate
                output_tokens = len(raw_response.split()) * 1.3
                total_tokens = input_tokens + output_tokens
        else:
            # Test mode - use shared test response generator
            results = create_test_response(items_to_analyze)
            raw_response = f"TEST MODE RESPONSE\n{json.dumps(results, indent=2)}"

            # For test mode, estimate token count
            input_tokens = int(len((system_message + user_message).split()) * 1.3)
            output_tokens = int(len(raw_response.split()) * 1.3)
            total_tokens = input_tokens + output_tokens

            parsing_metadata = {'parse_success': True, 'parse_method': 'test_mode'}

            return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, parsing_metadata

        # Parse the response using shared JSON parser
        results, parse_success, parse_method = parse_json_response(raw_response, items_to_analyze)

        parsing_metadata = {
            'parse_success': parse_success,
            'parse_method': parse_method
        }

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens, parsing_metadata

    except Exception as e:
        items_key = config.JSON_ITEMS_KEY
        context_key = config.JSON_CONTEXT_KEY
        results = create_segment_error_response(
            text_segment.get(items_key, []), "GPT", str(e)
        )
        error_response = f"ERROR: {str(e)}"
        # Create fallback messages for error case
        system_message = "Error occurred before message creation"
        user_message = f"Error prompt for: {text_segment.get(context_key, 'unknown text')}"
        parsing_metadata = {'parse_success': False, 'parse_method': 'error'}
        return results, (system_message, user_message), error_response, 0, 0, 0, parsing_metadata


def encode_text_gpt(
    text_data: Dict[str, Any],
    coordinator: Optional[Any] = None
) -> Tuple[Dict[str, Any], Tuple[str, str], str, int, int, int]:
    """
    GPT-specific implementation for encoding a complete plaintext file into TEI XML.

    Args:
        text_data: Dictionary with 'filename' and 'content' keys containing plaintext file data.
        coordinator: LLMProcessingCoordinator instance for accessing shared methods.
            If None, uses fallback placeholder messages.

    Returns:
        Tuple containing:
            - results: Dictionary with 'tei_xml' (str) and 'success' (bool) keys.
            - (system_message, user_message): Tuple of system and user prompts used.
            - raw_response: Raw response text from GPT API.
            - total_tokens: Total token count (input + output).
            - input_tokens: Input token count.
            - output_tokens: Output token count.

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

        # Call the OpenAI API (if enabled in config)
        if config.ENABLE_API_CALLS:
            # Determine which parameters to use based on model
            api_params = {
                "model": config.MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            }

            # Reasoning models (o1, o3) use max_completion_tokens
            is_reasoning_model = any(model_name in config.MODEL_NAME.lower() for model_name in [
                'o1-', 'o3-'
            ])
            
            # GPT-5 models don't support token limit parameters in Chat Completions API
            is_gpt5 = 'gpt-5' in config.MODEL_NAME.lower()

            if is_reasoning_model:
                api_params['max_completion_tokens'] = config.MAX_TOKENS
            elif not is_gpt5:
                # Standard models use max_tokens
                api_params['max_tokens'] = config.MAX_TOKENS
                api_params['temperature'] = config.TEMPERATURE
            # For GPT-5, don't set any token limit parameter

            client = get_openai_client()
            response = client.chat.completions.create(**api_params)

            # Get the raw response content
            raw_response = response.choices[0].message.content.strip()

            # Get exact token usage from API response
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # Fallback: estimate token counts
                input_tokens = int(len((system_message + user_message).split()) * 1.3)
                output_tokens = int(len(raw_response.split()) * 1.3)
                total_tokens = input_tokens + output_tokens
        else:
            # Test mode - return realistic fake TEI XML response
            raw_response = create_test_tei_response("GPT", filename)

            # For test mode, estimate token count
            input_tokens = int(len((system_message + user_message).split()) * 1.3)
            output_tokens = int(len(raw_response.split()) * 1.3)
            total_tokens = input_tokens + output_tokens

        # Parse the response to extract TEI XML
        tei_xml = extract_xml_from_response(raw_response)

        # Create results dictionary
        results = create_text_encoding_result(tei_xml)

        return results, (system_message, user_message), raw_response, total_tokens, input_tokens, output_tokens

    except Exception as e:
        results = create_text_encoding_error(str(e))
        error_response = f"ERROR: {str(e)}"
        # Create fallback messages for error case
        system_message = "Error occurred before message creation"
        user_message = f"Error prompt for file: {text_data.get('filename', 'unknown')}"
        return results, (system_message, user_message), error_response, 0, 0, 0


if __name__ == "__main__":
    print("Please run: python llm_processing.py")
