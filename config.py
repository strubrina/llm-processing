"""Configuration module for LLM processing.

This module contains all configuration settings for the LLM processing pipeline,
including model selection, API keys, prompt configuration, and input/output paths.

API keys are loaded from keys.py (which should not be committed to version control).
See keys.py.example for the expected format.
"""

import os

# =============================================================================
# API KEYS
# =============================================================================
# Import API keys (optional - only needed when ENABLE_API_CALLS = True)
try:
    import keys
    OPENAI_API_KEY = keys.OPENAI
    ANTHROPIC_API_KEY = keys.ANTHROPIC
except (ImportError, AttributeError):
    # Keys not available - only a problem if API calls are enabled
    OPENAI_API_KEY = None
    ANTHROPIC_API_KEY = None


# =============================================================================
# TESTING AND GENERAL SETTINGS
# =============================================================================

# Set to True to enable API calls, False for test mode (prompts only)
ENABLE_API_CALLS = False

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model Selection - Uncomment ONE model to use
#MODEL_NAME = "gpt-5-mini-2025-08-07"         # OpenAI GPT-5
# MODEL_NAME = "claude-sonnet-4-5-20250929"    # Anthropic Claude Sonnet 4.5
# MODEL_NAME = "qwen3-14B-Q6"                # Alibaba Qwen3 (local)
MODEL_NAME = "qwen3-14B-IQ2_XS"                # Alibaba Qwen3 (local) - small model for CPU!
# MODEL_NAME = "olmo2-32B-instruct-Q4"         # AI2 OLMo2 32B (local)
# MODEL_NAME = "gpt-4o-mini"    # DeepSeek R1 (local)

# Qwen-specific: Enable/disable thinking mode
QWEN_USE_THINKING = False


# Local Model Paths - Adjust based on your hardware

# GPU paths (requires ~20 GB VRAM):
# MODEL_PATH_QWEN3 = os.path.join('..', '..', '..', 'models', 'Qwen_Qwen3-14B-Q6_K.gguf')
# MODEL_PATH_OLMO2 = os.path.join('..', '..', '..', 'models', 'OLMo-2-0325-32B-Instruct-Q4_K_S.gguf')

# CPU paths (requires ~8-9 GB RAM):
MODEL_PATH_QWEN3 = os.path.join('models', 'Qwen_Qwen3-14B-IQ2_XS.gguf')



# Model Parameters
TEMPERATURE = 0.6
MAX_TOKENS = 7000

# Hardware Acceleration
USE_GPU = False  # Set to False to run on CPU only (slower but works without GPU)

# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================

# Prompt version selection - switch between different prompt designs
# Available versions should be in prompts/{version}/ directories
# Each version must contain prompt.txt (required)
# Optional files: encoding_rules.txt, few_shot_examples.txt
PROMPT_VERSION = "editorial_interventions"

# Derived path to prompt directory (do not modify)
PROMPT_DIR = os.path.join("prompts", PROMPT_VERSION)

# User message template file (optional)
# This file should be in the prompt folder specified by PROMPT_VERSION
# You can use different files for different workflows by changing this value
USER_MESSAGE = "user_message.txt"


# =============================================================================
# INPUT/OUTPUT SETTINGS
# =============================================================================

# Input Type: "txt" for a folder containing .txt files, or "json" for a single JSON file
INPUT_TYPE = "json"  # Options: "txt" or "json"

# Input Path:
#   - For "txt": path to directory containing .txt files
#   - For "json": path to the JSON file
INPUT_PATH = os.path.join("..", "data", "input", "dummy.json")

# JSON Processing Mode (only used when INPUT_TYPE = "json")
#   - "key_extraction": Extracts and analyzes specific keys from JSON objects
#   - "object_processing": Processes complete JSON objects as units
JSON_PROCESSING_MODE = "key_extraction"  # Options: "key_extraction" or "object_processing"

# Output: Directory for generated output files
OUTPUT_DIR = os.path.join("..", "data", "output", "test")

# Output file extension
# Options: ".xml" for XML files, ".json" for JSON output
OUTPUT_EXTENSION = ".json"

# XML output type (only used when OUTPUT_EXTENSION = ".xml")
# Options: "tei" for TEI XML encoding, "rdf" for RDF-XML output
XML_OUTPUT_TYPE = "rdf"  # Options: "tei" or "rdf"

# JSON output mode (only used when OUTPUT_EXTENSION = ".json" for text processing)
#   - "raw": Extract content after <think> tags (anything between { and })
#   - "json-array": Create properly structured JSON array with validated objects
JSON_OUTPUT_MODE = "json-array"  # Options: "raw" or "json-array"


# =============================================================================
# JSON KEY EXTRACTION CONFIGURATION
# (Only used when JSON_PROCESSING_MODE = "key_extraction")
# =============================================================================
# Configure the type of key extraction workflow to use

# JSON Extraction Type:
#   - "tei_encoding": TEI encoding workflow (context + text segments to encode)
#   - "information_extraction": Extract specific key-value pairs from JSON
JSON_EXTRACTION_TYPE = "tei_encoding"  # Options: "tei_encoding" or "information_extraction"

# Key extraction output format
#   - "xml_mapping": Create XML update mapping files for TEI encoding workflows
#   - "json": Output as JSON (uses JSON_OUTPUT_MODE for raw/json-array)
KEY_EXTRACTION_OUTPUT_FORMAT = "xml_mapping"  # Options: "xml_mapping" or "json"


# =============================================================================
# TEI ENCODING WORKFLOW CONFIGURATION
# (Only used when JSON_EXTRACTION_TYPE = "tei_encoding")
# =============================================================================
# For workflows that encode text segments with TEI markup

# Key containing the context text
JSON_CONTEXT_KEY = "context"

# Key containing the text segments to encode
# Can be either a list of strings or a single string
JSON_ITEMS_KEY = "bracketed_sequences"

# Keys to preserve as metadata in the output
JSON_METADATA_KEYS = ["element_id", "filename", "xpath", "index"]

# Labels used in user prompts
JSON_CONTEXT_LABEL = "Context"
JSON_ITEMS_LABEL = "Bracketed Sequences"


# =============================================================================
# XML UPDATE MAPPING KEYS (for TEI encoding workflow with xml_mapping output)
# =============================================================================
# Configure which JSON keys identify elements for XML update operations.
# These map to your input JSON structure.

# Key containing the filename to update
XML_MAPPING_FILENAME_KEY = "filename"

# Key containing the element identifier
XML_MAPPING_ELEMENT_ID_KEY = "element_id"

# Key containing the XPath location
XML_MAPPING_XPATH_KEY = "xpath"

# =============================================================================
# JSON OUTPUT FIELD MAPPING (for key_extraction mode)
# =============================================================================
# Configure which field names the LLM uses in its JSON output.
# Update these if your prompt generates different field names.

# Field name for the TEI encoding in LLM output
JSON_OUTPUT_TEI_FIELD = "tei"  # e.g., "tei" or "tei_encoding"

# Field name for the intervention type in LLM output
JSON_OUTPUT_TYPE_FIELD = "type"  # e.g., "type" or "intervention_type"

# Field name for the explanation in LLM output (optional)
JSON_OUTPUT_EXPLANATION_FIELD = None  # Set to None if not used


# =============================================================================
# INFORMATION EXTRACTION WORKFLOW CONFIGURATION
# (Only used when JSON_EXTRACTION_TYPE = "information_extraction")
# =============================================================================
# For workflows that extract specific metadata fields from JSON objects

# Keys to extract from each JSON object (array of key names)
# Example: ["dateline", "signature", "salutation"]
JSON_DATA_KEYS = ["dateline", "signature"]

# Labels for data keys used in user prompts (must match order and length of JSON_DATA_KEYS)
# If not provided or empty, will use the key names directly
# Example: ["Dateline", "Signature", "Salutation"]
JSON_DATA_LABELS = ["Dateline", "Signature"]

# Keys from the input JSONto preserve as metadata in the output
# These will be included in the output JSON but not processed by the LLM
JSON_METADATA_KEYS_INFO = ["letter"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_openai_api_key() -> str:
    """
    Get OpenAI API key with validation.

    Returns:
        OpenAI API key string.

    Raises:
        ValueError: If ENABLE_API_CALLS is True but OPENAI_API_KEY is not set.
    """
    if ENABLE_API_CALLS and not OPENAI_API_KEY:
        raise ValueError(
            "ENABLE_API_CALLS is True but OPENAI_API_KEY is not set. "
            "Please create keys.py with your API key."
        )
    return OPENAI_API_KEY


def get_anthropic_api_key() -> str:
    """
    Get Anthropic API key with validation.

    Returns:
        Anthropic API key string.

    Raises:
        ValueError: If ENABLE_API_CALLS is True but ANTHROPIC_API_KEY is not set.
    """
    if ENABLE_API_CALLS and not ANTHROPIC_API_KEY:
        raise ValueError(
            "ENABLE_API_CALLS is True but ANTHROPIC_API_KEY is not set. "
            "Please create keys.py with your API key."
        )
    return ANTHROPIC_API_KEY
