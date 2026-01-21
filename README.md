# LLM Processing for TEI XML Generation

Automated TEI (Text Encoding Initiative) XML generation from plaintext sources using Large Language Models.

## Overview

This project processes plaintext and generates TEI XML encoded documents. It supports multiple LLM providers and allows for flexible prompt engineering and configuration.

**Key Features:**
- **Multi-model support**: OpenAI GPT, Anthropic Claude, Alibaba Qwen, OLMo
- **Flexible configuration**: Easily switch between models, prompts, and parameters
- **Configurable JSON processing**: Adapt to any JSON structure by configuring key mappings
- **Comprehensive metrics**: Token usage tracking, cost estimation, processing time
- **GPU monitoring**: Track GPU utilization for local models
- **Batch processing**: Process multiple text files with error handling and logging
- **Test mode**: Generate prompts without making API calls for development

## Supported Workflows

### 1. Text Processing Workflow
Processes complete plaintext files into TEI XML. This workflow:
- Takes plaintext files from a configured directory
- Generates TEI XML encoded documents
- Supports batch processing of multiple files
- Saves outputs with comprehensive logging

**Input**: Plaintext files (`.txt` files in a directory)
**Output**: TEI XML files with full encoding

### 2. JSON Processing Workflow
Processes JSON files with two modes:

#### a) Key Extraction Mode
Extracts and analyzes specific keys from JSON objects, providing TEI encodings for the extracted values. This mode is configurable to work with any JSON structure by specifying which keys to extract and analyze.

**Input**: JSON file with objects containing keys to extract
**Output**: XML update mappings with TEI encodings

#### b) Object Processing Mode
Processes complete JSON objects as units, generating direct output files.

**Input**: JSON file with any JSON structure
**Output**: Direct output files (XML/RDF)

## Requirements

### Python Version
- **Required**: Python 3.8 or higher
- **Tested with**: Python 3.10, 3.11
- **Recommended**: Python 3.10 or 3.11 for best compatibility

### Core Dependencies (Required)
```bash
anthropic>=0.66.0       # Claude API
openai>=1.99.0          # GPT API
pandas>=2.3.0           # Data analysis
lxml>=6.0.0             # XML processing
```

**Note**: For reproducibility, record exact versions used for published results:
```bash
pip freeze > requirements_versions.txt
```

### Optional Dependencies (For Local Models)
Only required if using local models (Qwen, OLMo):
```bash
llama-cpp-python>=0.3.16    # Local model inference
torch>=2.8.0                # GPU support and diagnostics
gputil                      # GPU monitoring
psutil                      # System resource monitoring
```

**Important**: For GPU support with local models, install `llama-cpp-python` with CUDA wheels (see Installation section below).

## Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Important for Local Models (Qwen, OLMo):**
   Please also install llama-cpp-python with CUDA support:
   ```bash
   pip uninstall llama-cpp-python
   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
   ```
   This ensures proper GPU acceleration. Without this step, local models may run very slowly or fail to utilize your GPU.

2. **Set up API keys** (for cloud models only)

   You can provide API keys in one of two ways:

   **Option A: Using a `keys.py` file** (default)

   Create a `keys.py` file in the project root and add your API keys:
   ```python
   OPENAI = "your-openai-api-key-here"
   ANTHROPIC = "your-anthropic-api-key-here"
   ```

   **Option B: Using environment variables** (recommended for production)

   Set environment variables instead:
   ```bash
   # On Linux/macOS:
   export OPENAI_API_KEY="your-openai-api-key-here"
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"

   # On Windows (PowerShell):
   $env:OPENAI_API_KEY="your-openai-api-key-here"
   $env:ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   ```

   Then modify `config.py` to read from environment variables:
   ```python
   import os

   # In the API KEYS section, replace the keys.py import with:
   OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
   ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
   ```

3. **Configure the model and settings**

   Edit `config.py` to select your desired model and settings (see Configuration section below).

## Usage

### Basic Usage

**Interactive Mode** (default):
```bash
python llm_processing.py
```

The script will:
1. Prompt you to select a workflow (Text Processing or JSON Processing)
2. Load input files from the configured directory
3. Process each file using the selected LLM
4. Save TEI XML outputs and processing logs

**Command-Line Mode** (with argparse):
```bash
# Run text processing workflow directly
python llm_processing.py --workflow text
# or
python llm_processing.py -w text

# Run JSON processing workflow directly
python llm_processing.py --workflow json
# or
python llm_processing.py -w json

# Show version
python llm_processing.py --version

# Show help
python llm_processing.py --help
```

### Test Mode (No API Calls)

To generate prompts without making API calls:

```python
# In config.py
ENABLE_API_CALLS = False
```

This is useful for:
- Testing prompt configurations
- Validating input files
- Development and debugging
- Estimating costs before processing

### Processing Custom Text Files

1. Place your plaintext files in `data/input/` (or a custom directory)
2. Update `config.py`:
   ```python
   INPUT_TYPE = "txt"
   INPUT_PATH = "data/input"  # Your input directory
   ```
3. Run `python llm_processing.py`

### Processing JSON Files

1. Prepare your JSON file with the appropriate structure
2. Update `config.py`:
   ```python
   INPUT_TYPE = "json"
   INPUT_PATH = "data/input/json/your_file.json"
   JSON_PROCESSING_MODE = "key_extraction"  # or "object_processing"

   # If using key_extraction mode, configure the keys:
   JSON_CONTEXT_KEY = "your_context_field"
   JSON_ITEMS_KEY = "your_items_field"
   JSON_METADATA_KEYS = ["id", "filename", "xpath"]
   ```
3. Run `python llm_processing.py`

### Creating Few-Shot Examples

If using a prompt version that includes few-shot examples:

```bash
python create_few_shot_examples.py
```

This extracts examples from paired XML/TXT files in `data/original_sample/few-shot_sample/`.
It requires to have a txt subfolder with the plaintext files and all the corresponding XML reference files in the main folder.

## Configuration

Edit `config.py` to customize the processing:

### Model Selection

**OpenAI GPT:**
```python
MODEL_NAME = "gpt-5-mini-2025-08-07"
TEMPERATURE = 0.3 # for cloud models 0.3 is recommended
MAX_TOKENS = 8000
```

**Anthropic Claude:**
```python
MODEL_NAME = "claude-sonnet-4-5-20250929"
TEMPERATURE = 0.3
MAX_TOKENS = 8000
```

**Local Models (Qwen):**
```python
MODEL_NAME = "qwen3-14B-Q6"
MODEL_PATH_QWEN3 = '../../../models/Qwen_Qwen3-14B-Q6_K.gguf'
QWEN_USE_THINKING = True  # Enable/disable thinking mode
TEMPERATURE = 0.6 # for local models this is the recommended temperature setting
MAX_TOKENS = 8000
```

**Local Models (OLMo):**
```python
MODEL_NAME = "olmo2-32B-instruct-Q4"
MODEL_PATH_OLMO2 = '../../../models/OLMo-2-0325-32B-Instruct-Q4_K_S.gguf'
TEMPERATURE = 0.6
MAX_TOKENS = 8000
```

**Note on Model Compatibility:**
The model processors have been tested with:
- **GPT**: GPT-4, GPT-5
- **Claude**: Claude 4 / 4.5 (Sonnet)
- **Qwen**: Qwen3
- **OLMo**: OLMo2

Newer model versions may have different API parameters or response formats. If you encounter issues with newer models, the processor code in `processors/` may need to be adapted to accommodate API changes.

### Prompt Configuration

Switch between different prompt versions:
```python
PROMPT_VERSION = "prompts_editorial_interventions"  # Example: prompts_editorial_interventions
```

Each prompt version directory should be in `prompts/{version}/` and contain:
- `prompt.txt` (required) - Main prompt instructions for system message
- `encoding_rules.txt` (optional) - TEI encoding guidelines (added to system message)
- `few_shot_examples.txt` (optional) - Input-output examples (added to system message)
- `user_message.txt` (optional) - User message template (if not provided, raw content is sent)

```python
USER_MESSAGE = "user_message.txt"  # Optional template file in prompt directory
```


### Input/Output Paths

```python
# Input type: "txt" for plaintext files, or "json" for JSON processing
INPUT_TYPE = "txt"  # Options: "txt" or "json"
# Note: When INPUT_TYPE = "txt", the system always processes .txt files

# Input path (directory for txt, file path for json)
INPUT_PATH = "data/input/json/editorial_interventions"

# JSON Processing Mode (only used when INPUT_TYPE = "json")
JSON_PROCESSING_MODE = "key_extraction"  # Options: "key_extraction" or "object_processing"

# Output directory for generated files
OUTPUT_DIR = "data/output"
OUTPUT_EXTENSION = ".xml"
```

### JSON Key Mapping Configuration

When using `JSON_PROCESSING_MODE = "key_extraction"`, you can configure which keys from your JSON structure to use:

```python
# =============================================================================
# JSON KEY MAPPING (for key_extraction mode)
# =============================================================================

# Key containing the context/text to analyze
JSON_CONTEXT_KEY = "full_element_text"

# Key containing the items to extract and process (expects a list)
JSON_ITEMS_KEY = "bracketed_sequences"

# Keys to preserve as metadata in the output (list of key names)
JSON_METADATA_KEYS = ["element_id", "filename", "xpath", "index"]
```

**Example:** If your JSON structure uses different key names:

```json
{
  "id": "123",
  "source_file": "letter.xml",
  "text_context": "Some text with [annotations]",
  "annotations": ["[annotation1]", "[annotation2]"]
}
```

You would configure:

```python
JSON_CONTEXT_KEY = "text_context"
JSON_ITEMS_KEY = "annotations"
JSON_METADATA_KEYS = ["id", "source_file"]
```

This allows the same code to work with different JSON formats without modification. See `WORKFLOW_ANALYSIS.md` for more details on the JSON processing workflows.

### API Control

```python
# Set to False to generate prompts without making API calls
ENABLE_API_CALLS = True
```

## Output Structure

Processing results are organized in timestamped directories:

```
data/output/{model_name}/processing_{timestamp}/
├── letter1.xml                    # LLM-generated TEI XML files
├── letter2.xml
├── ...
└── log/
    ├── processing_metadata.json   # Processing metrics and costs
    ├── llm_responses.txt          # Raw LLM responses
    └── prompts_user.txt           # User prompts sent to LLM
```

### Processing Metadata

The `processing_metadata.json` file contains:
- Model information (name, type, version)
- Processing time and timestamp
- Token usage (input/output/total)
- Cost estimation (for API models)
- GPU utilization (for local models)
- Temperature and prompt version

### Example Metadata:
```json
{
  "timestamp": "20250106_143022",
  "model_instance": "claude-sonnet-4-5-20250929",
  "model_type": "claude",
  "prompt_version": "prompt_v5",
  "temperature": 0.3,
  "processing_time": "00:05:23",
  "token_count": 145623,
  "input_tokens": 98234,
  "output_tokens": 47389,
  "estimated_cost_usd": 4.26
}
```

## Project Structure

```
llm_processing/
├── llm_processing.py              # Main coordinator script
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── processors/                    # Model-specific implementations
│   ├── __init__.py
│   ├── gpt.py                    # OpenAI GPT
│   ├── claude.py                 # Anthropic Claude
│   ├── qwen.py                   # Alibaba Qwen
│   └── olmo.py                   # OLMo
├── utils/                         # Shared utility functions
│   ├── __init__.py
│   └── utils.py                  # Utility functions for processors
├── prompts/                       # Prompt versions
│   └── {prompt_version}/         # e.g., prompts_editorial_interventions/
│       ├── prompt.txt
│       ├── encoding_rules.txt
│       └── few_shot_examples.txt
└── data/
    ├── input/                     # Input files (txt or json)
    ├── output/                    # Generated TEI XML files
    └── original_sample/           # Sample files for examples
```

## Cost Estimation

The tool automatically calculates costs for API-based models:

**OpenAI GPT-5 Pricing (per million tokens):**
- GPT-5-nano: $0.05 input / $0.40 output
- GPT-5-mini: $0.25 input / $2.00 output
- GPT-5: $1.25 input / $10.00 output

**Anthropic Claude Pricing (per million tokens):**
- Claude Sonnet 4: $3.00 input / $15.00 output

For any other models please add pricing.

## Troubleshooting

### "MODEL_NAME not defined" error
Ensure `config.py` has exactly one uncommented `MODEL_NAME` line.

### "API key not found" error
1. Verify `keys.py` exists and contains your API keys
2. Or set `ENABLE_API_CALLS = False` for test mode

### Local model fails to load
1. Verify the model path in `config.py` points to a valid `.gguf` file
2. Ensure `llama-cpp-python` is installed with GPU support if needed
3. Check that you have sufficient RAM/VRAM for the model

### Local models running very slowly or not using GPU
This is a common issue with `llama-cpp-python`. The default pip installation doesn't include GPU support.

**Solution:**
```bash
pip uninstall llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

This installs the CUDA 12.1 version with proper GPU acceleration. Adjust the `cu121` suffix if you're using a different CUDA version (e.g., `cu118` for CUDA 11.8).

### No output files generated
1. Check console output for errors
2. Verify input directory contains `.txt` files
3. Ensure output directory has write permissions
4. Review `log/llm_responses.txt` for LLM errors

## Development

### Adding a New Model

1. Create a new processor in `processors/`:
   ```python
   # processors/new_model_processor.py
   import config
   from utils.utils import (
       create_segment_error_response,
       create_test_response,
       create_test_tei_response,
       create_text_encoding_error,
       create_text_encoding_result,
       extract_tei_xml_from_response,
       parse_json_response,
       validate_text_data,
       validate_text_segment
   )

   def analyze_letter_new_model(letter_data, coordinator=None):
       # Implementation
       pass
   ```

2. Register in `llm_processing.py`:
   ```python
   self.processors["newmodel"] = {
       "name": "New Model",
       "module": "processors.new_model_processor",
       "letter_function": "analyze_letter_new_model"
   }
   ```

3. Add detection in `detect_model_from_config()`:
   ```python
   elif "newmodel" in model_name_lower:
       return "newmodel"
   ```

### Creating a New Prompt Version

1. Create directory: `prompts/prompt_v6/`
2. Add required files:
   - `prompt.txt` - Main instructions (required)
   - `encoding_rules.txt` - Guidelines (optional)
   - `few_shot_examples.txt` - Examples (optional)
   - `user_message.txt` - User message template (optional - if omitted, raw content is sent)
3. Update `config.py`:
   ```python
   PROMPT_VERSION = "prompt_v6"
   USER_MESSAGE = "user_message.txt"  # Optional, defaults to this filename
   ```

### Using Different JSON Structures

The JSON key extraction workflow is designed to work with any JSON structure. To adapt it to your JSON format:

1. Identify your JSON structure:
   - Which field contains the context text? → `JSON_CONTEXT_KEY`
   - Which field contains the items to analyze? → `JSON_ITEMS_KEY`
   - Which fields should be preserved as metadata? → `JSON_METADATA_KEYS`

2. Update `config.py` with your field names:
   ```python
   JSON_CONTEXT_KEY = "your_context_field"
   JSON_ITEMS_KEY = "your_items_field"
   JSON_METADATA_KEYS = ["field1", "field2", "field3"]
   ```

3. Ensure your JSON file matches the expected structure:
   - `JSON_CONTEXT_KEY` should contain a string
   - `JSON_ITEMS_KEY` should contain a list
   - `JSON_METADATA_KEYS` can be any fields you want preserved

See `WORKFLOW_ANALYSIS.md` for detailed information about the JSON processing workflows.

## Reproducibility

For reproducing published results, see the comprehensive reproducibility guide below.

### Code Version

**Current Version**: 1.0.0

For published results, record the git commit hash:
```bash
git rev-parse HEAD
```

### Environment Requirements

#### Python Version
- **Required**: Python 3.8 or higher
- **Tested with**: Python 3.10, 3.11

#### System Dependencies

**For Local Models (Qwen, OLMo):**
- **CUDA Toolkit** (for GPU processing): Version 11.8, 12.1, 12.4, or 13.0
- **NVIDIA GPU** with sufficient VRAM (see model-specific requirements below)
- **CPU alternative**: Models can run on CPU (slower, requires 8-9 GB RAM minimum)

**For Cloud Models (GPT, Claude):**
- Internet connection for API access
- Valid API keys (create `keys.py` file with your API keys)

### Model Versions

#### OpenAI GPT Models
- **Model Name**: As specified in `config.MODEL_NAME`
- **API Version**: Current as of 2025
- **Example**: `gpt-5-mini-2025-08-07`

#### Anthropic Claude Models
- **Model Name**: As specified in `config.MODEL_NAME`
- **API Version**: Current as of 2025
- **Example**: `claude-sonnet-4-5-20250929`

#### Local Models

**Qwen3:**
- **Model File**: GGUF format
- **Recommended Quantization**: Q6_K for GPU (14B model), IQ2_XS for CPU
- **File Size**: ~8-10 GB (Q6_K), ~4-5 GB (IQ2_XS)
- **VRAM Requirements**:
  - Q6_K: ~20 GB VRAM
  - IQ2_XS: ~8-9 GB RAM (CPU mode)
- **Model Path**: Configured in `config.MODEL_PATH_QWEN3`

**OLMo2:**
- **Model File**: GGUF format
- **Recommended Quantization**: Q4_K_S
- **File Size**: ~20 GB
- **VRAM Requirements**: ~24 GB VRAM
- **Model Path**: Configured in `config.MODEL_PATH_OLMO2`

### Critical Settings for Reproducibility

All settings are in `config.py`. For published results, record:

1. **Model Selection**:
   ```python
   MODEL_NAME = "qwen3-14B-Q6"  # or other model
   ```

2. **Temperature** (affects randomness):
   ```python
   TEMPERATURE = 0.1  # Lower = more deterministic
   ```

3. **Max Tokens**:
   ```python
   MAX_TOKENS = 10000
   ```

4. **Prompt Version**:
   ```python
   PROMPT_VERSION = "prompts_editorial_interventions"
   ```

5. **Hardware Settings**:
   ```python
   USE_GPU = True  # or False for CPU
   ```

6. **Qwen Thinking Mode** (if using Qwen):
   ```python
   QWEN_USE_THINKING = False  # or True
   ```

### Non-Deterministic Operations

**Temperature Settings**:
- **Cloud Models (GPT, Claude)**: Temperature affects output randomness
  - Lower temperature (0.1-0.3) = more deterministic, consistent results
  - Higher temperature (0.6-1.0) = more creative, variable results
- **Local Models (Qwen, OLMo)**: Temperature also affects randomness
  - Recommended: 0.1-0.6 depending on use case

**Note**: Even with temperature=0.1, LLM outputs may vary slightly between runs due to:
- Model internal randomness
- API response variations (for cloud models)
- Floating-point precision differences

For maximum reproducibility:
- Use the lowest temperature that produces acceptable results
- Run multiple times and compare outputs
- Document any observed variations

### Reproducing Published Results

#### Step 1: Record Configuration

Before running, record the exact configuration used:
```bash
# Save git commit hash
git rev-parse HEAD > reproducibility_info.txt

# Save config.py contents
cp config.py config_backup.py

# Record Python version
python --version >> reproducibility_info.txt

# Record package versions
pip freeze > requirements_versions.txt
```

#### Step 2: Set Configuration

1. Edit `config.py` to match published configuration
2. Ensure model files are in correct locations (for local models)
3. Verify API keys are set (for cloud models)

#### Step 3: Run Processing

```bash
# Interactive mode
python llm_processing.py

# Or with workflow selection
python llm_processing.py --workflow text
python llm_processing.py --workflow json
```

#### Step 4: Verify Outputs

Outputs are saved to:
- `{OUTPUT_DIR}/{MODEL_NAME}/processing_{timestamp}/` - Generated files
- `{OUTPUT_DIR}/{MODEL_NAME}/processing_{timestamp}/log/` - Logs and metadata

Compare outputs with published results:
- File structure should match
- File naming conventions should match
- Content should be similar (allowing for LLM variability)

### Environment-Specific Considerations

**GPU vs CPU:**
- **GPU**: Faster processing, requires CUDA setup
- **CPU**: Slower but more portable, no special setup needed
- Results should be identical, only speed differs

**Operating System:**
- **Windows**: Tested and working
- **Linux**: Should work (not extensively tested)
- **macOS**: May work with CPU mode (not tested)

**CUDA Versions:**
- Different CUDA versions may require different llama-cpp-python wheels
- Results should be identical regardless of CUDA version
- If GPU issues occur, fall back to CPU mode

### Known Variations

**Expected Variations:**
1. **LLM Output Variability**: Even with low temperature, outputs may vary slightly
2. **Token Counting**: Approximate counts may differ slightly between runs
3. **Timing Information**: Processing times will vary based on system load

**Unexpected Variations:**
If results differ significantly from published results, check:
1. Model version/quantization matches
2. Temperature setting matches
3. Prompt version matches
4. Input data matches exactly
5. Python package versions match (especially llama-cpp-python)

