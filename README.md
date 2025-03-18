# Document Processing and Chat Application System

A powerful system for processing various document formats into vector embeddings and providing an interactive chat interface for querying document content using multiple AI models.

## Copyright and License

Copyright © 2025 MKM Research Labs. All rights reserved.

This software is provided under license by MKM Research Labs. Use, reproduction, distribution, or modification of this code is subject to the terms and conditions of the license agreement provided with this software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Overview

This system consists of two main components:

1. **Document Processor** (`pdf.py`): Handles the processing of various document formats (PDF, EPUB, Microsoft Office) into vector embeddings using FAISS for efficient similarity search.

2. **Chat Application** (`app.py`): A Flask-based web application that provides an interface for querying processed documents using various AI models, including local LM Studio models and cloud-based services like Anthropic's Claude and Perplexity.

## Features

### Document Processing
- Supports multiple document formats:
  - PDF (.pdf)
  - EPUB (.epub)
  - Word Documents (.doc, .docx)
  - PowerPoint (.ppt, .pptx)
  - Excel (.xls, .xlsx)
- Incremental document processing
- Progress tracking with tqdm
- Document chunking with overlap
- Vector embedding generation using HuggingFace models
- FAISS index for efficient similarity search
- Automatic handling of file updates

### Chat Application
- Web-based interface for document queries
- Multiple AI model support:
  - Local models via LM Studio:
    - Mistral 7B Instruct v0.2
    - Deepseek Llama 8B
    - Deepseek Qwen 1.5B
  - Cloud-based models:
    - Perplexity Sonar (multiple variants)
    - Claude 3.5 Sonnet
    - Claude 3 Opus
- Chat history saving and retrieval
- Source document tracking
- Context-aware responses

## System Requirements

### Software Requirements

- **Python**: Version 3.9-3.11 (recommended)
- **LM Studio**: For local model inference
- **Required Models in LM Studio**:
  - Mistral 7B Instruct v0.2
  - Deepseek-Ri1-Sistill-Llama-8B-Q3_k_k
  - DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M

### Hardware Requirements

- **RAM**: Minimum 16GB recommended, 8GB may work for smaller document sets
- **Storage**: At least 10GB free space for models and document processing
- **CPU**: Multi-core processor recommended
- **GPU**: Optional but recommended for faster processing and model inference

### API Keys (for cloud models)

The application requires API keys for cloud-based models:
- `ANTHROPIC_API_KEY`: For Claude models
- `PERPLEXITY_API_KEY`: For Perplexity Sonar models

## Installation

### Automatic Setup

1. Run the setup script which will create a virtual environment and install all dependencies:
```bash
python setup.py
```

2. The setup script will create:
   - Virtual environment in `venv/` directory
   - Required directory structure
   - Launcher scripts for various operations

### Manual Setup

If automatic setup fails, you can manually install the required dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install core dependencies
pip install langchain langchain_community langchain_huggingface faiss-cpu
pip install torch torchvision
pip install transformers sentence-transformers
pip install pypdf docx2txt ebooklib beautifulsoup4 tqdm
pip install unstructured python-pptx openpyxl
pip install flask werkzeug requests anthropic
```

## Setup and Configuration

1. Install LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/)
2. Download and install the required models in LM Studio:
   - Mistral 7B Instruct v0.2
   - Deepseek-Ri1-Sistill-Llama-8B-Q3_k_k
   - DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M
3. Configure API keys in `app.py`:
   - Replace `ANTHROPIC_API_KEY='xxx'` with your Anthropic API key
   - Replace `PERPLEXITY_API_KEY='yyy'` with your Perplexity API key

## Usage

### Document Processing

1. Place your documents in the `docs` folder
2. Run the document processor using the provided launcher script:
   - Windows: `process_docs.bat`
   - macOS/Linux: `./process_docs.sh`

Or manually run:
```bash
python src/pdf.py
```

Optional arguments:
- `--force`: Force reprocessing of all documents
- `--no-progress`: Disable progress bars
- `--max-docs N`: Maximum number of documents to process (default: 10)
- `--alt-embeddings`: Use alternative embedding models

### Starting the Chat Application

1. Start LM Studio and ensure it's running in local inference mode on port 1234
2. Run the application using the provided launcher script:
   - Windows: `run_app.bat`
   - macOS/Linux: `./run_app.sh`

Or manually run:
```bash
python src/app.py
```

3. Access the chat interface at `http://localhost:5000`

## File Structure

```
├── src/                  # Source code directory
│   ├── pdf.py            # Document processor
│   ├── app.py            # Flask application
│   ├── faiss_index/      # Vector store
│   ├── templates/        # HTML templates
│   ├── static/           # Static assets
│   └── all_chats.json    # Chat history
├── docs/                 # Document storage
├── venv/                 # Virtual environment
├── setup.py              # Setup script
├── run_app.bat/sh        # Application launcher
├── process_docs.bat/sh   # Document processor launcher
└── README.md             # This file
```

## Troubleshooting

### LM Studio Connection Issues
- Ensure LM Studio is running before starting the app
- Verify LM Studio is running on port 1234
- Check that the required models are loaded in LM Studio

### API Connection Issues
- Verify API keys are correctly set in `app.py`
- Check internet connection for cloud model access
- Ensure API service quotas have not been exceeded

### Document Processing Issues
- Check file formats are supported
- Ensure documents are in the `docs` directory
- Try using the `--force` flag to reprocess documents
- Use `--alt-embeddings` if the default embedding model fails

## Contributing

Please submit bug reports, feature requests, and pull requests through the project's issue tracker.
