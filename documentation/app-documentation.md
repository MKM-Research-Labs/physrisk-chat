# Chat Application Documentation

The `app.py` module is a Flask-based web application that provides an interactive chat interface for querying document collections processed by the DocumentProcessor. It leverages both local LM Studio models and cloud-based AI services to provide intelligent responses based on document content.

## Overview

This application serves as the frontend interface for a document question-answering system. It connects to a FAISS vector database of processed documents and uses various AI models to generate answers to user queries based on relevant document content.

## Features

- **Web-based Chat Interface**: Clean, responsive UI for asking questions about your documents
- **Multiple AI Model Support**: 
  - Local models via LM Studio (Mistral, Deepseek)
  - Cloud-based models (Perplexity Sonar, Claude)
- **Document Context Retrieval**: Automatically finds and uses relevant document sections
- **Source Attribution**: Shows which documents were used to answer questions
- **Chat History**: Save and retrieve past conversations
- **Document Summaries**: Access document summaries when available

## Prerequisites

Before using the application:

1. Run the `setup.py` script to install all dependencies
2. Process documents using `pdf.py` to create the vector database
3. Install and run LM Studio for local model inference
4. (Optional) Configure API keys for cloud-based models

## Configuration

### API Keys

At the top of the `app.py` file, locate and replace the placeholder API keys:

```python
# API Keys. User needs to insert own API keys here
ANTHROPIC_API_KEY='xxx'  # Replace with your Anthropic API key
PERPLEXITY_API_KEY='yyy'  # Replace with your Perplexity API key
```

### API Endpoints

The application communicates with:

- Local LM Studio server at `http://localhost:1234/v1/chat/completions`
- Perplexity API at `https://api.perplexity.ai/chat/completions`
- Anthropic API at `https://api.anthropic.com/v1/messages`

### Available Models

The application supports the following models:

| Model ID | Display Name | Type |
|----------|--------------|------|
| `mistral-7b-instruct-v0.2` | Mistral 7B Instruct v0.2 | Local (LM Studio) |
| `Deepseek-Ri1-Sistill-Llama-8B-Q3_k_k` | Deepseek Llama 8B | Local (LM Studio) |
| `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M` | Deepseek Qwen 1.5B | Local (LM Studio) |
| `sonar` | Perplexity Sonar | Cloud (Perplexity) |
| `sonar-pro` | Perplexity Sonar Pro | Cloud (Perplexity) |
| `sonar-reasoning` | Perplexity Sonar Reasoning | Cloud (Perplexity) |
| `sonar-reasoning-pro` | Perplexity Sonar Reasoning Pro | Cloud (Perplexity) |
| `claude-3.5-sonnet` | Claude 3.5 Sonnet | Cloud (Anthropic) |
| `claude-3-opus` | Claude 3 Opus | Cloud (Anthropic) |

## Starting the Application

### Using Launcher Scripts

After running `setup.py`, use the generated launcher script:

- Windows: `run_app.bat`
- macOS/Linux: `./run_app.sh`

### Manual Start

```bash
# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows

# Navigate to the src directory
cd src

# Run the application
python app.py
```

The server will start on port 5000, and you can access the application at `http://localhost:5000`.

## Application Architecture

### Core Components

1. **Flask Web Server**: Handles HTTP requests and serves the application
2. **Vector Store Interface**: Connects to FAISS index for document retrieval
3. **Model Handlers**: Routes queries to appropriate AI models
4. **Chat Storage**: Maintains conversation history

### Model Handling Logic

The application uses two main handlers for model requests:

- `handle_local_model()`: Manages requests to LM Studio running locally
- `handle_api_model()`: Manages requests to cloud API services (Perplexity and Claude)

### Data Flow

1. User submits a query with a selected model
2. Application retrieves relevant document sections from FAISS
3. Context and query are sent to the selected model
4. Response is received and returned to the user along with source information

## API Routes

### `/`
- **Method**: GET
- **Description**: Serves the main chat interface
- **Returns**: HTML template for the chat application

### `/query`
- **Method**: POST
- **Description**: Process a user query and generate a response
- **Request Body**:
  ```json
  {
    "query": "User question text",
    "model": "model-id-from-available-models"
  }
  ```
- **Response**:
  ```json
  {
    "response": "AI generated answer",
    "sources": [
      {"file": "document1.pdf", "page": "5"},
      {"file": "document2.docx", "page": "2"}
    ]
  }
  ```
- **Error Response**:
  ```json
  {
    "error": "Error message description"
  }
  ```

### `/save_chat`
- **Method**: POST
- **Description**: Save a chat conversation to history
- **Request Body**:
  ```json
  {
    "chat": {
      "title": "Chat title",
      "messages": [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer", "sources": [...]}
      ]
    }
  }
  ```
- **Response**:
  ```json
  {
    "message": "Chat saved successfully",
    "chat_id": "chat_20250318_123045"
  }
  ```

### `/get_chats`
- **Method**: GET
- **Description**: Retrieve all saved chat histories
- **Response**:
  ```json
  {
    "chats": [
      {
        "id": "chat_20250317_152233",
        "title": "Financial Report Discussion",
        "messages": [...]
      },
      {
        "id": "chat_20250318_091512",
        "title": "Technical Documentation Questions",
        "messages": [...]
      }
    ]
  }
  ```

### `/get_summarised_files`
- **Method**: GET
- **Description**: Retrieve summaries of processed documents if available
- **Response**: JSON object with document summaries

## Error Handling

The application includes comprehensive error handling for:

- Invalid request formats
- Missing required parameters
- Invalid model selections
- FAISS search failures
- LM Studio connection issues
- API connection problems
- Internal server errors

Errors are returned with appropriate HTTP status codes and descriptive messages.

## Chat Storage

Chat histories are stored in `all_chats.json` located in the application directory. The structure is:

```json
{
  "chats": [
    {
      "id": "chat_20250318_123045",
      "title": "Chat Title",
      "messages": [
        {
          "role": "user",
          "content": "Question text"
        },
        {
          "role": "assistant",
          "content": "Answer text",
          "sources": [
            {"file": "document.pdf", "page": "5"}
          ]
        }
      ]
    }
  ]
}
```

## Technical Details

### Text Replacement

The application includes a term replacement function (`replace_terms`) that can be customized to substitute specific terms in responses. This can be useful for:

- Replacing internal codenames with proper terminology
- Standardizing product names or terminology
- Correcting common errors in source documents

### Document Retrieval

The application uses FAISS to perform semantic search with the following parameters:

- Retrieves up to 10 most relevant document chunks per query
- Uses the `similarity_search` method for retrieval
- Combines chunks into a single context with source attribution

### Model Parameters

#### Local Models (LM Studio)
- Temperature: 0.7
- Max Tokens: 1000

#### Claude Models
- Model: claude-3-sonnet-20240229
- Temperature: 0.3
- Max Tokens: 2048

## Troubleshooting

### Common Issues

#### LM Studio Connection Failed
- **Symptom**: "Local model request failed" error message
- **Solution**: 
  1. Ensure LM Studio is running
  2. Verify it's in local inference mode
  3. Check that it's using the default port (1234)
  4. Verify the selected model is loaded in LM Studio

#### API Request Failures
- **Symptom**: "API request failed" error message
- **Solution**:
  1. Verify API keys are correctly configured
  2. Check internet connection
  3. Ensure you haven't exceeded API usage limits
  4. Verify the API endpoints haven't changed

#### No Relevant Documents Found
- **Symptom**: "No relevant documents found" error message
- **Solution**:
  1. Make sure documents have been processed with `pdf.py`
  2. Check the FAISS index exists and isn't corrupted
  3. Try rephrasing your query
  4. Process more relevant documents if available

#### JSON Parsing Errors
- **Symptom**: Server errors or invalid responses
- **Solution**:
  1. Check for malformed requests
  2. Ensure API responses are in expected format
  3. Look for encoding issues in document content

## Debugging

The application includes detailed logging to assist with troubleshooting:

- Server startup information (port, paths, available models)
- Request processing steps
- Error messages with stack traces
- Document metadata inspection

To enable more verbose logging, set `debug=True` in the `app.run()` call (enabled by default).

## Performance Considerations

- **Concurrency**: The default Flask server handles one request at a time
- **Response Time**: Varies based on:
  - Selected model (local models are typically faster)
  - Document collection size
  - Query complexity
  - Network conditions (for API models)
- **Memory Usage**: Increases with:
  - FAISS index size
  - Number of concurrent users
  - Document context length

## Security Notes

- API keys should be stored securely, not hardcoded in the script
- The application is designed for local network use, not public exposure
- No authentication is implemented in the base application
- Consider using environment variables for API keys in production
