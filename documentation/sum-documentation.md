# Document Summarizer Documentation

The `DocumentSummarizer` module (`sum.py`) is a specialized tool that processes documents from the `docs` folder, creates a temporary FAISS index for each document, and generates summaries using a local LM Studio model. These summaries are stored in a central JSON file for easy access through the chat application.

## Overview

`DocumentSummarizer` enhances the document processing system by adding automatic summarization capabilities. It uses the same document loading and processing infrastructure as the main `DocumentProcessor` but focuses on creating concise, AI-generated summaries of each document that can be displayed in the chat interface.

## Features

- **Automatic Document Summary Generation**: Creates summaries of all documents in the docs folder
- **Local Model Integration**: Uses local LM Studio models (primarily Mistral) for summary generation
- **Persistent Storage**: Stores summaries directly in JSON for easy retrieval by the chat application
- **Incremental Processing**: Only summarizes new or modified documents, tracking changes via file hashes
- **Fallback Mechanisms**: Multiple fallback strategies when the primary summarization method fails
- **Status Tracking**: Detailed reporting on document summarization status

## Prerequisites

Before using the `DocumentSummarizer`, ensure you have:

1. Run the `setup.py` script to install required dependencies
2. Processed documents using `pdf.py` (recommended but not required)
3. Started LM Studio with a Mistral model running on the default port (1234)
4. Python 3.9 or higher installed

## Basic Usage

```bash
# Run with default settings (process up to 20 documents)
python sum.py

# Process only 5 documents
python sum.py --max-docs 5

# Force reprocessing of all documents, even if already summarized
python sum.py --force

# Show status of all documents without processing them
python sum.py --list-status

# Clean the summary database before processing
python sum.py --clean

# Use alternative embedding models if the default model fails
python sum.py --alt-embeddings

# Disable progress bars
python sum.py --no-progress
```

## Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--max-docs N` | Maximum number of documents to process (default: 20) |
| `--no-progress` | Disable progress bars during processing |
| `--force` | Force reprocessing of all documents, even if already summarized |
| `--list-status` | Show summarization status of all documents without processing |
| `--clean` | Clear all existing summaries from the JSON database |
| `--alt-embeddings` | Use alternative embedding models if the default model fails |

## How It Works

The `DocumentSummarizer` processes documents in several steps:

1. **Document Selection**: Identifies documents in the `docs` folder that need summarization
2. **Vector Index Creation**: Creates a temporary FAISS index for each document
3. **Summary Generation**: Queries a local LM Studio model with document content to generate a summary
4. **Storage**: Saves the summary directly in the `summarised_files.json` file
5. **Status Tracking**: Records the processing status and document hash for change detection

## Summary Storage

Summaries are stored in `src/summarised_files.json` with the following structure:

```json
{
  "document_name.pdf": {
    "hash": "1615481422.1234567",
    "summarised_date": "2025-03-18T15:30:45.123456",
    "num_chunks": 42,
    "summary": "Full text of the summary...",
    "summary_type": "FULL"
  },
  "another_document.docx": {
    "hash": "1615481789.7654321",
    "summarised_date": "2025-03-18T15:35:22.654321",
    "num_chunks": 18,
    "summary": "Full text of the summary...",
    "summary_type": "BASIC_FALLBACK"
  }
}
```

### Summary Types

The system records different summary types:

- `FULL`: Complete AI-generated summary using the Mistral model
- `BASIC_FALLBACK`: Simple summary generated when the AI model fails
- `EMPTY_DOCUMENT`: Placeholder summary for empty documents
- `ERROR_DOCUMENT`: Placeholder summary for documents that had processing errors

## Advanced Usage

### Integrating with the Chat Application

The summaries created by `DocumentSummarizer` are automatically available in the chat application through the `/get_summarised_files` API endpoint. This allows users to see document summaries without querying the AI model.

### Custom Model Prompts

The system uses a specific prompt template for the Mistral model:

```
Context information about {document_name}:
{context}

Question: can you summarise the document {document_name}

Please provide a detailed summary of the document based on the context provided.
```

This prompt can be modified in the `query_mistral` method if you want to adjust the summarization style.

### Handling Large Documents

For particularly large documents, the system automatically reduces the context size to avoid overwhelming the model:

- First attempt: Uses up to 50 chunks
- Second attempt: Reduces to 25 chunks
- Third attempt: Reduces to 17 chunks

If all attempts fail, it falls back to a basic summary generator.

## Error Handling and Fallbacks

The system incorporates multiple layers of error handling:

1. **Embedding Failures**: If the default embedding model fails, alternative models are attempted
2. **API Failures**: If the Mistral model is unavailable, multiple retries with reduced context are attempted
3. **Document Processing Errors**: If a document can't be processed, a basic summary is generated from the filename
4. **Empty Documents**: Special handling for empty or unreadable documents

## Performance Considerations

- **Processing Time**: Summarization is relatively slow (typically 10-30 seconds per document)
- **Memory Usage**: Creating vector embeddings can be memory-intensive for large documents
- **Local Model Requirements**: LM Studio needs to be running with a suitable model (like Mistral)
- **Storage**: Summaries are stored efficiently as text in a single JSON file

## Technical Details

### Supported Document Formats

The summarizer supports the same document formats as the main `DocumentProcessor`:

- PDF (.pdf)
- EPUB (.epub)
- Word Documents (.doc, .docx)
- PowerPoint (.ppt, .pptx)
- Excel (.xls, .xlsx)

### Enhanced EPUB Processing

Like the main `DocumentProcessor`, the `DocumentSummarizer` includes an enhanced EPUB loader (`ImprovedEPubLoader`) that:

- Extracts chapter structure
- Preserves formatting
- Handles tables and lists
- Extracts metadata from the EPUB file

### Document Chunking

Documents are split using `RecursiveCharacterTextSplitter` with:

- `chunk_size=3000`: Each chunk contains approximately 3000 characters
- `chunk_overlap=500`: Adjacent chunks share 500 characters for context continuity

## Troubleshooting

### Common Issues

1. **Model Connection Errors**
   - Ensure LM Studio is running on port 1234
   - Verify that a suitable model (like Mistral) is loaded in LM Studio
   - Check LM Studio logs for errors

2. **Empty or Failed Summaries**
   - Try running with the `--force` flag to regenerate summaries
   - Use the `--alt-embeddings` flag if embedding model issues occur
   - Check document format and content for compatibility issues

3. **Status Tracking Issues**
   - Use `--list-status` to view the current state of all documents
   - Use `--clean` to reset the database if tracking becomes corrupted
   - Verify write permissions for the `summarised_files.json` file

### Monitoring and Logs

The summarizer provides detailed console output with:

- Per-document processing status
- Success/failure indicators
- Error details when problems occur
- Summary of results after batch processing

## Integration with Other Components

### Chat Application Integration

The summaries generated by `DocumentSummarizer` are directly accessible to the chat application (`app.py`) via the `get_summarised_files` endpoint, which reads from the same JSON file.

### Document Processor Compatibility

The `DocumentSummarizer` uses many of the same components as the main `DocumentProcessor` (`pdf.py`), including the document loaders, text splitters, and embedding models. This ensures consistent treatment of documents across both systems.

## Usage Examples

### Summarizing All Documents

```bash
# First, clean the database and process all documents from scratch
python sum.py --clean

# Then process up to 50 documents
python sum.py --max-docs 50
```

### Checking Document Status

```bash
# See which documents have summaries and which need processing
python sum.py --list-status
```

### Refreshing Changed Documents

```bash
# Only process documents that have changed since last run
python sum.py
```

### Update Summaries for All Documents

```bash
# Force regeneration of summaries for all documents
python sum.py --force
```

## Viewing Summaries

Summaries can be viewed in several ways:

1. **Chat Interface**: Summaries are displayed in the chat application
2. **Direct JSON Access**: Examine the `summarised_files.json` file
3. **Status Report**: Use `--list-status` to see a preview of summaries

## Security and Limitations

- The summarizer is designed for local use only
- No data is sent to external services; all processing is done locally
- Large documents may exceed the context window of the local model, resulting in less comprehensive summaries
- The quality of summaries depends on the capability of the local LM Studio model
