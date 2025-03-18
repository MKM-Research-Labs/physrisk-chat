# DocumentProcessor Documentation

The `DocumentProcessor` module is a powerful document processing system that converts various document formats into vector embeddings for use in semantic search and retrieval applications.

## Overview

`DocumentProcessor` handles the extraction, processing, and vectorization of content from multiple document formats including PDF, EPUB, and Microsoft Office files. It uses FAISS (Facebook AI Similarity Search) to store and retrieve document chunks based on semantic similarity.

## Features

- **Multi-format Support**: Process PDFs, EPUBs, Word documents, PowerPoint presentations, and Excel spreadsheets
- **Incremental Processing**: Only processes new or modified files
- **Smart Chunking**: Divides documents into semantically meaningful chunks with configurable overlap
- **Efficient Embeddings**: Uses Hugging Face sentence transformers for generating embeddings
- **Progress Tracking**: Visual progress bars for lengthy operations
- **Error Handling**: Robust error recovery and detailed logging
- **Metadata Preservation**: Maintains document source, page, and structural information

## Prerequisites

Before using the `DocumentProcessor`, ensure you have:

1. Installed all required dependencies via the provided `setup.py` script
2. Created a `docs` directory in your project root to store documents for processing
3. Python 3.9 or higher installed

## Basic Usage

```python
from pdf import DocumentProcessor

# Initialize the processor
processor = DocumentProcessor()

# Process all documents in the docs folder
processor.process_documents()

# To force reprocessing of all documents
processor.process_documents(force_reprocess=True)
```

## Command Line Usage

The module can also be run directly from the command line:

```bash
python pdf.py
python pdf.py --force
python pdf.py --no-progress
python pdf.py --max-docs 15
python pdf.py --alt-embeddings
```

### Command Line Arguments

- `--force`: Force reprocessing of all documents, even if unchanged
- `--no-progress`: Hide progress bars during processing
- `--max-docs N`: Limit processing to N documents (default is 10)
- `--alt-embeddings`: Use alternative embedding models (useful if primary model fails)

## Configuration

The `DocumentProcessor` class can be configured by modifying class attributes:

```python
processor = DocumentProcessor()

# Change maximum number of documents to process
processor.max_documents = 20

# Disable progress bars
processor.show_progress = False

# Use alternative embeddings model
processor.setup_alternative_embeddings()
```

## Advanced Usage

### Processing Specific Document Types

```python
# Get the appropriate loader for a specific file
file_path = "docs/my_document.pdf"
loader = processor.get_loader_for_file(file_path)

# Load and process a single document
docs = loader.load()
```

### Customizing Text Processing

The `DocumentProcessor` includes a text sanitization method that can be extended:

```python
# Override the sanitize_text method in a subclass
class CustomDocumentProcessor(DocumentProcessor):
    def sanitize_text(self, text):
        # Apply custom text processing logic
        text = super().sanitize_text(text)
        # Additional custom processing
        return text
```

### Creating Custom Loaders

The system allows for custom loader classes to handle specialized document formats. Custom loaders should follow the pattern established by `ImprovedEPubLoader` in the module.

## EPUB Processing

The module includes an enhanced EPUB loader (`ImprovedEPubLoader`) that:

- Extracts chapter structure
- Preserves formatting
- Handles tables and lists
- Extracts metadata from the EPUB file

## Document Chunking

Documents are split into chunks using a `RecursiveCharacterTextSplitter` with the following default settings:

- `chunk_size=3000`: Each chunk contains approximately 3000 characters
- `chunk_overlap=500`: Adjacent chunks share 500 characters for context continuity

## Embedding Model

By default, the processor uses the `all-MiniLM-L6-v2` model from Hugging Face for generating embeddings. Alternative models can be configured using the `--alt-embeddings` flag or the `setup_alternative_embeddings()` method.

## Error Handling

The processor includes comprehensive error handling at multiple levels:

- File-level errors are logged and processing continues with remaining files
- Document-specific errors are recorded in the `processed_files.json` file
- Embedding errors trigger fallback to alternative models
- All errors are reported in the console output

## Technical Details

### Directory Structure

- `docs/` - Place documents here for processing
- `faiss_index/` - FAISS vector store location
- `processed_files.json` - Tracks processed documents and their status

### Supported File Formats

| Format | Extensions | Loader |
|--------|------------|--------|
| PDF | .pdf | PyPDFLoader |
| EPUB | .epub | ImprovedEPubLoader |
| Word | .doc, .docx | Docx2txtLoader |
| PowerPoint | .ppt, .pptx | UnstructuredPowerPointLoader |
| Excel | .xls, .xlsx | UnstructuredExcelLoader |

### Metadata Fields

Each document chunk contains standardized metadata:

- `source`: Original filename
- `file_path`: Full path to the source file
- `page`: Page number or equivalent position
- Format-specific fields (varies by document type)

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Run `setup.py` to install all dependencies
   - For EPUB processing, ensure `EbookLib` and `beautifulsoup4` are installed

2. **Embedding Model Failures**
   - Use the `--alt-embeddings` flag to try alternative models
   - Check that Hugging Face transformers are properly installed

3. **Memory Issues with Large Documents**
   - Reduce the `max_documents` setting
   - Process documents in smaller batches

4. **Slow Processing**
   - Processing time depends on document size and complexity
   - First-time processing is slower due to embedding generation

## Logging and Monitoring

The processor provides detailed console output during operation:

- File processing status
- Chunk generation counts
- Error messages for failed operations
- Progress bars for long-running operations

## Performance Considerations

- Initial processing of documents is CPU and memory intensive
- Subsequent runs (with unchanged documents) are much faster
- Vector embedding generation is the most resource-intensive operation
- For large document collections, consider increasing RAM allocation
