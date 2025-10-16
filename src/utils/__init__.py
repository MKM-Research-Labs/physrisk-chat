# src/utils/__init__.py
"""
Utility functions package for document processing.
"""
from .text_utils import sanitize_text, standardize_metadata
from .file_utils import (
    get_file_hash, 
    load_json_file, 
    save_json_file, 
    get_supported_files,
    ensure_directory_exists,
    create_processed_record
)
from .diagnostics import diagnose_pdf, evaluate_diagnostics, generate_recommendation

# Export commonly used functions at package level
__all__ = [
    'sanitize_text',
    'standardize_metadata',
    'get_file_hash',
    'load_json_file',
    'save_json_file',
    'get_supported_files',
    'ensure_directory_exists',
    'create_processed_record',
    'diagnose_pdf',
    'evaluate_diagnostics',
    'generate_recommendation'
]
