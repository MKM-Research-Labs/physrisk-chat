# Copyright (c) 2025 MKM Research Labs. All rights reserved.
# 
# This software is provided under license by MKM Research Labs. 
# Use, reproduction, distribution, or modification of this code is subject to the 
# terms and conditions of the license agreement provided with this software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# src/utils/text_utils.py
"""
Text processing utilities for document processing.
"""
import re
import unicodedata
import os
from langchain.schema import Document

# Try to import replacement terms, use empty function if not available
try:
    from ..rep import replace_terms
except ImportError:
    def replace_terms(text):
        return text

def sanitize_text(text):
    """
    Enhanced text sanitization to ensure compatibility with embedding models
    """
    if not isinstance(text, str):
        text = str(text)
        
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove control characters and normalize whitespace
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    # Apply domain-specific term replacements if available
    text = replace_terms(text)
    
    # Handle empty texts
    if not text:
        text = "Empty document section"
        
    return text

def standardize_metadata(chunks, docs_type):
    """
    Ensure all chunks have standardized metadata fields.
    This helps maintain consistency across different document types.
    """
    standardized_chunks = []

    for chunk in chunks:
        metadata = chunk.metadata.copy()
    
        # Ensure 'source' exists
        if 'source' not in metadata:
            metadata['source'] = os.path.basename(metadata.get('file_path', 'Unknown'))
    
        # Standardize page field - make sure every document has 'page'
        if 'page' not in metadata:
            # Try alternate fields in order of preference
            page_value = metadata.get('page_number')
            if page_value is None:
                page_value = metadata.get('item_id')
            if page_value is None:
                page_value = "1"  # Default fallback
        
            # Set the standardized page field
            metadata['page'] = page_value
        
        # Add docs_type to metadata to distinguish between misc and phys documents
        metadata['docs_type'] = docs_type
    
        # Create a new document with standardized metadata
        standardized_chunk = Document(
            page_content=chunk.page_content,
            metadata=metadata
        )
        standardized_chunks.append(standardized_chunk)

    return standardized_chunks