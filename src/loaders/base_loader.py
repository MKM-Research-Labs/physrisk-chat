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
# src/loaders/base_loader.py
"""
Base document loader interface defining the contract that all 
document loaders should implement.
"""
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
import os

class BaseLoader(ABC):
    """
    Abstract base class for document loaders.
    All document loaders should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def __init__(self, file_path: str):
        """
        Initialize the document loader.
        
        Args:
            file_path (str): Path to the document file
        """
        self.file_path = file_path
    
    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load and parse a document, returning a list of Documents.
        
        Returns:
            List[Document]: List of Document objects with text content and metadata
        """
        pass
    
    def handle_error(self, error: Exception) -> List[Document]:
        """
        Handle document processing errors by creating an error document.
        
        Args:
            error (Exception): The exception that occurred
            
        Returns:
            List[Document]: A list with a single Document containing error information
        """
        import os
        
        return [Document(
            page_content=f"Error processing document: {str(error)}",
            metadata={
                "source": os.path.basename(self.file_path),
                "file_path": self.file_path,
                "error": str(error)
            }
        )]
    
    def get_file_extension(self) -> str:
        """
        Get the file extension of the document.
        
        Returns:
            str: The lowercase file extension including the dot (e.g., '.pdf')
        """
        import os
        _, ext = os.path.splitext(self.file_path)
        return ext.lower()
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and sanitize text to handle encoding and special character issues.
        This is a utility method that can be used by implementing classes.
        
        Args:
            text (str): The text to clean
            
        Returns:
            str: Cleaned text
        """
        try:
            from ..utils.text_utils import sanitize_text
            return sanitize_text(text)
        except (ImportError, AttributeError):
            # Fallback to basic cleaning
            if not isinstance(text, str):
                text = str(text)
                
            # Remove control characters and normalize whitespace
            import re
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Strip leading and trailing whitespace
            text = text.strip()
            
            # Handle empty texts
            if not text:
                text = "Empty document section"
                
            return text