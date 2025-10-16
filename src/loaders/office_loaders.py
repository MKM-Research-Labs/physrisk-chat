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
# src/loaders/office_loaders.py
"""
Loaders for Microsoft Office document formats (Word, PowerPoint, Excel).
These are enhanced wrappers around LangChain's loaders with improved
metadata handling and error recovery.
"""
import os
from langchain.schema import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)

# Import utilities
from ..utils.text_utils import sanitize_text

class DocxLoader:
    """
    Enhanced loader for Microsoft Word documents (.docx)
    """
    
    def __init__(self, file_path):
        """
        Initialize the Word document loader.
        
        Args:
            file_path (str): Path to the DOCX file
        """
        self.file_path = file_path
        self.base_loader = Docx2txtLoader(file_path)
    
    def load(self):
        """
        Load and parse Word document, returning a list of Documents.
        
        Returns:
            list: List of Document objects with text content and metadata
        """
        try:
            # Use the base loader to get documents
            docs = self.base_loader.load()
            
            # Enhance metadata and sanitize content
            enhanced_docs = []
            for i, doc in enumerate(docs):
                # Add additional metadata
                metadata = doc.metadata.copy()
                metadata.update({
                    "source": os.path.basename(self.file_path),
                    "file_path": self.file_path,
                    "page": i + 1,
                    "doc_type": "docx"
                })
                
                # Create new document with enhanced metadata and sanitized content
                enhanced_doc = Document(
                    page_content=sanitize_text(doc.page_content),
                    metadata=metadata
                )
                enhanced_docs.append(enhanced_doc)
            
            return enhanced_docs
            
        except Exception as e:
            print(f"Error processing Word document {self.file_path}: {str(e)}")
            # Return document with error information
            return [Document(
                page_content=f"Error processing Word document",
                metadata={
                    "source": os.path.basename(self.file_path),
                    "file_path": self.file_path,
                    "error": str(e)
                }
            )]

# Alias DocLoader to use the same implementation for .doc files
DocLoader = DocxLoader

class PowerPointLoader:
    """
    Enhanced loader for Microsoft PowerPoint documents (.pptx, .ppt)
    """
    
    def __init__(self, file_path):
        """
        Initialize the PowerPoint document loader.
        
        Args:
            file_path (str): Path to the PowerPoint file
        """
        self.file_path = file_path
        self.base_loader = UnstructuredPowerPointLoader(file_path)
    
    def load(self):
        """
        Load and parse PowerPoint document, returning a list of Documents.
        
        Returns:
            list: List of Document objects with text content and metadata
        """
        try:
            # Use the base loader to get documents
            docs = self.base_loader.load()
            
            # Enhance metadata and sanitize content
            enhanced_docs = []
            for i, doc in enumerate(docs):
                # Add additional metadata
                metadata = doc.metadata.copy()
                metadata.update({
                    "source": os.path.basename(self.file_path),
                    "file_path": self.file_path,
                    "slide": metadata.get("page_number", i + 1),  # Use page_number as slide if available
                    "page": metadata.get("page_number", i + 1),   # Standard page field
                    "doc_type": "powerpoint"
                })
                
                # Create new document with enhanced metadata and sanitized content
                enhanced_doc = Document(
                    page_content=sanitize_text(doc.page_content),
                    metadata=metadata
                )
                enhanced_docs.append(enhanced_doc)
            
            return enhanced_docs
            
        except Exception as e:
            print(f"Error processing PowerPoint document {self.file_path}: {str(e)}")
            # Return document with error information
            return [Document(
                page_content=f"Error processing PowerPoint document",
                metadata={
                    "source": os.path.basename(self.file_path),
                    "file_path": self.file_path,
                    "error": str(e)
                }
            )]

class ExcelLoader:
    """
    Enhanced loader for Microsoft Excel documents (.xlsx, .xls)
    """
    
    def __init__(self, file_path, mode="elements"):
        """
        Initialize the Excel document loader.
        
        Args:
            file_path (str): Path to the Excel file
            mode (str): Extraction mode for UnstructuredExcelLoader
                        ("elements" extracts tables, "single" for entire document)
        """
        self.file_path = file_path
        self.mode = mode
        self.base_loader = UnstructuredExcelLoader(file_path, mode=mode)
    
    def load(self):
        """
        Load and parse Excel document, returning a list of Documents.
        
        Returns:
            list: List of Document objects with text content and metadata
        """
        try:
            # Use the base loader to get documents
            docs = self.base_loader.load()
            
            # Enhance metadata and sanitize content
            enhanced_docs = []
            sheet_counter = {}  # Track elements per sheet
            
            for i, doc in enumerate(docs):
                # Get sheet name or default to "unknown"
                sheet_name = doc.metadata.get("sheet_name", "unknown")
                
                # Increment counter for this sheet
                if sheet_name not in sheet_counter:
                    sheet_counter[sheet_name] = 1
                else:
                    sheet_counter[sheet_name] += 1
                
                # Add additional metadata
                metadata = doc.metadata.copy()
                metadata.update({
                    "source": os.path.basename(self.file_path),
                    "file_path": self.file_path,
                    "sheet_element": sheet_counter[sheet_name],  # Element number within sheet
                    "page": f"{sheet_name}_{sheet_counter[sheet_name]}",  # Standard page field
                    "doc_type": "excel"
                })
                
                # Create new document with enhanced metadata and sanitized content
                enhanced_doc = Document(
                    page_content=sanitize_text(doc.page_content),
                    metadata=metadata
                )
                enhanced_docs.append(enhanced_doc)
            
            return enhanced_docs
            
        except Exception as e:
            print(f"Error processing Excel document {self.file_path}: {str(e)}")
            # Return document with error information
            return [Document(
                page_content=f"Error processing Excel document",
                metadata={
                    "source": os.path.basename(self.file_path),
                    "file_path": self.file_path,
                    "error": str(e)
                }
            )]