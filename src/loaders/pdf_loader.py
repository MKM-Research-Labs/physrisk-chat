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
# src/loaders/pdf_loader.py
"""
PDF document loader implementation with enhanced handling for problematic PDFs.
Incorporates multiple fallback mechanisms for handling difficult PDF files.
"""
import os
import warnings
import traceback
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import base loader
from .base_loader import BaseLoader

# Import utilities
from ..utils.text_utils import sanitize_text

# Suppress common PDF-related warnings
warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")
warnings.filterwarnings("ignore", message=".*ignoring.*") 

class EnhancedPDFLoader(BaseLoader):
    """
    Enhanced PDF loader that handles problematic PDFs with binary content and encoding issues.
    This class adds robust fallback methods when standard PDF loaders fail.
    """
    
    def __init__(self, file_path):
        """
        Initialize the PDF loader.
        
        Args:
            file_path (str): Path to the PDF file
        """
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500
        )

    def load(self):
        """
        Load PDF with multiple fallback mechanisms for handling problematic files.
        Returns a list of Document objects with page content and metadata.
        """
        documents = []
        
        # Try the primary method with PyPDFLoader from LangChain
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(self.file_path)
            documents = loader.load()
            print(f"Successfully loaded PDF with primary method: {len(documents)} pages")
            
            # Add additional metadata and clean text
            enhanced_docs = []
            for i, doc in enumerate(documents):
                metadata = doc.metadata.copy()
                metadata.update({
                    "source": os.path.basename(self.file_path),
                    "file_path": self.file_path,
                    "extraction_method": "langchain_pypdf"
                })
                
                enhanced_doc = Document(
                    page_content=self._clean_text(doc.page_content),
                    metadata=metadata
                )
                enhanced_docs.append(enhanced_doc)
            
            return enhanced_docs
            
        except Exception as e:
            primary_error = str(e)
            print(f"Primary PDF loading method failed: {primary_error}")
            
            # First fallback: Try with PyPDF2 directly with error handling
            try:
                print("Attempting first fallback method with PyPDF2...")
                import PyPDF2
                
                with open(self.file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file, strict=False)
                    
                    for i, page in enumerate(pdf_reader.pages):
                        # Use a try-except block for each page
                        try:
                            # Extract text with error handling for each page
                            text = ""
                            try:
                                text = page.extract_text() or ""
                            except Exception as page_error:
                                text = f"[Error extracting text: {str(page_error)}]"
                                
                            # Create Document with available text
                            if text.strip():
                                # Clean the text to avoid encoding issues
                                text = self._clean_text(text)
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        "source": os.path.basename(self.file_path),
                                        "file_path": self.file_path,
                                        "page": i + 1,
                                        "total_pages": len(pdf_reader.pages),
                                        "extraction_method": "pypdf2_direct"
                                    }
                                )
                                documents.append(doc)
                        except Exception as page_error:
                            print(f"  Error processing page {i+1}: {str(page_error)}")
                            # Try to create a document with error info
                            documents.append(Document(
                                page_content=f"[Error on page {i+1}]",
                                metadata={
                                    "source": os.path.basename(self.file_path),
                                    "file_path": self.file_path,
                                    "page": i + 1,
                                    "error": str(page_error),
                                    "extraction_method": "pypdf2_direct"
                                }
                            ))
                            
                if documents:
                    print(f"First fallback successful: extracted {len(documents)} pages with content")
                    return documents
                        
            except Exception as fallback1_error:
                print(f"First fallback method failed: {str(fallback1_error)}")
                
            # Second fallback: Try with pdfplumber
            try:
                print("Attempting second fallback method with pdfplumber...")
                try:
                    import pdfplumber
                except ImportError:
                    # Install pdfplumber if not available
                    import subprocess
                    print("Installing pdfplumber...")
                    subprocess.check_call(["pip", "install", "pdfplumber"])
                    import pdfplumber
                
                with pdfplumber.open(self.file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text() or ""
                            
                            # Clean the text
                            text = self._clean_text(text)
                            
                            if text.strip():
                                doc = Document(
                                    page_content=text,
                                    metadata={
                                        "source": os.path.basename(self.file_path),
                                        "file_path": self.file_path,
                                        "page": i + 1,
                                        "total_pages": len(pdf.pages),
                                        "extraction_method": "pdfplumber"
                                    }
                                )
                                documents.append(doc)
                        except Exception as page_error:
                            print(f"  Error with pdfplumber on page {i+1}: {str(page_error)}")
                
                if documents:
                    print(f"Second fallback successful: extracted {len(documents)} pages with content")
                    return documents
                    
            except Exception as fallback2_error:
                print(f"Second fallback method failed: {str(fallback2_error)}")
                
            # Try binary-safe extraction method
            try:
                print("Attempting binary-safe extraction method...")
                documents = self._binary_safe_extraction()
                if documents:
                    print(f"Binary-safe extraction successful: {len(documents)} pages")
                    return documents
            except Exception as binary_error:
                print(f"Binary-safe extraction failed: {str(binary_error)}")
            
            # Third fallback: Try with pdf2image + OCR if available
            try:
                print("Attempting OCR fallback with pdf2image and pytesseract...")
                documents = self._ocr_extraction()
                if documents:
                    print(f"OCR fallback successful: extracted {len(documents)} pages with content")
                    return documents
                    
            except Exception as fallback3_error:
                print(f"OCR fallback method failed: {str(fallback3_error)}")
            
            # Last resort: Create a document with error information
            error_doc = Document(
                page_content=f"Error processing PDF document. Original error: {primary_error}",
                metadata={
                    "source": os.path.basename(self.file_path),
                    "file_path": self.file_path,
                    "error": primary_error,
                    "extraction_method": "failed"
                }
            )
            return [error_doc]
    
    def _clean_text(self, text):
        """
        Clean and sanitize text to handle encoding and special character issues.
        
        Args:
            text: The text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            # Convert to string if not already
            try:
                if hasattr(text, 'decode'):
                    text = text.decode('utf-8', errors='replace')
                else:
                    text = str(text)
            except Exception:
                text = str(text)
        
        # Use sanitize_text from utils
        try:
            return sanitize_text(text)
        except (ImportError, AttributeError):
            # Fallback to local implementation if utils not available
            # Handle binary data that might be in the text
            text = ''.join(char for char in text if ord(char) < 128 or ord(char) > 160)
            
            # Replace problematic characters
            text = text.replace('\x00', '')
            
            # Normalize whitespace
            import re
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text if text else "Empty page content"
    
    def _binary_safe_extraction(self):
        """
        Attempt a binary-safe extraction for problematic PDFs.
        This is a specialized method for PDFs with binary content issues.
        
        Returns:
            list: List of Document objects extracted with binary-safe methods
        """
        documents = []
        
        try:
            # Custom binary-safe extraction
            import PyPDF2
            with open(self.file_path, 'rb') as file:
                # Use a more permissive reader setting
                pdf = PyPDF2.PdfReader(file, strict=False)
                
                # Process each page with careful binary handling
                for i in range(len(pdf.pages)):
                    try:
                        page = pdf.pages[i]
                        
                        # Extract text content carefully
                        text = ""
                        try:
                            text = page.extract_text()
                        except Exception as text_error:
                            # Try accessing page contents directly if text extraction fails
                            try:
                                contents = page.get("/Contents")
                                if contents:
                                    if isinstance(contents, list):
                                        # Handle content streams
                                        raw_text = ""
                                        for content in contents:
                                            try:
                                                if hasattr(content, "get_data"):
                                                    raw_text += content.get_data().decode('utf-8', errors='replace')
                                            except:
                                                pass
                                        text = raw_text
                                    elif hasattr(contents, "get_data"):
                                        # Single content stream
                                        text = contents.get_data().decode('utf-8', errors='replace')
                            except Exception as content_error:
                                print(f"Could not extract content from page {i+1}: {str(content_error)}")
                        
                        # Skip pages with no extractable content
                        if not text or not text.strip():
                            continue
                        
                        # Create document with cleaned text
                        clean_text = self._clean_text(text)
                        if clean_text:
                            doc = Document(
                                page_content=clean_text,
                                metadata={
                                    "source": os.path.basename(self.file_path),
                                    "file_path": self.file_path,
                                    "page": i + 1,
                                    "extraction_method": "binary_safe"
                                }
                            )
                            documents.append(doc)
                    
                    except Exception as page_error:
                        print(f"Error in binary-safe extraction for page {i+1}: {str(page_error)}")
        
        except Exception as e:
            print(f"Binary-safe extraction completely failed: {str(e)}")
        
        return documents
    
    def _ocr_extraction(self):
        """
        Extract text using OCR as a last resort.
        Requires pdf2image and pytesseract to be installed.
        
        Returns:
            list: List of Document objects with OCR-extracted text
        """
        documents = []
        
        try:
            # Try to import required packages
            try:
                import pdf2image
                import pytesseract
                from PIL import Image
            except ImportError:
                # Install dependencies
                import subprocess
                print("Installing pdf2image and pytesseract...")
                subprocess.check_call(["pip", "install", "pdf2image", "pytesseract", "pillow"])
                import pdf2image
                import pytesseract
                from PIL import Image
            
            # Convert PDF to images
            images = pdf2image.convert_from_path(self.file_path)
            
            # Process each page
            for i, image in enumerate(images):
                try:
                    # Extract text using OCR
                    text = pytesseract.image_to_string(image)
                    
                    # Skip empty results
                    if not text or not text.strip():
                        continue
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=self._clean_text(text),
                        metadata={
                            "source": os.path.basename(self.file_path),
                            "file_path": self.file_path,
                            "page": i + 1,
                            "total_pages": len(images),
                            "extraction_method": "ocr"
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"OCR error on page {i+1}: {str(e)}")
            
            return documents
        
        except Exception as e:
            print(f"OCR extraction failed: {str(e)}")
            return []