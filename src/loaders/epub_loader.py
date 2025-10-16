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
# src/loaders/epub_loader.py
"""
EPUB document loader implementation.
"""
import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ImprovedEPubLoader:
    """
    Enhanced loader for EPUB files that extracts structured content
    with metadata including chapter information.
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500
        )
    
    def load(self):
        """
        Load and parse EPUB document, returning a list of Documents.
        Each Document contains text content with metadata about source,
        chapter, and position in document.
        """
        try:
            # Load the EPUB file
            book = epub.read_epub(self.file_path)
            
            # Get book metadata
            book_title = self._get_book_title(book)
            book_author = self._get_book_author(book)
            
            # Extract all documents
            documents = []
            
            # Process book items in order
            items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            
            # Sort items to maintain document flow
            # This tries to respect the spine order if available
            if book.spine:
                spine_ids = [item[0] for item in book.spine]
                items.sort(key=lambda x: spine_ids.index(x.id) if x.id in spine_ids else float('inf'))
            
            # Track chapter information
            current_chapter = "Unknown Chapter"
            chapter_num = 0
            
            for i, item in enumerate(items):
                # Try to determine if this is a chapter start
                html_content = item.get_content().decode('utf-8', errors='replace')
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Look for chapter title candidates
                chapter_candidates = soup.find_all(['h1', 'h2', 'h3'])
                if chapter_candidates:
                    chapter_title = chapter_candidates[0].get_text(strip=True)
                    if chapter_title:
                        chapter_num += 1
                        current_chapter = f"Chapter {chapter_num}: {chapter_title}"
                
                # Get clean text content
                text = self._html_to_text(soup)
                
                if not text.strip():
                    continue  # Skip empty content
                
                # Create document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(self.file_path),
                        "file_path": self.file_path,
                        "book_title": book_title,
                        "book_author": book_author,
                        "chapter": current_chapter,
                        "item_id": item.id,
                        "page_number": i + 1,
                        "page": i + 1
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error processing EPUB file: {e}")
            # Return empty document with error information
            return [Document(
                page_content="Error processing document",
                metadata={
                    "source": os.path.basename(self.file_path),
                    "file_path": self.file_path,
                    "error": str(e)
                }
            )]
    
    def _get_book_title(self, book):
        """Extract book title from metadata"""
        try:
            title = book.get_metadata('DC', 'title')
            if title:
                return title[0][0]
        except:
            pass
        return os.path.basename(self.file_path)
    
    def _get_book_author(self, book):
        """Extract book author from metadata"""
        try:
            creator = book.get_metadata('DC', 'creator')
            if creator:
                return creator[0][0]
        except:
            pass
        return "Unknown Author"
    
    def _html_to_text(self, soup):
        """Convert HTML to clean text while preserving important structure"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Handle special elements to preserve structure
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            # Add newlines around headings
            heading.insert_before('\n\n')
            heading.insert_after('\n')
        
        for para in soup.find_all('p'):
            para.insert_after('\n\n')
        
        for li in soup.find_all('li'):
            li.insert_before('• ')
            li.insert_after('\n')
        
        # Get text and clean up whitespace
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text