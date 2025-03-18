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



import os
import numpy as np
import json
import warnings
import re
import unicodedata
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Optional import for replacement terms
try:
    from rep import replace_terms
except ImportError:
    def replace_terms(text):
        return text

warnings.filterwarnings("ignore", message="Ignoring wrong pointing object")
warnings.filterwarnings("ignore", message=".*ignoring.*") 

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


class DocumentProcessor:
    def __init__(self):
        # Set default paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.docs_folder = os.path.join(os.path.dirname(current_dir), 'docs')
        self.processed_files_path = os.path.join(current_dir, 'processed_files.json')
        self.faiss_path = os.path.join(current_dir, 'faiss_index')
        
        # Initialize the text embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500
        )
        self.show_progress = True  # Flag to control progress bar display
        self.max_documents = 10
        
        # Define supported formats AFTER the class methods are available
        self.SUPPORTED_FORMATS = {
            '.pdf': PyPDFLoader,
            '.epub': self._get_epub_loader,  # Now correctly refers to an instance method
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.pptx': UnstructuredPowerPointLoader, 
            '.ppt': UnstructuredPowerPointLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader
        }
        
        # Install dependencies if needed
        self._ensure_dependencies()

    def _ensure_dependencies(self):
        """Ensure required dependencies are installed"""
        try:
            import ebooklib
            import bs4
        except ImportError:
            print("Installing required dependencies for EPUB processing...")
            import subprocess
            subprocess.check_call(["pip", "install", "EbookLib", "beautifulsoup4"])
            print("Dependencies installed successfully")

    def _get_epub_loader(self, file_path):
        """Factory function to return our custom EPUB loader"""
        return ImprovedEPubLoader(file_path)

    def standardize_metadata(self, chunks):
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
        
            # Create a new document with standardized metadata
            standardized_chunk = Document(
                page_content=chunk.page_content,
                metadata=metadata
            )
            standardized_chunks.append(standardized_chunk)
    
        return standardized_chunks
    
    def load_processed_files(self):
        """Load the record of processed files from JSON"""
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, 'r') as f:
                return json.load(f)
        return {}

    def save_processed_files(self, processed_files):
        """Save the record of processed files to JSON"""
        with open(self.processed_files_path, 'w') as f:
            json.dump(processed_files, f, indent=2)

    def get_file_hash(self, filepath):
        """Get file modification time as a simple hash"""
        return str(os.path.getmtime(filepath))

    def get_loader_for_file(self, file_path):
        """Get the appropriate loader based on file extension"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}")
        
        loader_class_or_factory = self.SUPPORTED_FORMATS[ext]
        
        # If it's a method (factory function), call it with the file path
        if callable(loader_class_or_factory) and not isinstance(loader_class_or_factory, type):
            return loader_class_or_factory(file_path)
        
        # Otherwise, it's a class, so return it
        return loader_class_or_factory
    
    def sanitize_text(self, text):
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

    def process_single_document(self, filename, processed_files, force_reprocess=False):
        """Process a single document and update the index"""
        file_path = os.path.join(self.docs_folder, filename)
        file_hash = self.get_file_hash(file_path)
        
        # Skip if file hasn't changed
        if not force_reprocess and filename in processed_files and processed_files[filename]['hash'] == file_hash:
            return None
        
        try:
            print(f"\nProcessing {filename}...")
            loader_class_or_instance = self.get_loader_for_file(file_path)
            
            # Handle both class and instance returns
            if isinstance(loader_class_or_instance, type):
                # It's a class, instantiate it
                if file_path.lower().endswith(('.xlsx', '.xls')):
                    loader = loader_class_or_instance(file_path, mode="elements")
                else:
                    loader = loader_class_or_instance(file_path)
            else:
                # It's already an instance (from factory function)
                loader = loader_class_or_instance
            
            # Load pages with progress bar
            try:
                pages = loader.load()
                
                if not pages:
                    print(f"⚠ Document {filename} appears to be empty or unreadable")
                    # Record in processed files that this document was empty
                    processed_files[filename] = {
                        'hash': file_hash,
                        'processed_date': datetime.now().isoformat(),
                        'num_chunks': 0,
                        'status': 'EMPTY'
                    }
                    self.save_processed_files(processed_files)
                    return None
                
                # Sanitize page content before splitting
                sanitized_pages = []
                for page in pages:
                    # Create a new Document object with sanitized content
                    sanitized_page = Document(
                        page_content=self.sanitize_text(page.page_content),
                        metadata=page.metadata
                    )
                    sanitized_pages.append(sanitized_page)
                
                chunks = []
                
                # Show progress for document splitting
                with tqdm(total=len(sanitized_pages), desc="Splitting document", 
                         disable=not self.show_progress) as pbar:
                    for page in sanitized_pages:
                        page_chunks = self.text_splitter.split_documents([page])
                        chunks.extend(page_chunks)
                        pbar.update(1)
                
                if not chunks:
                    print(f"⚠ No chunks were generated from {filename}")
                    # Record document was processed but generated no chunks
                    processed_files[filename] = {
                        'hash': file_hash,
                        'processed_date': datetime.now().isoformat(),
                        'num_chunks': 0,
                        'status': 'NO_CHUNKS'
                    }
                    self.save_processed_files(processed_files)
                    return None
                
                # Update processed files record
                processed_files[filename] = {
                    'hash': file_hash,
                    'processed_date': datetime.now().isoformat(),
                    'num_chunks': len(chunks),
                    'status': 'SUCCESS'
                }
                
                print(f"✓ Generated {len(chunks)} chunks from {filename}")
                if chunks:
                    chunks = self.standardize_metadata(chunks)
                return chunks
                
            except IndexError as e:
                print(f"⚠ Index error with {filename}: {str(e)} - document might be empty or corrupted")
                # Record document had an error during processing
                processed_files[filename] = {
                    'hash': file_hash,
                    'processed_date': datetime.now().isoformat(),
                    'num_chunks': 0,
                    'status': 'ERROR',
                    'error': str(e)
                }
                self.save_processed_files(processed_files)
                return None
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            # Record failure in processed files
            processed_files[filename] = {
                'hash': file_hash,
                'processed_date': datetime.now().isoformat(),
                'num_chunks': 0,
                'status': 'ERROR',
                'error': str(e)
            }
            self.save_processed_files(processed_files)
            return None

    def process_documents(self, force_reprocess=False):
        """Process all supported documents with progress tracking"""
        processed_files = {} if force_reprocess else self.load_processed_files()
        
        all_chunks = []
        
        # Get list of supported files
        supported_extensions = tuple(self.SUPPORTED_FORMATS.keys())
        document_files = [
            f for f in os.listdir(self.docs_folder) 
            if f.lower().endswith(supported_extensions) and
            os.path.isfile(os.path.join(self.docs_folder, f))
        ]
        
        if not document_files:
            print(f"No supported files found in docs folder. Supported formats: {', '.join(supported_extensions)}")
            return False
        
        total_documents = len(document_files)
    
        # First, identify all files that need processing
        files_needing_processing = []
        
        for filename in document_files:
            file_path = os.path.join(self.docs_folder, filename)
            file_hash = self.get_file_hash(file_path)
        
            if (force_reprocess or 
                filename not in processed_files or 
                processed_files[filename]['hash'] != file_hash):
                files_needing_processing.append(filename)
                
        if not files_needing_processing:
            print(f"\nNo files need processing out of {total_documents} total documents")
            return True
    
        # Apply document limit if specified
        files_to_process = files_needing_processing
        
        if self.max_documents is not None:
            if len(files_needing_processing) > self.max_documents:
                files_to_process = files_needing_processing[:self.max_documents]
                print(f"\nFound {len(files_needing_processing)} files needing processing")
                print(f"Limiting to {self.max_documents} files due to max_documents setting")
            else:
                print(f"\nProcessing all {len(files_needing_processing)} files that need updating")
    
        print(f"\nProcessing {len(files_to_process)} out of {total_documents} total documents")
        # Get the original indices of the files to process
        file_indices = {filename: idx for idx, filename in enumerate(document_files, 1)}

        # Process each file with progress bar
        with tqdm(total=len(files_to_process), desc="Processing documents",
                 unit="file", disable=not self.show_progress) as pbar:
            for filename in files_to_process:
                doc_position = file_indices[filename]
                pbar.set_description(f"Processing document {doc_position}/{total_documents}")
                chunks = self.process_single_document(filename, processed_files, force_reprocess)
                if chunks:
                    all_chunks.extend(chunks)
                pbar.update(1)

        # Update the index if we have new content
        if all_chunks:
            print(f"\nProcessing new content ({len(all_chunks)} chunks)...")
            
            # Try to load and merge with existing index
            vector_store = None
            if os.path.exists(self.faiss_path) and not force_reprocess:
                try:
                    print("Loading existing index...")
                    existing_store = FAISS.load_local(
                        self.faiss_path, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # Process in smaller batches with robust error handling
                    print(f"Adding {len(all_chunks)} new chunks to existing store")
                    texts = []
                    metadatas = []
                    
                    # Prepare and validate all texts
                    with tqdm(total=len(all_chunks), desc="Preparing text data",
                            disable=not self.show_progress) as pbar:
                        for chunk in all_chunks:
                            # Additional sanitization before embedding
                            clean_text = self.sanitize_text(chunk.page_content)
                            if clean_text and len(clean_text.strip()) > 10:  # Minimum text length
                                texts.append(clean_text)
                                metadatas.append(chunk.metadata)
                            pbar.update(1)
                    
                    # Add to existing store in batches
                    batch_size = 10  # Smaller batch size for more stability
                    with tqdm(total=len(texts), desc="Adding to vector store",
                            disable=not self.show_progress) as pbar:
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i+batch_size]
                            batch_metadatas = metadatas[i:i+batch_size]
                            
                            try:
                                # Use add_texts method which handles embedding internally
                                existing_store.add_texts(
                                    texts=batch_texts, 
                                    metadatas=batch_metadatas
                                )
                            except Exception as e:
                                print(f"⚠ Warning: Batch failed, trying one by one: {str(e)}")
                                # Fall back to one-by-one processing if batch fails
                                for j in range(len(batch_texts)):
                                    try:
                                        existing_store.add_texts(
                                            texts=[batch_texts[j]], 
                                            metadatas=[batch_metadatas[j]]
                                        )
                                    except Exception as e2:
                                        print(f"⚠ Skipping problematic text: {str(e2)[:100]}...")
                            
                            pbar.update(len(batch_texts))
                    
                    vector_store = existing_store
                    print("✓ Successfully merged new documents")
                    
                except Exception as e:
                    print(f"❌ Error during merge: {str(e)}")
                    print("Creating new index with all documents...")
                    try:
                        # Clean and prepare texts for direct creation
                        texts = []
                        metadatas = []
                        
                        with tqdm(total=len(all_chunks), desc="Preparing text data",
                                disable=not self.show_progress) as pbar:
                            for chunk in all_chunks:
                                clean_text = self.sanitize_text(chunk.page_content)
                                if clean_text and len(clean_text.strip()) > 10:
                                    texts.append(clean_text)
                                    metadatas.append(chunk.metadata)
                                pbar.update(1)
                        
                        print(f"Creating vector store from {len(texts)} clean texts...")
                        vector_store = FAISS.from_texts(texts, self.embeddings, metadatas)
                    except Exception as e2:
                        print(f"❌ Error creating new index: {str(e2)}")
                        # Attempt with alternative embedding model as fallback
                        try:
                            print("Attempting with alternative embedding model...")
                            alt_embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                                encode_kwargs={'normalize_embeddings': True}
                            )
                            vector_store = FAISS.from_texts(texts, alt_embeddings, metadatas)
                        except Exception as e3:
                            print(f"❌ Final attempt failed: {str(e3)}")
                            return False
            else:
                try:
                    print("Creating new index...")
                    # Clean and prepare all texts
                    texts = []
                    metadatas = []
                    
                    with tqdm(total=len(all_chunks), desc="Preparing text data",
                            disable=not self.show_progress) as pbar:
                        for chunk in all_chunks:
                            clean_text = self.sanitize_text(chunk.page_content)
                            if clean_text and len(clean_text.strip()) > 10:
                                texts.append(clean_text)
                                metadatas.append(chunk.metadata)
                            pbar.update(1)
                    
                    print(f"Creating vector store from {len(texts)} documents...")
                    vector_store = FAISS.from_texts(texts, self.embeddings, metadatas)
                    
                except Exception as e:
                    print(f"❌ Error creating index: {str(e)}")
                    # Try with alternative embedding model
                    try:
                        print("Attempting with alternative embedding model...")
                        alt_embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                            encode_kwargs={'normalize_embeddings': True}
                        )
                        vector_store = FAISS.from_texts(texts, alt_embeddings, metadatas)
                    except Exception as e2:
                        print(f"❌ Alternative model failed: {str(e2)}")
                        return False
            
            # Save index and update processed files
            try:
                # Check if vector_store was successfully created
                if vector_store is None:
                    print("❌ Failed to create or update vector store")
                    return False
                    
                with tqdm(total=1, desc="Saving index", 
                         disable=not self.show_progress) as pbar:
                    vector_store.save_local(self.faiss_path)
                    self.save_processed_files(processed_files)
                    pbar.update(1)
                
                # Verify the index
                try:
                    print("\nVerifying index...")
                    with tqdm(total=1, desc="Verification", 
                             disable=not self.show_progress) as pbar:
                        final_store = FAISS.load_local(
                            self.faiss_path,
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        # Only try verification if index was loaded successfully
                        if final_store:
                            _ = final_store.similarity_search("test", k=1)
                            pbar.update(1)
                            print("✓ Index verification successful")
                        else:
                            print("⚠ Index verification skipped: could not load index")
                except Exception as e:
                    print(f"⚠ Index verification failed: {str(e)}")
                    # Continue anyway since the index was saved
                
                return True
                
            except Exception as e:
                print(f"❌ Error saving index: {str(e)}")
                return False
        else:
            print("\nNo new or modified files to process")
            self.save_processed_files(processed_files)

        # Clean up processed_files.json
        try:
            print("\nCleaning up processed files record...")
            current_files = set(os.listdir(self.docs_folder))
            processed_files = {k: v for k, v in processed_files.items() if k in current_files}
            self.save_processed_files(processed_files)
        except Exception as e:
            print(f"⚠ Warning during cleanup: {str(e)}")
            
        return True
    
    def setup_alternative_embeddings(self):
        """Set up alternative embeddings when the default model encounters issues"""
        try:
            print("Setting up alternative embedding model...")
            # Use a more robust sentence-transformers model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✓ Successfully set up alternative embeddings")
            return True
        except Exception as e:
            print(f"❌ Error setting up alternative embeddings: {str(e)}")
            
            # Try another fallback option
            try:
                print("Trying MPNet base model as fallback...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-mpnet-base-v2",
                    encode_kwargs={'normalize_embeddings': True}
                )
                print("✓ Successfully set up MPNet embeddings")
                return True
            except Exception as e2:
                print(f"❌ Error setting up fallback embeddings: {str(e2)}")
                return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process PDF, EPUB, and Microsoft Office documents for vector search')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocessing of all documents')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')
    parser.add_argument('--max-docs', type=int, default=10,
                       help='Maximum number of documents to process (default:10)')
    parser.add_argument('--alt-embeddings', action='store_true',
                       help='Use alternative embedding models')
    args = parser.parse_args()
    
    processor = DocumentProcessor()
    processor.max_documents = args.max_docs
    
    if args.no_progress:
        processor.show_progress = False
        
    if args.alt_embeddings:
        processor.setup_alternative_embeddings()
    
    processor.process_documents(force_reprocess=args.force)