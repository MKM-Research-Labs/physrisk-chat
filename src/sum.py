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


#!/usr/bin/env python3
"""
Document Summarizer with JSON Content Storage

This script:
1. Takes documents from the docs folder
2. Creates a FAISS index for each document
3. Calls the Mistral model with the query "can you summarise the document <document name>"
4. Stores the summaries directly in the summarised_files.json file
5. Maintains document processing history and summaries in a single JSON file

"""
import os
import json
import requests
import warnings
import re
import unicodedata
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
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

# Suppress warnings
warnings.filterwarnings("ignore", message="Ignoring wrong pointing object*")
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
                        "page_number": i + 1
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

class DocumentSummarizer:
    def __init__(self):
        # Set up paths
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.docs_folder = os.path.join(os.path.dirname(self.current_dir), 'docs')
        self.faiss_path = os.path.join(self.current_dir, 'temp_faiss_index')
        self.summarised_files_path = os.path.join(self.current_dir, 'summarised_files.json')
        
        # Set up embeddings with proper configuration to avoid the TextEncodeInput error
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500
        )
        
        # Flag to control progress bar display
        self.show_progress = True
        
        # Define supported formats using the improved EPUB loader
        self.SUPPORTED_FORMATS = {
            '.pdf': PyPDFLoader,
            '.epub': self._get_epub_loader,  # Use factory method for the improved EPUB loader
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

    def load_summarised_files(self):
        """Load the record of summarised files from JSON"""
        if os.path.exists(self.summarised_files_path):
            with open(self.summarised_files_path, 'r') as f:
                return json.load(f)
        return {}

    def save_summarised_files(self, summarised_files):
        """Save the record of summarised files to JSON"""
        with open(self.summarised_files_path, 'w') as f:
            json.dump(summarised_files, f, indent=2)

    def get_file_hash(self, filepath):
        """Get file modification time as a simple hash"""
        return str(os.path.getmtime(filepath))
    
    def sanitize_text(self, text):
        """
        Enhanced text sanitization to ensure compatibility with embedding models.
        This fixes the TextEncodeInput error by ensuring clean, properly formatted text.
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
        
        # Handle empty texts
        if not text:
            text = "Empty document section"
            
        return text

    def get_documents(self, max_docs=20):
        """Get up to max_docs supported documents from the docs folder that aren't summarized yet"""
        supported_extensions = tuple(self.SUPPORTED_FORMATS.keys())
        
        # Load list of already summarized files
        summarised_files = self.load_summarised_files()
        
        # Get all supported documents in the docs folder
        all_document_files = [
            f for f in os.listdir(self.docs_folder) 
            if f.lower().endswith(supported_extensions) and 
            os.path.isfile(os.path.join(self.docs_folder, f))
        ]
        
        if not all_document_files:
            raise ValueError(f"No supported files found in docs folder. Supported formats: {', '.join(supported_extensions)}")
        
        # Filter out documents that have already been summarized
        unsummarised_files = []
        for file in all_document_files:
            file_path = os.path.join(self.docs_folder, file)
            file_hash = self.get_file_hash(file_path)
            
            if file not in summarised_files or summarised_files[file]['hash'] != file_hash:
                unsummarised_files.append(file)
                
        if not unsummarised_files:
            print("All files have already been summarized. Use --force to reprocess them.")
            return all_document_files[:max_docs]  # Return already summarized files if no new ones
            
        print(f"Found {len(unsummarised_files)} document(s) that need summarizing out of {len(all_document_files)} total documents.")
        
        # Return up to max_docs documents that need summarizing
        return unsummarised_files[:max_docs]

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
  
    def process_document(self, filename, summarised_files):
        """Process a single document and create a FAISS index for it with robust embedding handling"""
        file_path = os.path.join(self.docs_folder, filename)
        file_hash = self.get_file_hash(file_path)
    
        # Check if file is already summarised and hash hasn't changed
        if filename in summarised_files and summarised_files[filename]['hash'] == file_hash:
            print(f"File {filename} already summarised and hasn't changed. Skipping.")
            return None, False
    
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
        
            # Load pages with progress bar and handle empty documents
            try:
                pages = loader.load()
            
                if not pages:
                    print(f"⚠ Document {filename} appears to be empty or unreadable")
                    return None, "EMPTY"
                
                # Sanitize page content before splitting to avoid embedding errors
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
                    return None, "EMPTY"
                
                print(f"✓ Generated {len(chunks)} chunks from {filename}")
                
                # Prepare the texts and metadata for embedding
                texts = []
                metadatas = []
                
                # Additional sanitization for embedding
                with tqdm(total=len(chunks), desc="Preparing texts for embedding",
                         disable=not self.show_progress) as pbar:
                    for chunk in chunks:
                        # Additional sanitization before embedding
                        clean_text = self.sanitize_text(chunk.page_content)
                        if clean_text and len(clean_text.strip()) > 10:  # Minimum text length
                            texts.append(clean_text)
                            metadatas.append(chunk.metadata)
                        pbar.update(1)
                
                if not texts:
                    print(f"⚠ No valid text chunks remained after sanitization for {filename}")
                    return None, "EMPTY"
                
                print(f"✓ Prepared {len(texts)} valid chunks for embedding")
                
                # Create FAISS index with careful error handling
                print("Creating new FAISS index...")
                
                try:
                    # Create vector store with explicit text and metadata
                    # Use smaller batch sizes to avoid memory issues
                    batch_size = 5  # Smaller batch size for stability
                    
                    # Initialize empty vector store
                    vector_store = None
                    
                    # Process in batches
                    with tqdm(total=len(texts), desc="Creating embeddings",
                              disable=not self.show_progress) as pbar:
                        
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i+batch_size]
                            batch_metadatas = metadatas[i:i+batch_size]
                            
                            try:
                                # For first batch, create the initial vector store
                                if vector_store is None:
                                    vector_store = FAISS.from_texts(
                                        batch_texts, 
                                        self.embeddings, 
                                        batch_metadatas
                                    )
                                else:
                                    # Add subsequent batches to the existing store
                                    vector_store.add_texts(
                                        batch_texts,
                                        batch_metadatas
                                    )
                                pbar.update(len(batch_texts))
                                
                            except Exception as e:
                                print(f"⚠ Batch failed, trying one by one: {str(e)[:100]}...")
                                # If batch fails, try one by one
                                for j in range(len(batch_texts)):
                                    try:
                                        if vector_store is None:
                                            vector_store = FAISS.from_texts(
                                                [batch_texts[j]], 
                                                self.embeddings, 
                                                [batch_metadatas[j]]
                                            )
                                        else:
                                            vector_store.add_texts(
                                                [batch_texts[j]],
                                                [batch_metadatas[j]]
                                            )
                                        pbar.update(1)
                                    except Exception as e2:
                                        print(f"⚠ Skipping problematic text: {str(e2)[:100]}...")
                                        pbar.update(1)
                                        continue
                        
                    # If we still couldn't create a vector store, try with alternative embeddings
                    if vector_store is None:
                        print("⚠ Failed with primary embedding model, trying alternative model...")
                        alt_embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                            encode_kwargs={'normalize_embeddings': True}
                        )
                        vector_store = FAISS.from_texts(texts[:10], alt_embeddings, metadatas[:10])
                
                    # Save index
                    with tqdm(total=1, desc="Saving index", 
                            disable=not self.show_progress) as pbar:
                        vector_store.save_local(self.faiss_path)
                        pbar.update(1)
                
                    print("✓ Index creation successful")
                    return vector_store, True
                    
                except Exception as e:
                    print(f"❌ Error creating FAISS index: {str(e)}")
                    # Try one last time with minimal text and alternative embeddings
                    try:
                        print("Attempting with minimal text and alternative embeddings...")
                        alt_embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                            encode_kwargs={'normalize_embeddings': True}
                        )
                        
                        # Take just a few high-quality texts
                        minimal_texts = [t for t in texts if len(t) > 50][:5]
                        if not minimal_texts:
                            minimal_texts = ["Content placeholder for document: " + filename]
                            
                        minimal_metadatas = metadatas[:len(minimal_texts)]
                        if len(minimal_metadatas) < len(minimal_texts):
                            minimal_metadatas.extend([{"source": filename}] * (len(minimal_texts) - len(minimal_metadatas)))
                            
                        vector_store = FAISS.from_texts(
                            minimal_texts,
                            alt_embeddings,
                            minimal_metadatas
                        )
                        
                        vector_store.save_local(self.faiss_path)
                        print("✓ Created minimal emergency index")
                        return vector_store, True
                    except Exception as e2:
                        print(f"❌ All embedding attempts failed: {str(e2)}")
                        return None, "ERROR"
                
            except IndexError as e:
                print(f"⚠ Index error with {filename}: {str(e)} - document might be empty or corrupted")
                return None, "EMPTY"
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, "ERROR"
    
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
    
    def query_mistral(self, document_name, vector_store):
        """Query Mistral model for document summary with error handling and retries"""
        # Construct query
        query = f"can you summarise the document {document_name}"
        print(f"\nQuerying Mistral with: '{query}'")
        
        # Get context from FAISS - start with fewer chunks if needed
        max_chunks = 50  # Start with fewer chunks
        
        for attempt in range(3):  # Try up to 3 times with reducing context
            try:
                # Adjust context size based on attempt number
                num_chunks = max(2, max_chunks // (attempt + 1))
                
                print(f"Attempt {attempt+1}: Using {num_chunks} chunks for context...")
                docs = vector_store.similarity_search(query, num_chunks)
                
                if not docs:
                    print("No relevant document chunks found, using generic summary approach")
                    # Fall back to very basic summary with just document name
                    return self.generate_basic_summary(document_name)
                
                # Build context with progressively smaller amounts of text
                context = "\n\n".join([
                    f"From {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content[:20 if attempt > 0 else 1000]}" 
                    for doc in docs
                ])
                
                # Call Mistral model
                headers = {
                    "Content-Type": "application/json"
                }
                
                # Format prompt with context and query
                prompt = f"""Context information about {document_name}:
{context[:10000]}  # Limit context to avoid overloading model

Question: {query}

Please provide a detailed summary of the document based on the context provided."""

                payload = {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }

                # Make request to local LM Studio server (Mistral model)
                response = requests.post(
                    "http://localhost:1234/v1/chat/completions",  # LM Studio default address
                    headers=headers,
                    json=payload,
                    timeout=300  # Add timeout
                )
                
                response.raise_for_status()
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0 and "message" in response_data["choices"][0]:
                    response_text = response_data["choices"][0]["message"].get("content", "")
                else:
                    raise ValueError("Unexpected API response format")
                
                print("✓ Successfully received response from Mistral")
                return response_text

            except requests.exceptions.RequestException as e:
                print(f"⚠ Attempt {attempt+1} failed: {str(e)}")
                if attempt == 2:  # Last attempt
                    print("All API attempts failed, using backup summary generation method")
                    return self.generate_basic_summary(document_name)
                # Otherwise, try again with reduced context
                
            except Exception as e:
                print(f"⚠ Unexpected error in attempt {attempt+1}: {str(e)}")
                if attempt == 2:  # Last attempt
                    print("All API attempts failed, using backup summary generation method")
                    return self.generate_basic_summary(document_name)
                    
    def generate_basic_summary(self, document_name):
        """Generate a basic summary when API calls fail"""
        print("Generating basic summary based on document metadata...")
        
        # Extract meaningful information from filename
        base_name = os.path.splitext(document_name)[0]
        
        # Clean up the name for better readability
        clean_name = base_name.replace('_', ' ').replace('-', ' ')
        
        # Generate a simple summary using the filename
        summary = f"""# Summary of {clean_name}

This document appears to be about {clean_name}.

The automatic summarization using the Mistral model was not successful due to technical limitations. This is a placeholder summary generated based on the document filename.

## Key Information:
- Document name: {document_name}
- Summary generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Method: Automated filename-based placeholder (API summarization failed)

To get a better summary of this document, you may want to:
1. Try running the summarizer again later
2. Adjust the model parameters
3. Process this document manually

## Note:
This is a fallback summary created because the API-based summarization was unsuccessful.
"""
        return summary

    def run(self, max_docs=20):
        """Run the document summarization pipeline for up to max_docs documents"""
        try:
            # Load the existing summarised files record
            summarised_files = self.load_summarised_files()
            
            # Step 1: Get up to max_docs documents
            document_names = self.get_documents(max_docs)
            print(f"Selected {len(document_names)} documents: {', '.join(document_names)}")
            
            results = []
            
            # Process each document
            for i, document_name in enumerate(document_names, 1):
                print(f"\n{'='*110}")
                print(f"Processing document {i}/{len(document_names)}: {document_name}")
                print(f"{'='*110}")
                
                try:
                    # Get file path and hash
                    file_path = os.path.join(self.docs_folder, document_name)
                    file_hash = self.get_file_hash(file_path)
                    
                    # Step 2: Create FAISS index for the document
                    vector_store, is_new_processing = self.process_document(document_name, summarised_files)
                    
                    if is_new_processing == False:
                        # Skip summarization if document was already processed and hasn't changed
                        results.append((document_name, True, "Used existing summary", "SKIPPED - Already summarized"))
                        continue
                    elif is_new_processing == "EMPTY" or is_new_processing == "ERROR":
                        # Document is empty or had an error during processing
                        basic_summary = self.generate_basic_summary(document_name)

                        # Update the record with the basic summary
                        summarised_files[document_name] = {
                            'hash': file_hash,
                            'summarised_date': datetime.now().isoformat(),
                            'num_chunks': 0,  # Use 0 for empty documents or errors
                            'summary': basic_summary,
                            'summary_type': 'EMPTY_DOCUMENT' if is_new_processing == "EMPTY" else 'ERROR_DOCUMENT'
                        }
                        self.save_summarised_files(summarised_files)
                        results.append((document_name, True, "Basic summary stored for empty/error document", "BASIC_FALLBACK"))
                        continue  # Skip to next document since this one had issues
                    
                    try:
                        # Step 3: Query Mistral model for summary
                        summary = self.query_mistral(document_name, vector_store)
                        
                        # Update the summarised files record with the summary content directly
                        summarised_files[document_name] = {
                            'hash': file_hash,
                            'summarised_date': datetime.now().isoformat(),
                            'num_chunks': len(vector_store.index_to_docstore_id) if vector_store else 0,
                            'summary': summary,  # Store the summary directly in the JSON
                            'summary_type': 'FULL'  # Indicate this is a full AI-generated summary
                        }
                        
                        # Save the updated record
                        self.save_summarised_files(summarised_files)
                        
                        print(f"✓ Summarization complete! Summary stored in JSON record")
                        results.append((document_name, True, "Summary stored in JSON", "SUCCESS"))
                        
                    except Exception as e:
                        print(f"❌ Error processing document {document_name}: {str(e)}")
                        # Even if we have an error, try to create a basic summary
                        try:
                            # Generate a basic fallback summary
                            basic_summary = self.generate_basic_summary(document_name)
                            
                            # Update the record with the basic summary
                            summarised_files[document_name] = {
                                'hash': file_hash,
                                'summarised_date': datetime.now().isoformat(),
                                'num_chunks': len(vector_store.index_to_docstore_id) if vector_store else 0,
                                'summary': basic_summary,  # Store the basic summary directly in JSON
                                'summary_type': 'BASIC_FALLBACK'  # Indicate this is a fallback summary
                            }
                            self.save_summarised_files(summarised_files)
                            
                            print(f"✓ Basic fallback summary stored in JSON record")
                            results.append((document_name, True, "Basic summary stored in JSON", "BASIC_FALLBACK"))
                        except Exception as inner_e:
                            print(f"❌ Even basic summary failed: {str(inner_e)}")
                            results.append((document_name, False, str(e), "FAILED"))
                except Exception as e:
                    print(f"❌ Error processing document {document_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()  # This will print the full stack trace
                    results.append((document_name, False, str(e), "FAILED"))
            
            # Print summary of results
            print("\n" + "="*110)
            print("SUMMARY OF PROCESSING RESULTS")
            print("="*110)
            for doc, success, details, status in results:
                status_display = f"✓ {status}" if success else f"❌ {status}"
                
                # Add color indicators for different statuses
                if status == "SUCCESS":
                    status_color = "\033[92m"  # Green
                elif status == "BASIC_FALLBACK":
                    status_color = "\033[93m"  # Yellow
                elif status == "SKIPPED - Already summarized":
                    status_color = "\033[94m"  # Blue
                else:
                    status_color = "\033[91m"  # Red
                
                reset_color = "\033[0m"
                print(f"{status_color}{status_display}{reset_color}: {doc} - {details}")
            
            # Final confirmation of summarised_files.json
            print(f"\nThe summarised_files.json has been updated at: {self.summarised_files_path}")
            print(f"Total documents tracked: {len(summarised_files)}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error in summarization pipeline: {str(e)}")
            return False

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Summarize documents and store summaries directly in JSON')
    parser.add_argument('--max-docs', type=int, default=20,
                        help='Maximum number of documents to process (default: 20)')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bars')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of documents even if already summarized')
    parser.add_argument('--list-status', action='store_true',
                        help='List the summarization status of all documents without processing')
    parser.add_argument('--clean', action='store_true',
                        help='Clear the summarised_files.json before processing')
    parser.add_argument('--alt-embeddings', action='store_true',
                        help='Use alternative embedding models if default model fails')
    args = parser.parse_args()
    
    print("=" * 110)
    print("Document Summarizer with JSON Content Storage")
    print("=" * 110)
    
    summarizer = DocumentSummarizer()
    
    # Use alternative embeddings if requested
    if args.alt_embeddings:
        summarizer.setup_alternative_embeddings()
    
    # Clean the summarised_files.json if requested
    if args.clean:
        print("Cleaning summarised_files.json - clearing all existing records")
        summarizer.save_summarised_files({})
    
    # Just list status without processing if requested
    if args.list_status:
        print("DOCUMENT SUMMARIZATION STATUS")
        print("=" * 110)
        
        # Get all supported documents
        supported_extensions = tuple(summarizer.SUPPORTED_FORMATS.keys())
        all_docs = [f for f in os.listdir(summarizer.docs_folder) 
                   if f.lower().endswith(supported_extensions) and
                   os.path.isfile(os.path.join(summarizer.docs_folder, f))]
        
        # Load summarized files data
        summarised_files = summarizer.load_summarised_files()
        
        # Check status for each file
        summarized_count = 0
        unsummarized_count = 0
        changed_count = 0
        
        for doc in sorted(all_docs):
            doc_path = os.path.join(summarizer.docs_folder, doc)
            current_hash = summarizer.get_file_hash(doc_path)
            
            if doc in summarised_files:
                if summarised_files[doc]['hash'] == current_hash:
                    summary_type = summarised_files[doc].get('summary_type', 'FULL')
                    print(f"✓ SUMMARIZED ({summary_type}): {doc}")
                    summarized_count += 1
                else:
                    print(f"⚠ CHANGED   : {doc} (needs updating)")
                    changed_count += 1
            else:
                print(f"✗ PENDING   : {doc}")
                unsummarized_count += 1
        
        print("\nSUMMARY:")
        print(f"Total Documents: {len(all_docs)}")
        print(f"  - Summarized: {summarized_count}")
        print(f"  - Changed (needs update): {changed_count}")
        print(f"  - Pending: {unsummarized_count}")
        print("")
        print(f"Summarization Record: {summarizer.summarised_files_path}")
        
        # Print summary sample for the first summarized document if available
        if summarized_count > 0:
            for doc in sorted(all_docs):
                if doc in summarised_files and summarised_files[doc]['hash'] == summarizer.get_file_hash(os.path.join(summarizer.docs_folder, doc)):
                    summary = summarised_files[doc].get('summary', '')
                    if summary:
                        print("\nSample Summary (first 200 characters):")
                        print("-" * 60)
                        print(f"{summary[:200]}...")
                        break
        
    else:
        # Normal processing mode
        print(f"Processing up to {args.max_docs} documents")
        print(f"Storing summaries directly in: {summarizer.summarised_files_path}")
        print("=" * 110)
        
        if args.no_progress:
            summarizer.show_progress = False
        
        # If force flag is set, create an empty summarised_files.json to force reprocessing
        if args.force:
            print("Force flag set - reprocessing all documents regardless of previous summarization")
            summarizer.save_summarised_files({})
        
        summarizer.run(args.max_docs)