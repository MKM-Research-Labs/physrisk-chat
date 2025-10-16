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

# src/document_processor.py
"""
Core document processing functionality.
Handles processing documents and creating vector embeddings.
"""
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import from local modules
from .config import CONFIG, paths, get_collection_config
from .loaders import get_loader_for_file, SUPPORTED_EXTENSIONS
from .utils.text_utils import sanitize_text, standardize_metadata
from .utils.file_utils import (
    get_file_hash,
    load_json_file,
    save_json_file,
    get_supported_files,
    ensure_directory_exists,
    create_processed_record
)
from .utils.diagnostics import diagnose_pdf

class DocumentProcessor:
    """
    Handles document processing, indexing, and vector embedding.
    """
    
    def __init__(self, docs_type="misc"):
        """
        Initialize the document processor for a specific document collection.
        
        Args:
            docs_type (str): Type of document collection ("misc" or "phys")
        """
        # Store the docs_type for reference
        self.docs_type = docs_type
        
        # Get collection configuration from centralized config
        self.collection_config = get_collection_config(docs_type)
        
        # Configure paths from centralized config
        self.docs_folder = self.collection_config['docs_folder']
        self.processed_files_path = self.collection_config['processed_file']
        self.faiss_path = self.collection_config['faiss_index']
        
        # Initialize the text embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG['embedding_model'],  # Updated key name
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create text splitter with configuration settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG['chunk_size'],
            chunk_overlap=CONFIG['chunk_overlap']
        )
        
        # Processing settings
        self.show_progress = True  # Flag to control progress bar display
        self.max_documents = CONFIG['default_max_docs']
        
        # Ensure directories exist
        ensure_directory_exists(self.docs_folder)
        ensure_directory_exists(os.path.dirname(self.processed_files_path))
        ensure_directory_exists(self.faiss_path)

    def diagnose_and_report_problematic_files(self):
        """
        Run diagnostics on all files that have errors in the processed_files list
        and generate a detailed report to help fix issues.
        
        Returns:
            list: List of diagnostic reports for problematic files
        """
        processed_files = self.load_processed_files()
    
        # Find files with errors
        error_files = {filename: info for filename, info in processed_files.items() 
                       if info.get('status') == 'ERROR'}
    
        if not error_files:
            print(f"No error files found in {self.processed_files_path}")
            return []
    
        print(f"\nRunning diagnostics on {len(error_files)} problematic files in {self.docs_folder}...")
    
        diagnostic_reports = []
    
        for filename, info in error_files.items():
            file_path = os.path.join(self.docs_folder, filename)
        
            if not os.path.exists(file_path):
                print(f"⚠️ Cannot diagnose {filename} - file not found")
                continue
            
            print(f"Diagnosing {filename}...")
            try:
                diagnostic = diagnose_pdf(file_path)
                diagnostic_reports.append(diagnostic)
            
                # Print key findings
                print(f"  Assessment: {diagnostic['overall_assessment']}")
                if diagnostic['issues_detected']:
                    print(f"  Issues: {', '.join(diagnostic['issues_detected'])}")
                print(f"  Recommendation: {diagnostic['recommended_approach']}")
            except Exception as e:
                print(f"⚠️ Error during diagnosis: {str(e)}")
    
        # Generate a report file
        if diagnostic_reports:
            report_path = os.path.join(os.path.dirname(self.processed_files_path), 
                                       f'pdf_diagnostic_report_{self.docs_type}.json')
            try:
                save_json_file(report_path, diagnostic_reports)
                print(f"\nDiagnostic report saved to {report_path}")
            except Exception as e:
                print(f"\nError saving diagnostic report: {str(e)}")
    
        return diagnostic_reports

    def handle_problematic_document(self, filename, processed_files, error_msg=None):
        """
        Handle problematic documents by updating processed_files with error status
        and attempting recovery if possible.
        
        Args:
            filename (str): Name of the problematic file
            processed_files (dict): Dictionary of processed file records
            error_msg (str, optional): Error message from the processing attempt
            
        Returns:
            list or None: List of recovered document chunks if successful, None otherwise
        """
        file_path = os.path.join(self.docs_folder, filename)
        file_hash = get_file_hash(file_path)
    
        # Record failure in processed files
        processed_files[filename] = {
            'hash': file_hash,
            'processed_date': datetime.now().isoformat(),
            'num_chunks': 0,
            'status': 'ERROR_HANDLED',  # Mark as handled error
            'error': error_msg or "Unknown error during processing"
        }
    
        # For binary error specifically
        if error_msg and ('Invalid Elementary Object' in error_msg or 
                          "can't concat str to ByteStringObject" in error_msg or
                          "stream filter 'FlateDecode'" in error_msg):
            print(f"⚠️ Detected binary content issue in {filename}")
            try:
                # Attempt binary-safe extraction using enhanced PDF loader
                from src.loaders.pdf_loader import EnhancedPDFLoader
                print(f"Applying special binary-safe handling for {filename}...")
            
                # Use the enhanced loader directly
                loader = EnhancedPDFLoader(file_path)
                pages = loader.load()
            
                if pages and len(pages) > 0:
                    # Sanitize page content
                    sanitized_pages = []
                    for page in pages:
                        # Skip pages with error messages
                        if isinstance(page.page_content, str) and page.page_content.startswith('[Error'):
                            continue
                        
                        # Create a new Document object with sanitized content
                        sanitized_page = Document(
                            page_content=sanitize_text(page.page_content),
                            metadata=page.metadata
                        )
                        sanitized_pages.append(sanitized_page)
                
                    # Split into chunks if we have valid content
                    if sanitized_pages:
                        chunks = self.text_splitter.split_documents(sanitized_pages)
                    
                        if chunks:
                            chunks = standardize_metadata(chunks, self.docs_type)
                            # Update processed files to successful
                            processed_files[filename] = {
                                'hash': file_hash,
                                'processed_date': datetime.now().isoformat(),
                                'num_chunks': len(chunks),
                                'status': 'RECOVERED',  # Mark as recovered
                                'recovery_method': 'binary_safe_extraction'
                            }
                            print(f"✅ Successfully recovered {len(chunks)} chunks from {filename}")
                            return chunks
            
                print(f"⚠️ Recovery attempt failed for {filename}")
                return None
            
            except Exception as recovery_error:
                print(f"⚠️ Error during recovery attempt: {str(recovery_error)}")
                # Update the error message to include recovery attempt failure
                processed_files[filename]['error'] += f" | Recovery failed: {str(recovery_error)}"
                return None
    
        return None
    
    def load_processed_files(self):
        """
        Load the record of processed files from JSON.
        
        Returns:
            dict: Dictionary of processed file records
        """
        return load_json_file(self.processed_files_path, default={})

    def save_processed_files(self, processed_files):
        """
        Save the record of processed files to JSON.
        
        Args:
            processed_files (dict): Dictionary of processed file records
            
        Returns:
            bool: True if successful, False otherwise
        """
        return save_json_file(self.processed_files_path, processed_files)

    def process_single_document(self, filename, processed_files, force_reprocess=False):
        """
        Process a single document and generate chunks for indexing.
        
        Args:
            filename (str): Name of the document file
            processed_files (dict): Dictionary of processed file records
            force_reprocess (bool): Whether to reprocess even if unchanged
            
        Returns:
            list or None: List of document chunks if successful, None otherwise
        """
        file_path = os.path.join(self.docs_folder, filename)
        file_hash = get_file_hash(file_path)
        
        # Skip if file hasn't changed
        if not force_reprocess and filename in processed_files and processed_files[filename]['hash'] == file_hash:
            return None
        
        try:
            print(f"\nProcessing {filename} from {self.docs_type}...")
            loader = get_loader_for_file(file_path)
            
            # Load pages with progress bar
            try:
                pages = loader.load()
                
                if not pages:
                    print(f"⚠️ Document {filename} appears to be empty or unreadable")
                    # Record in processed files that this document was empty
                    processed_files[filename] = create_processed_record(
                        filename, file_path, status='EMPTY', num_chunks=0
                    )
                    self.save_processed_files(processed_files)
                    return None
                
                # Sanitize page content before splitting
                sanitized_pages = []
                for page in pages:
                    # Create a new Document object with sanitized content
                    sanitized_page = Document(
                        page_content=sanitize_text(page.page_content),
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
                    print(f"⚠️ No chunks were generated from {filename}")
                    # Record document was processed but generated no chunks
                    processed_files[filename] = create_processed_record(
                        filename, file_path, status='NO_CHUNKS', num_chunks=0
                    )
                    self.save_processed_files(processed_files)
                    return None
                
                # Update processed files record
                processed_files[filename] = create_processed_record(
                    filename, file_path, status='SUCCESS', num_chunks=len(chunks)
                )
                
                print(f"✅ Generated {len(chunks)} chunks from {filename}")
                if chunks:
                    chunks = standardize_metadata(chunks, self.docs_type)
                return chunks
                
            except IndexError as e:
                print(f"⚠️ Index error with {filename}: {str(e)} - document might be empty or corrupted")
                # Try recovery for problematic documents
                recovery_chunks = self.handle_problematic_document(filename, processed_files, str(e))
                if recovery_chunks:
                    return recovery_chunks
    
                # If recovery failed, proceed with the existing error handling
                processed_files[filename] = create_processed_record(
                    filename, file_path, status='ERROR', num_chunks=0, error=str(e)
                )
                self.save_processed_files(processed_files)
                return None

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                # Try recovery for problematic documents
                recovery_chunks = self.handle_problematic_document(filename, processed_files, str(e))
                if recovery_chunks:
                    return recovery_chunks
        
                # If recovery failed, proceed with the existing error handling
                processed_files[filename] = create_processed_record(
                    filename, file_path, status='ERROR', num_chunks=0, error=str(e)
                )
                self.save_processed_files(processed_files)
                return None
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {str(e)}")
            # Record failure in processed files
            processed_files[filename] = create_processed_record(
                filename, file_path, status='ERROR', num_chunks=0, error=str(e)
            )
            self.save_processed_files(processed_files)
            return None

    def process_documents(self, force_reprocess=False):
        """
        Process all supported documents with progress tracking.
        
        Args:
            force_reprocess (bool): Whether to reprocess all documents
            
        Returns:
            bool: True if processing completed successfully, False otherwise
        """
        processed_files = {} if force_reprocess else self.load_processed_files()
        
        all_chunks = []
        
        # Get list of supported files
        document_files = get_supported_files(self.docs_folder, SUPPORTED_EXTENSIONS)
        
        if not document_files:
            print(f"No supported files found in {self.docs_folder}.")
            print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
            return False
        
        total_documents = len(document_files)
    
        # First, identify all files that need processing
        files_needing_processing = []
        
        for filename in document_files:
            file_path = os.path.join(self.docs_folder, filename)
            file_hash = get_file_hash(file_path)
        
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
                            clean_text = sanitize_text(chunk.page_content)
                            if clean_text and len(clean_text.strip()) > 10:  # Minimum text length
                                texts.append(clean_text)
                                metadatas.append(chunk.metadata)
                            pbar.update(1)
                    
                    # Add to existing store in batches
                    batch_size = 5  # Smaller batch size for more stability
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
                                print(f"⚠️ Warning: Batch failed, trying one by one: {str(e)}")
                                # Fall back to one-by-one processing if batch fails
                                for j in range(len(batch_texts)):
                                    try:
                                        existing_store.add_texts(
                                            texts=[batch_texts[j]], 
                                            metadatas=[batch_metadatas[j]]
                                        )
                                    except Exception as e2:
                                        print(f"⚠️ Skipping problematic text: {str(e2)[:100]}...")
                            
                            pbar.update(len(batch_texts))
                    
                    vector_store = existing_store
                    print("✅ Successfully merged new documents")
                    
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
                                clean_text = sanitize_text(chunk.page_content)
                                if clean_text and len(clean_text.strip()) > 10:
                                    texts.append(clean_text)
                                    metadatas.append(chunk.metadata)
                                pbar.update(1)
                        
                        print(f"Creating vector store from {len(texts)} clean texts...")
                        vector_store = FAISS.from_texts(texts, self.embeddings, metadatas)
                    except Exception as e2:
                        print(f"❌ Error creating new index: {str(e2)}")
                        return self._try_with_alternative_embeddings(texts, metadatas)
            else:
                try:
                    print("Creating new index...")
                    # Clean and prepare all texts
                    texts = []
                    metadatas = []
                    
                    with tqdm(total=len(all_chunks), desc="Preparing text data",
                            disable=not self.show_progress) as pbar:
                        for chunk in all_chunks:
                            clean_text = sanitize_text(chunk.page_content)
                            if clean_text and len(clean_text.strip()) > 10:
                                texts.append(clean_text)
                                metadatas.append(chunk.metadata)
                            pbar.update(1)
                    
                    print(f"Creating vector store from {len(texts)} documents...")
                    vector_store = FAISS.from_texts(texts, self.embeddings, metadatas)
                    
                except Exception as e:
                    print(f"❌ Error creating index: {str(e)}")
                    return self._try_with_alternative_embeddings(texts, metadatas)
            
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
                            print("✅ Index verification successful")
                        else:
                            print("⚠️ Index verification skipped: could not load index")
                except Exception as e:
                    print(f"⚠️ Index verification failed: {str(e)}")
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
            print(f"⚠️ Warning during cleanup: {str(e)}")
            
        return True
    
    def _try_with_alternative_embeddings(self, texts, metadatas):
        """
        Try creating the index with alternative embedding models when the primary model fails.
        
        Args:
            texts (list): List of text chunks to embed
            metadatas (list): List of metadata for each chunk
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Try with alternative embedding model
        try:
            print("Attempting with alternative embedding model...")
            alt_embeddings = HuggingFaceEmbeddings(
                model_name=CONFIG['fallback_embedding_model'],
                encode_kwargs={'normalize_embeddings': True}
            )
            vector_store = FAISS.from_texts(texts, alt_embeddings, metadatas)
            
            # Save if successful
            vector_store.save_local(self.faiss_path)
            print("✅ Successfully created index with alternative embedding model")
            
            # Update the instance embeddings to match what was used
            self.embeddings = alt_embeddings
            return True
        except Exception as e2:
            print(f"❌ Alternative model failed: {str(e2)}")
            
            # Last resort - try with another fallback model
            try:
                print("Trying last resort embedding model...")
                last_resort_embeddings = HuggingFaceEmbeddings(
                    model_name=CONFIG['last_resort_model'],
                    encode_kwargs={'normalize_embeddings': True}
                )
                vector_store = FAISS.from_texts(texts, last_resort_embeddings, metadatas)
                
                # Save if successful
                vector_store.save_local(self.faiss_path)
                print("✅ Successfully created index with last resort embedding model")
                
                # Update the instance embeddings to match what was used
                self.embeddings = last_resort_embeddings
                return True
            except Exception as e3:
                print(f"❌ All embedding attempts failed: {str(e3)}")
                return False
    
    def setup_alternative_embeddings(self):
        """
        Set up alternative embeddings when the default model encounters issues.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("Setting up alternative embedding model...")
            # Use a more robust sentence-transformers model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=CONFIG['fallback_embedding_model'],
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Successfully set up alternative embeddings")
            return True
        except Exception as e:
            print(f"❌ Error setting up alternative embeddings: {str(e)}")
            
            # Try another fallback option
            try:
                print("Trying MPNet base model as fallback...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=CONFIG['last_resort_model'],
                    encode_kwargs={'normalize_embeddings': True}
                )
                print("✅ Successfully set up MPNet embeddings")
                return True
            except Exception as e2:
                print(f"❌ Error setting up fallback embeddings: {str(e2)}")
                return False


# Create a simple package-level helper function to initialize processors
def get_processor(collection_type):
    """
    Get a document processor for a specific collection type.
    
    Args:
        collection_type (str): Collection type ('misc' or 'phys')
        
    Returns:
        DocumentProcessor: Initialized document processor
    """
    return DocumentProcessor(docs_type=collection_type)