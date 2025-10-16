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

# src/document_summariser.py
"""
Document summarization functionality.
Safely reads from existing FAISS indexes to extract document chunks for summarization.
"""
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Import from local modules
from .config import CONFIG, get_collection_config
from .utils.file_utils import (
    get_file_hash,
    load_json_file,
    save_json_file,
    ensure_directory_exists
)


class DocumentSummariser:
    """
    Handles document summarization using existing FAISS indexes.
    Creates a read-only copy to prevent corruption of production indexes.
    """
    
    def __init__(self, docs_type="misc"):
        """
        Initialize the document summariser for a specific document collection.
        
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
        
        # Configure summary file path
        if self.docs_type == "misc":
            self.summary_file = CONFIG['summarised_files_misc_path']
        elif self.docs_type == "phys":
            self.summary_file = CONFIG['summarised_files_phys_path']
        else:
            raise ValueError(f"Unsupported docs_type: {self.docs_type}")
        
        # Initialize the text embeddings model (same as used in processing)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG['embedding_model'],
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Processing settings
        self.show_progress = True
        self.max_documents = CONFIG['default_max_docs']
        
        # Ensure directories exist
        ensure_directory_exists(self.docs_folder)
        ensure_directory_exists(os.path.dirname(self.summary_file))

    def load_processed_files(self):
        """
        Load the record of processed files from JSON.
        
        Returns:
            dict: Dictionary of processed file records
        """
        return load_json_file(self.processed_files_path, default={})

    def create_safe_faiss_copy(self):
        """
        Create a temporary read-only copy of the FAISS index for safe access.
        
        Returns:
            tuple: (temp_path, vector_store) or (None, None) if failed
        """
        if not os.path.exists(self.faiss_path):
            print(f"FAISS index not found at {self.faiss_path}")
            return None, None
        
        try:
            # Create temporary directory for the copy
            temp_dir = os.path.join(os.path.dirname(self.summary_file), f'temp_faiss_copy_{self.docs_type}')
            
            # Remove existing temp directory if it exists
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Copy the FAISS index files
            print(f"Creating safe copy of FAISS index...")
            shutil.copytree(self.faiss_path, temp_dir)
            
            # Load the copied index
            print(f"Loading FAISS copy...")
            vector_store = FAISS.load_local(
                temp_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            print(f"Successfully loaded FAISS copy with {vector_store.index.ntotal} vectors")
            return temp_dir, vector_store
            
        except Exception as e:
            print(f"Error creating FAISS copy: {str(e)}")
            return None, None

    def get_document_chunks_from_faiss(self, vector_store, document_name):
        """
        Extract all chunks belonging to a specific document from FAISS index.
        
        Args:
            vector_store (FAISS): The vector store
            document_name (str): Name of the document to extract chunks for
            
        Returns:
            list: List of document chunks with content and metadata
        """
        try:
            document_chunks = []
            
            # Get all document IDs from the vector store
            all_doc_ids = list(vector_store.docstore._dict.keys())
            
            # Filter chunks by document source
            for doc_id in all_doc_ids:
                try:
                    doc = vector_store.docstore._dict[doc_id]
                    doc_source = doc.metadata.get('source', '')
                    
                    # Match document by various source formats
                    if (document_name in doc_source or 
                        os.path.basename(doc_source) == document_name or
                        os.path.splitext(os.path.basename(doc_source))[0] == os.path.splitext(document_name)[0]):
                        
                        document_chunks.append({
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'doc_id': doc_id
                        })
                        
                except Exception:
                    # Skip problematic chunks
                    continue
            
            if document_chunks:
                # Sort by page number for better context
                document_chunks.sort(key=lambda x: x['metadata'].get('page', 0))
                print(f"Found {len(document_chunks)} chunks for {document_name}")
                return document_chunks
            else:
                print(f"No chunks found for {document_name} in FAISS index")
                return []
                
        except Exception as e:
            print(f"Error extracting chunks for {document_name}: {str(e)}")
            return []

    def query_local_model(self, document_name, document_chunks):
        """
        Query local model for document summary using FAISS-extracted chunks.
        
        Args:
            document_name (str): The name of the document
            document_chunks (list): List of document chunks from FAISS
        
        Returns:
            str: The generated summary
        """
        import requests
        
        if not document_chunks:
            return self.generate_basic_summary(document_name)
        
        # Limit chunks for efficiency
        max_chunks = min(25, len(document_chunks))
        
        for attempt in range(3):
            try:
                # Reduce chunks on retry
                num_chunks = max(2, max_chunks // (attempt + 1))
                selected_chunks = document_chunks[:num_chunks]
                
                print(f"Attempt {attempt+1}: Using {num_chunks} chunks for local model...")
                
                # Build context from FAISS chunks
                context_parts = []
                for chunk in selected_chunks:
                    page_info = f"Page {chunk['metadata'].get('page', 'N/A')}"
                    content_length = 500 if attempt == 0 else 300
                    content = chunk['content'][:content_length]
                    context_parts.append(f"[{page_info}] {content}")
                
                context = "\n\n".join(context_parts)
                
                # Simple prompt for local model
                prompt = f"""Document: {document_name}

Content from FAISS index:
{context[:6000]}

Write a comprehensive summary of this document in 2-3 paragraphs."""

                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.4,
                    "max_tokens": 600
                }

                response = requests.post(
                    "http://localhost:1234/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=120
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                if ("choices" in response_data and 
                    len(response_data["choices"]) > 0 and 
                    "message" in response_data["choices"][0]):
                    
                    print("Successfully received response from local model")
                    return response_data["choices"][0]["message"].get("content", "")
                else:
                    raise ValueError("Unexpected API response format")

            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == 2:
                    return self.generate_basic_summary(document_name)

    def generate_basic_summary(self, document_name):
        """
        Generate a basic summary when other methods fail.
        
        Args:
            document_name (str): The name of the document
        
        Returns:
            str: The generated basic summary
        """
        base_name = os.path.splitext(document_name)[0]
        clean_name = base_name.replace('_', ' ').replace('-', ' ')
        
        return f"""# Summary of {clean_name}

This document appears to be about {clean_name}.

The automatic summarization using the local model was not successful due to technical limitations. This is a placeholder summary generated based on the document filename.

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

    def summarize_documents(self, max_docs=None, force_reprocess=False, clean=False):
        """
        Summarize documents using existing FAISS index safely.
        Creates a temporary copy of FAISS index to prevent corruption.
        
        Args:
            max_docs (int, optional): Maximum number of documents to summarize
            force_reprocess (bool): Whether to reprocess even if already summarized
            clean (bool): Whether to clear existing summaries before processing
        
        Returns:
            bool: True if summarization completed successfully, False otherwise
        """
        # Create safe copy of FAISS index
        temp_faiss_path, vector_store = self.create_safe_faiss_copy()
        
        if not vector_store:
            print("Failed to create safe FAISS copy - cannot proceed")
            return False
        
        try:
            # Load existing summaries
            summarised_files = {} if clean else load_json_file(self.summary_file, default={})
            
            # Get documents to summarize from processed files
            processed_files = self.load_processed_files()
            processed_success = [f for f, info in processed_files.items() 
                                 if info.get('status') == 'SUCCESS']
            
            # Apply document limit
            if max_docs is None:
                max_docs = self.max_documents
            
            if max_docs and len(processed_success) > max_docs:
                document_files = processed_success[:max_docs]
            else:
                document_files = processed_success
            
            if not document_files:
                print("No successfully processed documents found to summarize.")
                return False
            
            print(f"\nSummarizing {len(document_files)} documents in {self.docs_type} collection")
            print("Using safe copy of FAISS index for chunk extraction")
            
            results = []
            
            # Process each document
            with tqdm(total=len(document_files), desc="Summarizing documents", 
                     disable=not self.show_progress) as pbar:
                
                for document_name in document_files:
                    pbar.set_description(f"Summarizing {document_name}")
                    
                    try:
                        # Get file hash for change tracking
                        document_path = os.path.join(self.docs_folder, document_name)
                        file_hash = get_file_hash(document_path)
                        
                        # Skip if already summarized and hasn't changed
                        if (not force_reprocess and 
                            document_name in summarised_files and 
                            summarised_files[document_name]['hash'] == file_hash):
                            results.append((document_name, True, "Used existing summary", "SKIPPED"))
                            pbar.update(1)
                            continue
                        
                        # Extract chunks from FAISS index
                        document_chunks = self.get_document_chunks_from_faiss(vector_store, document_name)
                        
                        if document_chunks:
                            # Get summary from local model
                            summary = self.query_local_model(document_name, document_chunks)
                            
                            # Store the summary
                            summarised_files[document_name] = {
                                'hash': file_hash,
                                'summarised_date': datetime.now().isoformat(),
                                'num_chunks': len(document_chunks),
                                'summary': summary,
                                'summary_type': 'FULL'
                            }
                            
                            results.append((document_name, True, "Summary from FAISS chunks", "SUCCESS"))
                        else:
                            # Generate basic fallback summary
                            basic_summary = self.generate_basic_summary(document_name)
                            
                            summarised_files[document_name] = {
                                'hash': file_hash,
                                'summarised_date': datetime.now().isoformat(),
                                'num_chunks': 0,
                                'summary': basic_summary,
                                'summary_type': 'BASIC_FALLBACK'
                            }
                            
                            results.append((document_name, True, "Basic summary", "BASIC_FALLBACK"))
                        
                        # Save after each document
                        save_json_file(self.summary_file, summarised_files)
                        
                    except Exception as e:
                        print(f"Error summarizing {document_name}: {str(e)}")
                        results.append((document_name, False, str(e), "FAILED"))
                    
                    pbar.update(1)
            
            # Print results summary
            print("\n" + "="*60)
            print("SUMMARIZATION RESULTS")
            print("="*60)
            
            success_count = sum(1 for _, success, _, _ in results if success)
            
            for doc, success, details, status in results:
                status_symbol = "✓" if success else "✗"
                print(f"{status_symbol} {status}: {doc}")
            
            print(f"\nCompleted: {success_count}/{len(results)} documents")
            print(f"Summary file: {self.summary_file}")
            
            return success_count > 0
            
        finally:
            # Clean up temporary FAISS copy
            if temp_faiss_path and os.path.exists(temp_faiss_path):
                try:
                    shutil.rmtree(temp_faiss_path)
                    print("Cleaned up temporary FAISS copy")
                except Exception as e:
                    print(f"Warning: Could not clean up temporary FAISS copy: {str(e)}")


def get_summariser(collection_type):
    """
    Get a document summariser for a specific collection type.
    
    Args:
        collection_type (str): Collection type ('misc' or 'phys')
        
    Returns:
        DocumentSummariser: Initialized document summariser
    """
    return DocumentSummariser(docs_type=collection_type)