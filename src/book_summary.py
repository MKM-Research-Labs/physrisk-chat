#!/usr/bin/env python3
"""
Enhanced script that processes all books from proc_files_{knowledge}.json:
1) Loads FAISS index once at startup
2) Goes through each book and checks if summary exists
3) If no summary exists, creates one using local LLM
4) Saves results to summarised_files_{knowledge}.json

Usage: python3 book_summary.py [misc|phys]
"""

import os
import sys
import json
import pickle
import requests
from datetime import datetime
import hashlib

# Add the project root to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Required dependencies not installed: {e}")
    print("Please install: pip install faiss-cpu sentence-transformers numpy")
    sys.exit(1)

# Local LLM configuration
LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"
LOCAL_LLM_MODEL = "deepseek-r1-distill-qwen-1.5b"


def get_paths(knowledge_type):
    """Get file paths based on knowledge type (misc or phys)"""
    return {
        'faiss_index': f"../faiss_{knowledge_type}/index.faiss",
        'faiss_pkl': f"../faiss_{knowledge_type}/index.pkl",
        'proc_files': f"proc_files_{knowledge_type}.json",
        'output_file': f"summarised_files_{knowledge_type}.json"
    }


def load_json_file(file_path, default=None):
    """Load JSON file with error handling"""
    if default is None:
        default = {}
    
    if not os.path.exists(file_path):
        return default
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return default


def load_faiss_index(paths):
    """Load FAISS index and metadata once at startup"""
    print(f"📂 Loading FAISS index from {paths['faiss_index']}...")
    try:
        index = faiss.read_index(paths['faiss_index'])
        with open(paths['faiss_pkl'], 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
        
        # Handle LangChain tuple structure
        if isinstance(metadata, tuple) and len(metadata) >= 2:
            docstore, index_to_docstore_id = metadata
            print(f"   Found docstore: {type(docstore)}")
            print(f"   Found index mapping: {type(index_to_docstore_id)} with {len(index_to_docstore_id)} entries")
            print(f"   ✅ Using LangChain structure with {len(index_to_docstore_id)} mappings")
            return index, (docstore, index_to_docstore_id)
        
        return index, metadata
    except Exception as e:
        print(f"❌ Error loading FAISS: {e}")
        return None, None


def initialize_embedding_model():
    """Initialize the embedding model once at startup"""
    print("🧠 Loading embedding model (one-time operation)...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded")
        return model
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return None


def search_book_content(book_name, embedding_model, index, metadata, top_k=5):
    """Search for relevant content about a specific book"""
    try:
        # Handle LangChain structure
        if isinstance(metadata, tuple) and len(metadata) == 2:
            docstore, index_to_docstore_id = metadata
            
            query = f"provide a summary of the book {book_name}"
            
            query_embedding = embedding_model.encode([query])
            query_embedding = np.array(query_embedding, dtype=np.float32)
            
            distances, indices = index.search(query_embedding, top_k)
            
            relevant_chunks = []
            
            for i, idx in enumerate(indices[0]):
                try:
                    if idx in index_to_docstore_id:
                        doc_id = index_to_docstore_id[idx]
                        document = docstore.search(doc_id)
                        if document and hasattr(document, 'page_content'):
                            content = document.page_content
                            relevant_chunks.append(content)
                        
                except Exception as e:
                    continue
            
            if relevant_chunks:
                result = "\n\n".join(relevant_chunks[:3])
                return result
            else:
                return ""
        
        else:
            return ""
            
    except Exception as e:
        return ""


def call_llm_for_summary(book_name, context):
    """Call the local LLM to generate a summary"""
    try:
        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that creates comprehensive book summaries. Analyze the provided content and create detailed, structured summaries. Do not include any thinking process or preamble - provide only the final summary."
                },
                {
                    "role": "user", 
                    "content": f"""Please provide a comprehensive summary of the book "{book_name}" based on the following content:

{context}

Please structure your summary to include:
1. Main themes and topics
2. Key concepts and ideas
3. Important takeaways
4. Target audience and relevance

Provide a detailed, well-organized summary."""
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.7,
            "stream": False
        }
        
        response = requests.post(LOCAL_LLM_URL, json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            raw_summary = result["choices"][0]["message"]["content"].strip()
            
            # Clean up the summary by removing thinking process
            cleaned_summary = clean_summary_text(raw_summary)
            return cleaned_summary
        else:
            return None
            
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        return None


def clean_summary_text(text):
    """Remove thinking process and other unwanted elements from LLM output"""
    # Remove <think>...</think> blocks
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove common preamble phrases
    preamble_patterns = [
        r'^.*?(?=###|\*\*|1\.|Main themes|Key concepts|Important takeaways)',
        r'^.*?(?=The book)',
        r'^.*?(?=This book)'
    ]
    
    for pattern in preamble_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple blank lines to double
    text = text.strip()
    
    return text


def save_summaries(summaries_data, output_path):
    """Save the summaries data to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summaries_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Error saving summaries: {e}")
        return False


def main():
    """Main execution function"""
    # Parse command line arguments
    if len(sys.argv) != 2 or sys.argv[1] not in ['misc', 'phys']:
        print("Usage: python3 book_summary.py [misc|phys]")
        print("  misc - Process miscellaneous books")
        print("  phys - Process physics books")
        return 1
    
    knowledge_type = sys.argv[1]
    paths = get_paths(knowledge_type)
    
    print("=" * 80)
    print(f"🚀 BATCH BOOK SUMMARY PROCESSOR ({knowledge_type.upper()})")
    print("=" * 80)
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Load all required resources once at startup
    print("\n" + "="*50)
    print("INITIALIZATION PHASE")
    print("="*50)
    
    # Load FAISS index once
    index, metadata = load_faiss_index(paths)
    if index is None or metadata is None:
        print("❌ Failed to load FAISS index")
        return 1
    
    # Load embedding model once
    embedding_model = initialize_embedding_model()
    if embedding_model is None:
        print("❌ Failed to load embedding model")
        return 1
    
    # Load processed files list
    print(f"📚 Loading processed files from {paths['proc_files']}...")
    proc_files = load_json_file(paths['proc_files'])
    if not proc_files:
        print(f"❌ Could not load {paths['proc_files']}")
        return 1
    print(f"✅ Found {len(proc_files)} processed books")
    
    # Load existing summaries
    print(f"📄 Loading existing summaries from {paths['output_file']}...")
    existing_summaries = load_json_file(paths['output_file'], default={})
    print(f"✅ Found {len(existing_summaries)} existing summaries")
    
    # Step 2: Process each book
    print("\n" + "="*50)
    print("BOOK PROCESSING PHASE")
    print("="*50)
    
    books_to_process = []
    books_already_done = []
    
    # Check which books need processing
    for book_name in proc_files.keys():
        if book_name in existing_summaries:
            books_already_done.append(book_name)
        else:
            books_to_process.append(book_name)
    
    print(f"\n📊 Processing Status:")
    print(f"  📚 Total books: {len(proc_files)}")
    print(f"  ✅ Already summarized: {len(books_already_done)}")
    print(f"  ⏳ Need processing: {len(books_to_process)}")
    
    if not books_to_process:
        print("\n🎉 All books already have summaries! Nothing to do.")
        return 0
    
    # Process books that need summaries
    print(f"\n🔄 Processing {len(books_to_process)} books...")
    successful_summaries = 0
    failed_summaries = 0
    
    for i, book_name in enumerate(books_to_process, 1):
        print(f"\n📖 [{i}/{len(books_to_process)}] Processing: {book_name[:60]}...")
        
        # Search for relevant content
        print("  🔍 Searching for relevant content...")
        context = search_book_content(book_name, embedding_model, index, metadata)
        
        if not context:
            print("  ⚠️  No relevant content found, skipping")
            failed_summaries += 1
            continue
        
        print(f"  ✅ Found {len(context)} characters of context")
        
        # Generate summary
        print("  🤖 Generating summary with LLM...")
        summary = call_llm_for_summary(book_name, context)
        
        if summary:
            # Add to existing summaries
            existing_summaries[book_name] = {
                "hash": "generated",
                "summarised_date": datetime.now().isoformat(),
                "summary": summary,
                "summary_type": "FULL",
                "method": "FAISS_SEARCH_LLM",
                "model": LOCAL_LLM_MODEL,
                "knowledge_type": knowledge_type
            }
            
            print(f"  ✅ Summary generated ({len(summary)} characters)")
            successful_summaries += 1
            
            # Save after each successful summary
            save_summaries(existing_summaries, paths['output_file'])
            
        else:
            print("  ❌ Failed to generate summary")
            failed_summaries += 1
    
    # Final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"✅ Successfully processed: {successful_summaries}")
    print(f"❌ Failed to process: {failed_summaries}")
    print(f"📄 Total summaries now: {len(existing_summaries)}")
    print(f"💾 Saved to: {paths['output_file']}")
    
    if successful_summaries > 0:
        print(f"\n🎉 Successfully added {successful_summaries} new book summaries!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())