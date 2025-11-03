#!/usr/bin/env python3
"""
Unified entry point for the MKM Research Document Processor and Q&A Application.
Handles document processing, summarization, and web interface.
"""
import sys
import os
import argparse
import json
import hashlib
from docs_misc.cleanup_misc import main as cleanup_misc
from docs_phys.cleanup_phys import main as cleanup_phys
from src.app import DocumentQAApp
from src.document_processor import DocumentProcessor
from src.config import get_collection_config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='MKM Research Document Processor and Q&A Application'
    )
    
    # Mode selection
    parser.add_argument('-web-only', action='store_true',
                       help='Skip all processing and go directly to web interface')
    parser.add_argument('-process-only', action='store_true',
                       help='Only process documents without starting web interface')
    parser.add_argument('-summarize-only', action='store_true',
                       help='Only run summarization without document processing or web interface')
    parser.add_argument('-list-summaries', action='store_true',
                       help='List all summarized documents without processing')
    
    # Web server options
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web server on (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind the web server to (default: 127.0.0.1)')
    parser.add_argument('--debug', action='store_true', default=True,
                       help='Run Flask in debug mode (default: True)')
    
    # Collection selection
    parser.add_argument('--collection', choices=['misc', 'phys', 'all'], default='all',
                       help='Document collection to process (default: all)')
    
    # Document processing options
    parser.add_argument('--max-docs', type=int, default=50,
                       help='Maximum number of documents to process (default: 50)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of all documents')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress bars')
    parser.add_argument('--alt-embeddings', action='store_true',
                       help='Use alternative embedding models')
    parser.add_argument('--diagnose', action='store_true',
                       help='Run diagnostics on problematic files')
    
    # Summarization options
    parser.add_argument('--summarize', action='store_true',
                       help='Run document summarization after processing')
    parser.add_argument('--clean-summaries', action='store_true',
                       help='Clear existing summaries before processing')
    
    return parser.parse_args()


def load_json_file(file_path, default=None):
    """Load JSON file with error handling."""
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


def get_file_hash(file_path):
    """Get SHA-256 hash of file content."""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    except IOError as e:
        print(f"Warning: Could not hash {file_path}: {e}")
        return ""


def process_collection(collection_type, args):
    """Process a specific document collection."""
    print(f"\n{'='*50}")
    print(f"Processing {collection_type.upper()} document collection")
    print(f"{'='*50}")
    
    processor = DocumentProcessor(docs_type=collection_type)
    processor.max_documents = args.max_docs
    processor.show_progress = not args.no_progress
    
    if args.alt_embeddings:
        processor.setup_alternative_embeddings()
    
    if args.diagnose:
        processor.diagnose_and_report_problematic_files()
        return True
    
    # Process documents
    try:
        processor.process_documents(force_reprocess=args.force)
        print(f"✅ {collection_type.upper()} collection processing completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error processing {collection_type} collection: {str(e)}")
        return False


def summarize_collection(collection_type, args):
    """Run summarization for a specific collection."""
    print(f"\n{'='*50}")
    print(f"Summarizing {collection_type.upper()} document collection")
    print(f"{'='*50}")
    
    try:
        processor = DocumentProcessor(docs_type=collection_type)
        success = processor.summarize_documents(
            max_docs=args.max_docs,
            force_reprocess=args.force,
            clean=args.clean_summaries
        )
        
        if success:
            print(f"✅ {collection_type.upper()} collection summarization completed successfully")
            return True
        else:
            print(f"❌ {collection_type.upper()} collection summarization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error summarizing {collection_type} collection: {str(e)}")
        return False


def list_collection_summaries(collection_type):
    """List summary status for a specific collection."""
    processor = DocumentProcessor(docs_type=collection_type)
    collection_config = get_collection_config(collection_type)
    summary_file = collection_config['summary_file']
    docs_folder = collection_config['docs_folder']
    
    # Get all documents in the folder
    all_docs = [f for f in os.listdir(docs_folder) 
                if os.path.isfile(os.path.join(docs_folder, f))]
    
    # Load summarized files data
    summarised_files = load_json_file(summary_file, default={})
    
    # Check status for each file
    summarized_count = 0
    unsummarized_count = 0
    changed_count = 0
    
    for doc in sorted(all_docs):
        doc_path = os.path.join(docs_folder, doc)
        current_hash = get_file_hash(doc_path)
        
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
    
    print(f"\nSUMMARY:")
    print(f"Total Documents: {len(all_docs)}")
    print(f"  - Summarized: {summarized_count}")
    print(f"  - Changed (needs update): {changed_count}")
    print(f"  - Pending: {unsummarized_count}")
    print(f"\nSummarization Record: {summary_file}")
    
    # Print summary sample for the first summarized document if available
    if summarized_count > 0:
        for doc in sorted(all_docs):
            if doc in summarised_files:
                stored_hash = summarised_files[doc]['hash']
                current_hash = get_file_hash(os.path.join(docs_folder, doc))
                if stored_hash == current_hash:
                    summary = summarised_files[doc].get('summary', '')
                    if summary:
                        print("\nSample Summary (first 200 characters):")
                        print("-" * 60)
                        print(f"{summary[:200]}...")
                        break


def run_cleanup():
    """Run cleanup for all document collections."""
    print("\n🧹 Cleaning up misc documents...")
    cleanup_misc()
    
    print("🧹 Cleaning up physics documents...")
    cleanup_phys()


def run_processing(args):
    """Run document processing for selected collections."""
    collections = []
    if args.collection in ['misc', 'all']:
        collections.append('misc')
    if args.collection in ['phys', 'all']:
        collections.append('phys')
    
    results = {}
    for collection in collections:
        results[collection] = process_collection(collection, args)
    
    return all(results.values()), results


def run_summarization(args):
    """Run summarization for selected collections."""
    collections = []
    if args.collection in ['misc', 'all']:
        collections.append('misc')
    if args.collection in ['phys', 'all']:
        collections.append('phys')
    
    results = {}
    for collection in collections:
        results[collection] = summarize_collection(collection, args)
    
    return all(results.values()), results


def list_summaries(args):
    """List summaries for selected collections."""
    if args.collection in ['misc', 'all']:
        print(f"\n{'='*50}")
        print(f"SUMMARIES FOR MISC DOCUMENTS")
        print(f"{'='*50}")
        list_collection_summaries('misc')
    
    if args.collection in ['phys', 'all']:
        print(f"\n{'='*50}")
        print(f"SUMMARIES FOR PHYS DOCUMENTS")
        print(f"{'='*50}")
        list_collection_summaries('phys')


def start_web_application(args):
    """Start the Flask web application."""
    print("\n" + "=" * 60)
    print("STARTING WEB APPLICATION")
    print("=" * 60)
    print(f"🚀 Starting web server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        app = DocumentQAApp()
        app.run(debug=args.debug, port=args.port, host=args.host)
    except KeyboardInterrupt:
        print("\n⛔️ Shutting down gracefully...")
        return 0
    except Exception as e:
        print(f"❌ Error starting web application: {str(e)}")
        return 1
    
    return 0


def main():
    """Main entry point"""
    # Check if this is the Flask reloader process
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    
    args = parse_arguments()
    
    # Only run document operations in the main process (not the reloader)
    if not is_reloader_process:
        
        # Handle list-summaries mode
        if args.list_summaries:
            list_summaries(args)
            return 0
        
        # Handle summarize-only mode
        if args.summarize_only:
            print("=" * 60)
            print("RUNNING SUMMARIZATION ONLY")
            print("=" * 60)
            
            all_success, results = run_summarization(args)
            
            if all_success:
                print("\n✅ All summarization completed successfully!")
                return 0
            else:
                print("\n❌ Some summarization failed!")
                for collection, success in results.items():
                    if not success:
                        print(f"  - {collection.upper()} collection summarization failed")
                return 1
        
        # Handle web-only mode
        if args.web_only:
            print("⏭️ Skipping document processing as requested...")
        
        # Handle process-only or standard mode
        elif args.process_only or not args.web_only:
            print("=" * 60)
            print("STARTING DOCUMENT PROCESSING")
            print("=" * 60)
            
            # Run cleanup
            run_cleanup()
            
            # Process documents
            all_success, results = run_processing(args)
            
            if not all_success:
                print("\n❌ Some document processing failed!")
                for collection, success in results.items():
                    if not success:
                        print(f"  - {collection.upper()} collection failed")
                
                if args.process_only or not args.summarize:
                    user_input = input("\nContinue anyway? (y/N): ")
                    if user_input.lower() != 'y':
                        return 1
            else:
                print("\n✅ All document processing completed successfully!")
            
            # Run summarization if requested
            if args.summarize:
                print("\n" + "=" * 60)
                print("STARTING DOCUMENT SUMMARIZATION")
                print("=" * 60)
                
                all_success_sum, results_sum = run_summarization(args)
                
                if not all_success_sum:
                    print("\n❌ Some summarization failed!")
                    for collection, success in results_sum.items():
                        if not success:
                            print(f"  - {collection.upper()} collection summarization failed")
                    
                    if not args.process_only:
                        user_input = input("\nContinue to web interface anyway? (y/N): ")
                        if user_input.lower() != 'y':
                            return 1
                else:
                    print("\n✅ All summarization completed successfully!")
            
            # Exit if process-only mode
            if args.process_only:
                print("\nProcessing complete!")
                return 0
    
    # Start web application (runs in both parent and reloader processes)
    return start_web_application(args)


if __name__ == "__main__":
    sys.exit(main())