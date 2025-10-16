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
# SOFTWARE.# src/config.py
"""
Centralized configuration management for MKM Research Document Q&A Application.
Provides consistent path handling and configuration across all modules.
"""
import os
from pathlib import Path
from typing import Dict, Any

class PathManager:
    """Centralized path management for the application."""
    
    def __init__(self):
        # Get the project root directory (where main files are located)
        self.project_root = Path(__file__).parent.parent.absolute()
        
        # Core directories
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        
        # Document collections
        self.docs_misc_dir = self.project_root / "docs_misc"
        self.docs_phys_dir = self.project_root / "docs_phys"
        
        # FAISS indices
        self.faiss_dir = self.project_root / "faiss_indices"
        self.faiss_misc_dir = self.project_root / "faiss_misc"
        self.faiss_phys_dir = self.project_root / "faiss_phys"
        
        # Templates and static files
        self.templates_dir = self.project_root / "templates"
        self.static_dir = self.project_root / "static"
        
        # Chat and summary files
        self.chats_file = self.src_dir / "all_chats.json"
        self.summarised_files_misc = self.src_dir / "summarised_files_misc.json"
        self.summarised_files_phys = self.src_dir / "summarised_files_phys.json"
        self.summarised_files_main = self.src_dir / "summarised_files.json"  
        
        # Processing files
        self.processed_files_misc = self.src_dir / "proc_files_misc.json"
        self.processed_files_phys = self.src_dir / "proc_files_phys.json"
        
        # Ensure critical directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all critical directories exist."""
        critical_dirs = [
            self.data_dir,
            self.docs_misc_dir,
            self.docs_phys_dir,
            self.faiss_misc_dir,
            self.faiss_phys_dir,
            self.templates_dir,
            self.static_dir
        ]
        
        for directory in critical_dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_docs_dir(self, collection_type: str) -> Path:
        """Get the documents directory for a specific collection."""
        if collection_type == "misc":
            return self.docs_misc_dir
        elif collection_type == "phys":
            return self.docs_phys_dir
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")
    
    def get_faiss_dir(self, collection_type: str) -> Path:
        """Get the FAISS index directory for a specific collection."""
        if collection_type == "misc":
            return self.faiss_misc_dir
        elif collection_type == "phys":
            return self.faiss_phys_dir
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")
    
    def get_summary_file(self, collection_type: str) -> Path:
        """Get the summary file path for a specific collection."""
        if collection_type == "misc":
            return self.summarised_files_misc
        elif collection_type == "phys":
            return self.summarised_files_phys
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")
    
    def get_processed_file(self, collection_type: str) -> Path:
        """Get the processed files tracking file for a specific collection."""
        if collection_type == "misc":
            return self.processed_files_misc
        elif collection_type == "phys":
            return self.processed_files_phys
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")

# Initialize the path manager
paths = PathManager()

# Legacy CONFIG dictionary for backward compatibility
CONFIG = {
    # Document processing settings
    'chunk_size': 3000,
    'chunk_overlap': 500,
    'default_max_docs': 50,
    
    # Model settings
    'embedding_model': "all-MiniLM-L6-v2",
    'alternative_embedding_models': [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    ],
    
    # File extensions
    'supported_extensions': ['.pdf', '.epub', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls'],
    
    # Collections configuration
    'collections': {
        'misc': {
            'name': 'Miscellaneous Knowledge',
            'description': 'General documents and miscellaneous content',
            'docs_folder': str(paths.docs_misc_dir),
            'faiss_index': str(paths.faiss_misc_dir),
            'summary_file': str(paths.summarised_files_misc),
            'processed_file': str(paths.processed_files_misc)
        },
        'phys': {
            'name': 'Physical Knowledge', 
            'description': 'Physics and physical science documents',
            'docs_folder': str(paths.docs_phys_dir),
            'faiss_index': str(paths.faiss_phys_dir),
            'summary_file': str(paths.summarised_files_phys),
            'processed_file': str(paths.processed_files_phys)
        }
    },
    
    # Legacy path fields for backward compatibility
    'docs_misc_folder': str(paths.docs_misc_dir),
    'docs_phys_folder': str(paths.docs_phys_dir),
    'faiss_misc_path': str(paths.faiss_misc_dir),
    'faiss_phys_path': str(paths.faiss_phys_dir),
    'summarised_files_misc_path': str(paths.summarised_files_misc),
    'summarised_files_phys_path': str(paths.summarised_files_phys),
    'all_chats_path': str(paths.chats_file),
    
    # App settings
    'templates_folder': str(paths.templates_dir),
    'static_folder': str(paths.static_dir),
    'project_root': str(paths.project_root)
}

def get_collection_config(collection_type: str) -> Dict[str, Any]:
    """Get configuration for a specific collection."""
    if collection_type not in CONFIG['collections']:
        raise ValueError(f"Unknown collection type: {collection_type}. Available: {list(CONFIG['collections'].keys())}")
    
    return CONFIG['collections'][collection_type]

def get_all_collections() -> Dict[str, Dict[str, Any]]:
    """Get configuration for all collections."""
    return CONFIG['collections']

def validate_paths():
    """Validate that all configured paths exist and are accessible."""
    issues = []
    
    # Check critical directories
    critical_paths = [
        ('Project Root', paths.project_root),
        ('Source Directory', paths.src_dir),
        ('Templates Directory', paths.templates_dir),
        ('Static Directory', paths.static_dir)
    ]
    
    for name, path in critical_paths:
        if not path.exists():
            issues.append(f"{name} does not exist: {path}")
        elif not path.is_dir():
            issues.append(f"{name} is not a directory: {path}")
    
    # Check collection directories
    for collection_type, config in CONFIG['collections'].items():
        docs_dir = Path(config['docs_folder'])
        faiss_dir = Path(config['faiss_index'])
        
        if not docs_dir.exists():
            issues.append(f"Documents directory for {collection_type} does not exist: {docs_dir}")
        
        # FAISS directories are created as needed, so just warn if missing
        if not faiss_dir.exists():
            print(f"Warning: FAISS index directory for {collection_type} does not exist: {faiss_dir}")
    
    if issues:
        raise RuntimeError(f"Path validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues))
    
    print("✓ All critical paths validated successfully")

def print_configuration():
    """Print the current configuration for debugging."""
    print("=" * 60)
    print("MKM RESEARCH DOCUMENT Q&A - CONFIGURATION")
    print("=" * 60)
    print(f"Project Root: {paths.project_root}")
    print(f"Source Directory: {paths.src_dir}")
    print()
    
    print("COLLECTIONS:")
    for collection_type, config in CONFIG['collections'].items():
        print(f"  {collection_type.upper()} - {config['name']}")
        print(f"    Documents: {config['docs_folder']}")
        print(f"    FAISS Index: {config['faiss_index']}")
        print(f"    Summaries: {config['summary_file']}")
        print(f"    Processed: {config['processed_file']}")
        print()
    
    print("FILES:")
    print(f"  Chat History: {paths.chats_file}")
    print(f"  Templates: {paths.templates_dir}")
    print(f"  Static Files: {paths.static_dir}")
    print()
    
    print("SETTINGS:")
    print(f"  Chunk Size: {CONFIG['chunk_size']}")
    print(f"  Chunk Overlap: {CONFIG['chunk_overlap']}")
    print(f"  Default Max Docs: {CONFIG['default_max_docs']}")
    print(f"  Embedding Model: {CONFIG['embedding_model']}")
    print("=" * 60)

# Validate configuration on import
if __name__ == "__main__":
    try:
        validate_paths()
        print_configuration()
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        exit(1)
else:
    # Only validate critical paths during normal import
    try:
        if not paths.project_root.exists():
            raise RuntimeError(f"Project root does not exist: {paths.project_root}")
    except Exception as e:
        print(f"Warning: Configuration issue detected: {e}")