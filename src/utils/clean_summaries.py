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
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


class SummaryCleaner:
    """
    A class to clean and manage summary JSON files for document collections.
    
    This class provides methods to clean summary files based on different modes:
    - fallback_only: Remove only BASIC_FALLBACK entries
    - force_all: Keep entries but force reprocessing by changing hashes
    - clean_all: Remove all entries (start fresh)
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the SummaryCleaner.
        
        Args:
            base_dir: Base directory path. If None, will auto-detect from script location.
        """
        if base_dir is None:
            # Auto-detect base directory - go up two directories from this script
            self.base_dir = Path(__file__).resolve().parents[2]
        else:
            self.base_dir = Path(base_dir).resolve()
        
        # Configure paths
        self.summary_files = {
            'misc': self.base_dir / 'src' / 'summarised_files_misc.json',
            'phys': self.base_dir / 'src' / 'summarised_files_phys.json'
        }
        
        # For backward compatibility
        self.legacy_summary = self.base_dir / 'src' / 'summarised_files.json'
        
        # Available cleaning modes
        self.MODES = ['fallback_only', 'force_all', 'clean_all']
        
        # Available collections
        self.COLLECTIONS = ['misc', 'phys']
    
    def _get_files_to_process(self, docs_type: Optional[str] = None) -> List[Tuple[str, Path]]:
        """
        Get list of files to process based on docs_type.
        
        Args:
            docs_type: Type of document collection ("misc", "phys", or None for both)
            
        Returns:
            List of tuples (collection_type, file_path)
        """
        files_to_process = []
        
        if docs_type == "misc":
            files_to_process.append(('misc', self.summary_files['misc']))
        elif docs_type == "phys":
            files_to_process.append(('phys', self.summary_files['phys']))
        else:
            # Process both
            files_to_process.extend([
                ('misc', self.summary_files['misc']),
                ('phys', self.summary_files['phys']),
                ('legacy', self.legacy_summary)
            ])
        
        return files_to_process
    
    def _ensure_file_exists(self, file_path: Path) -> None:
        """
        Ensure the file exists, creating it as empty JSON if it doesn't.
        
        Args:
            file_path: Path to the file to check/create
        """
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not file_path.exists():
            # Create empty JSON if file doesn't exist
            with open(file_path, 'w') as f:
                json.dump({}, f)
            print(f"Created empty file: {file_path}")
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load JSON data from file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the JSON data
            
        Raises:
            Exception: If file cannot be loaded
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {str(e)}")
    
    def _save_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Save JSON data to file.
        
        Args:
            file_path: Path to the JSON file
            data: Dictionary containing the data to save
            
        Raises:
            Exception: If file cannot be saved
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise Exception(f"Error saving {file_path}: {str(e)}")
    
    def _clean_fallback_only(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Clean data by removing only BASIC_FALLBACK entries.
        
        Args:
            data: Original data dictionary
            
        Returns:
            Tuple of (cleaned_data, fallback_count)
        """
        fallback_count = sum(1 for info in data.values() 
                           if info.get('summary_type') == 'BASIC_FALLBACK')
        
        cleaned_data = {k: v for k, v in data.items() 
                       if v.get('summary_type') != 'BASIC_FALLBACK'}
        
        return cleaned_data, fallback_count
    
    def _clean_force_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean data by modifying hash values to force reprocessing.
        
        Args:
            data: Original data dictionary
            
        Returns:
            Modified data dictionary
        """
        cleaned_data = data.copy()
        for key in cleaned_data:
            if 'hash' in cleaned_data[key]:
                # Change hash to force reprocessing
                cleaned_data[key]['hash'] = "modified_" + cleaned_data[key]['hash']
        
        return cleaned_data
    
    def _clean_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean data by removing all entries.
        
        Args:
            data: Original data dictionary
            
        Returns:
            Empty dictionary
        """
        return {}
    
    def clean_file(self, file_path: Path, collection_type: str, mode: str) -> Dict[str, int]:
        """
        Clean a single summary file.
        
        Args:
            file_path: Path to the file to clean
            collection_type: Type of collection being cleaned
            mode: Cleaning mode
            
        Returns:
            Dictionary with cleaning statistics
        """
        print(f"Processing {collection_type} summary file: {file_path}")
        
        # Ensure file exists
        self._ensure_file_exists(file_path)
        
        # Load existing file
        data = self._load_json_file(file_path)
        
        # Count before cleaning
        total_before = len(data)
        
        # Apply cleaning based on mode
        if mode == "clean_all":
            cleaned_data = self._clean_all(data)
            print(f"Removing all {total_before} entries")
            removed_count = total_before
            
        elif mode == "fallback_only":
            cleaned_data, fallback_count = self._clean_fallback_only(data)
            print(f"Removing {fallback_count} fallback entries")
            removed_count = fallback_count
            
        elif mode == "force_all":
            cleaned_data = self._clean_force_all(data)
            print(f"Modified hashes for {len(cleaned_data)} entries to force reprocessing")
            removed_count = 0  # No entries removed, just modified
            
        else:
            raise ValueError(f"Unknown cleaning mode: {mode}")
        
        # Save cleaned data
        self._save_json_file(file_path, cleaned_data)
        
        # Count after cleaning
        total_after = len(cleaned_data)
        
        print(f"Cleaned {file_path}")
        print(f"  - Entries before: {total_before}")
        print(f"  - Entries after: {total_after}")
        
        return {
            'total_before': total_before,
            'total_after': total_after,
            'removed_count': removed_count
        }
    
    def clean(self, docs_type: Optional[str] = None, mode: str = "fallback_only") -> Dict[str, Dict[str, int]]:
        """
        Clean summary JSON files based on the selected mode.
        
        Args:
            docs_type: Type of document collection to clean ("misc", "phys", or None for both)
            mode: Cleaning mode ("fallback_only", "force_all", or "clean_all")
            
        Returns:
            Dictionary containing cleaning statistics for each processed file
            
        Raises:
            ValueError: If invalid docs_type or mode is provided
        """
        # Validate inputs
        if docs_type is not None and docs_type not in self.COLLECTIONS:
            raise ValueError(f"Invalid docs_type: {docs_type}. Must be one of: {self.COLLECTIONS}")
        
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: {self.MODES}")
        
        # Print base directory for verification
        print(f"Base directory: {self.base_dir}")
        
        # Get files to process
        files_to_process = self._get_files_to_process(docs_type)
        
        # Process each file
        results = {}
        for collection_type, file_path in files_to_process:
            try:
                stats = self.clean_file(file_path, collection_type, mode)
                results[collection_type] = stats
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                results[collection_type] = {'error': str(e)}
        
        return results
    
    def get_stats(self, docs_type: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about the current state of summary files.
        
        Args:
            docs_type: Type of document collection to check ("misc", "phys", or None for both)
            
        Returns:
            Dictionary containing statistics for each file
        """
        files_to_process = self._get_files_to_process(docs_type)
        stats = {}
        
        for collection_type, file_path in files_to_process:
            try:
                self._ensure_file_exists(file_path)
                data = self._load_json_file(file_path)
                
                total_entries = len(data)
                fallback_entries = sum(1 for info in data.values() 
                                     if info.get('summary_type') == 'BASIC_FALLBACK')
                
                stats[collection_type] = {
                    'total_entries': total_entries,
                    'fallback_entries': fallback_entries,
                    'processed_entries': total_entries - fallback_entries,
                    'file_path': str(file_path)
                }
                
            except Exception as e:
                stats[collection_type] = {'error': str(e)}
        
        return stats


def main():
    """Main entry point for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean summary JSON files')
    parser.add_argument('--collection', choices=['misc', 'phys'], 
                       help='Collection to clean (default: both)')
    parser.add_argument('--mode', choices=['fallback_only', 'force_all', 'clean_all'], 
                       default='fallback_only',
                       help='Cleaning mode: remove fallbacks only, force reprocessing of all, or clean all')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics about summary files without cleaning')
    args = parser.parse_args()
    
    # Create cleaner instance
    cleaner = SummaryCleaner()
    
    if args.stats:
        # Just show stats
        stats = cleaner.get_stats(args.collection)
        print("\nSummary File Statistics:")
        print("=" * 50)
        for collection, data in stats.items():
            if 'error' in data:
                print(f"{collection.upper()}: Error - {data['error']}")
            else:
                print(f"{collection.upper()}:")
                print(f"  Total entries: {data['total_entries']}")
                print(f"  Fallback entries: {data['fallback_entries']}")
                print(f"  Processed entries: {data['processed_entries']}")
                print(f"  File: {data['file_path']}")
    else:
        # Clean files
        results = cleaner.clean(args.collection, args.mode)
        print("\nCleaning completed!")
        
        # Show summary
        total_processed = sum(1 for r in results.values() if 'error' not in r)
        total_errors = sum(1 for r in results.values() if 'error' in r)
        
        print(f"Files processed successfully: {total_processed}")
        if total_errors > 0:
            print(f"Files with errors: {total_errors}")


if __name__ == "__main__":
    main()