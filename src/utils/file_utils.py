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

# src/utils/file_utils.py
"""
File utility functions for document processing.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Union, Any

def get_file_hash(filepath: str) -> str:
    """
    Get file modification time as a simple hash.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        str: Modification time as a string to use as a hash
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return str(os.path.getmtime(filepath))

def load_json_file(filepath: str, default: Any = None) -> Any:
    """
    Load JSON data from a file with error handling.
    
    Args:
        filepath (str): Path to the JSON file
        default: Default value to return if file doesn't exist or is invalid
        
    Returns:
        The parsed JSON data or the default value
    """
    if not os.path.exists(filepath):
        return default if default is not None else {}
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {filepath}: {str(e)}")
        return default if default is not None else {}
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return default if default is not None else {}

def save_json_file(filepath: str, data: Any) -> bool:
    """
    Save data to a JSON file with error handling.
    
    Args:
        filepath (str): Path to save the JSON file
        data: Data to save as JSON
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file {filepath}: {str(e)}")
        return False

def get_supported_files(directory: str, supported_extensions: tuple) -> List[str]:
    """
    Get a list of files with supported extensions in a directory.
    
    Args:
        directory (str): Directory to search for files
        supported_extensions (tuple): Tuple of supported file extensions (e.g., ('.pdf', '.docx'))
        
    Returns:
        List[str]: List of filenames (not full paths) of supported files
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return []
    
    return [
        f for f in os.listdir(directory) 
        if f.lower().endswith(supported_extensions) and
        os.path.isfile(os.path.join(directory, f))
    ]

def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory (str): Directory path to ensure exists
        
    Returns:
        bool: True if directory exists or was created, False on error
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {str(e)}")
        return False

def create_processed_record(
    filename: str, 
    file_path: str, 
    status: str = "SUCCESS", 
    num_chunks: int = 0, 
    error: str = None
) -> Dict[str, Any]:
    """
    Create a standard record for a processed file.
    
    Args:
        filename (str): Name of the file
        file_path (str): Full path to the file
        status (str): Processing status ("SUCCESS", "ERROR", etc.)
        num_chunks (int): Number of chunks generated
        error (str, optional): Error message if status is "ERROR"
        
    Returns:
        Dict[str, Any]: Record entry for the processed file
    """
    record = {
        'hash': get_file_hash(file_path),
        'processed_date': datetime.now().isoformat(),
        'num_chunks': num_chunks,
        'status': status
    }
    
    if error:
        record['error'] = error
    
    return record