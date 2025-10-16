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

# src/utils/diagnostics.py
"""
Diagnostic tools for document processing, particularly for problematic PDFs.
"""
import os
import json
from datetime import datetime
import pdfplumber
import PyPDF2

def diagnose_pdf(file_path):
    """
    Diagnostic function to analyze problematic PDF files and identify issues.
    
    Args:
        file_path: Path to the PDF file to diagnose
        
    Returns:
        dict: Diagnostic information about the PDF
    """
    
    diagnostics = {
        "file_name": os.path.basename(file_path),
        "file_path": file_path,
        "file_size_bytes": os.path.getsize(file_path),
        "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
        "issues_detected": [],
        "binary_content_detected": False,
        "corrupt_structure_detected": False,
        "password_protected": False,
        "parsing_attempts": []
    }
    
    # Check if file exists and is readable
    if not os.path.exists(file_path):
        diagnostics["issues_detected"].append("File not found")
        return diagnostics
    
    # Try reading the raw bytes to check for obvious binary content
    try:
        with open(file_path, 'rb') as f:
            first_bytes = f.read(1024)  # Read first 1KB
            
            # Check for PDF header
            if not first_bytes.startswith(b'%PDF'):
                diagnostics["issues_detected"].append("Not a standard PDF file (missing %PDF header)")
            
            # Check for binary content issues
            if b'\x00' in first_bytes:
                diagnostics["binary_content_detected"] = True
                diagnostics["issues_detected"].append("Contains null bytes which may cause parsing issues")
            
            # Look for specific problematic byte sequences
            if b'\x86H' in first_bytes:
                diagnostics["issues_detected"].append("Contains problematic byte sequence '\\x86H'")
    except Exception as e:
        diagnostics["issues_detected"].append(f"Error reading file: {str(e)}")
    
    # Try parsing with PyPDF2
    try:
        attempt_info = {"parser": "PyPDF2", "success": False, "error": None, "pages": 0}
        
        with open(file_path, 'rb') as f:
            try:
                pdf = PyPDF2.PdfReader(f, strict=False)
                attempt_info["success"] = True
                attempt_info["pages"] = len(pdf.pages)
                
                # Check if encrypted
                if pdf.is_encrypted:
                    diagnostics["password_protected"] = True
                    diagnostics["issues_detected"].append("PDF is password protected")
                
                # Try accessing a few pages to check structure
                for i in range(min(3, len(pdf.pages))):
                    try:
                        page = pdf.pages[i]
                        _ = page.extract_text()
                    except Exception as page_error:
                        attempt_info["error"] = f"Error extracting text from page {i+1}: {str(page_error)}"
                        diagnostics["corrupt_structure_detected"] = True
                        diagnostics["issues_detected"].append(f"Page {i+1} structure issue: {str(page_error)}")
                        break
                
            except Exception as e:
                attempt_info["error"] = str(e)
                if "Password required" in str(e):
                    diagnostics["password_protected"] = True
                    diagnostics["issues_detected"].append("PDF is password protected")
                else:
                    diagnostics["corrupt_structure_detected"] = True
                    diagnostics["issues_detected"].append(f"PDF structure issue: {str(e)}")
        
        diagnostics["parsing_attempts"].append(attempt_info)
    except ImportError:
        diagnostics["parsing_attempts"].append({
            "parser": "PyPDF2", 
            "success": False, 
            "error": "PyPDF2 library not available"
        })
    
    # Additional diagnostics code omitted for brevity...
    
    # Generate recommendation
    diagnostics["overall_assessment"] = evaluate_diagnostics(diagnostics)
    diagnostics["recommended_approach"] = generate_recommendation(diagnostics)
    
    return diagnostics

def evaluate_diagnostics(diagnostics):
    """Evaluate diagnostic results and provide an overall assessment"""
    successful_attempts = sum(1 for attempt in diagnostics["parsing_attempts"] if attempt["success"])
    
    if successful_attempts == 0:
        return "Critical: All parsing methods failed"
    elif diagnostics["corrupt_structure_detected"]:
        return "Problematic: PDF has corrupt structure but might be partially recoverable"
    elif diagnostics["binary_content_detected"]:
        return "Caution: Binary content detected that might cause parsing issues"
    elif diagnostics["password_protected"]:
        return "Protected: PDF requires password to access"
    else:
        return "Likely parseable with proper handling"

def generate_recommendation(diagnostics):
    """Generate recommendations based on diagnostics"""
    if diagnostics["password_protected"]:
        return "Need password to process"
    
    successful_attempts = [a for a in diagnostics["parsing_attempts"] if a["success"]]
    if successful_attempts:
        best_attempt = max(successful_attempts, key=lambda x: x.get("pages", 0))
        return f"Use {best_attempt['parser']} for best results"
    elif diagnostics["binary_content_detected"] or diagnostics["corrupt_structure_detected"]:
        return "Try OCR-based extraction using pdf2image and Tesseract"
    else:
        return "Try alternative PDF libraries or tools"