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
"""
MKM Research Document Q&A Application (v6)
------------------------------------------

Description:
    This application provides a document query interface using Retrieval Augmented Generation (RAG).
    Users can interact with locally hosted LLMs through LM Studio or cloud-based APIs (Perplexity, Anthropic).
    The system retrieves relevant document chunks using FAISS vector indexing and generates responses
    based on the provided context.

Key Features:
    - Multiple LLM options (local and cloud-based)
    - FAISS vector search for document retrieval
    - Chat history management
    - Document summary access
    - Web-based UI
    - Multiple knowledge bases via different FAISS indices

LM Studio Configuration:
    When using local models through LM Studio, proper prompt template configuration
    is required, especially for Mistral 7B. To configure:
    
    1. Open LM Studio and select your Mistral 7B model
    2. Navigate to model settings (gear icon or menu)
    3. Find the "Prompt Template" section
    4. Replace the default template with a simplified version like:
       ```
       {% for message in messages %}
       {% if message.role == 'user' %}
       User: {{ message.content }}
       {% elif message.role == 'assistant' %}
       Assistant: {{ message.content }}
       {% endif %}
       {% endfor %}
       ```
    5. Save changes and restart the server if necessary
    
    This ensures compatibility with the application's message format and prevents
    "Error rendering prompt with jinja template" errors.
"""

from flask import Flask, render_template, request, jsonify, url_for
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import requests
import subprocess
import sys
import os
import json
import traceback
import argparse
from typing import List, Dict
from http import HTTPStatus
import anthropic
from src.rep import replace_terms
import re
from src.config import CONFIG, paths, get_collection_config


class DocumentQAApp:
    """Main application class for the Document Q&A system"""
    
    def __init__(self):
        # API Keys - user needs to define own API Keys
        self.ANTHROPIC_API_KEY = ''
        self.PERPLEXITY_API_KEY = ''
        
        # API URLs
        self.PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
        self.ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        static_folder=str(paths.static_dir),      
                        template_folder=str(paths.templates_dir))
        
        # Set up chat storage
        self.CHATS_FILE = str(paths.chats_file)
        self._init_chat_storage()
        
        # Configure available FAISS indices
        self.AVAILABLE_INDICES = {
            collection_type: {
                "path": config["faiss_index"],
                "display_name": config["name"]
            }
            for collection_type, config in CONFIG['collections'].items()
        }
        
        # Store the active index key globally
        self.ACTIVE_INDEX_KEY = "misc"
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create a dictionary to store loaded vector stores
        self.vector_stores = {}
        
        # Define available models
        self.AVAILABLE_MODELS = {
            "mistral-7b-instruct-v0.2": "Mistral 7B Instruct v0.2",
            "cogito-v1-preview-llama-3b": "Cogito v1 3B",
            "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M": "Deepseek Qwen 1.5B",
            "sonar": "Perplexity Sonar",
            "sonar-pro": "Perplexity Sonar Pro",
            "sonar-reasoning": "Perplexity Sonar Reasoning",
            "sonar-reasoning-pro": "Perplexity Sonar Reasoning Pro",
            "claude-3.5-sonnet": "Claude 3.5 Sonnet",
            "claude-3-opus": "Claude 3 Opus"
        }
        
        # Load the default index on startup
        self._load_default_index()
        
        # Setup routes
        self._setup_routes()
    
    def _init_chat_storage(self):
        """Initialize chat storage if it doesn't exist"""
        if not os.path.exists(self.CHATS_FILE):
            with open(self.CHATS_FILE, 'w', encoding='utf-8') as f:
                json.dump({"chats": []}, f)
    
    def _load_default_index(self):
        """Load the default index on startup"""
        try:
            self.vector_stores["misc"] = self._load_faiss_index("misc")
        except Exception as e:
            print(f"Warning: Failed to load default index: {str(e)}")
    
    def _load_faiss_index(self, index_key):
        """Load a FAISS index by key"""
        if index_key not in self.AVAILABLE_INDICES:
            raise ValueError(f"Invalid index key: {index_key}")
        
        index_path = self.AVAILABLE_INDICES[index_key]["path"]
        print(f"Loading FAISS index from: {index_path}")
        
        if index_key not in self.vector_stores:
            try:
                self.vector_stores[index_key] = FAISS.load_local(
                    index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Successfully loaded index: {index_key}")
            except Exception as e:
                print(f"Error loading index {index_key}: {str(e)}")
                print(traceback.format_exc())
                raise
        
        return self.vector_stores[index_key]
    
    def _handle_local_model(self, query: str, context: str, model: str) -> str:
        """Handle requests to local LM Studio models with improved error handling"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Simplify the prompt structure
            system_message = "You are a helpful assistant that answers questions based on the provided context."
            
            # Split context into chunks if it's very large
            max_context_length = 4000  # Adjust this based on your model's limitations
            if len(context) > max_context_length:
                context = context[:max_context_length] + "... [context truncated for length]"
            
            # Use the OpenAI-compatible messages format
            payload = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # Log the request for debugging (but truncate for readability)
            truncated_payload = payload.copy()
            if len(context) > 100:
                truncated_context = context[:100] + "... [truncated for log]"
                truncated_payload["messages"][1]["content"] = f"Context: {truncated_context}\n\nQuestion: {query}"
            print(f"Sending request to LM Studio: {json.dumps(truncated_payload, indent=2)}")
            
            # Make request to local LM Studio server
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=300  # Add timeout to prevent hanging
            )
            
            # Better error handling
            if response.status_code != 200:
                print(f"Error response from LM Studio: Status {response.status_code}")
                print(f"Response content: {response.text}")
                return f"Error from local model: {response.status_code} - {response.text}"
            
            response_json = response.json()
            response_text = response_json["choices"][0]["message"]["content"]
            return replace_terms(response_text)

        except requests.exceptions.RequestException as e:
            print(f"Request exception: {str(e)}")
            return f"Local model request failed. Make sure LM Studio is running on port 1234. Error: {str(e)}"
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            return "Received invalid response from local model"
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            print(traceback.format_exc())
            return f"Unexpected error with local model: {str(e)}"

    def _handle_api_model(self, query: str, context: str, model: str) -> str:
        """Handle requests to API-based models (Perplexity and Claude)"""
        try:
            if "sonar" in model:
                headers = {
                    "Authorization": f"Bearer {self.PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": f"Use this context to answer: {context}"},
                        {"role": "user", "content": query}
                    ]
                }

                
                try:
                    response = requests.post(self.PERPLEXITY_API_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    response_text = response.json()["choices"][0]["message"]["content"]
                    return replace_terms(response_text)
                except requests.exceptions.HTTPError as http_err:
                    print(f"HTTP error occurred: {http_err}")
                    print(f"Response content: {response.text}")
                    raise Exception(f"Perplexity API request failed: {str(http_err)}")
                except requests.exceptions.RequestException as req_err:
                    raise Exception(f"Perplexity API request failed: {str(req_err)}")

                
            elif "claude" in model:
                client = anthropic.Anthropic(api_key=self.ANTHROPIC_API_KEY)
                try:
                    message = client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=2048,
                        temperature=0.3,
                        system="",
                        messages=[
                            {
                                "role": "user",
                                "content": f"Context: {context}\n\nQuestion: {query}"
                            }
                        ]
                    )
                    print("Response from Claude:", message)  # Debug print
                    response_text = message.content[0].text
                    return replace_terms(response_text)
                except Exception as e:
                    print(f"Claude API Error: {str(e)}")
                    raise

            else:
                raise ValueError(f"Unsupported model: {model}")

                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def _setup_routes(self):
        """Setup all Flask routes"""
        
        @self.app.route('/')
        def home():
            return render_template('chat.html')

        @self.app.route('/get_summarised_files', methods=['GET'])
        def get_summarised_files():
            """API endpoint to retrieve document summaries from the summarised_files.json file"""
            try:
                # Use the active index to determine which summary file to use
                collection_config = get_collection_config(self.ACTIVE_INDEX_KEY)
                summary_file_path = collection_config['summary_file']
                
                # Check if the file exists
                if not os.path.exists(summary_file_path):
                    return jsonify({})
                    
                # Read and return the file content
                with open(summary_file_path, 'r', encoding='utf-8') as f:
                    summarised_files = json.load(f)
                    
                return jsonify(summarised_files)
            
            except Exception as e:
                print(f"Error retrieving summarised files: {str(e)}")
                return jsonify({'error': f'Error retrieving document summaries: {str(e)}'}), 500
            
        @self.app.route('/get_available_indices', methods=['GET'])
        def get_available_indices():
            """API endpoint to retrieve the list of available FAISS indices"""
            try:
                indices_info = {
                    key: {
                        "display_name": info["display_name"],
                        "active": key == self.ACTIVE_INDEX_KEY
                    }
                    for key, info in self.AVAILABLE_INDICES.items()
                }
                
                return jsonify({
                    "indices": indices_info,
                    "active_index": self.ACTIVE_INDEX_KEY
                })
            except Exception as e:
                print(f"Error retrieving indices: {str(e)}")
                print(traceback.format_exc())
                return jsonify({'error': f'Error retrieving indices: {str(e)}'}), 500   

        @self.app.route('/switch_index', methods=['POST'])
        def switch_index():
            """API endpoint to switch the active FAISS index"""
            try:
                # Validate request data
                if not request.is_json:
                    return jsonify({'error': 'Request must be JSON'}), 400
                    
                request_data = request.get_json()
                index_key = request_data.get('index_key')
                
                # Validate index key
                if not index_key or index_key not in self.AVAILABLE_INDICES:
                    return jsonify({
                        'error': f'Invalid index key. Valid options are: {list(self.AVAILABLE_INDICES.keys())}'
                    }), 400
                
                # Log that we're switching indexes
                print(f"Switching active index from {self.ACTIVE_INDEX_KEY} to {index_key}")
                
                # Update the active index
                self.ACTIVE_INDEX_KEY = index_key
                
                # Force reload the index even if it was previously loaded
                # This ensures any changes to the index are picked up
                try:
                    print(f"Forcing reload of index {index_key}")
                    # Remove the index from loaded vector_stores if it exists
                    if index_key in self.vector_stores:
                        del self.vector_stores[index_key]
                        
                    # Load the index fresh
                    self._load_faiss_index(index_key)
                    print(f"Successfully reloaded index: {index_key}")
                except Exception as e:
                    print(f"Error loading index {index_key}: {str(e)}")
                    print(traceback.format_exc())
                    return jsonify({
                        'error': f'Failed to load index {index_key}: {str(e)}'
                    }), 500
                
                return jsonify({
                    'message': f'Switched to index: {self.AVAILABLE_INDICES[index_key]["display_name"]}',
                    'active_index': index_key,
                    'indices': {
                        k: {
                            "display_name": v["display_name"],
                            "active": k == index_key
                        } for k, v in self.AVAILABLE_INDICES.items()
                    }
                })
            except Exception as e:
                print(f"Error switching index: {str(e)}")
                print(traceback.format_exc())
                return jsonify({'error': f'Error switching index: {str(e)}'}), 500

        @self.app.route('/query', methods=['POST'])
        def query():
            try:
                # Validate request data
                if not request.is_json:
                    return jsonify({'error': 'Request must be JSON'}), HTTPStatus.BAD_REQUEST
                    
                request_data = request.get_json()
                user_query = request_data.get('query')
                selected_model = request_data.get('model')
                
                # Validate required fields
                if not user_query or not selected_model:
                    return jsonify({
                        'error': 'Missing required fields: query and model'
                    }), HTTPStatus.BAD_REQUEST
                    
                # Use AVAILABLE_MODELS dictionary for validation
                if selected_model not in self.AVAILABLE_MODELS:
                    self.app.logger.error(f"Invalid model received: {selected_model} vs valid {list(self.AVAILABLE_MODELS.keys())}")
                    return jsonify({
                        'error': f'Invalid model selection. Valid models are: {list(self.AVAILABLE_MODELS.keys())}'
                    }), HTTPStatus.BAD_REQUEST

                try:
                    # Get the active FAISS index
                    active_vector_store = self.vector_stores.get(self.ACTIVE_INDEX_KEY)
                    
                    # If index not loaded, try to load it
                    if active_vector_store is None:
                        try:
                            active_vector_store = self._load_faiss_index(self.ACTIVE_INDEX_KEY)
                        except Exception as index_error:
                            self.app.logger.error(f"Failed to load index {self.ACTIVE_INDEX_KEY}: {str(index_error)}")
                            return jsonify({
                                'error': f'Knowledge base error: Failed to load index {self.ACTIVE_INDEX_KEY}'
                            }), HTTPStatus.INTERNAL_SERVER_ERROR
                    
                    # Get context from FAISS
                    docs = active_vector_store.similarity_search(user_query, 10)
                    if not docs:
                        return jsonify({
                            'error': 'No relevant documents found'
                        }), HTTPStatus.NOT_FOUND
                        
                    context = "\n\n".join([
                        f"From {doc.metadata.get('source', 'Unknown source')} (Page {doc.metadata.get('page', doc.metadata.get('page_number', 'Unknown page'))}): {doc.page_content}" 
                        for doc in docs
                    ])
                    
                    for i, doc in enumerate(docs):
                        self.app.logger.debug(f"Doc {i} metadata: {doc.metadata}")

                    # Route to appropriate handler based on model type
                    if selected_model in ["mistral-7b-instruct-v0.2", "cogito-v1-preview-llama-3b", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"]:
                        response = self._handle_local_model(user_query, context, selected_model)
                        
                        # Check if the response contains an error message
                        if response.startswith("Error") or "failed" in response.lower():
                            # Return error without sources
                            return jsonify({
                                'error': response
                            }), HTTPStatus.INTERNAL_SERVER_ERROR
                    else:
                        response = self._handle_api_model(user_query, context, selected_model)

                    
                    # Clean up source metadata for the response
                    cleaned_sources = []
                    for doc in docs:
                        try:
                            source_entry = {
                                "file": doc.metadata.get("source", "Unknown source"),
                                "page": str(doc.metadata.get("page", 
                                        doc.metadata.get("page_number", 
                                        doc.metadata.get("item_id", "1"))))
                            }
                            cleaned_sources.append(source_entry)
                        except Exception as e:
                            self.app.logger.error(f"Error processing source metadata: {str(e)}")
                            # Add a fallback source entry
                            cleaned_sources.append({
                                "file": "Unknown source",
                                "page": "1"
                            })
                    
                    return jsonify({
                        'response': response,
                        'sources': cleaned_sources
                    }), HTTPStatus.OK

                except Exception as e:
                    self.app.logger.error(f"Error in query processing: {str(e)}")
                    self.app.logger.error(f"Stack trace: {traceback.format_exc()}")
                    return jsonify({
                        'error': f'Error processing query: {str(e)}'
                    }), HTTPStatus.INTERNAL_SERVER_ERROR
                
            except Exception as e:
                self.app.logger.error(f"Server error: {str(e)}")
                self.app.logger.error(f"Stack trace: {traceback.format_exc()}")
                return jsonify({
                    'error': f'Server error: {str(e)}'
                }), HTTPStatus.INTERNAL_SERVER_ERROR

        @self.app.route('/save_chat', methods=['POST'])
        def save_chat():
            try:
                chat_data = request.json.get('chat')
                if not chat_data:
                    return jsonify({'error': 'No chat data provided'}), 400

                chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                try:
                    with open(self.CHATS_FILE, 'r', encoding='utf-8') as f:
                        all_chats = json.load(f)
                        if not isinstance(all_chats, dict) or 'chats' not in all_chats:
                            all_chats = {"chats": []}
                except (FileNotFoundError, json.JSONDecodeError):
                    all_chats = {"chats": []}
                
                chat_data['id'] = chat_id
                all_chats['chats'].append(chat_data)
                
                with open(self.CHATS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_chats, f, ensure_ascii=False, indent=2)

                return jsonify({
                    'message': 'Chat saved successfully',
                    'chat_id': chat_id
                })

            except Exception as e:
                print(f"Error saving chat: {str(e)}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/get_chats', methods=['GET'])
        def get_chats():
            try:
                try:
                    with open(self.CHATS_FILE, 'r', encoding='utf-8') as f:
                        all_chats = json.load(f)
                        if not isinstance(all_chats, dict) or 'chats' not in all_chats:
                            all_chats = {"chats": []}
                except (FileNotFoundError, json.JSONDecodeError):
                    all_chats = {"chats": []}
                    with open(self.CHATS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(all_chats, f, ensure_ascii=False, indent=2)
                
                return jsonify(all_chats)
            except Exception as e:
                print(f"Error getting chats: {str(e)}")
                # Return empty chat structure instead of error
                return jsonify({"chats": []})

        @self.app.route('/analyze_sources', methods=['POST'])
        def analyze_sources():
            """API endpoint to analyze query sources for visualization"""
            try:
                # Validate request data
                if not request.is_json:
                    return jsonify({'error': 'Request must be JSON'}), 400
                    
                request_data = request.get_json()
                sources = request_data.get('sources')
                query = request_data.get('query')
                
                if not sources or not isinstance(sources, list):
                    return jsonify({'error': 'Invalid sources data'}), 400
                
                # Group sources by file name
                source_counts = {}
                for source in sources:
                    file_name = source.get('file', '').replace('/Users/newdavid/Documents/MKMChat/docs/', '')
                    if not file_name:
                        continue
                        
                    if file_name not in source_counts:
                        source_counts[file_name] = {
                            'count': 0,
                            'pages': set()
                        }
                        
                    source_counts[file_name]['count'] += 1
                    source_counts[file_name]['pages'].add(source.get('page', '1'))
                
                # Extract year from filename if possible
                def extract_year(filename):
                    year_match = re.search(r'(19|20)\d{2}', filename)
                    return int(year_match.group(0)) if year_match else None
                
                # Date grouping
                date_groups = {
                    'Recent (2023-2025)': 0,
                    'Newer (2020-2022)': 0,
                    'Recent (2017-2019)': 0,
                    'Older (2014-2016)': 0,
                    'Archive (Before 2014)': 0
                }
                
                # Process each source for date categorization
                for file_name in source_counts:
                    year = extract_year(file_name)
                    if year:
                        if year >= 2023:
                            date_groups['Recent (2023-2025)'] += source_counts[file_name]['count']
                        elif year >= 2020:
                            date_groups['Newer (2020-2022)'] += source_counts[file_name]['count']
                        elif year >= 2017:
                            date_groups['Recent (2017-2019)'] += source_counts[file_name]['count']
                        elif year >= 2014:
                            date_groups['Older (2014-2016)'] += source_counts[file_name]['count']
                        else:
                            date_groups['Archive (Before 2014)'] += source_counts[file_name]['count']
                
                # Format response
                top_sources = []
                for file_name, data in source_counts.items():
                    top_sources.append({
                        'name': file_name[:20] + ('...' if len(file_name) > 20 else ''),
                        'count': data['count'],
                        'pages': len(data['pages'])
                    })
                
                # Sort by count, descending
                top_sources.sort(key=lambda x: x['count'], reverse=True)
                top_sources = top_sources[:5]  # Take top 5
                
                # Format date distribution
                date_distribution = []
                for date_range, count in date_groups.items():
                    if count > 0:
                        date_distribution.append({
                            'name': date_range,
                            'count': count
                        })
                
                # Generate estimated relevance distribution (in a real implementation,
                # this would use embeddings to calculate actual relevance)
                total_sources = len(sources)
                relevance_distribution = [
                    {'name': 'Very High', 'value': int(total_sources * 0.3) or 1},
                    {'name': 'High', 'value': int(total_sources * 0.4) or 1},
                    {'name': 'Medium', 'value': int(total_sources * 0.2) or 1},
                    {'name': 'Low', 'value': int(total_sources * 0.1) or 1}
                ]
                
                # Extract keywords from filenames
                terms = {}
                for filename in source_counts:
                    words = re.sub(r'\.\w+$', '', filename)  # Remove file extension
                    words = re.sub(r'[^a-zA-Z0-9 ]', ' ', words)  # Remove special chars
                    words = [w for w in words.split() if len(w) > 3]  # Split and filter short words
                    
                    for word in words:
                        if word not in terms:
                            terms[word] = 0
                        terms[word] += source_counts[filename]['count']
                
                # Get top keywords
                top_keywords = []
                for text, value in sorted(terms.items(), key=lambda x: x[1], reverse=True)[:10]:
                    top_keywords.append({'text': text, 'value': value})
                
                return jsonify({
                    'totalResults': len(sources),
                    'topSources': top_sources,
                    'relevanceDistribution': relevance_distribution,
                    'dateDistribution': date_distribution,
                    'topKeywords': top_keywords,
                    'query': query
                })
                
            except Exception as e:
                print(f"Error analyzing sources: {str(e)}")
                print(traceback.format_exc())
                return jsonify({'error': f'Error analyzing sources: {str(e)}'}), 500
    
    def run(self, debug=True, port=5000, host='127.0.0.1'):
        """Run the Flask application"""
        print("Starting server...")
        
        # Print available indices
        print("Available FAISS indices:")
        for key, info in self.AVAILABLE_INDICES.items():
            print(f"  - {key}: {info['display_name']} ({info['path']})")
            
        print(f"Default active index: {self.ACTIVE_INDEX_KEY}")
        print("Template directory:", os.path.join(os.path.dirname(__file__), 'templates'))
        print("Static directory:", os.path.join(os.path.dirname(__file__), 'static'))
        print(f"Chat history file: {self.CHATS_FILE}")
        print("Available models:", list(self.AVAILABLE_MODELS.keys()))
        
        self.app.run(debug=debug, port=port, host=host)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MKM Research Document Q&A Application')
    return parser.parse_args()


# For backward compatibility when running directly
if __name__ == '__main__':
    args = parse_arguments()
    app = DocumentQAApp()
    app.run(debug=True, port=5000)