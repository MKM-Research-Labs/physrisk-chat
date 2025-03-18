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


from flask import Flask, render_template, request, jsonify, url_for
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import requests
import os
import json
import traceback
from typing import List, Dict
from http import HTTPStatus
import anthropic
from rep import replace_terms

# API Keys.  User needs to insert own API keys here

ANTHROPIC_API_KEY='xxx'
PERPLEXITY_API_KEY='yyy'

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def handle_local_model(query: str, context: str, model: str) -> str:
    """Handle requests to local LM Studio models"""
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        # Format the prompt with context and query
        prompt = f"""Context: {context}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""

        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # Make request to local LM Studio server
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",  # LM Studio default address
            headers=headers,
            json=payload
        )
        
        response.raise_for_status()
        response_text = response.json()["choices"][0]["message"]["content"]
        return replace_terms(response_text)

    except requests.exceptions.RequestException as e:
        raise Exception(f"Local model request failed: {str(e)}")


def handle_api_model(query: str, context: str, model: str) -> str:
    """Handle requests to API-based models (Perplexity and Claude)"""
    try:
        if "sonar" in model:
            headers = {
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
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
                response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
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
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
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


# Initialize Flask app and set up paths
app = Flask(__name__, 
            static_folder='static',      
            template_folder='templates'   
)

# Set up chat storage using all_chats.json
CHATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_chats.json")

# Initialize chat storage if it doesn't exist
if not os.path.exists(CHATS_FILE):
    with open(CHATS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"chats": []}, f)

# Initialize embeddings and load FAISS index
current_dir = os.path.dirname(os.path.abspath(__file__))
faiss_path = os.path.join(current_dir, "faiss_index")
print(f"Loading FAISS index from: {faiss_path}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

# Define available models
# Replace existing model list with:
AVAILABLE_MODELS = {
    "mistral-7b-instruct-v0.2": "Mistral 7B Instruct v0.2",
    "Deepseek-Ri1-Sistill-Llama-8B-Q3_k_k": "Deepseek Llama 8B",
    "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M": "Deepseek Qwen 1.5B",
    "sonar": "Perplexity Sonar",
    "sonar-pro": "Perplexity Sonar Pro",
    "sonar-reasoning": "Perplexity Sonar Reasoning",
    "sonar-reasoning-pro": "Perplexity Sonar Reasoning Pro",
    "claude-3.5-sonnet": "Claude 3.5 Sonnet",
    "claude-3-opus": "Claude 3 Opus"
}

from flask import Flask, jsonify
import os
import json


@app.route('/')
def home():
    return render_template('chat
                           .html')

@app.route('/get_summarised_files', methods=['GET'])

def get_summarised_files():
    """API endpoint to retrieve document summaries from the summarised_files.json file"""
    try:
        # Get the path to the summarised_files.json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        summarised_files_path = os.path.join(current_dir, 'summarised_files.json')
        
        # Check if the file exists
        if not os.path.exists(summarised_files_path):
            # Return an empty object if the file doesn't exist
            return jsonify({})
            
        # Read and return the file content
        with open(summarised_files_path, 'r', encoding='utf-8') as f:
            summarised_files = json.load(f)
            
        return jsonify(summarised_files)
    
    except Exception as e:
        print(f"Error retrieving summarised files: {str(e)}")
        return jsonify({'error': f'Error retrieving document summaries: {str(e)}'}), 500


@app.route('/query', methods=['POST'])
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
        if selected_model not in AVAILABLE_MODELS:
            app.logger.error(f"Invalid model received: {selected_model} vs valid {list(AVAILABLE_MODELS.keys())}")
            return jsonify({
                'error': f'Invalid model selection. Valid models are: {list(AVAILABLE_MODELS.keys())}'
            }), HTTPStatus.BAD_REQUEST

        try:
            # Get context from FAISS
            docs = vector_store.similarity_search(user_query, 10)
            if not docs:
                return jsonify({
                    'error': 'No relevant documents found'
                }), HTTPStatus.NOT_FOUND
                
            context = "\n\n".join([
                f"From {doc.metadata.get('source', 'Unknown source')} (Page {doc.metadata.get('page', doc.metadata.get('page_number', 'Unknown page'))}): {doc.page_content}" 
                for doc in docs
            ])
            
            for i, doc in enumerate(docs):
                app.logger.debug(f"Doc {i} metadata: {doc.metadata}")

            # Route to appropriate handler based on model type
            if selected_model in ["mistral-7b-instruct-v0.2", "Deepseek-Ri1-Sistill-Llama-8B-Q3_k_k", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"]:
                response = handle_local_model(user_query, context, selected_model)
            else:
                response = handle_api_model(user_query, context, selected_model)

            
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
                    app.logger.error(f"Error processing source metadata: {str(e)}")
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
            app.logger.error(f"Error in query processing: {str(e)}")
            app.logger.error(f"Stack trace: {traceback.format_exc()}")
            return jsonify({
                'error': f'Error processing query: {str(e)}'
            }), HTTPStatus.INTERNAL_SERVER_ERROR
        
    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        app.logger.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), HTTPStatus.INTERNAL_SERVER_ERROR

@app.route('/save_chat', methods=['POST'])
def save_chat():
    try:
        chat_data = request.json.get('chat')
        if not chat_data:
            return jsonify({'error': 'No chat data provided'}), 400

        chat_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with open(CHATS_FILE, 'r', encoding='utf-8') as f:
                all_chats = json.load(f)
                if not isinstance(all_chats, dict) or 'chats' not in all_chats:
                    all_chats = {"chats": []}
        except (FileNotFoundError, json.JSONDecodeError):
            all_chats = {"chats": []}
        
        chat_data['id'] = chat_id
        all_chats['chats'].append(chat_data)
        
        with open(CHATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_chats, f, ensure_ascii=False, indent=2)

        return jsonify({
            'message': 'Chat saved successfully',
            'chat_id': chat_id
        })

    except Exception as e:
        print(f"Error saving chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_chats', methods=['GET'])
def get_chats():
    try:
        try:
            with open(CHATS_FILE, 'r', encoding='utf-8') as f:
                all_chats = json.load(f)
                if not isinstance(all_chats, dict) or 'chats' not in all_chats:
                    all_chats = {"chats": []}
        except (FileNotFoundError, json.JSONDecodeError):
            all_chats = {"chats": []}
            with open(CHATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_chats, f, ensure_ascii=False, indent=2)
        
        return jsonify(all_chats)
    except Exception as e:
        print(f"Error getting chats: {str(e)}")
        # Return empty chat structure instead of error
        return jsonify({"chats": []})

if __name__ == '__main__':
    print("Starting server...")
    print(f"FAISS index location: {faiss_path}")
    print("Template directory:", os.path.join(os.path.dirname(__file__), 'templates'))
    print("Static directory:", os.path.join(os.path.dirname(__file__), 'static'))
    print(f"Chat history file: {CHATS_FILE}")
    print("Available models:", list(AVAILABLE_MODELS.keys()))
    app.run(debug=True, port=5000)