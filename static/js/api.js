/**
 * MKM Research Labs - API Service
 * 
 * Copyright (c) 2025 MKM Research Labs. All rights reserved.
 * 
 * This software is provided under license by MKM Research Labs. 
 * Use, reproduction, distribution, or modification of this code is subject to the 
 * terms and conditions of the license agreement provided with this software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.


 * Handles all API communication with the backend
 */
const ApiService = (() => {
    // Private properties
    const apiEndpoints = {
      query: '/query',
      saveChat: '/save_chat',
      getChats: '/get_chats',
      getSummarizedFiles: '/get_summarised_files',
      getAvailableIndices: '/get_available_indices',
      switchIndex: '/switch_index'
    };
  
    /**
     * Make API request
     * @param {string} url - API endpoint
     * @param {string} method - HTTP method
     * @param {Object} data - Request payload (for POST)
     * @returns {Promise} - Promise with response data
     */
    const makeRequest = async (url, method = 'GET', data = null) => {
      try {
        const options = {
          method,
          headers: {
            'Content-Type': 'application/json'
          }
        };
  
        if (data && method !== 'GET') {
          options.body = JSON.stringify(data);
        }
  
        const response = await fetch(url, options);
        
        // Parse JSON response
        const responseData = await response.json();
        
        // Check for API error
        if (!response.ok) {
          throw new Error(responseData.error || `Server returned ${response.status}`);
        }
        
        return responseData;
      } catch (error) {
        console.error(`API Error (${url}):`, error);
        throw error;
      }
    };
  
    // Public methods
    return {
      /**
       * Send query to the LLM
       * @param {string} query - User query text
       * @param {string} model - Selected model name
       * @returns {Promise} - Promise with response and sources
       */
      sendQuery: (query, model) => {
        return makeRequest(apiEndpoints.query, 'POST', { query, model });
      },
  
      /**
       * Save chat history
       * @param {Array} messages - Chat messages
       * @returns {Promise} - Promise with save status
       */
      saveChat: (messages) => {
        return makeRequest(apiEndpoints.saveChat, 'POST', {
          chat: {
            timestamp: new Date().toISOString(),
            messages
          }
        });
      },
  
      /**
       * Get chat history
       * @returns {Promise} - Promise with chat history
       */
      getChats: () => {
        return makeRequest(apiEndpoints.getChats);
      },
  
      /**
       * Get document summaries
       * @returns {Promise} - Promise with document summaries
       */
      getDocumentSummaries: () => {
        return makeRequest(apiEndpoints.getSummarizedFiles);
      },
  
      /**
       * Get available knowledge bases
       * @returns {Promise} - Promise with knowledge base info
       */
      getKnowledgeBases: () => {
        return makeRequest(apiEndpoints.getAvailableIndices);
      },
  
      /**
       * Switch active knowledge base
       * @param {string} indexKey - Knowledge base key to switch to
       * @returns {Promise} - Promise with switch status
       */
      switchKnowledgeBase: (indexKey) => {
        return makeRequest(apiEndpoints.switchIndex, 'POST', { index_key: indexKey });
      }
    };
  })();