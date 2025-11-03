/**
 * MKM Research Labs - Loading Indicator Utility
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
 * 
 * Handles display and management of loading indicator
 */
const LoaderUtils = (() => {
  // Private variables
  let chatContainer = null;
  
  /**
   * Get or create the loader element
   * @private
   * @returns {HTMLElement} - Loader element
   */
  const getOrCreateLoader = () => {
    // Get chat container if not already cached
    if (!chatContainer) {
      chatContainer = document.getElementById('chat-container');
      if (!chatContainer) {
        console.error('Chat container not found');
        return null;
      }
    }
    
    // Try to find existing loader
    let loaderElement = document.getElementById('query-loader');
    
    // If loader doesn't exist (removed by clearing chat), create it
    if (!loaderElement) {
      loaderElement = document.createElement('div');
      loaderElement.id = 'query-loader';
      loaderElement.className = 'loader-container';
      loaderElement.innerHTML = `
        <div class="hourglass-loader"></div>
        <div class="loader-text">Processing your query...</div>
      `;
      
      // Append to chat container
      chatContainer.appendChild(loaderElement);
    }
    
    return loaderElement;
  };
  
  /**
   * Show the loading indicator
   * @param {string} message - Optional custom loading message
   */
  const showLoader = (message = 'Processing your query...') => {
    const loaderElement = getOrCreateLoader();
    if (!loaderElement) return;
    
    // Update message if provided
    const textElement = loaderElement.querySelector('.loader-text');
    if (textElement && message) {
      textElement.textContent = message;
    }
    
    // Show loader
    loaderElement.classList.add('active');
    if (chatContainer) {
      chatContainer.classList.add('chat-container-loading');
    }
  };
  
  /**
   * Hide the loading indicator
   */
  const hideLoader = () => {
    const loaderElement = document.getElementById('query-loader');
    
    if (loaderElement) {
      loaderElement.classList.remove('active');
    }
    
    if (chatContainer) {
      chatContainer.classList.remove('chat-container-loading');
    }
  };
  
  /**
   * Check if loader is currently visible
   * @returns {boolean}
   */
  const isLoading = () => {
    const loaderElement = document.getElementById('query-loader');
    if (!loaderElement) return false;
    return loaderElement.classList.contains('active');
  };
  
  /**
   * Initialize the loader (called on page load)
   */
  const initLoader = () => {
    chatContainer = document.getElementById('chat-container');
    if (!chatContainer) {
      console.error('Chat container not found');
      return;
    }
    
    // Create initial loader element
    getOrCreateLoader();
  };
  
  // Public API
  return {
    /**
     * Initialize the loader (called on page load)
     */
    init: initLoader,
    
    /**
     * Show the loading indicator
     * @param {string} message - Optional custom loading message
     */
    show: showLoader,
    
    /**
     * Hide the loading indicator
     */
    hide: hideLoader,
    
    /**
     * Check if loader is currently visible
     * @returns {boolean}
     */
    isActive: isLoading
  };
})();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    LoaderUtils.init();
  });
} else {
  LoaderUtils.init();
}