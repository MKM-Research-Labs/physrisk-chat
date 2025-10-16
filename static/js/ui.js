/**
 * MKM Research Labs - UI Utilities
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


 * Common UI management functions
 */
const UI = (() => {
    // Private methods and properties
    const getElement = (selector) => document.querySelector(selector);
    const getAllElements = (selector) => document.querySelectorAll(selector);
    
    // Public methods
    return {
      /**
       * Show notification toast
       * @param {string} message - Message to display
       * @param {number} duration - Duration in ms (default: 3000)
       */
      showNotification: (message, duration = 3000) => {
        const notification = getElement('#notification');
        notification.textContent = message;
        notification.classList.add('show');
        
        setTimeout(() => {
          notification.classList.remove('show');
        }, duration);
      },
      
      /**
       * Toggle sidebar visibility
       */
      toggleSidebar: () => {
        const sidebar = getElement('#sidebar');
        const toggleButton = getElement('#toggle-sidebar');
        
        if (sidebar.classList.contains('w-64')) {
          sidebar.classList.remove('w-64');
          sidebar.classList.add('w-0');
          toggleButton.textContent = '→';
        } else {
          sidebar.classList.remove('w-0');
          sidebar.classList.add('w-64');
          toggleButton.textContent = '←';
        }
      },
      
      /**
       * Switch active tab
       * @param {string} tabId - ID of tab to activate
       */
      switchTab: (tabId) => {
        // Update active tab styling
        getAllElements('.tab-btn').forEach(tab => {
          if (tab.id === tabId) {
            tab.classList.add('active');
            tab.classList.remove('bg-gray-300');
          } else {
            tab.classList.remove('active');
            tab.classList.add('bg-gray-300');
          }
        });
        
        // Show/hide appropriate panels
        const targetPanelId = getElement(`#${tabId}`).getAttribute('data-panel');
        
        getAllElements('.panel-container').forEach(panel => {
          if (panel.id === targetPanelId) {
            panel.classList.remove('hidden');
          } else {
            panel.classList.add('hidden');
          }
        });
      },
      
      /**
       * Create loading placeholder
       * @returns {HTMLElement} - Loading placeholder element
       */
      createLoadingPlaceholder: () => {
        const placeholder = document.createElement('div');
        placeholder.className = 'loading-placeholder';
        return placeholder;
      },
      
      /**
       * Create empty state message
       * @param {string} message - Message to display
       * @returns {HTMLElement} - Empty state element
       */
      createEmptyState: (message) => {
        const emptyState = document.createElement('div');
        emptyState.className = 'text-center p-4 text-gray-500';
        emptyState.textContent = message;
        return emptyState;
      },
      
      /**
       * Set active knowledge base indicator
       * @param {string} name - Knowledge base display name
       */
      setActiveKnowledgeBase: (name) => {
        const indicator = getElement('#active-kb-indicator');
        indicator.textContent = name;
      },
      
      /**
       * Show modal
       * @param {string} modalId - Modal element ID
       */
      showModal: (modalId) => {
        const modal = getElement(`#${modalId}`);
        modal.classList.add('show');
      },
      
      /**
       * Hide modal
       * @param {string} modalId - Modal element ID
       */
      hideModal: (modalId) => {
        const modal = getElement(`#${modalId}`);
        modal.classList.remove('show');
      },
      
      /**
       * Clear container
       * @param {string} containerId - Container element ID
       */
      clearContainer: (containerId) => {
        const container = getElement(`#${containerId}`);
        container.innerHTML = '';
        return container;
      },
      
      /**
       * Get element by ID
       * @param {string} id - Element ID
       * @returns {HTMLElement} - Element
       */
      getElement,
      
      /**
       * Get elements by selector
       * @param {string} selector - CSS selector
       * @returns {NodeList} - Matching elements
       */
      getAllElements
    };
  })();