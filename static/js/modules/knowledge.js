/**
 * MKM Research Labs - Knowledge Base Management
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


 * Handles knowledge base switching and display
 */
const KnowledgeManager = (() => {
    // Private properties
    let activeKnowledgeBase = "misc";
    
    /**
     * Load available knowledge bases from API
     */
    const loadAvailableKnowledgeBases = async () => {
      try {
        // Get knowledge base list container and add loading state
        const kbList = UI.getElement('#knowledge-bases-list');
        kbList.innerHTML = '';
        kbList.appendChild(UI.createLoadingPlaceholder());
        
        // Get knowledge bases from API
        const data = await ApiService.getKnowledgeBases();
        
        // Update active knowledge base
        if (data && !data.error) {
          activeKnowledgeBase = data.active_index;
          const activeDisplayName = data.indices[activeKnowledgeBase]?.display_name || activeKnowledgeBase;
          UI.setActiveKnowledgeBase(activeDisplayName);
        }
        
        // Clear container
        kbList.innerHTML = '';
        
        // Add knowledge base items
        if (!data || !data.indices || Object.keys(data.indices).length === 0) {
          kbList.appendChild(UI.createEmptyState('No knowledge bases available.'));
          return;
        }
        
        Object.entries(data.indices).forEach(([key, info]) => {
          const isActive = key === activeKnowledgeBase;
          
          const kbElement = document.createElement('div');
          kbElement.className = `kb-item ${isActive ? 'kb-item-active' : ''}`;
          
          kbElement.innerHTML = `
            <div class="flex items-center">
              <div class="flex-1">
                <div class="kb-item-name">${info.display_name}</div>
                <div class="kb-item-key">${key}</div>
              </div>
              ${isActive ? '<div class="kb-item-active-badge">✓ Active</div>' : ''}
            </div>
          `;
          
          // Add click event to switch knowledge base
          if (!isActive) {
            kbElement.addEventListener('click', () => {
              switchKnowledgeBase(key);
            });
          }
          
          kbList.appendChild(kbElement);
        });
      } catch (error) {
        console.error('Failed to load knowledge bases:', error);
        
        const kbList = UI.getElement('#knowledge-bases-list');
        kbList.innerHTML = '';
        
        const errorElement = document.createElement('div');
        errorElement.className = 'p-3 bg-red-100 text-red-800 rounded-lg mt-2';
        errorElement.textContent = `Error: ${error.message}`;
        kbList.appendChild(errorElement);
        
        // Fallback to client-side knowledge bases
        provideFallbackKnowledgeBases();
      }
    };
    
    /**
     * Provide fallback knowledge bases if API fails
     */
    const provideFallbackKnowledgeBases = () => {
      const kbList = UI.getElement('#knowledge-bases-list');
      
      // Use default value if activeKnowledgeBase isn't set
      if (!activeKnowledgeBase) {
        activeKnowledgeBase = "misc";
      }
      
      // Update the knowledge base indicator in the header
      UI.setActiveKnowledgeBase(activeKnowledgeBase === "misc" ? "Miscellaneous Knowledge" : "Physical Knowledge");
      
      // Hardcoded fallback for testing
      const fallbackIndices = {
        "misc": {
          "display_name": "Miscellaneous Knowledge",
          "active": activeKnowledgeBase === "misc"
        },
        "phys": {
          "display_name": "Physical Knowledge",
          "active": activeKnowledgeBase === "phys"
        }
      };
      
      // Add fallback knowledge base items
      Object.entries(fallbackIndices).forEach(([key, info]) => {
        const isActive = info.active;
        
        const kbElement = document.createElement('div');
        kbElement.className = `kb-item ${isActive ? 'kb-item-active' : ''}`;
        
        kbElement.innerHTML = `
          <div class="flex items-center">
            <div class="flex-1">
              <div class="kb-item-name">${info.display_name}</div>
              <div class="kb-item-key">${key}</div>
            </div>
            ${isActive ? '<div class="kb-item-active-badge">✓ Active</div>' : ''}
          </div>
        `;
        
        // Add click event to switch knowledge base
        if (!isActive) {
          kbElement.addEventListener('click', () => {
            switchKnowledgeBase(key);
          });
        }
        
        kbList.appendChild(kbElement);
      });
      
      // Add a notice about client-side only mode
      const noticeElement = document.createElement('div');
      noticeElement.className = 'mt-4 p-2 text-xs bg-yellow-100 text-yellow-800 rounded';
      noticeElement.textContent = 'Using client-side mode. Changes will not persist on server.';
      kbList.appendChild(noticeElement);
    };
    
    /**
     * Switch to a different knowledge base
     * @param {string} indexKey - Knowledge base key to switch to
     */
    const switchKnowledgeBase = async (indexKey) => {
      try {
        // Create and display warning modal
        const modalId = 'kb-warning-modal';
        const warningModal = document.createElement('div');
        warningModal.className = 'modal show';
        warningModal.id = modalId;
        
        warningModal.innerHTML = `
          <div class="modal-content" style="max-width: 30rem;">
            <div class="modal-header">
              <h3 class="text-lg font-bold">Switching Knowledge Base</h3>
              <button class="close-btn" data-modal-id="${modalId}">&times;</button>
            </div>
            <div class="modal-body">
              <p class="mb-4">Switching to ${indexKey === 'misc' ? 'Miscellaneous' : 'Physical'} Knowledge Base. This may take a moment as the new index loads.</p>
              <div class="flex justify-center">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              </div>
              <div id="kb-status" class="mt-3 text-sm text-blue-600"></div>
            </div>
          </div>
        `;
        
        document.body.appendChild(warningModal);
        
        // Add close button functionality
        warningModal.querySelector('.close-btn').addEventListener('click', () => {
          warningModal.remove();
        });
        
        // Disable knowledge base items while switching
        const kbItems = UI.getAllElements('.kb-item');
        kbItems.forEach(item => {
          item.classList.add('opacity-50', 'pointer-events-none');
        });
        
        // Status updates
        const kbStatus = warningModal.querySelector('#kb-status');
        const updateStatus = (message) => {
          if (kbStatus) kbStatus.textContent = message;
          console.log(message);
        };
        
        updateStatus("Sending request to server...");
        
        try {
          // Call server API to switch index
          updateStatus("Contacting server...");
          const startTime = Date.now();
          
          const data = await ApiService.switchKnowledgeBase(indexKey);
          
          if (data.error) {
            throw new Error(data.error);
          }
          
          const loadTime = ((Date.now() - startTime) / 1000).toFixed(1);
          updateStatus(`Index loaded in ${loadTime} seconds`);
          
          // Add slight delay to show the success message
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          // Update the active knowledge base
          activeKnowledgeBase = data.active_index;
          
          // Show success notification
          UI.showNotification(data.message);
          
          // Update the KB indicator in the header
          const activeDisplayName = data.indices?.[activeKnowledgeBase]?.display_name || 
            (activeKnowledgeBase === 'misc' ? 'Miscellaneous Knowledge' : 'Physical Knowledge');
          UI.setActiveKnowledgeBase(activeDisplayName);
          
          // Remove the warning modal
          setTimeout(() => {
            warningModal.remove();
          }, 500);
          
          // Reload the knowledge base list
          setTimeout(() => loadAvailableKnowledgeBases(), 500);
        } catch (error) {
          console.error('Server API error:', error);
          updateStatus(`Error: ${error.message}`);
          
          // Delay showing error
          await new Promise(resolve => setTimeout(resolve, 2000));
          
          // FALLBACK: Implement client-side switching logic
          // Update active knowledge base
          activeKnowledgeBase = indexKey;
          
          // Show fallback notification
          UI.showNotification(`Switched to ${indexKey === 'misc' ? 'Miscellaneous' : 'Physical'} Knowledge (client-only mode)`);
          
          // Update the KB indicator
          UI.setActiveKnowledgeBase(indexKey === 'misc' ? 'Miscellaneous Knowledge' : 'Physical Knowledge');
          
          // Remove the warning modal
          setTimeout(() => {
            warningModal.remove();
          }, 500);
          
          // Reload the knowledge base list
          setTimeout(() => loadAvailableKnowledgeBases(), 500);
        }
      } catch (error) {
        console.error('Failed to switch knowledge base:', error);
        UI.showNotification(`Error switching knowledge base: ${error.message}`);
      }
    };
    
    // Public methods
    return {
      /**
       * Initialize knowledge manager
       */
      init: () => {
        // Initial load of knowledge base info
        loadAvailableKnowledgeBases();
      },
      
      /**
       * Load available knowledge bases
       */
      loadAvailableKnowledgeBases,
      
      /**
       * Switch to a different knowledge base
       * @param {string} indexKey - Knowledge base key
       */
      switchKnowledgeBase
    };
  })();