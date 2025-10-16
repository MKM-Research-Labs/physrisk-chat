/**
 * MKM Research Labs - Document Management
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


 * Handles document summaries and interactions
 */
const DocumentManager = (() => {
    // Private properties
    let documentSummaries = {};
    
    /**
     * Load document summaries from API
     */
    const loadDocumentSummaries = async () => {
      try {
        // Get document list container and add loading state
        const documentList = UI.getElement('#document-list');
        documentList.innerHTML = '';
        documentList.appendChild(UI.createLoadingPlaceholder());
        
        // Get document summaries from API
        const data = await ApiService.getDocumentSummaries();
        
        // Save document summaries
        documentSummaries = data || {};
        
        // Clear container
        documentList.innerHTML = '';
        
        // Add document items
        const documentNames = Object.keys(documentSummaries).sort();
        
        if (documentNames.length === 0) {
          documentList.appendChild(UI.createEmptyState('No document summaries available.'));
          return;
        }
        
        documentNames.forEach(docName => {
          const docInfo = documentSummaries[docName];
          const docElement = document.createElement('div');
          docElement.className = 'document-item';
          docElement.setAttribute('data-name', docName);
          
          // Check summary type for icon
          const summaryType = docInfo.summary_type || 'FULL';
          const iconClass = summaryType === 'FULL' ? 'text-green-500' : 'text-yellow-500';
          const iconSymbol = summaryType === 'FULL' ? '✓' : '⚠';
          
          docElement.innerHTML = `
            <div class="flex items-center">
              <span class="${iconClass} mr-2">${iconSymbol}</span>
              <div class="flex-1">
                <div class="document-item-title">${docName}</div>
                <div class="document-item-date">
                  ${new Date(docInfo.summarised_date).toLocaleDateString()}
                </div>
              </div>
            </div>
          `;
          
          // Add click event to show document summary
          docElement.addEventListener('click', () => {
            showDocumentSummary(docName);
          });
          
          documentList.appendChild(docElement);
        });
        
        // Apply search filter if exists
        const searchTerm = UI.getElement('#doc-search').value;
        if (searchTerm) {
          filterDocuments();
        }
      } catch (error) {
        console.error('Failed to load document summaries:', error);
        UI.showNotification('Failed to load document summaries');
        
        const documentList = UI.getElement('#document-list');
        documentList.innerHTML = '';
        
        const errorElement = document.createElement('div');
        errorElement.className = 'p-3 bg-red-100 text-red-800 rounded-lg mt-2';
        errorElement.textContent = `Error: ${error.message}`;
        documentList.appendChild(errorElement);
      }
    };
    
    /**
     * Show document summary in modal
     * @param {string} docName - Document name
     */
    const showDocumentSummary = (docName) => {
      const docInfo = documentSummaries[docName];
      
      if (!docInfo || !docInfo.summary) {
        UI.showNotification('No summary available for this document');
        return;
      }
      
      ModalManager.showDocumentSummary(docName, docInfo.summary);
    };
    
    /**
     * Filter documents by search term
     */
    const filterDocuments = () => {
      const searchTerm = UI.getElement('#doc-search').value.toLowerCase();
      const documentElements = UI.getAllElements('#document-list .document-item');
      
      documentElements.forEach(element => {
        const docName = element.getAttribute('data-name').toLowerCase();
        if (docName.includes(searchTerm)) {
          element.classList.remove('hidden');
        } else {
          element.classList.add('hidden');
        }
      });
    };
    
    // Public methods
    return {
      /**
       * Initialize document manager
       */
      init: () => {
        // Add document search functionality
        UI.getElement('#doc-search').addEventListener('input', filterDocuments);
      },
      
      /**
       * Load document summaries
       */
      loadDocumentSummaries,
      
      /**
       * Show document summary
       * @param {string} docName - Document name
       */
      showDocumentSummary
    };
  })();