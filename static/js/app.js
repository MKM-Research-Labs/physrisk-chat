/**
 * MKM Research Labs - Main Application
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


 * Initializes and coordinates all modules
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('MKM Research Labs Application Starting...');
    
    // Initialize UI event listeners
    initializeUIEvents();
    
    // Initialize modules in the correct order
    initializeModules();
    
    console.log('Application Initialized Successfully');
  });
  
  /**
   * Initialize global UI event listeners
   */
  function initializeUIEvents() {
    // Handle sidebar toggle
    UI.getElement('#toggle-sidebar').addEventListener('click', UI.toggleSidebar);
    
    // Handle tab switching
    const tabButtons = UI.getAllElements('.tab-btn');
    tabButtons.forEach(tab => {
      tab.addEventListener('click', () => {
        const tabId = tab.id;
        UI.switchTab(tabId);
        
        // Load data for the selected tab if needed
        if (tabId === 'document-summaries-tab') {
          DocumentManager.loadDocumentSummaries();
        } else if (tabId === 'knowledge-base-tab') {
          KnowledgeManager.loadAvailableKnowledgeBases();
        }
      });
    });
    
    // Initialize audio preferences
    AudioUtils.initVoicePreferences();
    
    // Initialize loading indicator
    if (typeof LoaderUtils !== 'undefined') {
      LoaderUtils.init();
      console.log('✓ Loading indicator initialized');
    } else {
      console.warn('LoaderUtils not found - loading indicator disabled');
    }
  }
  
  /**
   * Initialize all modules
   */
  function initializeModules() {
    // Initialize in dependency order
    ModalManager.init();
    ChatManager.init();
    DocumentManager.init();
    KnowledgeManager.init();

    // Initialize visualization module if available
    if (VisualizationManager) {
      VisualizationManager.init();
    }
    
    // Handle window events
    window.addEventListener('beforeunload', () => {
      // Stop any audio playback
      AudioUtils.stopSpeaking();
    });
  }