/**
 * MKM Research Labs - Modal Management
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


 * Handles modal dialogs for document summaries and chat previews
 */
const ModalManager = (() => {
  // Private properties
  let currentDocument = null;
  let currentPreviewChat = null;
  
  /**
   * Initialize all modal event listeners
   */
  const initializeModals = () => {
    // Get modal elements
    const summaryModal = UI.getElement('#summary-modal');
    const chatPreviewModal = UI.getElement('#chat-preview-modal');
    const closeSummaryBtn = UI.getElement('#close-modal');
    const closeChatBtn = UI.getElement('#close-chat-modal');
    
    // Document summary modal buttons
    const listenSummaryBtn = UI.getElement('#listen-summary');
    const copySummaryBtn = UI.getElement('#copy-summary');
    const exportSummaryBtn = UI.getElement('#export-summary-pdf');
    const useSummaryBtn = UI.getElement('#use-summary');
    
    // Chat preview modal buttons
    const listenChatBtn = UI.getElement('#listen-chat');
    const copyChatBtn = UI.getElement('#copy-chat');
    const exportChatBtn = UI.getElement('#export-chat-pdf');
    const loadChatBtn = UI.getElement('#load-chat');
    
    // Close modal events
    closeSummaryBtn.addEventListener('click', () => {
      AudioUtils.stopSpeaking();
      UI.hideModal('summary-modal');
    });
    
    closeChatBtn.addEventListener('click', () => {
      AudioUtils.stopSpeaking();
      UI.hideModal('chat-preview-modal');
    });
    
    // Close on click outside modal
    window.addEventListener('click', (e) => {
      if (e.target === summaryModal) {
        AudioUtils.stopSpeaking();
        UI.hideModal('summary-modal');
      }
      if (e.target === chatPreviewModal) {
        AudioUtils.stopSpeaking();
        UI.hideModal('chat-preview-modal');
      }
    });
    
    // Document summary modal actions - FIXED LISTENER
    listenSummaryBtn.addEventListener('click', function() {
      const summaryText = UI.getElement('#modal-content').textContent;
      const icon = document.getElementById('listen-icon');
      
      // If already speaking, stop
      if (responsiveVoice.isPlaying()) {
        responsiveVoice.cancel();
        icon.textContent = '🔊';
        this.innerHTML = '<span id="listen-icon">🔊</span> Listen';
        return;
      }
      
      // Get selected voice
      const voice = document.getElementById('voice-select').value;
      
      // Start speaking
      icon.textContent = '⏹️';
      this.innerHTML = '<span id="listen-icon">⏹️</span> Stop';
      
      responsiveVoice.speak(summaryText, voice, {
        onend: function() {
          icon.textContent = '🔊';
          listenSummaryBtn.innerHTML = '<span id="listen-icon">🔊</span> Listen';
        }
      });
    });
    
    copySummaryBtn.addEventListener('click', () => {
      const summaryText = UI.getElement('#modal-content').textContent;
      ExportUtils.copyToClipboard(summaryText);
    });
    
    exportSummaryBtn.addEventListener('click', () => {
      if (currentDocument) {
        const summaryText = UI.getElement('#modal-content').textContent;
        ExportUtils.exportDocumentToPDF(currentDocument, summaryText);
      } else {
        UI.showNotification('No document summary to export');
      }
    });
    
    useSummaryBtn.addEventListener('click', () => {
      if (currentDocument) {
        const summaryText = UI.getElement('#modal-content').textContent;
        const queryInput = UI.getElement('#query-input');
        queryInput.value = `Tell me more about ${currentDocument} based on this summary: ${summaryText.substring(0, 200)}...`;
        UI.hideModal('summary-modal');
        queryInput.focus();
      }
    });
    
    // Chat preview modal actions - FIXED LISTENER
    listenChatBtn.addEventListener('click', function() {
      if (!currentPreviewChat) return;
      
      const chatContent = UI.getElement('#chat-modal-content').textContent;
      const icon = document.getElementById('chat-listen-icon');
      
      // If already speaking, stop
      if (responsiveVoice.isPlaying()) {
        responsiveVoice.cancel();
        icon.textContent = '🔊';
        this.innerHTML = '<span id="chat-listen-icon">🔊</span> Listen';
        return;
      }
      
      // Get selected voice
      const voice = document.getElementById('chat-voice-select').value;
      
      // Start speaking
      icon.textContent = '⏹️';
      this.innerHTML = '<span id="chat-listen-icon">⏹️</span> Stop';
      
      responsiveVoice.speak(chatContent, voice, {
        onend: function() {
          icon.textContent = '🔊';
          listenChatBtn.innerHTML = '<span id="chat-listen-icon">🔊</span> Listen';
        }
      });
    });
    
    copyChatBtn.addEventListener('click', () => {
      if (currentPreviewChat) {
        const chatText = ExportUtils.formatChatForCopy(currentPreviewChat.messages);
        ExportUtils.copyToClipboard(chatText);
      }
    });
    
    exportChatBtn.addEventListener('click', () => {
      if (currentPreviewChat) {
        ExportUtils.exportChatToPDF(currentPreviewChat);
      }
    });
    
    loadChatBtn.addEventListener('click', () => {
      if (currentPreviewChat) {
        ChatManager.loadChat(currentPreviewChat);
        UI.hideModal('chat-preview-modal');
      }
    });
  };
  
  // Public methods
  return {
    /**
     * Initialize modal manager
     */
    init: () => {
      initializeModals();
    },
    
    /**
     * Show document summary modal
     * @param {string} docName - Document name
     * @param {string} summary - Document summary text
     */
    showDocumentSummary: (docName, summary) => {
      currentDocument = docName;
      
      const modalTitle = UI.getElement('#modal-title');
      const modalContent = UI.getElement('#modal-content');
      
      modalTitle.textContent = docName;
      modalContent.textContent = summary;
      
      UI.showModal('summary-modal');
    },
    
    /**
     * Show chat preview modal
     * @param {Object} chat - Chat object with messages and metadata
     */
    showChatPreview: (chat) => {
      currentPreviewChat = chat;
      
      // Set modal title using first user message or date
      const modalTitle = UI.getElement('#chat-modal-title');
      const modalContent = UI.getElement('#chat-modal-content');
      
      let titleText = 'Chat History';
      const date = new Date(chat.timestamp);
      let previewText = '';
      
      for (const message of chat.messages) {
        if (message.role === 'user') {
          previewText = message.content;
          break;
        }
      }
      
      if (previewText) {
        titleText = previewText.length > 40 ? 
          previewText.substring(0, 40) + '...' : previewText;
      } else {
        titleText = `Chat from ${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
      }
      
      modalTitle.textContent = titleText;
      
      // Render chat messages in the modal
      modalContent.innerHTML = '';
      
      let lastRole = '';
      
      chat.messages.forEach(message => {
        if (message.role === 'sources') {
          // Add sources as a special block
          const sourcesDiv = document.createElement('div');
          sourcesDiv.className = 'chat-preview-sources';
          sourcesDiv.innerHTML = message.content.replace(/\n/g, '<br>');
          modalContent.appendChild(sourcesDiv);
        } else if (message.role !== 'error') {
          // Add a role header if it's different from the last message
          if (message.role !== lastRole) {
            const roleHeader = document.createElement('div');
            roleHeader.className = 'chat-preview-role';
            roleHeader.textContent = message.role === 'user' ? 'User:' : 'Assistant:';
            modalContent.appendChild(roleHeader);
            lastRole = message.role;
          }
          
          // Add the message content
          const messageDiv = document.createElement('div');
          messageDiv.className = `chat-preview-message ${
            message.role === 'user' ? 'chat-preview-user' : 'chat-preview-assistant'
          }`;
          messageDiv.textContent = message.content;
          modalContent.appendChild(messageDiv);
        }
      });
      
      UI.showModal('chat-preview-modal');
    }
  };
})();