/**
 * MKM Research Labs - Chat Management
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
 * Handles chat messaging, history, and interactions
 */
const ChatManager = (() => {
  // Private properties
  let chatMessages = [];
  let currentChatId = null;
  
  /**
   * Initialize chat form submission
   */
  const initChatForm = () => {
    const form = UI.getElement('#query-form');
    const input = UI.getElement('#query-input');
    const modelSelect = UI.getElement('#model-select');
    const chatContainer = UI.getElement('#chat-container');
    const newChatButton = UI.getElement('#new-chat');
    
    // Restore last used model
    const lastModel = StorageUtils.getLastModel();
    if (lastModel && modelSelect.querySelector(`option[value="${lastModel}"]`)) {
      modelSelect.value = lastModel;
    }
    
    // Save model selection
    modelSelect.addEventListener('change', () => {
      StorageUtils.saveLastModel(modelSelect.value);
    });
    
    // New chat button
    newChatButton.addEventListener('click', () => {
      chatContainer.innerHTML = '';
      chatMessages = [];
      currentChatId = null;
      input.value = '';
      input.focus();
    });
    
    // Form submission
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const query = input.value.trim();
      if (!query) return;
      
      appendMessage('user', query);
      input.value = '';
      
      try {
        const selectedModel = modelSelect.value;
        const response = await ApiService.sendQuery(query, selectedModel);
        
        if (response.error) {
          appendMessage('error', response.error);
        } else {
          appendMessage('assistant', response.response);
          
          if (response.sources && Array.isArray(response.sources) && response.sources.length > 0) {
            appendSources(response.sources);
          }
        }
        
        // Auto-save after each interaction
        autoSaveChat();
      } catch (error) {
        console.error('Query error:', error);
        appendMessage('error', 'Failed to get response: ' + error.message);
        autoSaveChat();
      }
    });
  };
  
  /**
   * Create a message element
   * @param {string} role - Message role (user, assistant, error)
   * @param {string} content - Message content
   * @param {number} messageIndex - Index in chatMessages array
   * @returns {HTMLElement} - Message element
   */
  const createMessageElement = (role, content, messageIndex) => {
    const messageWrapper = document.createElement('div');
    messageWrapper.className = 'message-wrapper';
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role === 'user' ? 'message-user' : role === 'error' ? 'message-error' : 'message-assistant'}`;
    
    // Format content with proper spacing
    if (role === 'assistant') {
      const formattedContent = content
        .replace(/\n\n+/g, '\n\n')
        .trim();
      
      messageDiv.textContent = formattedContent;
    } else {
      messageDiv.textContent = content;
    }
    
    // Add copy button for assistant messages
    if (role === 'assistant') {
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-message-btn';
      copyButton.textContent = 'Copy';
      copyButton.onclick = (e) => {
        e.preventDefault();
        copyMessageWithSources(content, messageIndex);
      };
      messageWrapper.appendChild(copyButton);
      
      // Attach visualization if available
      if (VisualizationManager) {
        VisualizationManager.attachVisualizationToMessage(messageWrapper, messageIndex, role);
      }
    }
    
    messageWrapper.appendChild(messageDiv);
    return messageWrapper;
  };
  
  /**
   * Append a message to the chat
   * @param {string} role - Message role
   * @param {string} content - Message content
   */
  const appendMessage = (role, content) => {
    const messageIndex = chatMessages.length;
    const messageElement = createMessageElement(role, content, messageIndex);
    
    UI.getElement('#chat-container').appendChild(messageElement);
    UI.getElement('#chat-container').scrollTop = UI.getElement('#chat-container').scrollHeight;
    
    chatMessages.push({ role, content });
  };
  
  /**
   * Append sources to the chat
   * @param {Array} sources - Sources array
   */
  const appendSources = (sources) => {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'message-sources';
    
    // Clean and group sources by filename and collect page numbers
    const groupedSources = sources.reduce((acc, source) => {
      // Remove the path prefix
      const cleanFile = source.file.replace('/Users/newdavid/Documents/MKMChat/docs/', '');
      
      if (!acc[cleanFile]) {
        acc[cleanFile] = new Set();
      }
      acc[cleanFile].add(source.page);
      return acc;
    }, {});
    
    // Create the sources list
    let sourcesList = document.createElement('ul');
    sourcesList.className = 'list-disc pl-5 mt-1';
    
    sourcesDiv.innerHTML = 'Sources:';
    
    // Format sources list
    const formattedSources = Object.entries(groupedSources).map(([file, pages]) => {
      const pageNumbers = Array.from(pages).sort((a, b) => a - b);
      return `${file} (p.${pageNumbers.join(', ')})`;
    });
    
    // Add each source as a list item
    formattedSources.forEach(sourceText => {
      const listItem = document.createElement('li');
      listItem.className = 'mb-1';
      listItem.textContent = sourceText;
      sourcesList.appendChild(listItem);
    });
    
    sourcesDiv.appendChild(sourcesList);
    UI.getElement('#chat-container').appendChild(sourcesDiv);
    
    // Save to chat history in the same list format
    chatMessages.push({
      role: 'sources',
      content: 'Sources:\n' + formattedSources.map(src => `• ${src}`).join('\n')
    });
  };
  
  /**
   * Copy message with sources to clipboard
   * @param {string} text - Message text
   * @param {number} messageIndex - Message index
   */
  const copyMessageWithSources = async (text, messageIndex) => {
    try {
      // Find the next sources message after this message
      let sourcesText = '';
      for (let i = messageIndex + 1; i < chatMessages.length; i++) {
        if (chatMessages[i].role === 'sources') {
          sourcesText = '\n\n' + chatMessages[i].content;
          break;
        }
      }
      
      // Combine message content with sources
      const fullText = text + sourcesText;
      
      await ExportUtils.copyToClipboard(fullText);
    } catch (err) {
      UI.showNotification('Failed to copy text');
      console.error('Copy failed:', err);
    }
  };
  
  /**
   * Auto-save chat history
   */
  const autoSaveChat = () => {
    if (chatMessages.length === 0) return;
    
    // Use a timeout to debounce saves
    if (autoSaveChat.timeout) {
      clearTimeout(autoSaveChat.timeout);
    }
    
    autoSaveChat.timeout = setTimeout(async () => {
      try {
        await ApiService.saveChat(chatMessages);
        UI.showNotification('Chat saved successfully!');
        loadChatHistory(); // Refresh the chat list
      } catch (error) {
        console.error('Failed to save chat:', error);
        UI.showNotification('Failed to save chat: ' + error.message);
      }
    }, 1000);
  };
  
  /**
   * Load chat history
   */
  const loadChatHistory = async () => {
    try {
      // Get chat container and add loading state
      const chatList = UI.getElement('#chat-list');
      chatList.innerHTML = '';
      chatList.appendChild(UI.createLoadingPlaceholder());
      
      // Get chat history from API
      const data = await ApiService.getChats();
      
      // Clear container
      chatList.innerHTML = '';
      
      // Add chat history buttons
      const chats = data?.chats || [];
      
      if (chats.length > 0) {
        chats.reverse().forEach(chat => {
          const date = new Date(chat.timestamp);
          const chatButton = document.createElement('button');
          chatButton.className = 'chat-item';
          
          // Extract first user message for preview
          let previewText = 'Empty chat';
          for (const message of chat.messages) {
            if (message.role === 'user') {
              previewText = message.content;
              break;
            }
          }
          
          chatButton.innerHTML = `
            <div class="chat-item-title">${date.toLocaleDateString()}</div>
            <div class="chat-item-time">${date.toLocaleTimeString()}</div>
            <div class="chat-item-preview">${previewText}</div>
          `;
          
          // Add event listener for chat preview
          chatButton.addEventListener('click', (e) => {
            if (e.ctrlKey || e.metaKey) {
              // Ctrl/Cmd + click: Load chat directly
              loadChat(chat);
            } else {
              // Regular click: Show preview
              ModalManager.showChatPreview(chat);
            }
          });
          
          chatList.appendChild(chatButton);
        });
      } else {
        chatList.appendChild(UI.createEmptyState('No chat history yet. Start a new chat!'));
      }
      
      // Apply search filter if exists
      const searchTerm = UI.getElement('#chat-search').value;
      if (searchTerm) {
        filterChats();
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
      UI.showNotification('Failed to load chat history');
      
      const chatList = UI.getElement('#chat-list');
      chatList.innerHTML = '';
      
      const errorElement = document.createElement('div');
      errorElement.className = 'p-3 bg-red-100 text-red-800 rounded-lg mt-2';
      errorElement.textContent = `Error: ${error.message}`;
      chatList.appendChild(errorElement);
    }
  };
  
  /**
   * Filter chats by search term
   */
  const filterChats = () => {
    const searchTerm = UI.getElement('#chat-search').value.toLowerCase();
    const chatButtons = Array.from(UI.getAllElements('#chat-list .chat-item'));
    
    chatButtons.forEach(button => {
      const chatText = button.textContent.toLowerCase();
      if (chatText.includes(searchTerm)) {
        button.classList.remove('hidden');
      } else {
        button.classList.add('hidden');
      }
    });
  };
  
  // Public methods
  return {
    /**
     * Initialize chat manager
     */
    init: () => {
      initChatForm();
      
      // Add chat search functionality
      UI.getElement('#chat-search').addEventListener('input', filterChats);
      
      // Initial load
      loadChatHistory();
    },
    
    /**
     * Load chat into main chat container
     * @param {Object} chat - Chat object with messages
     */
    loadChat: (chat) => {
      const chatContainer = UI.getElement('#chat-container');
      chatContainer.innerHTML = '';
      
      chatMessages = chat.messages;
      currentChatId = chat.id;
      
      chatMessages.forEach((message, index) => {
        if (message.role === 'sources') {
          const sourcesDiv = document.createElement('div');
          sourcesDiv.className = 'message-sources';
          sourcesDiv.innerHTML = message.content;
          chatContainer.appendChild(sourcesDiv);
        } else {
          const messageElement = createMessageElement(message.role, message.content, index);
          chatContainer.appendChild(messageElement);
        }
      });
      
      chatContainer.scrollTop = chatContainer.scrollHeight;
    },
    
    /**
     * Reload chat history
     */
    reloadChatHistory: loadChatHistory,
    
    /**
     * Get current chat messages
     * @returns {Array} - Chat messages
     */
    getChatMessages: () => {
      return [...chatMessages];
    }
  };
})();