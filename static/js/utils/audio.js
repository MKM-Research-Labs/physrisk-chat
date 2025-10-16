/**
 * MKM Research Labs - Audio Utilities
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


 * Handles text-to-speech functionality
 */
const AudioUtils = (() => {
    // Private methods and properties
    let isSpeaking = false;
    
    /**
     * Set button state based on speech status
     * @param {HTMLElement} button - Button element
     * @param {boolean} playing - Whether speech is playing
     */
    const updateButtonState = (button, playing) => {
      const iconSpan = button.querySelector('span');
      if (playing) {
        iconSpan.textContent = '⏸️';
        button.innerHTML = button.innerHTML.replace('Listen', 'Pause');
      } else {
        iconSpan.textContent = '🔊';
        button.innerHTML = button.innerHTML.replace('Pause', 'Listen');
        button.innerHTML = button.innerHTML.replace('Resume', 'Listen');
      }
    };
    
    /**
     * Check if ResponsiveVoice is available
     * @returns {boolean} - Whether ResponsiveVoice is available
     */
    const isResponsiveVoiceAvailable = () => {
      return typeof responsiveVoice !== 'undefined' && responsiveVoice !== null;
    };
    
    // Public methods
    return {
      /**
       * Initialize voice preferences
       */
      initVoicePreferences: () => {
        const savedVoice = StorageUtils.getPreferredVoice();
        
        // Set voice in both select elements
        ['voice-select', 'chat-voice-select'].forEach(selectId => {
          const select = document.getElementById(selectId);
          if (select) {
            // Find matching option
            for (let i = 0; i < select.options.length; i++) {
              if (select.options[i].value === savedVoice) {
                select.selectedIndex = i;
                break;
              }
            }
            
            // Add change listener
            select.addEventListener('change', () => {
              StorageUtils.savePreferredVoice(select.value);
              // Sync other select
              const otherSelectId = selectId === 'voice-select' ? 'chat-voice-select' : 'voice-select';
              const otherSelect = document.getElementById(otherSelectId);
              if (otherSelect) {
                otherSelect.value = select.value;
              }
            });
          }
        });
      },
      
      /**
       * Speak text using ResponsiveVoice
       * @param {string} text - Text to speak
       * @param {HTMLElement} button - Button element to update
       */
      speak: (text, button) => {
        // Check if ResponsiveVoice is available
        if (!isResponsiveVoiceAvailable()) {
          UI.showNotification('Text-to-speech is not available. Please check your internet connection.');
          return;
        }
        
        // If already speaking
        if (responsiveVoice.isPlaying()) {
          // If paused, resume
          if (button.querySelector('span').textContent.includes('▶️')) {
            responsiveVoice.resume();
            updateButtonState(button, true);
          } else {
            // If playing, pause
            responsiveVoice.pause();
            button.querySelector('span').textContent = '▶️';
            button.innerHTML = button.innerHTML.replace('Pause', 'Resume');
          }
          return;
        }
        
        // Start new speech
        const voiceSelect = button.id.includes('listen-chat') 
          ? document.getElementById('chat-voice-select') 
          : document.getElementById('voice-select');
        
        const selectedVoice = voiceSelect.value;
        
        // Setup speech parameters
        const speechParams = {
          pitch: 1,
          rate: 1,
          volume: 1,
          onstart: () => {
            isSpeaking = true;
            updateButtonState(button, true);
          },
          onend: () => {
            isSpeaking = false;
            updateButtonState(button, false);
          },
          onerror: (e) => {
            console.error('Speech synthesis error:', e);
            UI.showNotification('Speech synthesis failed');
            isSpeaking = false;
            updateButtonState(button, false);
          }
        };
        
        responsiveVoice.speak(text, selectedVoice, speechParams);
      },
      
      /**
       * Stop speaking
       */
      stopSpeaking: () => {
        if (isResponsiveVoiceAvailable() && responsiveVoice.isPlaying()) {
          responsiveVoice.cancel();
          isSpeaking = false;
        }
      },
      
      /**
       * Format chat messages for speech
       * @param {Array} messages - Chat messages
       * @returns {string} - Formatted text for speech
       */
      formatChatForSpeech: (messages) => {
        let speechText = '';
        
        messages.forEach(message => {
          if (message.role !== 'sources' && message.role !== 'error') {
            const roleText = message.role === 'user' ? 'User: ' : 'Assistant: ';
            speechText += roleText + message.content + '. ';
          }
        });
        
        return speechText;
      }
    };
  })();