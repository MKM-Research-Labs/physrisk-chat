/**
 * MKM Research Labs - Storage Utilities
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


 * Handles local storage operations
 */
const StorageUtils = (() => {
    // Private methods and properties
    const STORAGE_KEYS = {
      PREFERRED_VOICE: 'preferredVoice',
      LAST_MODEL: 'lastUsedModel'
    };
    
    /**
     * Get item from local storage
     * @param {string} key - Storage key
     * @param {*} defaultValue - Default value if key not found
     * @returns {*} - Stored value or default
     */
    const getItem = (key, defaultValue = null) => {
      try {
        const value = localStorage.getItem(key);
        return value !== null ? JSON.parse(value) : defaultValue;
      } catch (error) {
        console.error(`Error getting item from storage (${key}):`, error);
        return defaultValue;
      }
    };
    
    /**
     * Set item in local storage
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     */
    const setItem = (key, value) => {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch (error) {
        console.error(`Error setting item in storage (${key}):`, error);
      }
    };
    
    // Public methods
    return {
      /**
       * Save preferred voice
       * @param {string} voice - Voice name
       */
      savePreferredVoice: (voice) => {
        setItem(STORAGE_KEYS.PREFERRED_VOICE, voice);
      },
      
      /**
       * Get preferred voice
       * @returns {string} - Preferred voice name
       */
      getPreferredVoice: () => {
        return getItem(STORAGE_KEYS.PREFERRED_VOICE, 'UK English Female');
      },
      
      /**
       * Save last used model
       * @param {string} model - Model name
       */
      saveLastModel: (model) => {
        setItem(STORAGE_KEYS.LAST_MODEL, model);
      },
      
      /**
       * Get last used model
       * @returns {string} - Last used model name
       */
      getLastModel: () => {
        return getItem(STORAGE_KEYS.LAST_MODEL, 'claude-3.5-sonnet');
      }
    };
  })();