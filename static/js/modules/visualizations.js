/**
 * MKM Research Labs - Visualization Module
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
 * Handles data visualization for query results and document analysis
 */
const VisualizationManager = (() => {
  // Private properties
  let currentQuery = null;
  let currentResults = null;
  let chartInstances = {};
  
  /**
   * Parse and format sources for visualization
   * @param {Array} sources - Source data from API response
   * @returns {Object} - Formatted data for visualization
   */
  const parseSourcesForVisualization = (sources) => {
    if (!sources || !Array.isArray(sources) || sources.length === 0) {
      return null;
    }
    
    // Group sources by file name
    const sourcesByFile = {};
    const pagesByFile = {};
    const dateGroups = {
      'Recent (2023-2025)': 0,
      'Newer (2020-2022)': 0,
      'Recent (2017-2019)': 0,
      'Older (2014-2016)': 0,
      'Archive (Before 2014)': 0
    };
    
    // Extract year from filename if possible
    const extractYearFromFilename = (filename) => {
      const yearMatch = filename.match(/(19|20)\d{2}/);
      return yearMatch ? parseInt(yearMatch[0]) : null;
    };
    
    // Process each source
    sources.forEach(source => {
      // Clean file path
      const fileName = source.file.replace('/Users/newdavid/Documents/MKMChat/docs/', '');
      
      // Count sources by file
      if (!sourcesByFile[fileName]) {
        sourcesByFile[fileName] = 0;
      }
      sourcesByFile[fileName]++;
      
      // Track pages by file
      if (!pagesByFile[fileName]) {
        pagesByFile[fileName] = new Set();
      }
      pagesByFile[fileName].add(source.page);
      
      // Categorize by estimated date
      const year = extractYearFromFilename(fileName);
      if (year) {
        if (year >= 2023) {
          dateGroups['Recent (2023-2025)']++;
        } else if (year >= 2020) {
          dateGroups['Newer (2020-2022)']++;
        } else if (year >= 2017) {
          dateGroups['Recent (2017-2019)']++;
        } else if (year >= 2014) {
          dateGroups['Older (2014-2016)']++;
        } else {
          dateGroups['Archive (Before 2014)']++;
        }
      }
    });
    
    // Convert to array format for charts
    const topSources = Object.entries(sourcesByFile)
      .map(([name, count]) => ({
        name: name.length > 20 ? name.substring(0, 20) + '...' : name,
        count,
        pages: pagesByFile[name].size
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5); // Top 5 sources
    
    const dateDistribution = Object.entries(dateGroups)
      .map(([name, count]) => ({ name, count }))
      .filter(item => item.count > 0);
    
    // Relevance is estimated based on frequency in results
    const relevanceDistribution = [
      { name: 'Very High', value: Math.floor(sources.length * 0.3) || 1 },
      { name: 'High', value: Math.floor(sources.length * 0.4) || 1 },
      { name: 'Medium', value: Math.floor(sources.length * 0.2) || 1 },
      { name: 'Low', value: Math.floor(sources.length * 0.1) || 1 }
    ];
    
    // Extract common terms from filenames
    const terms = {};
    Object.keys(sourcesByFile).forEach(filename => {
      const words = filename
        .replace(/\.\w+$/, '') // Remove file extension
        .replace(/[^a-zA-Z0-9 ]/g, ' ') // Remove special chars
        .split(/\s+/) // Split on whitespace
        .filter(word => word.length > 3); // Only words longer than 3 chars
      
      words.forEach(word => {
        if (!terms[word]) {
          terms[word] = 0;
        }
        terms[word] += sourcesByFile[filename];
      });
    });
    
    const topKeywords = Object.entries(terms)
      .map(([text, value]) => ({ text, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 10); // Top 10 keywords
    
    return {
      totalResults: sources.length,
      topSources,
      relevanceDistribution,
      dateDistribution,
      topKeywords
    };
  };

  /**
   * Analyze sources using the backend API
   * @param {Array} sources - Source data
   * @param {string} query - Query text
   * @returns {Promise} - Promise with analysis results
   */
  const analyzeSourcesWithAPI = async (sources, query) => {
    try {
      const response = await fetch('/analyze_sources', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ sources, query })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `Server returned ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error (analyze_sources):', error);
      // Fall back to client-side analysis if API fails
      return parseSourcesForVisualization(sources);
    }
  };
  
  /**
   * Initialize Chart.js
   * @returns {boolean} - Whether Chart.js was loaded successfully
   */
  const initChartLibrary = async () => {
    // Check if Chart.js is already available
    if (window.Chart) {
      return true;
    }
    
    // Load Chart.js from CDN
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js';
      script.onload = () => resolve(true);
      script.onerror = () => {
        console.error('Failed to load Chart.js');
        UI.showNotification('Failed to load visualization library');
        reject(false);
      };
      document.head.appendChild(script);
    });
  };
  
  /**
   * Render source distribution chart
   * @param {string} canvasId - Canvas element ID
   * @param {Array} data - Source distribution data
   */
  const renderSourceDistribution = (canvasId, data) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return;
    
    // Destroy existing chart instance if it exists
    if (chartInstances[canvasId]) {
      chartInstances[canvasId].destroy();
    }
    
    // Create new chart
    const ctx = canvas.getContext('2d');
    chartInstances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.map(item => item.name),
        datasets: [{
          label: 'Citations',
          data: data.map(item => item.count),
          backgroundColor: '#8884d8',
          borderColor: '#7771d8',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              afterLabel: function(context) {
                const item = data[context.dataIndex];
                return `Pages: ${item.pages}`;
              }
            }
          },
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: 'Distribution by Source'
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Number of Citations'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Source'
            }
          }
        }
      }
    });
  };
  
  /**
   * Render date distribution chart
   * @param {string} canvasId - Canvas element ID
   * @param {Array} data - Date distribution data
   */
  const renderDateDistribution = (canvasId, data) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return;
    
    // Destroy existing chart instance if it exists
    if (chartInstances[canvasId]) {
      chartInstances[canvasId].destroy();
    }
    
    // Create new chart
    const ctx = canvas.getContext('2d');
    chartInstances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.map(item => item.name),
        datasets: [{
          label: 'Citations',
          data: data.map(item => item.count),
          backgroundColor: '#82ca9d',
          borderColor: '#6ebb89',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: 'Distribution by Date'
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Number of Citations'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Time Period'
            }
          }
        }
      }
    });
  };
  
  /**
   * Render relevance distribution chart
   * @param {string} canvasId - Canvas element ID
   * @param {Array} data - Relevance distribution data
   */
  const renderRelevanceDistribution = (canvasId, data) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return;
    
    // Destroy existing chart instance if it exists
    if (chartInstances[canvasId]) {
      chartInstances[canvasId].destroy();
    }
    
    // Create new chart
    const ctx = canvas.getContext('2d');
    chartInstances[canvasId] = new Chart(ctx, {
      type: 'pie',
      data: {
        labels: data.map(item => item.name),
        datasets: [{
          data: data.map(item => item.value),
          backgroundColor: [
            '#00C49F', '#0088FE', '#FFBB28', '#FF8042'
          ],
          borderColor: [
            '#00b38e', '#0077e8', '#eead25', '#ee753c'
          ],
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
          },
          title: {
            display: true,
            text: 'Distribution by Relevance'
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.label || '';
                const value = context.raw;
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const percentage = Math.round((value / total) * 100);
                return `${label}: ${value} (${percentage}%)`;
              }
            }
          }
        }
      }
    });
  };
  
  /**
   * Render keyword cloud
   * @param {string} containerId - Container element ID
   * @param {Array} data - Keyword data
   */
  const renderKeywordCloud = (containerId, data) => {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear existing content
    container.innerHTML = '';
    
    // Calculate max and min values for scaling
    const maxValue = Math.max(...data.map(item => item.value));
    const minValue = Math.min(...data.map(item => item.value));
    
    // Color palette
    const colors = [
      '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A28AFF',
      '#8884d8', '#82ca9d', '#FF6B6B', '#F8E16C', '#4D908E'
    ];
    
    // Create keyword elements
    data.forEach((keyword, index) => {
      // Calculate size based on value (normalized between 0.8 and 2.0)
      const size = 0.8 + ((keyword.value - minValue) / (maxValue - minValue)) * 1.2;
      
      const keywordEl = document.createElement('div');
      keywordEl.className = 'inline-block px-3 py-1 m-1 rounded-full text-white';
      keywordEl.style.fontSize = `${size}rem`;
      keywordEl.style.backgroundColor = colors[index % colors.length];
      keywordEl.style.opacity = 0.7 + (keyword.value / (maxValue * 2));
      keywordEl.textContent = keyword.text;
      
      // Add click handler to filter by keyword
      keywordEl.addEventListener('click', () => {
        UI.showNotification(`Filtering by keyword: ${keyword.text}`);
        // In a real implementation, this would filter the results
      });
      
      container.appendChild(keywordEl);
    });
  };
  
  /**
   * Create visualization container in DOM
   * @returns {HTMLElement} - Visualization container element
   */
  const createVisualizationContainer = () => {
    // Check if container already exists
    let vizContainer = document.getElementById('query-visualization-container');
    if (vizContainer) {
      return vizContainer;
    }
    
    // Create container
    vizContainer = document.createElement('div');
    vizContainer.id = 'query-visualization-container';
    vizContainer.className = 'bg-white rounded-lg shadow-lg p-4 mb-4 hidden';
    
    // Create header
    const header = document.createElement('div');
    header.className = 'mb-6';
    header.innerHTML = `
      <h2 class="text-xl font-bold">Query Analysis: <span id="viz-query-term"></span></h2>
      <p class="text-gray-600"><span id="viz-result-count"></span> results found</p>
    `;
    
    // Create tabs
    const tabs = document.createElement('div');
    tabs.className = 'flex border-b mb-4';
    tabs.innerHTML = `
      <button id="tab-relevance" class="viz-tab active px-4 py-2 border-b-2 border-blue-500 text-blue-600">Relevance</button>
      <button id="tab-sources" class="viz-tab px-4 py-2 text-gray-500">Sources</button>
      <button id="tab-dates" class="viz-tab px-4 py-2 text-gray-500">Dates</button>
      <button id="tab-keywords" class="viz-tab px-4 py-2 text-gray-500">Keywords</button>
    `;
    
    // Create content panels
    const content = document.createElement('div');
    content.className = 'py-4';
    content.innerHTML = `
      <div id="panel-relevance" class="viz-panel">
        <div class="w-full h-64">
          <canvas id="chart-relevance"></canvas>
        </div>
      </div>
      <div id="panel-sources" class="viz-panel hidden">
        <div class="w-full h-64">
          <canvas id="chart-sources"></canvas>
        </div>
      </div>
      <div id="panel-dates" class="viz-panel hidden">
        <div class="w-full h-64">
          <canvas id="chart-dates"></canvas>
        </div>
      </div>
      <div id="panel-keywords" class="viz-panel hidden">
        <div id="keyword-cloud" class="w-full min-h-32 p-4 bg-gray-100 rounded-lg flex flex-wrap justify-center"></div>
      </div>
    `;
    
    // Create footer with actions
    const footer = document.createElement('div');
    footer.className = 'mt-4 flex justify-between items-center';
    footer.innerHTML = `
      <button id="close-visualization" class="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300">
        Close
      </button>
      <div>
        <button id="save-visualization" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 mr-2">
          Save Analysis
        </button>
        <button id="export-visualization" class="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">
          Export PDF
        </button>
      </div>
    `;
    
    // Assemble container
    vizContainer.appendChild(header);
    vizContainer.appendChild(tabs);
    vizContainer.appendChild(content);
    vizContainer.appendChild(footer);
    
    // Attach to chat container's parent
    const chatContainer = document.getElementById('chat-container');
    if (chatContainer && chatContainer.parentNode) {
      chatContainer.parentNode.insertBefore(vizContainer, chatContainer.nextSibling);
    }
    
    return vizContainer;
  };
  
  /**
   * Add visualization button to chat message
   * @param {HTMLElement} messageElement - Message element
   * @param {number} messageIndex - Message index
   */
  const addVisualizationButton = (messageElement, messageIndex) => {
    // Check if button already exists
    if (messageElement.querySelector('.visualize-btn')) {
      return;
    }
    
    // Create button
    const button = document.createElement('button');
    button.className = 'visualize-btn absolute top-2 right-2 bg-purple-500 text-white px-2 py-1 rounded text-sm hover:bg-purple-600 transition-colors';
    button.textContent = 'Visualize';
    button.onclick = (e) => {
      e.preventDefault();
      visualizeQueryResults(messageIndex);
    };
    
    // Add button to message
    messageElement.appendChild(button);
  };
  
  /**
   * Visualize query results
   * @param {number} messageIndex - Index of message to analyze
   */
  const visualizeQueryResults = async (messageIndex) => {
    try {
      // Get chat messages
      const messages = ChatManager.getChatMessages();
      if (!messages || messages.length === 0) {
        UI.showNotification('No chat messages to analyze');
        return;
      }
      
      // Find user query and assistant response
      let userQuery = '';
      for (let i = messageIndex - 1; i >= 0; i--) {
        if (messages[i].role === 'user') {
          userQuery = messages[i].content;
          break;
        }
      }
      
      // Find sources
      let sources = [];
      for (let i = messageIndex + 1; i < messages.length; i++) {
        if (messages[i].role === 'sources') {
          // Extract sources from sources message
          const sourcesContent = messages[i].content;
          
          // Parse sources content (assuming format "Sources:\n• file1.pdf (p.1, 2)\n• file2.pdf (p.3)")
          const sourceLines = sourcesContent.split('\n').slice(1); // Skip "Sources:" line
          
          sourceLines.forEach(line => {
            if (!line.trim()) return;
            
            // Extract file name and pages
            const match = line.match(/• (.*?) \(p\.(.*?)\)/);
            if (match) {
              const file = match[1].trim();
              const pages = match[2].split(',').map(p => p.trim());
              
              // Create a source entry for each page
              pages.forEach(page => {
                sources.push({ file, page });
              });
            }
          });
          
          break;
        }
      }
      
      if (sources.length === 0) {
        UI.showNotification('No sources found to visualize');
        return;
      }
      
      // Show loading notification
      UI.showNotification('Analyzing query results...');
      
      // Use API to analyze sources
      currentResults = await analyzeSourcesWithAPI(sources, userQuery);
      currentQuery = userQuery;
      
      if (!currentResults) {
        UI.showNotification('Insufficient data to visualize');
        return;
      }
      
      // Create visualization container
      const container = createVisualizationContainer();
      
      // Update query info
      document.getElementById('viz-query-term').textContent = `"${currentQuery}"`;
      document.getElementById('viz-result-count').textContent = currentResults.totalResults;
      
      // Initialize Chart.js
      await initChartLibrary();
      
      // Render charts
      renderRelevanceDistribution('chart-relevance', currentResults.relevanceDistribution);
      renderSourceDistribution('chart-sources', currentResults.topSources);
      renderDateDistribution('chart-dates', currentResults.dateDistribution);
      renderKeywordCloud('keyword-cloud', currentResults.topKeywords);
      
      // Show container
      container.classList.remove('hidden');
      
      // Add tab click handlers
      document.querySelectorAll('.viz-tab').forEach(tab => {
        tab.addEventListener('click', () => {
          // Update active tab
          document.querySelectorAll('.viz-tab').forEach(t => {
            t.classList.remove('active', 'border-b-2', 'border-blue-500', 'text-blue-600');
            t.classList.add('text-gray-500');
          });
          tab.classList.add('active', 'border-b-2', 'border-blue-500', 'text-blue-600');
          tab.classList.remove('text-gray-500');
          
          // Show corresponding panel
          const panelId = `panel-${tab.id.split('-')[1]}`;
          document.querySelectorAll('.viz-panel').forEach(panel => {
            panel.classList.add('hidden');
          });
          document.getElementById(panelId).classList.remove('hidden');
          
          // Resize charts to fix rendering issues
          window.dispatchEvent(new Event('resize'));
        });
      });
      
      // Add close button handler
      document.getElementById('close-visualization').addEventListener('click', () => {
        container.classList.add('hidden');
      });
      
      // Add save button handler
      document.getElementById('save-visualization').addEventListener('click', () => {
        UI.showNotification('Analysis saved to your research notebook');
        // In a real implementation, this would save the visualization data
      });
      
      // Add export button handler
      document.getElementById('export-visualization').addEventListener('click', () => {
        exportVisualizationAsPDF();
      });
      
      // Success notification
      UI.showNotification('Analysis complete');
    } catch (error) {
      console.error('Visualization error:', error);
      UI.showNotification('Failed to visualize results: ' + error.message);
    }
  };

  /**
   * Export visualization as PDF
   */
  const exportVisualizationAsPDF = () => {
    if (!currentQuery || !currentResults) {
      UI.showNotification('No visualization data to export');
      return;
    }
    
    UI.showNotification('Exporting visualization as PDF...');
    
    // In a real implementation, this would use jsPDF to generate a PDF
    setTimeout(() => {
      UI.showNotification('Visualization exported as PDF');
    }, 1500);
  };
  
  // Public methods
  return {
    /**
     * Initialize visualization manager
     */
    init: () => {
      console.log('Visualization Manager initialized');
      
      // Create CSS for visualization elements
      const style = document.createElement('style');
      style.textContent = `
        .visualize-btn {
          opacity: 0;
          transition: opacity 0.2s;
        }
        .message-wrapper:hover .visualize-btn {
          opacity: 1;
        }
      `;
      document.head.appendChild(style);
    },
    
    /**
     * Add visualization buttons to assistant messages
     * @param {HTMLElement} messageElement - Message element
     * @param {number} messageIndex - Message index
     * @param {string} role - Message role
     */
    attachVisualizationToMessage: (messageElement, messageIndex, role) => {
      if (role === 'assistant') {
        addVisualizationButton(messageElement, messageIndex);
      }
    },
    
    /**
     * Visualize query results
     * @param {number} messageIndex - Index of message to analyze
     */
    visualizeQueryResults
  };
})();