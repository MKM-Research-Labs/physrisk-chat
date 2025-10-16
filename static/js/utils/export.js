/**
 * MKM Research Labs - Export Utilities
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


 * Handles exporting content to PDF and clipboard
 */
const ExportUtils = (() => {
    // Initialize jsPDF when available
    let jsPDF;
    
    document.addEventListener('DOMContentLoaded', () => {
      if (window.jspdf) {
        jsPDF = window.jspdf.jsPDF;
      }
    });
    
    /**
     * Add header to PDF
     * @param {Object} pdf - jsPDF instance
     * @param {number} pageWidth - PDF page width
     */
    const addHeader = (pdf, pageWidth) => {
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(12);
      pdf.setTextColor(37, 99, 235); // Blue
      pdf.text("MKM Research Labs", pageWidth/2, 15, {align: 'center'});
    };
    
    /**
     * Add footer to PDF
     * @param {Object} pdf - jsPDF instance
     * @param {number} pageWidth - PDF page width
     * @param {number} pageHeight - PDF page height
     * @param {number} pageNumber - Current page number
     * @param {number} margin - Page margin
     */
    const addFooter = (pdf, pageWidth, pageHeight, pageNumber, margin) => {
      pdf.setFont('helvetica', 'normal');
      pdf.setFontSize(8);
      pdf.setTextColor(100, 100, 100);
      pdf.text(`Page ${pageNumber}`, pageWidth - margin, pageHeight - 10);
    };
    
    /**
     * Copy text to clipboard
     * @param {string} text - Text to copy
     * @returns {Promise} - Promise resolving when copy is complete
     */
    const copyToClipboard = async (text) => {
      try {
        await navigator.clipboard.writeText(text);
        UI.showNotification('Copied to clipboard!');
        return true;
      } catch (err) {
        console.error('Copy failed:', err);
        UI.showNotification('Failed to copy text');
        return false;
      }
    };
    
    // Public methods
    return {
      /**
       * Format chat for copying
       * @param {Array} messages - Chat messages
       * @returns {string} - Formatted text
       */
      formatChatForCopy: (messages) => {
        let formattedText = '';
        let lastRole = '';
  
        messages.forEach(message => {
          if (message.role === 'sources') {
            formattedText += `\n${message.content}\n\n`;
          } else {
            if (message.role !== lastRole) {
              const roleDisplay = message.role === 'user' ? 'User' : 'Assistant';
              formattedText += `\n${roleDisplay}:\n`;
              lastRole = message.role;
            }
            formattedText += `${message.content}\n\n`;
          }
        });
  
        return formattedText.trim();
      },
      
      /**
       * Export chat to PDF
       * @param {Object} chat - Chat object with messages
       */
      exportChatToPDF: async (chat) => {
        if (!chat || !jsPDF) {
          UI.showNotification('PDF generation is not available');
          return;
        }
        
        try {
          UI.showNotification('Generating PDF, please wait...');
          
          // Initialize PDF document
          const pdf = new jsPDF({
            orientation: 'portrait',
            unit: 'mm',
            format: 'a4',
            compress: true
          });
          
          // Set up the page dimensions and margins
          const pageWidth = pdf.internal.pageSize.getWidth();
          const pageHeight = pdf.internal.pageSize.getHeight();
          const margin = 20;
          const headerHeight = 15;
          const footerHeight = 10;
          
          let yPosition = headerHeight + margin;
          const lineHeight = 7;
          let pageNumber = 1;
          
          // Add title page elements
          pdf.setFont('helvetica', 'bold');
          pdf.setFontSize(18);
          pdf.setTextColor(37, 99, 235); // Blue
          pdf.text("Chat Export", pageWidth/2, yPosition, {align: 'center'});
          yPosition += lineHeight * 2;
          
          // Add date
          pdf.setFontSize(10);
          pdf.setTextColor(100, 100, 100);
          pdf.text(`Exported on ${new Date().toLocaleDateString()}`, pageWidth/2, yPosition, {align: 'center'});
          yPosition += lineHeight * 2;
          
          // Add first page header and footer
          addHeader(pdf, pageWidth);
          addFooter(pdf, pageWidth, pageHeight, pageNumber, margin);
          
          // Process each message
          for (const message of chat.messages) {
            // Handle sources differently
            if (message.role === 'sources') {
              // Add some space before sources
              yPosition += lineHeight;
              
              // Add sources header
              pdf.setFont('helvetica', 'bold');
              pdf.setFontSize(10);
              pdf.setTextColor(100, 100, 100);
              pdf.text('Sources:', margin, yPosition);
              yPosition += lineHeight;
              
              // Process sources
              pdf.setFont('helvetica', 'normal');
              pdf.setFontSize(8);
              const sourceLines = message.content.split('\n');
              
              for (let line of sourceLines) {
                if (line.trim() === 'Sources:') continue;
                
                // Check if we need a new page
                if (yPosition > pageHeight - footerHeight - margin) {
                  pdf.addPage();
                  pageNumber++;
                  yPosition = headerHeight + margin;
                  addHeader(pdf, pageWidth);
                  addFooter(pdf, pageWidth, pageHeight, pageNumber, margin);
                }
                
                const lines = pdf.splitTextToSize(line, pageWidth - (margin * 2));
                pdf.text(lines, margin, yPosition);
                yPosition += lines.length * 4; // Smaller line height for sources
              }
              
              // Add space after sources
              yPosition += lineHeight;
              
            } else if (message.role !== 'error') {
              // Check if we need a new page
              if (yPosition > pageHeight - footerHeight - margin - 20) {
                pdf.addPage();
                pageNumber++;
                yPosition = headerHeight + margin;
                addHeader(pdf, pageWidth);
                addFooter(pdf, pageWidth, pageHeight, pageNumber, margin);
              }
              
              // Add role header
              pdf.setFont('helvetica', 'bold');
              pdf.setFontSize(12);
              pdf.setTextColor(0, 0, 0);
              pdf.text(message.role === 'user' ? 'User:' : 'Assistant:', margin, yPosition);
              yPosition += lineHeight;
              
              // Add message content
              pdf.setFont('helvetica', 'normal');
              pdf.setFontSize(10);
              
              const messageLines = pdf.splitTextToSize(message.content, pageWidth - (margin * 2));
              
              // Check if message will fit on current page
              if (yPosition + (messageLines.length * 5) > pageHeight - footerHeight - margin) {
                // Calculate how many lines can fit on this page
                const linesPerPage = Math.floor((pageHeight - footerHeight - margin - yPosition) / 5);
                const firstPageLines = messageLines.slice(0, linesPerPage);
                const remainingLines = messageLines.slice(linesPerPage);
                
                // Add lines that fit on current page
                if (firstPageLines.length > 0) {
                  pdf.text(firstPageLines, margin, yPosition);
                }
                
                // Add new pages as needed for remaining content
                let currentLines = remainingLines;
                while (currentLines.length > 0) {
                  pdf.addPage();
                  pageNumber++;
                  yPosition = headerHeight + margin;
                  
                  // Add header and footer to new page
                  addHeader(pdf, pageWidth);
                  addFooter(pdf, pageWidth, pageHeight, pageNumber, margin);
                  
                  // Add continuation note
                  pdf.setFont('helvetica', 'bold');
                  pdf.setFontSize(12);
                  pdf.text(message.role === 'user' ? 'User (continued):' : 'Assistant (continued):', margin, yPosition);
                  yPosition += lineHeight;
                  
                  // Add chunk of content
                  pdf.setFont('helvetica', 'normal');
                  pdf.setFontSize(10);
                  
                  const maxLinesPerPage = Math.floor((pageHeight - footerHeight - margin - yPosition) / 5);
                  const pagePortion = currentLines.slice(0, maxLinesPerPage);
                  currentLines = currentLines.slice(maxLinesPerPage);
                  
                  pdf.text(pagePortion, margin, yPosition);
                  yPosition += pagePortion.length * 5;
                }
              } else {
                // If message fits on current page, add it directly
                pdf.text(messageLines, margin, yPosition);
                yPosition += messageLines.length * 5;
              }
              
              // Add spacing after each message
              yPosition += lineHeight;
            }
          }
          
          // Generate filename based on first user message or timestamp
          let filename = 'chat-export.pdf';
          for (const message of chat.messages) {
            if (message.role === 'user') {
              const truncatedTitle = message.content.substring(0, 30).replace(/[^a-z0-9]/gi, '_');
              filename = `chat-${truncatedTitle}.pdf`;
              break;
            }
          }
          
          pdf.save(filename);
          UI.showNotification('PDF exported successfully!');
        } catch (error) {
          console.error('PDF generation error:', error);
          UI.showNotification('Failed to generate PDF: ' + error.message);
        }
      },
      
      /**
       * Export document summary to PDF
       * @param {string} docName - Document name
       * @param {string} summary - Document summary text
       */
      exportDocumentToPDF: async (docName, summary) => {
        if (!docName || !summary || !jsPDF) {
          UI.showNotification('PDF generation is not available or missing content');
          return;
        }
        
        try {
          UI.showNotification('Generating PDF, please wait...');
          
          // Initialize PDF document
          const pdf = new jsPDF({
            orientation: 'portrait',
            unit: 'mm',
            format: 'a4',
            compress: true
          });
          
          // Set up the page dimensions and margins
          const pageWidth = pdf.internal.pageSize.getWidth();
          const pageHeight = pdf.internal.pageSize.getHeight();
          const margin = 20;
          const headerHeight = 15;
          const footerHeight = 10;
          
          let yPosition = headerHeight + margin;
          const lineHeight = 7;
          let pageNumber = 1;
          
          // Add title page elements
          pdf.setFont('helvetica', 'bold');
          pdf.setFontSize(18);
          pdf.setTextColor(37, 99, 235); // Blue
          pdf.text("Document Summary", pageWidth/2, yPosition, {align: 'center'});
          yPosition += lineHeight * 2;
          
          // Add document name
          pdf.setFontSize(14);
          pdf.setTextColor(0, 0, 0);
          pdf.text(docName, pageWidth/2, yPosition, {align: 'center'});
          yPosition += lineHeight * 2;
          
          // Add date
          pdf.setFontSize(10);
          pdf.setTextColor(100, 100, 100);
          pdf.text(`Exported on ${new Date().toLocaleDateString()}`, pageWidth/2, yPosition, {align: 'center'});
          yPosition += lineHeight * 3;
          
          // Add first page header and footer
          addHeader(pdf, pageWidth);
          addFooter(pdf, pageWidth, pageHeight, pageNumber, margin);
          
          // Add summary title
          pdf.setFont('helvetica', 'bold');
          pdf.setFontSize(12);
          pdf.setTextColor(0, 0, 0);
          pdf.text("Summary:", margin, yPosition);
          yPosition += lineHeight * 1.5;
          
          // Add summary content across pages
          pdf.setFont('helvetica', 'normal');
          pdf.setFontSize(10);
          
          const summaryLines = pdf.splitTextToSize(summary, pageWidth - (margin * 2));
          
          let currentIndex = 0;
          while (currentIndex < summaryLines.length) {
            // Calculate how many lines can fit on this page
            const maxLinesPerPage = Math.floor((pageHeight - footerHeight - margin - yPosition) / 5);
            const pageLines = summaryLines.slice(currentIndex, currentIndex + maxLinesPerPage);
            currentIndex += maxLinesPerPage;
            
            // Add the lines that fit on this page
            pdf.text(pageLines, margin, yPosition);
            yPosition += pageLines.length * 5;
            
            // If we have more lines, add a new page
            if (currentIndex < summaryLines.length) {
              pdf.addPage();
              pageNumber++;
              yPosition = headerHeight + margin;
              addHeader(pdf, pageWidth);
              addFooter(pdf, pageWidth, pageHeight, pageNumber, margin);
              
              // Add continuation note on new page
              pdf.setFont('helvetica', 'italic');
              pdf.setFontSize(10);
              pdf.text("(continued)", margin, yPosition);
              yPosition += lineHeight * 1.5;
              pdf.setFont('helvetica', 'normal');
            }
          }
          
          // Generate filename based on document name
          const filename = `doc-summary-${docName.substring(0, 30).replace(/[^a-z0-9]/gi, '_')}.pdf`;
          
          pdf.save(filename);
          UI.showNotification('Document summary exported as PDF successfully!');
        } catch (error) {
          console.error('PDF generation error:', error);
          UI.showNotification('Failed to generate PDF: ' + error.message);
        }
      },
      
      /**
       * Copy text to clipboard
       */
      copyToClipboard
    };
  })();