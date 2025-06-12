document.addEventListener('DOMContentLoaded', initialize);

// DOM Elements
const newChatBtn = document.querySelector('.new-chat-btn');
const chatHistory = document.querySelector('.sidebar_content');
const chatContainer = document.querySelector('.chat-container');
const messageInput = document.querySelector('.message-input');
const toggleButtons = document.querySelectorAll('.sidebar_toggle_button');
const fileInput = document.getElementById('file-upload');

// Event Listeners
newChatBtn.addEventListener('click', createNewChat);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

toggleButtons.forEach((btn) => {
    btn.addEventListener('click', toggleSidebar);
});

fileInput.addEventListener('change', handleFileSelection);

// Global state
let currentSessionId = null;
let chats = [];
let ragEnabled = false;
let systemInfo = {};

// Initialize application
function initialize() {
    // Load chat history
    fetchChats();
    
    // Load system info and RAG state
    loadSystemInfo();

    // Set initial sidebar state based on screen size
    if (window.innerWidth <= 768) {
        document.querySelector('.sidebar').style.display = 'none';
        document.querySelector('.sidebar-closed').style.display = 'flex';
    }

    // Listen for window resize events
    window.addEventListener('resize', () => {
        if (window.innerWidth <= 768) {
            document.querySelector('.sidebar').style.display = 'none';
            document.querySelector('.sidebar-closed').style.display = 'flex';
        }
    });
}

// Navigate to Settings
document.getElementById("settings-button").addEventListener("click", function () {
    window.location.href = '/settings';
});

// Load system information and update UI
function loadSystemInfo() {
    fetch('/system_info')
        .then(response => response.json())
        .then(data => {
            systemInfo = data;
            ragEnabled = data.rag_enabled;
            updateToggleButton();
            updateChatStatus();
        })
        .catch(error => {
            console.error('Error loading system info:', error);
        });
}

// Update toggle button text and state
function updateToggleButton() {
    const toggleBtn = document.getElementById("toggle-button");
    if (toggleBtn) {
        toggleBtn.textContent = ragEnabled ? "RAG: ON" : "RAG: OFF";
        toggleBtn.style.backgroundColor = ragEnabled ? "#4CAF50" : "#f44336";
        toggleBtn.title = ragEnabled ? 
            "RAG is enabled - using indexed documents" : 
            "RAG is disabled - using only LLM knowledge";
    }
}

// Update chat status indicator
function updateChatStatus() {
    // Create or update status indicator
    let statusIndicator = document.getElementById('chat-status');
    if (!statusIndicator) {
        statusIndicator = document.createElement('div');
        statusIndicator.id = 'chat-status';
        statusIndicator.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            z-index: 1000;
            transition: all 0.3s ease;
        `;
        document.body.appendChild(statusIndicator);
    }
    
    if (ragEnabled && systemInfo.rag_index_loaded) {
        const docCount = systemInfo.index_stats?.document_count || 0;
        statusIndicator.textContent = `RAG Active (${docCount} docs)`;
        statusIndicator.style.backgroundColor = '#4CAF50';
        statusIndicator.style.color = 'white';
        statusIndicator.style.display = 'block';
    } else if (ragEnabled && !systemInfo.rag_index_loaded) {
        statusIndicator.textContent = 'RAG Enabled (No docs)';
        statusIndicator.style.backgroundColor = '#ff9800';
        statusIndicator.style.color = 'white';
        statusIndicator.style.display = 'block';
    } else {
        statusIndicator.style.display = 'none';
    }
}

// RAG Toggle functionality
const toggleBtn = document.getElementById("toggle-button");
if (toggleBtn) {
    toggleBtn.addEventListener("click", async function () {
        try {
            const response = await fetch('/toggle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            ragEnabled = data.state;
            updateToggleButton();
            updateChatStatus();
            
            // Show user feedback
            showTemporaryMessage(
                ragEnabled ? 
                "RAG enabled - now using your indexed documents" : 
                "RAG disabled - using only LLM knowledge",
                ragEnabled ? "success" : "info"
            );
            
            console.log("RAG toggle state:", ragEnabled);
        } catch (error) {
            console.error('Error toggling RAG:', error);
            showTemporaryMessage("Error toggling RAG system", "error");
        }
    });
}

// Show temporary message to user
function showTemporaryMessage(message, type = "info") {
    const messageEl = document.createElement('div');
    messageEl.classList.add('message', 'bot-message', 'temp-message');
    
    // Style based on type
    if (type === "success") {
        messageEl.style.backgroundColor = '#d4edda';
        messageEl.style.color = '#155724';
        messageEl.style.border = '1px solid #c3e6cb';
    } else if (type === "error") {
        messageEl.style.backgroundColor = '#f8d7da';
        messageEl.style.color = '#721c24';
        messageEl.style.border = '1px solid #f5c6cb';
    } else {
        messageEl.style.backgroundColor = '#d1ecf1';
        messageEl.style.color = '#0c5460';
        messageEl.style.border = '1px solid #bee5eb';
    }
    
    messageEl.textContent = message;
    chatContainer.appendChild(messageEl);
    
    // Scroll to the new message
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    // Remove the message after a few seconds
    setTimeout(() => {
        if (messageEl.parentNode) {
            messageEl.remove();
        }
    }, 4000);
}

// Fetch chat sessions from server
function fetchChats() {
    fetch('/get_chats')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                chats = data.chats;
                renderChatHistory();

                // If there's no active chat, create one
                if (chats.length === 0) {
                    createNewChat();
                } else {
                    // Select the most recent chat
                    selectChat(chats[0].id);
                }
            }
        })
        .catch(error => {
            console.error('Error fetching chats:', error);
        });
}

// Render chat history in sidebar
function renderChatHistory() {
    chatHistory.innerHTML = '';

    if (chats.length === 0) {
        const emptyState = document.createElement('div');
        emptyState.className = 'empty-chat-state';
        emptyState.textContent = 'No chats yet';
        chatHistory.appendChild(emptyState);
        return;
    }

    console.log("Rendering chats:", chats);
    
    chats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.classList.add('chat-item');

        if (chat.id === currentSessionId) {
            chatItem.classList.add('active');
        }

        // Use preview or a default text
        chatItem.textContent = chat.preview || 'New Chat';
        chatItem.dataset.id = chat.id;
        chatItem.addEventListener('click', () => selectChat(chat.id));
        chatHistory.appendChild(chatItem);
    });
}

// Select a chat session
function selectChat(id) {
    currentSessionId = id;

    // Update active state in UI
    document.querySelectorAll('.chat-item').forEach(item => {
        item.classList.toggle('active', item.dataset.id === id);
    });

    // Fetch messages for this chat
    fetch('/get_chat_history')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderMessages(data.messages);
            }
        })
        .catch(error => {
            console.error('Error selecting chat:', error);
        });
}

// Render messages in chat container with RAG indicators
function renderMessages(messages) {
    chatContainer.innerHTML = '';

    if (!messages || messages.length === 0) {
        const welcomeMessage = document.createElement('div');
        welcomeMessage.classList.add('message', 'bot-message');
        
        // Show different welcome message based on RAG status
        if (ragEnabled && systemInfo.rag_index_loaded) {
            const docCount = systemInfo.index_stats?.document_count || 0;
            welcomeMessage.innerHTML = `
                <strong>Hello! How can I help you today?</strong><br>
                <small style="color: #666; font-size: 0.9em;">
                    💡 RAG is active with ${docCount} indexed documents. I can answer questions using your uploaded content.
                </small>
            `;
        } else if (ragEnabled && !systemInfo.rag_index_loaded) {
            welcomeMessage.innerHTML = `
                <strong>Hello! How can I help you today?</strong><br>
                <small style="color: #666; font-size: 0.9em;">
                    ℹ️ RAG is enabled but no documents are indexed yet. Upload documents in Settings to enhance my capabilities.
                </small>
            `;
        } else {
            welcomeMessage.textContent = 'Hello! How can I help you today?';
        }
        
        chatContainer.appendChild(welcomeMessage);
        return;
    }

    messages.forEach(message => {
        const messageEl = document.createElement('div');
        messageEl.classList.add('message');

        if (message.role === 'user') {
            messageEl.classList.add('user-message');
            messageEl.textContent = message.content;
        } else {
            messageEl.classList.add('bot-message');
            
            // Check if message contains source information (RAG was used)
            const content = message.content;
            if (content.includes('Sources:') || content.includes('Relevance Summary:')) {
                // This message used RAG - add an indicator
                const ragIndicator = document.createElement('div');
                ragIndicator.style.cssText = `
                    display: inline-block;
                    background: #4CAF50;
                    color: white;
                    font-size: 10px;
                    padding: 2px 6px;
                    border-radius: 3px;
                    margin-bottom: 5px;
                `;
                ragIndicator.textContent = '📚 RAG';
                ragIndicator.title = 'This response used your indexed documents';
                
                messageEl.appendChild(ragIndicator);
                messageEl.innerHTML += '<br>' + content.replace(/\n/g, '<br>');
            } else {
                messageEl.textContent = content;
            }
        }

        chatContainer.appendChild(messageEl);
    });

    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Create a new chat session
function createNewChat() {
    fetch('/new_chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Refresh chat list
                fetchChats();

                // Clear chat container and show welcome message
                renderMessages([]);

                // Set current session ID
                currentSessionId = data.session_id;

                // Clear input
                messageInput.value = '';
            }
        })
        .catch(error => {
            console.error('Error creating new chat:', error);
        });
}

// Send a message with RAG awareness
function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // Add user message to chat
    const userMessageEl = document.createElement('div');
    userMessageEl.classList.add('message', 'user-message');
    userMessageEl.textContent = message;
    chatContainer.appendChild(userMessageEl);

    // Clear input
    messageInput.value = '';

    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Show loading state with RAG indicator
    const botMessageEl = document.createElement('div');
    botMessageEl.classList.add('message', 'bot-message');
    
    if (ragEnabled && systemInfo.rag_index_loaded) {
        botMessageEl.innerHTML = '🔍 Searching documents and thinking...';
    } else if (ragEnabled && !systemInfo.rag_index_loaded) {
        botMessageEl.innerHTML = '🤔 Thinking... <small>(No documents to search)</small>';
    } else {
        botMessageEl.textContent = 'Thinking...';
    }
    
    chatContainer.appendChild(botMessageEl);

    // Send message to server
    fetch('/send_message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            botMessageEl.remove();

            // Display response from server
            const finalBotMessageEl = document.createElement('div');
            finalBotMessageEl.classList.add('message', 'bot-message');
            
            const responseText = data.response || data.error || "Sorry, I couldn't process your request.";
            
            // Check if RAG was used and add indicator
            if (responseText.includes('Sources:') || responseText.includes('Relevance Summary:')) {
                const ragIndicator = document.createElement('div');
                ragIndicator.style.cssText = `
                    display: inline-block;
                    background: #4CAF50;
                    color: white;
                    font-size: 10px;
                    padding: 2px 6px;
                    border-radius: 3px;
                    margin-bottom: 5px;
                `;
                ragIndicator.textContent = '📚 RAG';
                ragIndicator.title = 'This response used your indexed documents';
                
                finalBotMessageEl.appendChild(ragIndicator);
                finalBotMessageEl.innerHTML += '<br>' + responseText.replace(/\n/g, '<br>');
            } else {
                finalBotMessageEl.textContent = responseText;
            }
            
            chatContainer.appendChild(finalBotMessageEl);

            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Refresh system info (in case RAG state changed)
            loadSystemInfo();
        })
        .catch(error => {
            console.error('Error sending message:', error);

            // Remove loading message
            botMessageEl.remove();

            // Show error message
            const errorMessageEl = document.createElement('div');
            errorMessageEl.classList.add('message', 'bot-message', 'error-message');
            errorMessageEl.textContent = "Sorry, there was an error processing your message.";
            chatContainer.appendChild(errorMessageEl);

            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
}

// Toggle sidebar visibility
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const sidebarClosed = document.querySelector('.sidebar-closed');
    const mainContent = document.querySelector('.main-content');

    if (sidebar.style.display === 'none' || sidebar.style.display === '') {
        sidebar.style.display = 'flex';
        sidebarClosed.style.display = 'none';
        mainContent.style.marginLeft = 'var(--sidebar-width)';
    } else {
        sidebar.style.display = 'none';
        sidebarClosed.style.display = 'flex';
        mainContent.style.marginLeft = '0';
    }
}

// Handle file selection
function handleFileSelection(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    // Show selected files count in a temporary message
    showTemporaryMessage(
        `${files.length} file(s) selected. Go to Settings to upload and index them for RAG.`,
        "info"
    );
}

// Periodically refresh system info to keep RAG status current
setInterval(loadSystemInfo, 30000); // Every 30 seconds