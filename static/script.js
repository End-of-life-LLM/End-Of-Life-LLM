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

// Initialize application
function initialize() {
    // Load chat history
    fetchChats();

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


let isToggled = false;

const toggleBtn = document.getElementById("toggle-button");

toggleBtn.addEventListener("click", function () {
    isToggled = !isToggled;
    toggleBtn.textContent = isToggled ? "On" : "Off";
    console.log("Toggle state:", isToggled);
});

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

    chats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.classList.add('chat-item');

        if (chat.id === currentSessionId) {
            chatItem.classList.add('active');
        }

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
    fetch(`/select_chat/${id}`)
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

// Render messages in chat container
function renderMessages(messages) {
    chatContainer.innerHTML = '';

    if (!messages || messages.length === 0) {
        const welcomeMessage = document.createElement('div');
        welcomeMessage.classList.add('message', 'bot-message');
        welcomeMessage.textContent = 'Hello! How can I help you today?';
        chatContainer.appendChild(welcomeMessage);
        return;
    }

    messages.forEach(message => {
        const messageEl = document.createElement('div');
        messageEl.classList.add('message');

        if (message.role === 'user') {
            messageEl.classList.add('user-message');
        } else {
            messageEl.classList.add('bot-message');
        }

        messageEl.textContent = message.content;
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

                // Clear chat container
                chatContainer.innerHTML = '';
                const welcomeMessage = document.createElement('div');
                welcomeMessage.classList.add('message', 'bot-message');
                welcomeMessage.textContent = 'Hello! How can I help you today?';
                chatContainer.appendChild(welcomeMessage);

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

// Send a message
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

    // Show loading state
    const botMessageEl = document.createElement('div');
    botMessageEl.classList.add('message', 'bot-message');
    botMessageEl.textContent = 'Thinking...';
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
            finalBotMessageEl.textContent = data.response || data.error || "Sorry, I couldn't process your request.";
            chatContainer.appendChild(finalBotMessageEl);

            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
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
    const tempMessageEl = document.createElement('div');
    tempMessageEl.classList.add('message', 'bot-message', 'temp-message');
    tempMessageEl.textContent = `${files.length} file(s) selected. Go to settings to upload them, or just drag and drop into the chat to send.`;
    chatContainer.appendChild(tempMessageEl);

    // Scroll to the new message
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Remove the message after a few seconds
    setTimeout(() => {
        tempMessageEl.remove();
    }, 5000);
}