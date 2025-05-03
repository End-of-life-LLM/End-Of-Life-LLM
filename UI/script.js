// DOM Elements
const newChatBtn = document.querySelector('.new-chat-btn');
const chatHistory = document.querySelector('.chat-history');
const chatContainer = document.querySelector('.chat-container');
const messageInput = document.querySelector('.message-input');
const toggleSidebarBtn = document.querySelector('.toggle-sidebar');
const sidebar = document.querySelector('.sidebar');
const mainContent = document.querySelector('.main-content');

// Sample chat history data
let chats = [
    { id: 1, title: 'Chat about Python', active: true },
    { id: 2, title: 'Web development help' },
    { id: 3, title: 'AI model fine-tuning' }
];

// Sidebar state management
let sidebarVisible = true;

// Render chat history
function renderChatHistory() {
    chatHistory.innerHTML = '';
    chats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.classList.add('chat-item');
        if (chat.active) {
            chatItem.classList.add('active');
        }
        chatItem.textContent = chat.title;
        chatItem.dataset.id = chat.id;
        chatItem.addEventListener('click', () => selectChat(chat.id));
        chatHistory.appendChild(chatItem);
    });
}

// Select chat
function selectChat(id) {
    chats = chats.map(chat => ({
        ...chat,
        active: chat.id === id
    }));
    renderChatHistory();
    
    // Here you would load chat messages for the selected chat
    // This is a placeholder
    chatContainer.innerHTML = '<div class="message bot-message">Hello! How can I help you today?</div>';
    
    // On mobile, close sidebar after selecting a chat
    if (window.innerWidth <= 768) {
        toggleSidebar(false);
    }
}

// Send message
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
    
    // In a real app, you would send this message to your backend
    // and then display the response from the AI
    setTimeout(() => {
        // Simulate AI response
        const botMessageEl = document.createElement('div');
        botMessageEl.classList.add('message', 'bot-message');
        botMessageEl.textContent = "This is a placeholder response. In your actual implementation, this would be the response from the ChatGPT API.";
        chatContainer.appendChild(botMessageEl);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 1000);
}

// Create new chat
function createNewChat() {
    const newId = chats.length > 0 ? Math.max(...chats.map(c => c.id)) + 1 : 1;
    const newChat = {
        id: newId,
        title: `New Chat ${newId}`,
        active: true
    };
    
    chats = chats.map(chat => ({
        ...chat,
        active: false
    }));
    
    chats.unshift(newChat);
    renderChatHistory();
    
    // Clear chat container
    chatContainer.innerHTML = '<div class="message bot-message">Hello! How can I help you today?</div>';
}

// Toggle sidebar function
function toggleSidebar(forceState) {
    // If forceState is provided, use it, otherwise toggle current state
    sidebarVisible = forceState !== undefined ? forceState : !sidebarVisible;
    
    if (sidebarVisible) {
        sidebar.classList.remove('hidden');
        sidebar.classList.add('visible');
        //if (window.innerWidth > 768) {
          //  mainContent.classList.remove('full-width');
        //}
    } else {
        sidebar.classList.add('hidden');
        sidebar.classList.remove('visible');
      // if (window.innerWidth > 768) {
        //    mainContent.classList.add('full-width');
        //}
    }
}

// Event Listeners
newChatBtn.addEventListener('click', createNewChat);

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

toggleSidebarBtn.addEventListener('click', () => toggleSidebar());


// Initial setup
function initialize() {
    renderChatHistory();
    
    // Set initial sidebar state based on screen size
    if (window.innerWidth <= 768) {
        toggleSidebar(false);
    } else {
        toggleSidebar(true);
    }
    
    
    // Listen for window resize events
    window.addEventListener('resize', () => {
        if (window.innerWidth <= 768) {
            toggleSidebar(false);
        } else {
            toggleSidebar(true);
        }
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initialize);