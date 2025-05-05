const newChatBtn = document.querySelector('.new-chat-btn');
const chatHistory = document.querySelector('.sidebar_content');
const chatContainer = document.querySelector('.chat-container');
const messageInput = document.querySelector('.message-input');
const toggleButtons = document.querySelectorAll('.sidebar_toggle_button');

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

let chats = [
    { id: 1, title: '', active: true }
];



// Render chat history
function renderChatHistory() {
    chatHistory.innerHTML = '';
    chats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.classList.add('chat-item');
        if (chat.active) {
            chatItem.classList.add('active');
        }
        const firstSentence = chatContainer.textContent.split('.')[0] + '.';
        chatItem.textContent = firstSentence;
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

// JavaScript Function to Send Message
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
    
    // Send message to Python backend
    fetch('/send_message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        // Display response from Python
        const botMessageEl = document.createElement('div');
        botMessageEl.classList.add('message', 'bot-message');
        botMessageEl.textContent = data.response;
        chatContainer.appendChild(botMessageEl);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    })
    .catch(error => {
        console.error('Error:', error);
        // Show error message
        const errorMessageEl = document.createElement('div');
        errorMessageEl.classList.add('message', 'error-message');
        errorMessageEl.textContent = "Sorry, there was an error processing your message.";
        chatContainer.appendChild(errorMessageEl);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}


function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const sidebarClosed = document.querySelector('.sidebar-closed');

    const sidebarStyle = window.getComputedStyle(sidebar);

    if (sidebarStyle.display === 'flex') {
        sidebar.style.display = 'none';
        sidebarClosed.style.display = 'flex';
    } else {
        sidebar.style.display = 'flex';
        sidebarClosed.style.display = 'none';
    }
}



// Initial setup
function initialize() {
    renderChatHistory();
    
  /*  // Set initial sidebar state based on screen size
    if (window.innerWidth <= 768) {
        toggleSidebar(false);
    } else {
        toggleSidebar(true);
    }*/
    
    /*
    // Listen for window resize events
    window.addEventListener('resize', () => {
        if (window.innerWidth <= 768) {
            toggleSidebar(false);
        } else {
            toggleSidebar(true);
        }
    });*/
}


document.addEventListener('DOMContentLoaded', initialize);