const newChatBtn = document.querySelector('.new-chat-btn');
const chatHistory = document.querySelector('.sidebar_content');
const chatContainer = document.querySelector('.chat-container');


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


newChatBtn.addEventListener('click', createNewChat);




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