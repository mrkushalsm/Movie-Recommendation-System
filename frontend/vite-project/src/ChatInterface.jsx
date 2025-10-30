import React, { useState } from 'react';
import Message from './Message.jsx'; 

const sendMessageToBackend = async (messageText) => {
  console.log(`Sending message: "${messageText}" to backend...`);
 
  await new Promise(resolve => setTimeout(resolve, 500)); 
  return { text: `Echo: ${messageText}`, isUser: false };
};

function ChatInterface() {
  const [messages, setMessages] = useState([
    { text: 'Hello!I am the helpful assisatant!', isUser: false },
    { text: 'Ask related to movie..', isUser: true },
  ]);
  
  const [inputText, setInputText] = useState('');

 
  const handleSend = async (e) => {
    
    e.preventDefault(); 
    if (!inputText.trim()) return; 

    const userMessage = { text: inputText, isUser: true };
    
    
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    const messageToEcho = inputText; 
    setInputText(''); 
    

    const response = await sendMessageToBackend(messageToEcho);
    
    
    setMessages((prevMessages) => [...prevMessages, response]);
  };

  return (
    <div style={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh', 
      maxWidth: '600px', 
      margin: '0 auto', 
      border: '1px solid #ccc' 
    }}>
      
      
      <div style={{ flexGrow: 1, overflowY: 'auto', padding: '10px' }}>
        {messages.map((msg, index) => (
          
          <Message key={index} text={msg.text} isUser={msg.isUser} />
        ))}
      </div>

      
      <form onSubmit={handleSend} style={{ display: 'flex', padding: '10px', borderTop: '1px solid #ccc' }}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type a message..."
          style={{ flexGrow: 1, padding: '10px', borderRadius: '5px', border: '1px solid #ddd' }}
        />
        <button type="submit" style={{ padding: '10px 15px', marginLeft: '10px', borderRadius: '5px', border: 'none', backgroundColor: '#007bff', color: 'white', cursor: 'pointer' }}>
          Send
        </button>
      </form>
    </div>
  );
}

export default ChatInterface;