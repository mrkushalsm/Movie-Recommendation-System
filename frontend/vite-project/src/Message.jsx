import React from 'react';


function Message({ text, isUser }) {
  const messageStyle = {
    backgroundColor: isUser ? '#DCF8C6' : '#EAEAEA',
    alignSelf: isUser ? 'flex-end' : 'flex-start',
    borderRadius: '10px',
    padding: '8px 12px',
    margin: '5px',
    maxWidth: '70%',
    wordWrap: 'break-word',
  };

  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
      <div style={messageStyle}>
        {text}
      </div>
    </div>
  );
}

export default Message;