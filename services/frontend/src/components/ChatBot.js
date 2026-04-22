import React, { useState, useRef, useEffect } from 'react';
import { sendChatQuery, checkChatbotHealth } from '../api';
import './ChatBot.css';

/**
 * ChatBot component for interacting with the chatbot service
 */
const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isServiceReady, setIsServiceReady] = useState(false);
  const messagesEndRef = useRef(null);

  // Check chatbot service health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await checkChatbotHealth();
        setIsServiceReady(true);
      } catch (err) {
        setIsServiceReady(false);
        setError('Chatbot service is not available');
      }
    };

    checkHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!query.trim()) {
      return;
    }

    if (!isServiceReady) {
      setError('Chatbot service is not available. Please try again later.');
      return;
    }

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      text: query,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setLoading(true);
    setError(null);

    try {
      // Send query to chatbot API
      const response = await sendChatQuery(query);

      // Add bot response to chat
      const botMessage = {
        id: Date.now() + 1,
        text: response.response,
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      setError(
        err.response?.data?.detail || 'Failed to get response from chatbot'
      );
      console.error('Chat error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h2>Bird Chatbot</h2>
        <div className="status">
          <span
            className={`status-indicator ${isServiceReady ? 'ready' : 'unavailable'}`}
          ></span>
          <span>
            {isServiceReady ? 'Service Ready' : 'Service Unavailable'}
          </span>
        </div>
      </div>

      <div className="chatbot-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask me anything about birds!</p>
            <p className="hint">
              E.g., "What species is this bird?" or "Tell me about common
              songbirds"
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.sender}-message`}
          >
            <div className="message-content">
              <p>{message.text}</p>
            </div>
            <div className="message-time">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message bot-message loading">
            <div className="spinner"></div>
            <p>Thinking...</p>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {error && (
        <div className="error-message">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <form onSubmit={handleSubmit} className="chatbot-form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about birds..."
          disabled={loading || !isServiceReady}
          className="chat-input"
        />
        <button
          type="submit"
          disabled={loading || !isServiceReady || !query.trim()}
          className="chat-button"
        >
          {loading ? 'Sending...' : 'Send'}
        </button>
        {messages.length > 0 && (
          <button
            type="button"
            onClick={handleClearChat}
            className="clear-button"
            title="Clear chat history"
          >
            Clear
          </button>
        )}
      </form>
    </div>
  );
};

export default ChatBot;
