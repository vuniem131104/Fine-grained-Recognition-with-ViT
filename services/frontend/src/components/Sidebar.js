import React from 'react';
import { useAuth } from '../context/AuthContext';

const Sidebar = ({ onNewChat }) => {
  const { user, logout } = useAuth();

  return (
    <div className="sidebar">
      <div className="sidebar-logo">🐦</div>
      <div className="sidebar-title">BirdAI</div>
      <div className="sidebar-tagline">Bird species identification powered by AI</div>

      <button className="new-chat-btn" onClick={onNewChat}>
        <span className="new-chat-icon">＋</span>
        New Chat
      </button>

      <div className="sidebar-divider"></div>

      <div className="sidebar-section">How to use</div>
      <div className="sidebar-tip">
        <div className="tip-icon">+</div>
        <div className="tip-text">Tap <b>+</b> to attach a bird photo from your device</div>
      </div>
      <div className="sidebar-tip">
        <div className="tip-icon">↑</div>
        <div className="tip-text">Press <b>Enter</b> or the send button to submit</div>
      </div>
      <div className="sidebar-tip">
        <div className="tip-icon">🔍</div>
        <div className="tip-text">Get the top species match plus 3 alternatives with confidence scores</div>
      </div>
      <div className="sidebar-tip">
        <div className="tip-icon">💬</div>
        <div className="tip-text">Ask questions about birds in the chat</div>
      </div>

      {user && (
        <div className="sidebar-user">
          <div className="sidebar-user-info">
            <div className="sidebar-user-name">{user.full_name}</div>
            <div className="sidebar-user-email">{user.email}</div>
          </div>
          <button className="sidebar-logout-btn" onClick={logout} title="Sign out">
            ↩
          </button>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
