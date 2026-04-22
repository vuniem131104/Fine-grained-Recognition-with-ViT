import { useState, useRef } from 'react';
import './index.css';
import Sidebar from './components/Sidebar';
import MessageList from './components/MessageList';
import InputArea from './components/InputArea';
import LoginPage from './components/LoginPage';
import RegisterPage from './components/RegisterPage';
import { AuthProvider, useAuth } from './context/AuthContext';
import { sendChatQuery } from './api';

const toBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

const generateUUID = () => crypto.randomUUID();

function ChatApp() {
  const { user } = useAuth();
  const [messages, setMessages] = useState([]);
  const conversationId = useRef(generateUUID());

  const handleSubmitMessage = async (file, text) => {
    const userMessage = { role: 'user', kind: file ? 'image' : 'text' };
    if (file) {
      userMessage.imageUrl = URL.createObjectURL(file);
      userMessage.filename = file.name;
      if (text) userMessage.content = text;
    } else {
      userMessage.content = text;
    }

    setMessages(prev => [
      ...prev,
      userMessage,
      { role: 'assistant', kind: 'text', content: '', streaming: true },
    ]);

    try {
      const imgB64 = file ? await toBase64(file) : null;
      await sendChatQuery(
        text,
        imgB64,
        (chunk) => {
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.streaming) {
              updated[updated.length - 1] = { ...last, content: last.content + chunk };
            }
            return updated;
          });
        },
        user?.id ?? null,
        user ? conversationId.current : null,
      );

      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.streaming) {
          updated[updated.length - 1] = { ...last, streaming: false };
        }
        return updated;
      });
    } catch (error) {
      console.error('Chat failed:', error);
      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.streaming) {
          updated[updated.length - 1] = {
            ...last,
            content: `Error: ${error.message || 'Failed to get response'}`,
            streaming: false,
          };
        }
        return updated;
      });
    }
  };

  const handleClear = () => {
    setMessages([]);
    conversationId.current = generateUUID();
  };

  return (
    <div className="container">
      <Sidebar onNewChat={handleClear} />
      <div className="main-content">
        <div className="messages-container">
          <MessageList messages={messages} />
        </div>
        <InputArea onSubmitMessage={handleSubmitMessage} />
      </div>
    </div>
  );
}

function AppRoutes() {
  const { user } = useAuth();
  const [page, setPage] = useState('login');

  if (user) return <ChatApp />;

  if (page === 'register') {
    return <RegisterPage onNavigateLogin={() => setPage('login')} />;
  }
  return <LoginPage onNavigateRegister={() => setPage('register')} />;
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}
