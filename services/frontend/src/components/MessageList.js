import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ClassificationCard from './ClassificationCard';

const MessageList = ({ messages }) => {
  const endOfMessagesRef = useRef(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🐦</div>
        <div className="empty-title">BirdAI</div>
        <div className="empty-sub">Upload a bird photo or send a message</div>
      </div>
    );
  }

  return (
    <>
      {messages.map((msg, idx) => (
        <div key={idx} className={`msg-wrap ${msg.role === 'user' ? 'user' : ''}`}>
          {msg.role === 'user' ? (
            <div className="msg-row user">
              <div className="usr-stack">
                {msg.imageUrl && (
                  <img src={msg.imageUrl} alt="uploaded" className="msg-image" />
                )}
                {msg.content && (
                  <div className="bubble-usr">
                    <span>{msg.content}</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <>
              {msg.kind === 'classification' ? (
                <div className="msg-row">
                  <div className="bubble-bot">
                    Here's what I found:
                    <ClassificationCard prediction={msg.prediction} />
                  </div>
                </div>
              ) : (
                <div className="msg-row">
                  <div className="bubble-bot">
                    {msg.streaming && !msg.content ? (
                      <div className="loading-dots">
                        <span /><span /><span />
                      </div>
                    ) : (
                      <>
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            a: ({ href, children }) => (
                              <a href={href} target="_blank" rel="noopener noreferrer">
                                {children}
                              </a>
                            ),
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                        {msg.streaming && <span className="cursor-blink">▋</span>}
                      </>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      ))}
      <div ref={endOfMessagesRef} />
    </>
  );
};

export default MessageList;
