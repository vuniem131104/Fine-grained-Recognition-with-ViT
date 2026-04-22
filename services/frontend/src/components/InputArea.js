import React, { useState, useRef } from 'react';

const InputArea = ({ onSubmitMessage }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [textInput, setTextInput] = useState('');
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!selectedFile && !textInput.trim()) return;

    onSubmitMessage(selectedFile || null, textInput.trim());
    setSelectedFile(null);
    setTextInput('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="input-area">
      {selectedFile && (
        <div style={{ marginBottom: '8px' }}>
          <div className="file-preview">
            <span>{selectedFile.name}</span>
            <button
              className="file-preview-close"
              onClick={handleRemoveFile}
              type="button"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="input-form">
        <div className="file-input-wrapper">
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            onChange={handleFileChange}
          />
          <button
            type="button"
            className="file-upload-btn"
            onClick={() => fileInputRef.current?.click()}
          >
            +
          </button>
        </div>

        <input
          type="text"
          className="text-input"
          placeholder="Send a message..."
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          onKeyPress={handleKeyPress}
        />

        <button type="submit" className="send-btn">
          ↑
        </button>
      </form>
    </div>
  );
};

export default InputArea;
