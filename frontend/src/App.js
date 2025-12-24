import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState('');
  const [predictionStatus, setPredictionStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/jpg' || file.type === 'image/png')) {
      setSelectedFile(file);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files[0];
    if (file && (file.type === 'image/jpeg' || file.type === 'image/jpg' || file.type === 'image/png')) {
      setSelectedFile(file);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('https://defect-detection-concept-system.onrender.com/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        setPrediction(data.error);
        setPredictionStatus('error');
        setConfidence('');
      } else {
        setPrediction(data.prediction);
        setConfidence(data.confidence);
        setPredictionStatus('success');
      }
    } catch (error) {
      setPrediction('Failed to connect to server');
      setPredictionStatus('error');
      setConfidence('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header-section">
        <h1 className="main-title">Automated Real-Time Image-Driven Quality Inspection System for Industrial Defect Classification</h1>
      </header>

      <div className="action-bar">
        <div className="bar-buttons">
          <button className="action-btn upload-btn" onClick={() => fileInputRef.current?.click()}>
            <span className="btn-icon">⬆</span>Upload Image
          </button>
          {prediction && (
            <button className="action-btn delete-result-btn" onClick={() => {
              setPrediction('');
              setConfidence('');
              setPredictionStatus('');
              setSelectedFile(null);
            }}>
              <span className="btn-icon">✕</span>Delete Result
            </button>
          )}
        </div>
        <div className="logo-container">
          <img src="/logo.png" alt="Logo" className="company-logo" />
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/jpg,image/png"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      <div className="content-wrapper">
        {!selectedFile ? (
          <div
            className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="drop-icon">☁</div>
            <p className="drop-text">Drag and drop your image here, or click to select</p>
            <p className="drop-format">Supported formats: JPG, JPEG, PNG</p>
            <button className="choose-btn">Choose Image</button>
          </div>
        ) : (
          <div className="results-container">
            <div className="image-output-section">
              <div className="image-box">
                <h3 className="section-label">Original Image</h3>
                <img
                  src={URL.createObjectURL(selectedFile)}
                  alt="Uploaded"
                  className="preview-image"
                />
                <p className="file-name">{selectedFile.name}</p>
              </div>

              {prediction && (
                <div className="output-box">
                  <h3 className="section-label">Detection Result</h3>
                  <div className={`result-display ${predictionStatus === 'error' ? 'error' : ''}`}>
                    <div className="result-status">
                      <span className={`status-icon ${predictionStatus}`}>
                        {predictionStatus === 'success' ? '✓' : '✕'}
                      </span>
                    </div>
                    <p className="result-text">{prediction}</p>
                    {confidence && <p className="result-confidence">Confidence: {confidence}</p>}
                  </div>
                </div>
              )}
            </div>

            <button className="classify-btn" onClick={handleSubmit} disabled={loading}>
              {loading ? 'Classifying...' : 'Classify Image'}
            </button>
          </div>
        )}
      </div>

      <footer className="footer-section">
        <p>© Concet System Technology Solutions | Pune Maharashtra</p>
      </footer>
    </div>
  );
}

export default App;
