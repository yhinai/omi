import React, { useRef, useEffect, useState } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [alerts, setAlerts] = useState([]);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [cameraStatus, setCameraStatus] = useState('requesting');

  // Initialize camera and model
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Load TensorFlow.js model
        console.log('Loading COCO-SSD model...');
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        setIsModelLoaded(true);
        console.log('Model loaded successfully!');

        // Get camera access - optimized for MacBook
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { 
            width: { ideal: 1280, min: 640 }, 
            height: { ideal: 720, min: 480 },
            facingMode: 'user', // Front camera for better UX
            frameRate: { ideal: 30, min: 15 }
          }
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error initializing app:', error);
        setIsLoading(false);
      }
    };

    initializeApp();
  }, []);

  // Object detection loop
  useEffect(() => {
    if (!model || !videoRef.current) return;

    const detectObjects = async () => {
      if (videoRef.current && videoRef.current.readyState === 4) {
        const predictions = await model.detect(videoRef.current);
        
        // Filter for chairs and persons only
        const relevantDetections = predictions.filter(prediction => 
          prediction.class === 'person' || prediction.class === 'chair'
        );
        
        setDetections(relevantDetections);
        
        // Create alerts for detected objects
        if (relevantDetections.length > 0) {
          const newAlerts = relevantDetections.map(detection => ({
            id: Math.random(),
            class: detection.class,
            confidence: Math.round(detection.score * 100),
            timestamp: Date.now()
          }));
          
          setAlerts(prev => {
            // Keep only recent alerts (last 3 seconds)
            const filtered = prev.filter(alert => Date.now() - alert.timestamp < 3000);
            return [...filtered, ...newAlerts].slice(-5); // Keep max 5 alerts
          });
        }
      }
    };

    const interval = setInterval(detectObjects, 200); // Detect every 200ms
    return () => clearInterval(interval);
  }, [model]);

  // Auto-remove alerts after 3 seconds
  useEffect(() => {
    const timeout = setTimeout(() => {
      setAlerts(prev => prev.filter(alert => Date.now() - alert.timestamp < 3000));
    }, 100);
    return () => clearTimeout(timeout);
  }, [alerts]);

  const renderDetectionBoxes = () => {
    if (!videoRef.current || detections.length === 0) return null;
    
    const video = videoRef.current;
    const scaleX = video.offsetWidth / video.videoWidth;
    const scaleY = video.offsetHeight / video.videoHeight;

    return detections.map((detection, index) => {
      const [x, y, width, height] = detection.bbox;
      return (
        <div
          key={index}
          className="absolute border-3 border-pink-500 bg-pink-500 bg-opacity-20 rounded-lg"
          style={{
            left: x * scaleX,
            top: y * scaleY,
            width: width * scaleX,
            height: height * scaleY,
          }}
        >
          <div className="bg-pink-500 text-white px-2 py-1 text-xs rounded-t-lg font-bold">
            {detection.class} ({Math.round(detection.score * 100)}%)
          </div>
        </div>
      );
    });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-pink-800 to-red-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-white mx-auto mb-4"></div>
          <h2 className="text-white text-xl font-bold">
            {isModelLoaded ? 'Getting camera ready...' : 'Loading AI model...'}
          </h2>
          <p className="text-gray-300 mt-2">Setting up object detection</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black relative overflow-hidden">
      {/* Video Container - TikTok style */}
      <div className="relative w-full h-screen">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          muted
          playsInline
        />
        
        {/* Detection Overlay */}
        <div className="absolute inset-0">
          {renderDetectionBoxes()}
        </div>

        {/* Top UI Bar */}
        <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/60 to-transparent p-4 z-10">
          <div className="flex items-center justify-between">
            <h1 className="text-white text-xl font-bold">Object Detector</h1>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-white text-sm font-medium">LIVE</span>
            </div>
          </div>
        </div>

        {/* Alert Notifications - TikTok style */}
        <div className="absolute top-20 left-4 right-4 z-20 space-y-2">
          {alerts.map(alert => (
            <div
              key={alert.id}
              className="bg-gradient-to-r from-pink-500 to-purple-600 text-white px-4 py-3 rounded-full shadow-lg animate-bounce flex items-center space-x-3"
            >
              <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
              <span className="font-bold text-lg">
                🎯 Detected: {alert.class.toUpperCase()} ({alert.confidence}%)
              </span>
            </div>
          ))}
        </div>

        {/* Bottom Info Panel */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-6 z-10">
          <div className="text-center">
            <div className="text-white text-lg mb-2">
              Looking for: <span className="font-bold text-pink-400">People</span> & <span className="font-bold text-purple-400">Chairs</span>
            </div>
            
            {detections.length > 0 ? (
              <div className="flex flex-wrap justify-center gap-2 mt-3">
                {detections.map((detection, index) => (
                  <div
                    key={index}
                    className="bg-white/20 backdrop-blur-sm text-white px-3 py-1 rounded-full text-sm font-medium"
                  >
                    {detection.class} - {Math.round(detection.score * 100)}%
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-300 text-sm mt-2">
                👀 Scanning for objects...
              </div>
            )}
          </div>
        </div>

        {/* Floating Action Indicator */}
        <div className="absolute right-4 top-1/2 transform -translate-y-1/2 z-10">
          <div className="bg-white/20 backdrop-blur-sm rounded-full p-4">
            <div className="text-center">
              <div className="text-2xl mb-2">🔍</div>
              <div className="text-white text-xs font-bold">AI Vision</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;