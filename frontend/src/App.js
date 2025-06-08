import React, { useRef, useEffect, useState } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

function App() {
  const videoRef = useRef(null);
  const [model, setModel] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [cameraStatus, setCameraStatus] = useState('requesting');
  const [detectionCount, setDetectionCount] = useState(0);
  const [lastVoiceAnnouncement, setLastVoiceAnnouncement] = useState({});
  const [voiceInitialized, setVoiceInitialized] = useState(false);

  // Simple voice function that works on all platforms
  const speak = (text) => {
    try {
      // Cancel any existing speech
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
        
        // Small delay then speak
        setTimeout(() => {
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.rate = 0.8;
          utterance.volume = 1.0;
          utterance.pitch = 1.0;
          utterance.lang = 'en-US';
          
          utterance.onstart = () => console.log('🔊 Speaking:', text);
          utterance.onerror = (e) => console.error('Speech error:', e);
          utterance.onend = () => console.log('🔊 Speech finished');
          
          window.speechSynthesis.speak(utterance);
        }, 100);
      }
    } catch (error) {
      console.error('Voice error:', error);
    }
  };

  // Initialize voice on first user interaction
  const initializeVoice = () => {
    if (!voiceInitialized && window.speechSynthesis) {
      // Test speech to initialize
      speak('Voice ready');
      setVoiceInitialized(true);
      console.log('Voice initialized');
    }
  };

  // Voice announcement with cooldown
  const announceDetection = (objectClass, confidence) => {
    const now = Date.now();
    const lastAnnounced = lastVoiceAnnouncement[objectClass] || 0;
    const cooldownPeriod = 30000; // 30 seconds

    if (now - lastAnnounced > cooldownPeriod) {
      const message = `${objectClass} detected`;
      speak(message);
      
      setLastVoiceAnnouncement(prev => ({
        ...prev,
        [objectClass]: now
      }));
      
      console.log(`🔊 Announced: ${message}`);
    } else {
      const timeLeft = Math.round((cooldownPeriod - (now - lastAnnounced)) / 1000);
      console.log(`🔇 Cooldown: ${objectClass} - ${timeLeft}s remaining`);
    }
  };

  // Manual voice test
  const testVoice = () => {
    if (!voiceInitialized) {
      initializeVoice();
    } else {
      speak('Voice test successful');
    }
  };

  // Manual detection trigger
  const runManualDetection = async () => {
    if (!model || !videoRef.current) return;
    
    try {
      const video = videoRef.current;
      const predictions = await model.detect(video);
      console.log('Manual detection:', predictions.map(p => `${p.class} (${Math.round(p.score * 100)}%)`));
      
      const relevantDetections = predictions.filter(prediction => 
        prediction.class === 'person' || prediction.class === 'chair'
      );
      
      setDetections(relevantDetections);
      setDetectionCount(prev => prev + 1);
      
      if (relevantDetections.length > 0) {
        relevantDetections.forEach(detection => {
          announceDetection(detection.class, Math.round(detection.score * 100));
        });
      }
    } catch (error) {
      console.error('Detection error:', error);
    }
  };

  // Add click listener for voice initialization
  useEffect(() => {
    const handleClick = () => {
      if (!voiceInitialized) {
        initializeVoice();
      }
    };

    document.addEventListener('click', handleClick);
    document.addEventListener('touchstart', handleClick);
    
    return () => {
      document.removeEventListener('click', handleClick);
      document.removeEventListener('touchstart', handleClick);
    };
  }, [voiceInitialized]);

  // Initialize camera and model
  useEffect(() => {
    const initializeApp = async () => {
      try {
        await tf.ready();
        console.log('TensorFlow.js backend:', tf.getBackend());
        
        if (tf.getBackend() !== 'webgl') {
          await tf.setBackend('cpu');
        }
        
        console.log('Loading COCO-SSD model...');
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        setIsModelLoaded(true);
        console.log('Model loaded!');

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: { 
              width: { ideal: 1280, min: 320 }, 
              height: { ideal: 720, min: 240 },
              facingMode: 'environment'
            }
          });
          
          setCameraStatus('connected');
          
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
          }
        } catch (cameraError) {
          console.log('Camera failed, using test canvas');
          setCameraError(cameraError.message);
          setCameraStatus('error');
          createTestCanvas();
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Init error:', error);
        setIsLoading(false);
      }
    };

    const createTestCanvas = () => {
      if (videoRef.current) {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext('2d');
        
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, 640, 480);
        
        ctx.fillStyle = '#ff6b6b';
        ctx.fillRect(200, 150, 80, 200);
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText('Test Person', 210, 140);
        
        ctx.fillStyle = '#4ecdc4';
        ctx.fillRect(350, 250, 100, 80);
        ctx.fillText('Test Chair', 360, 240);
        
        const stream = canvas.captureStream(30);
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    };

    initializeApp();
  }, []);

  // Object detection loop
  useEffect(() => {
    if (!model || !videoRef.current || isLoading) return;

    const detectObjects = async () => {
      try {
        const video = videoRef.current;
        if (!video || video.readyState < 2) return;
        
        const predictions = await model.detect(video);
        const relevantDetections = predictions.filter(prediction => 
          prediction.class === 'person' || prediction.class === 'chair'
        );
        
        setDetections(relevantDetections);
        
        if (relevantDetections.length > 0 && voiceInitialized) {
          relevantDetections.forEach(detection => {
            announceDetection(detection.class, Math.round(detection.score * 100));
          });
        }
      } catch (error) {
        console.error('Detection error:', error);
      }
    };

    const interval = setInterval(detectObjects, 2000);
    return () => clearInterval(interval);
  }, [model, isLoading, voiceInitialized, lastVoiceAnnouncement]);

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
            {!isModelLoaded ? 'Loading AI model...' : 'Getting camera ready...'}
          </h2>
          <p className="text-gray-300 mt-2">Setting up object detection</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black relative overflow-hidden">
      <div className="relative w-full h-screen">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          muted
          playsInline
        />
        
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
              <div className="ml-4 text-white text-xs">
                🔊 Voice: {voiceInitialized ? 'ON' : 'Tap to enable'}
              </div>
            </div>
          </div>
        </div>

        {/* Debug Panel - Desktop only */}
        <div className="absolute top-20 right-4 bg-black/70 text-white p-3 rounded-lg text-xs max-w-xs z-30 hidden md:block">
          <div className="font-bold mb-2">🔧 Debug Info</div>
          <div>Backend: {tf.getBackend()}</div>
          <div>Camera: {cameraStatus}</div>
          <div>Model: {isModelLoaded ? 'loaded' : 'loading'}</div>
          <div>Voice: {voiceInitialized ? 'enabled' : 'click to enable'}</div>
          <div>Detections: {detections.length}</div>
          <div>Runs: {detectionCount}</div>
          <div className="mt-2">
            <div className="text-yellow-300 font-bold text-xs">Cooldowns:</div>
            {Object.entries(lastVoiceAnnouncement).map(([objectType, timestamp]) => {
              const timeLeft = Math.max(0, 30 - Math.floor((Date.now() - timestamp) / 1000));
              return (
                <div key={objectType} className="text-xs">
                  {objectType}: {timeLeft > 0 ? `${timeLeft}s` : 'ready'}
                </div>
              );
            })}
          </div>
          <button 
            onClick={runManualDetection}
            className="mt-2 bg-pink-500 hover:bg-pink-600 px-2 py-1 rounded text-white text-xs mr-1"
          >
            🔍 Detect
          </button>
          <button 
            onClick={testVoice}
            className="mt-2 bg-blue-500 hover:bg-blue-600 px-2 py-1 rounded text-white text-xs"
          >
            🔊 Voice
          </button>
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
            
            <div className="text-gray-400 text-xs mt-3">
              {voiceInitialized ? 
                '🔊 Voice announcements every 30 seconds' : 
                '📱 Tap anywhere to enable voice announcements'
              }
            </div>
          </div>
        </div>

        {/* Floating AI Indicator */}
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