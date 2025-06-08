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
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);

  // Initialize voice support and enable on user interaction
  useEffect(() => {
    // Check if speech synthesis is supported
    if ('speechSynthesis' in window) {
      setSpeechSupported(true);
      console.log('Speech synthesis supported');
      
      // On iOS, we need user interaction to enable speech
      const enableVoiceOnInteraction = () => {
        if (!voiceEnabled) {
          // Test voice with empty utterance to initialize on iOS
          const testUtterance = new SpeechSynthesisUtterance('');
          testUtterance.volume = 0;
          speechSynthesis.speak(testUtterance);
          setVoiceEnabled(true);
          console.log('Voice enabled after user interaction');
        }
      };

      // Add event listeners for user interaction
      const events = ['touchstart', 'click', 'keydown'];
      events.forEach(event => {
        document.addEventListener(event, enableVoiceOnInteraction, { once: true });
      });

      return () => {
        events.forEach(event => {
          document.removeEventListener(event, enableVoiceOnInteraction);
        });
      };
    } else {
      console.log('Speech synthesis not supported');
      setSpeechSupported(false);
    }
  }, [voiceEnabled]);

  // Voice announcement with cooldown and iOS compatibility
  const announceDetection = (objectClass, confidence) => {
    try {
      if (!speechSupported || !voiceEnabled) {
        console.log('🔇 Voice not available - speech synthesis not supported or not enabled');
        return;
      }

      const now = Date.now();
      const lastAnnounced = lastVoiceAnnouncement[objectClass] || 0;
      const cooldownPeriod = 30000; // 30 seconds

      // Only announce if 30 seconds have passed since last announcement for this object
      if (now - lastAnnounced > cooldownPeriod) {
        const message = `${objectClass} detected with ${confidence} percent confidence`;
        
        // Use Web Speech API for voice output with iOS-specific settings
        const utterance = new SpeechSynthesisUtterance(message);
        utterance.rate = 0.8; // Slower for better clarity on mobile
        utterance.pitch = 1.0;
        utterance.volume = 1.0; // Full volume for mobile
        utterance.lang = 'en-US'; // Explicit language for better compatibility
        
        // iOS-specific handling
        utterance.onstart = () => {
          console.log(`🔊 Voice started: ${message}`);
        };
        
        utterance.onerror = (event) => {
          console.error('Speech synthesis error:', event.error);
        };
        
        utterance.onend = () => {
          console.log('🔊 Voice announcement completed');
        };

        speechSynthesis.speak(utterance);
        console.log(`🔊 Voice: ${message}`);

        // Update last announcement time
        setLastVoiceAnnouncement(prev => ({
          ...prev,
          [objectClass]: now
        }));
      } else {
        const timeLeft = Math.round((cooldownPeriod - (now - lastAnnounced)) / 1000);
        console.log(`🔇 Voice cooldown: ${objectClass} - ${timeLeft}s remaining`);
      }
    } catch (error) {
      console.error('Voice announcement error:', error);
    }
  };

  // Manual detection trigger for testing
  const runManualDetection = async () => {
    if (!model || !videoRef.current) {
      console.log('Cannot run manual detection - missing model or video');
      return;
    }
    
    try {
      console.log('=== MANUAL DETECTION TRIGGERED ===');
      const video = videoRef.current;
      console.log('Video state:', {
        readyState: video.readyState,
        dimensions: `${video.videoWidth}x${video.videoHeight}`,
        playing: !video.paused,
        currentTime: video.currentTime
      });
      
      const predictions = await model.detect(video);
      console.log('Manual detection results:', predictions);
      setDetectionCount(prev => prev + 1);
      
      // Test with all objects, not just person/chair
      if (predictions.length > 0) {
        console.log('🎉 OBJECTS DETECTED:', predictions.map(p => `${p.class} (${Math.round(p.score * 100)}%)`));
        
        // Filter for chairs and persons only
        const relevantDetections = predictions.filter(prediction => 
          prediction.class === 'person' || prediction.class === 'chair'
        );
        
        setDetections(relevantDetections);
        
        // Voice announcements for detected objects
        if (relevantDetections.length > 0) {
          relevantDetections.forEach(detection => {
            announceDetection(detection.class, Math.round(detection.score * 100));
          });
        }
      }
      
    } catch (error) {
      console.error('Manual detection error:', error);
    }
  };

  // Initialize camera and model
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Configure TensorFlow.js backend for better compatibility
        await tf.ready();
        console.log('TensorFlow.js backend:', tf.getBackend());
        
        // Force CPU backend if WebGL fails
        if (tf.getBackend() === 'webgl') {
          console.log('Using WebGL backend');
        } else {
          console.log('WebGL not available, using CPU backend');
          await tf.setBackend('cpu');
        }
        
        // Load TensorFlow.js model
        console.log('Loading COCO-SSD model...');
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        setIsModelLoaded(true);
        console.log('Model loaded successfully!');

        // Get camera access - optimized for mobile devices
        try {
          const constraints = {
            video: { 
              width: { ideal: 1280, min: 320 }, 
              height: { ideal: 720, min: 240 },
              facingMode: 'environment', // Back camera for mobile, better for object detection
              frameRate: { ideal: 30, min: 15 }
            }
          };

          // Try back camera first, then front camera
          let stream;
          try {
            stream = await navigator.mediaDevices.getUserMedia(constraints);
          } catch (backCameraError) {
            console.log('Back camera failed, trying front camera');
            constraints.video.facingMode = 'user';
            stream = await navigator.mediaDevices.getUserMedia(constraints);
          }
          
          setCameraStatus('connected');
          
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
            console.log('Camera connected successfully!');
          }
        } catch (cameraError) {
          console.error('Camera error:', cameraError);
          setCameraError(cameraError.message);
          setCameraStatus('error');
          
          // If camera fails, create a test canvas for demo
          createTestCanvas();
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error initializing app:', error);
        setCameraError(error.message);
        setCameraStatus('error');
        setIsLoading(false);
      }
    };

    // Create test canvas when camera isn't available
    const createTestCanvas = () => {
      if (videoRef.current) {
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 480;
        const ctx = canvas.getContext('2d');
        
        // Draw test scene
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, 640, 480);
        
        // Draw person rectangle
        ctx.fillStyle = '#ff6b6b';
        ctx.fillRect(200, 150, 80, 200);
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText('Test Person', 210, 140);
        
        // Draw chair rectangle  
        ctx.fillStyle = '#4ecdc4';
        ctx.fillRect(350, 250, 100, 80);
        ctx.fillText('Test Chair', 360, 240);
        
        // Convert canvas to video stream
        const stream = canvas.captureStream(30);
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        console.log('Using test canvas for demo');
      }
    };

    initializeApp();
  }, []);

  // Object detection loop with better dependency management
  useEffect(() => {
    console.log('Detection effect triggered:', { model: !!model, video: !!videoRef.current, isLoading });
    
    if (!model) {
      console.log('No model available yet');
      return;
    }
    
    if (!videoRef.current) {
      console.log('No video element available yet');
      return;
    }

    if (isLoading) {
      console.log('Still loading, waiting...');
      return;
    }

    console.log('Starting detection interval...');

    const detectObjects = async () => {
      try {
        const video = videoRef.current;
        
        if (!video) {
          console.log('Video ref lost');
          return;
        }
        
        // Enhanced video readiness check
        if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
          console.log('Running detection...', { 
            readyState: video.readyState, 
            dimensions: `${video.videoWidth}x${video.videoHeight}`,
            backend: tf.getBackend()
          });
          
          const predictions = await model.detect(video);
          console.log('All predictions:', predictions.map(p => ({ class: p.class, score: p.score })));
          
          // Filter for chairs and persons only
          const relevantDetections = predictions.filter(prediction => 
            prediction.class === 'person' || prediction.class === 'chair'
          );
          
          console.log('Relevant detections:', relevantDetections);
          setDetections(relevantDetections);
          
          // Voice announcements for detected objects
          if (relevantDetections.length > 0) {
            console.log('Processing voice announcements for detections');
            relevantDetections.forEach(detection => {
              announceDetection(detection.class, Math.round(detection.score * 100));
            });
          }
        } else {
          console.log('Video not ready:', {
            readyState: video?.readyState,
            dimensions: `${video?.videoWidth || 0}x${video?.videoHeight || 0}`
          });
        }
      } catch (error) {
        console.error('Detection error:', error);
      }
    };

    // Initial detection after short delay
    const initialTimeout = setTimeout(() => {
      console.log('Running initial detection...');
      detectObjects();
    }, 2000);

    const interval = setInterval(() => {
      console.log('Interval tick - running detection');
      detectObjects();
    }, 2000); // 2 seconds interval for voice announcements
    
    return () => {
      console.log('Cleaning up detection interval');
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [model, isLoading, lastVoiceAnnouncement]);

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
            {!isModelLoaded ? 'Loading AI model...' : 
             cameraStatus === 'requesting' ? 'Requesting camera access...' : 
             'Getting camera ready...'}
          </h2>
          <p className="text-gray-300 mt-2">
            {!isModelLoaded ? 'Setting up object detection' : 
             'Please allow camera access when prompted'}
          </p>
          {cameraError && (
            <div className="mt-4 p-4 bg-red-500/20 border border-red-500 rounded-lg">
              <p className="text-red-300 font-medium">Camera Error:</p>
              <p className="text-red-200 text-sm mt-1">{cameraError}</p>
              <p className="text-gray-300 text-xs mt-2">
                Make sure to allow camera access in your browser settings
              </p>
            </div>
          )}
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
              <div className="ml-4 text-white text-xs flex items-center space-x-1">
                <span>🔊</span>
                <span>{speechSupported ? (voiceEnabled ? 'Voice: ON' : 'Tap to enable voice') : 'Voice: Not supported'}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Debug Panel - Hide on small screens */}
        <div className="absolute top-20 right-4 bg-black/70 text-white p-3 rounded-lg text-xs max-w-xs z-30 hidden md:block">
          <div className="font-bold mb-2">🔧 Debug Info</div>
          <div>Backend: {typeof tf !== 'undefined' ? tf.getBackend() : 'loading...'}</div>
          <div>Camera: {cameraStatus}</div>
          <div>Model: {isModelLoaded ? 'loaded' : 'loading'}</div>
          <div>Video Ready: {videoRef.current?.readyState || 0}/4</div>
          <div>Detections: {detections.length}</div>
          <div>Detection Runs: {detectionCount}</div>
          <div>Voice: {speechSupported ? (voiceEnabled ? 'enabled' : 'needs interaction') : 'not supported'}</div>
          <div className="mt-2">
            <div className="text-yellow-300 font-bold">Voice Cooldowns:</div>
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
            className="mt-2 bg-pink-500 hover:bg-pink-600 px-2 py-1 rounded text-white text-xs"
          >
            🔍 Test Detection
          </button>
          {cameraError && <div className="text-red-300 mt-1">Error: {cameraError}</div>}
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
              🔊 Voice announcements with 30-second cooldown
              {speechSupported && !voiceEnabled && (
                <div className="text-yellow-400 mt-1">
                  📱 Tap anywhere to enable voice on mobile
                </div>
              )}
            </div>
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