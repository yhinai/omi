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
  const [selectedVoice, setSelectedVoice] = useState('female');
  const [availableVoices, setAvailableVoices] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [navigationActive, setNavigationActive] = useState(false);
  const [currentRoute, setCurrentRoute] = useState(null);
  const [currentDirection, setCurrentDirection] = useState('');
  const [nextDirection, setNextDirection] = useState('');
  const [lastDirectionTime, setLastDirectionTime] = useState(0);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isChrome, setIsChrome] = useState(false);
  const [voiceRetryCount, setVoiceRetryCount] = useState(0);

  // Vehicle types for detection
  const vehicleTypes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train'];
  const detectionTargets = ['person', 'chair', ...vehicleTypes];

  // Initialize speech recognition
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = SpeechRecognition ? new SpeechRecognition() : null;

  // Detect browser for voice compatibility
  useEffect(() => {
    const isChromeBrowser = /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    setIsChrome(isChromeBrowser && isMobile);
    console.log('Browser detection:', { isChromeBrowser, isMobile, isChrome: isChromeBrowser && isMobile });
  }, []);

  // Initialize voices with Chrome iOS specific handling
  useEffect(() => {
    const loadVoices = () => {
      const voices = speechSynthesis.getVoices();
      setAvailableVoices(voices);
      console.log('Available voices:', voices.map(v => `${v.name} (${v.lang}) ${v.default ? '[DEFAULT]' : ''}`));
      
      // For Chrome iOS, sometimes voices load later
      if (isChrome && voices.length === 0) {
        console.log('Chrome iOS: No voices yet, will retry...');
        setTimeout(loadVoices, 500);
      }
    };

    loadVoices();
    speechSynthesis.onvoiceschanged = loadVoices;
    
    // Chrome iOS specific: Force voice loading
    if (isChrome) {
      setTimeout(loadVoices, 1000);
      setTimeout(loadVoices, 2000);
    }
  }, [isChrome]);

  // Get user location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          });
          console.log('Location:', position.coords.latitude, position.coords.longitude);
        },
        (error) => {
          console.error('Location error:', error);
          // Default to San Francisco if location fails
          setUserLocation({ lat: 37.7749, lng: -122.4194 });
        },
        { enableHighAccuracy: true }
      );
    }
  }, []);

  // Configure speech recognition
  useEffect(() => {
    if (recognition) {
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        console.log('Speech recognition started');
        setIsListening(true);
      };

      recognition.onend = () => {
        console.log('Speech recognition ended');
        setIsListening(false);
      };

      recognition.onresult = (event) => {
        const command = event.results[0][0].transcript.toLowerCase();
        console.log('Voice command:', command);
        handleVoiceCommand(command);
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
    }
  }, [recognition]);

  // Handle voice commands
  const handleVoiceCommand = async (command) => {
    if (command.includes('train') && command.includes('station')) {
      speak('Searching for nearest train station');
      await findNearestTransit('train');
    } else if (command.includes('bus') && command.includes('stop')) {
      speak('Searching for nearest bus stop');
      await findNearestTransit('bus');
    } else if (command.includes('subway') || command.includes('metro')) {
      speak('Searching for nearest subway station');
      await findNearestTransit('subway');
    } else {
      speak('Sorry, I did not understand that command. Try saying take me to train station');
    }
  };

  // Find nearest transit using Overpass API
  const findNearestTransit = async (type) => {
    if (!userLocation) {
      speak('Location not available');
      return;
    }

    try {
      let query = '';
      if (type === 'train') {
        query = `[out:json][timeout:25];
          (
            node["railway"="station"](around:2000,${userLocation.lat},${userLocation.lng});
            way["railway"="station"](around:2000,${userLocation.lat},${userLocation.lng});
          );
          out;`;
      } else if (type === 'bus') {
        query = `[out:json][timeout:25];
          (
            node["highway"="bus_stop"](around:1000,${userLocation.lat},${userLocation.lng});
          );
          out;`;
      } else if (type === 'subway') {
        query = `[out:json][timeout:25];
          (
            node["railway"="subway_entrance"](around:1000,${userLocation.lat},${userLocation.lng});
            node["public_transport"="station"]["railway"="subway"](around:1000,${userLocation.lat},${userLocation.lng});
          );
          out;`;
      }

      const response = await fetch('https://overpass-api.de/api/interpreter', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: query
      });

      const data = await response.json();
      console.log('Transit search results:', data);

      if (data.elements && data.elements.length > 0) {
        const nearest = data.elements[0];
        const distance = calculateDistance(
          userLocation.lat, userLocation.lng,
          nearest.lat, nearest.lon
        );
        
        const name = nearest.tags?.name || `${type} station`;
        speak(`Found ${name}, ${Math.round(distance)} meters away. Getting walking directions`);
        
        await getWalkingDirections(userLocation, { lat: nearest.lat, lng: nearest.lon });
      } else {
        speak(`No ${type} stations found nearby`);
      }
    } catch (error) {
      console.error('Transit search error:', error);
      speak('Sorry, unable to search for transit stations right now');
    }
  };

  // Get walking directions using OSRM with detailed street names
  const getWalkingDirections = async (start, end) => {
    try {
      const url = `https://router.project-osrm.org/route/v1/foot/${start.lng},${start.lat};${end.lng},${end.lat}?overview=full&geometries=geojson&steps=true&annotations=true`;
      
      const response = await fetch(url);
      const data = await response.json();
      
      if (data.routes && data.routes.length > 0) {
        const route = data.routes[0];
        const duration = Math.round(route.duration / 60);
        const distance = Math.round(route.distance);
        
        speak(`Walking route found. ${distance} meters, approximately ${duration} minutes. Starting navigation`);
        
        setCurrentRoute(route);
        setNavigationActive(true);
        setCurrentStepIndex(0);
        
        // Start with first direction
        if (route.legs && route.legs[0].steps && route.legs[0].steps.length > 0) {
          const firstStep = route.legs[0].steps[0];
          const firstDirection = formatNavigationDirection(firstStep);
          setCurrentDirection(firstDirection);
          speak(firstDirection);
          setLastDirectionTime(Date.now());
          
          // Set up location tracking for turn-by-turn
          startLocationTracking(route.legs[0].steps);
        }
      } else {
        speak('Unable to find walking route');
      }
    } catch (error) {
      console.error('Routing error:', error);
      speak('Sorry, unable to get directions right now');
    }
  };

  // Format navigation directions like real GPS apps
  const formatNavigationDirection = (step) => {
    const maneuver = step.maneuver;
    const distance = Math.round(step.distance);
    const streetName = step.name || 'the road';
    
    let direction = '';
    
    switch (maneuver.type) {
      case 'depart':
        direction = `Start by walking on ${streetName}`;
        break;
      case 'turn':
        const turnDirection = maneuver.modifier;
        if (turnDirection === 'left') {
          direction = `Turn left onto ${streetName}`;
        } else if (turnDirection === 'right') {
          direction = `Turn right onto ${streetName}`;
        } else if (turnDirection === 'slight left') {
          direction = `Turn slightly left onto ${streetName}`;
        } else if (turnDirection === 'slight right') {
          direction = `Turn slightly right onto ${streetName}`;
        } else {
          direction = `Turn ${turnDirection} onto ${streetName}`;
        }
        break;
      case 'continue':
        direction = `Continue on ${streetName}`;
        break;
      case 'merge':
        direction = `Merge onto ${streetName}`;
        break;
      case 'ramp':
        direction = `Take the ramp to ${streetName}`;
        break;
      case 'arrive':
        direction = 'You have arrived at your destination';
        break;
      default:
        direction = `Head ${maneuver.modifier || 'straight'} on ${streetName}`;
    }
    
    // Add distance for non-arrival instructions
    if (maneuver.type !== 'arrive' && distance > 0) {
      if (distance > 100) {
        direction += ` for ${Math.round(distance)} meters`;
      } else if (distance > 20) {
        direction += ` for ${distance} meters`;
      }
    }
    
    return direction;
  };

  // Start location tracking for navigation
  const startLocationTracking = (steps) => {
    if (!navigator.geolocation) return;
    
    const watchId = navigator.geolocation.watchPosition(
      (position) => {
        const currentLat = position.coords.latitude;
        const currentLng = position.coords.longitude;
        
        // Check if we're close to the next turn
        checkNavigationProgress(currentLat, currentLng, steps);
      },
      (error) => {
        console.error('Location tracking error:', error);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 5000
      }
    );
    
    // Store watch ID to clear later
    window.navigationWatchId = watchId;
  };

  // Check navigation progress and announce next directions
  const checkNavigationProgress = (currentLat, currentLng, steps) => {
    if (currentStepIndex >= steps.length) return;
    
    const currentStep = steps[currentStepIndex];
    const nextStepIndex = currentStepIndex + 1;
    
    if (nextStepIndex < steps.length) {
      const nextStep = steps[nextStepIndex];
      const nextStepLocation = nextStep.maneuver.location;
      
      // Calculate distance to next turn
      const distanceToNext = calculateDistance(
        currentLat, currentLng,
        nextStepLocation[1], nextStepLocation[0]
      );
      
      // If within 50 meters of next turn, announce it (with 30-second cooldown)
      if (distanceToNext < 50) {
        const now = Date.now();
        if (now - lastDirectionTime > 30000) { // 30 second cooldown
          const nextDirection = formatNavigationDirection(nextStep);
          setCurrentDirection(nextDirection);
          speak(`In ${Math.round(distanceToNext)} meters, ${nextDirection}`);
          setLastDirectionTime(now);
          setCurrentStepIndex(nextStepIndex);
        }
      }
    }
    
    // Check if we've arrived at destination
    if (currentStepIndex === steps.length - 1) {
      const destination = steps[steps.length - 1].maneuver.location;
      const distanceToDestination = calculateDistance(
        currentLat, currentLng,
        destination[1], destination[0]
      );
      
      if (distanceToDestination < 20) {
        speak('You have arrived at your destination');
        setCurrentDirection('You have arrived at your destination');
        setNavigationActive(false);
        if (window.navigationWatchId) {
          navigator.geolocation.clearWatch(window.navigationWatchId);
        }
      }
    }
  };

  // Stop navigation
  const stopNavigation = () => {
    setNavigationActive(false);
    setCurrentDirection('');
    setCurrentRoute(null);
    setCurrentStepIndex(0);
    if (window.navigationWatchId) {
      navigator.geolocation.clearWatch(window.navigationWatchId);
    }
    speak('Navigation stopped');
  };

  // Calculate distance between two points
  const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371000; // Earth's radius in meters
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  // Enhanced speak function with Chrome iOS fixes
  const speak = (text) => {
    try {
      if (!window.speechSynthesis) {
        console.log('Speech synthesis not available');
        return;
      }

      // Cancel any existing speech
      window.speechSynthesis.cancel();
      
      const speakWithDelay = () => {
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Chrome iOS specific settings
        if (isChrome) {
          utterance.volume = 1.0;
          utterance.rate = 0.9;
          utterance.pitch = selectedVoice === 'male' ? 0.8 : 1.0;
          utterance.lang = 'en-US';
          
          // For Chrome iOS, use default voice if available voices aren't working
          const voices = speechSynthesis.getVoices();
          if (voices.length > 0) {
            // Try to find a good default voice
            const defaultVoice = voices.find(v => v.default) || voices[0];
            if (defaultVoice) {
              utterance.voice = defaultVoice;
              console.log('Chrome iOS: Using voice:', defaultVoice.name);
            }
          }
        } else {
          // Regular voice selection for other browsers
          if (availableVoices.length > 0) {
            let voice;
            if (selectedVoice === 'male') {
              voice = availableVoices.find(v => 
                v.name.toLowerCase().includes('male') ||
                v.name.toLowerCase().includes('david') ||
                v.name.toLowerCase().includes('daniel') ||
                v.name.toLowerCase().includes('alex')
              );
            } else {
              voice = availableVoices.find(v => 
                v.name.toLowerCase().includes('female') ||
                v.name.toLowerCase().includes('susan') ||
                v.name.toLowerCase().includes('karen') ||
                v.name.toLowerCase().includes('victoria') ||
                v.name.toLowerCase().includes('samantha')
              );
            }
            
            if (voice) {
              utterance.voice = voice;
              console.log('Using voice:', voice.name);
            }
          }
          
          utterance.rate = selectedVoice === 'male' ? 0.7 : 0.8;
          utterance.pitch = selectedVoice === 'male' ? 0.7 : 1.1;
          utterance.volume = 1.0;
          utterance.lang = 'en-US';
        }
        
        utterance.onstart = () => {
          console.log('🔊 Speaking:', text);
          setVoiceRetryCount(0);
        };
        
        utterance.onerror = (e) => {
          console.error('Speech error:', e.error);
          
          // Chrome iOS retry logic
          if (isChrome && voiceRetryCount < 2) {
            console.log('Chrome iOS: Retrying speech...');
            setVoiceRetryCount(prev => prev + 1);
            setTimeout(() => speakWithDelay(), 500);
          }
        };
        
        utterance.onend = () => {
          console.log('🔊 Speech completed');
        };

        window.speechSynthesis.speak(utterance);
      };
      
      // Chrome iOS needs longer delay
      const delay = isChrome ? 300 : 100;
      setTimeout(speakWithDelay, delay);
      
    } catch (error) {
      console.error('Voice error:', error);
    }
  };

  // Start voice command listening
  const startListening = () => {
    if (recognition && !isListening) {
      recognition.start();
    }
  };

  // Chrome iOS specific voice initialization
  const initializeVoice = () => {
    if (!voiceInitialized && window.speechSynthesis) {
      if (isChrome) {
        // For Chrome iOS, try multiple initialization attempts
        console.log('Initializing voice for Chrome iOS...');
        
        // First attempt: Silent utterance
        const silentUtterance = new SpeechSynthesisUtterance('');
        silentUtterance.volume = 0.01;
        window.speechSynthesis.speak(silentUtterance);
        
        // Second attempt: Audible test
        setTimeout(() => {
          speak('Voice system ready for Chrome');
        }, 200);
      } else {
        speak('Voice system ready');
      }
      
      setVoiceInitialized(true);
      console.log('Voice initialized for', isChrome ? 'Chrome iOS' : 'other browser');
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

  // Voice announcement with cooldown
  const announceDetection = (objectClass, confidence) => {
    const now = Date.now();
    const lastAnnounced = lastVoiceAnnouncement[objectClass] || 0;
    const cooldownPeriod = 30000; // 30 seconds

    if (now - lastAnnounced > cooldownPeriod) {
      let message = '';
      if (vehicleTypes.includes(objectClass)) {
        message = `${objectClass} vehicle detected`;
      } else {
        message = `${objectClass} detected`;
      }
      
      speak(message);
      
      setLastVoiceAnnouncement(prev => ({
        ...prev,
        [objectClass]: now
      }));
      
      console.log(`🔊 Announced: ${message}`);
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
        detectionTargets.includes(prediction.class)
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

  // Test voice function
  const testVoice = () => {
    if (!voiceInitialized) {
      initializeVoice();
    } else {
      speak('Voice test successful');
    }
  };

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
        
        // Draw test objects
        ctx.fillStyle = '#ff6b6b';
        ctx.fillRect(200, 150, 80, 200);
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText('Test Person', 210, 140);
        
        ctx.fillStyle = '#4ecdc4';
        ctx.fillRect(350, 250, 100, 80);
        ctx.fillText('Test Chair', 360, 240);
        
        ctx.fillStyle = '#ffd93d';
        ctx.fillRect(100, 300, 120, 60);
        ctx.fillText('Test Car', 110, 290);
        
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
          detectionTargets.includes(prediction.class)
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
      const isVehicle = vehicleTypes.includes(detection.class);
      const boxColor = isVehicle ? 'border-yellow-500 bg-yellow-500' : 'border-pink-500 bg-pink-500';
      
      return (
        <div
          key={index}
          className={`absolute border-3 ${boxColor} bg-opacity-20 rounded-lg`}
          style={{
            left: x * scaleX,
            top: y * scaleY,
            width: width * scaleX,
            height: height * scaleY,
          }}
        >
          <div className={`${isVehicle ? 'bg-yellow-500' : 'bg-pink-500'} text-white px-2 py-1 text-xs rounded-t-lg font-bold`}>
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
          <p className="text-gray-300 mt-2">Setting up navigation assistant</p>
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
            <h1 className="text-white text-xl font-bold">Navigation Assistant</h1>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-white text-sm font-medium">LIVE</span>
              <div className="ml-4 text-white text-xs">
                🔊 Voice: {voiceInitialized ? 'ON' : 'Tap to enable'}
              </div>
            </div>
          </div>
        </div>

        {/* Voice Controls */}
        <div className="absolute top-20 left-4 bg-black/70 text-white p-3 rounded-lg z-30 md:block">
          <div className="font-bold mb-2">🎤 Voice Controls</div>
          <div className="space-y-2">
            <select 
              value={selectedVoice} 
              onChange={(e) => setSelectedVoice(e.target.value)}
              className="bg-gray-800 text-white text-xs p-1 rounded"
            >
              <option value="female">Female Voice</option>
              <option value="male">Male Voice</option>
            </select>
            
            <button 
              onClick={startListening}
              disabled={!recognition || isListening}
              className={`block w-full px-3 py-2 rounded text-xs font-bold ${
                isListening ? 'bg-red-500' : 'bg-blue-500 hover:bg-blue-600'
              }`}
            >
              {isListening ? '🎤 Listening...' : '🎤 Voice Command'}
            </button>
            
            <div className="text-xs text-gray-300">
              Say: "Take me to train station"
            </div>
          </div>
        </div>

        {/* Current Direction Display */}
        {navigationActive && currentDirection && (
          <div className="absolute top-1/2 left-4 right-4 transform -translate-y-1/2 bg-blue-600/95 text-white p-4 rounded-lg z-40 shadow-lg">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="font-bold text-lg mb-1">🧭 Navigation</div>
                <div className="text-sm leading-relaxed">{currentDirection}</div>
              </div>
              <button 
                onClick={stopNavigation}
                className="ml-4 bg-red-500 hover:bg-red-600 px-3 py-2 rounded text-sm font-bold"
              >
                Stop
              </button>
            </div>
          </div>
        )}

        {/* Navigation Status */}
        {navigationActive && (
          <div className="absolute top-20 right-4 bg-blue-600/90 text-white p-3 rounded-lg z-30 max-w-xs">
            <div className="font-bold text-sm">🗺️ Navigation Active</div>
            <div className="text-xs mt-1">Following walking route</div>
            <button 
              onClick={stopNavigation}
              className="mt-2 bg-red-500 hover:bg-red-600 px-2 py-1 rounded text-xs"
            >
              Stop Navigation
            </button>
          </div>
        )}

        {/* Debug Panel - Desktop only */}
        <div className="absolute bottom-4 right-4 bg-black/70 text-white p-3 rounded-lg text-xs max-w-xs z-30 hidden md:block">
          <div className="font-bold mb-2">🔧 Debug Info</div>
          <div>Backend: {tf.getBackend()}</div>
          <div>Camera: {cameraStatus}</div>
          <div>Voice: {voiceInitialized ? 'enabled' : 'click to enable'}</div>
          <div>Location: {userLocation ? '✓' : '✗'}</div>
          <div>Detections: {detections.length}</div>
          <div className="mt-2 space-x-1">
            <button 
              onClick={runManualDetection}
              className="bg-pink-500 hover:bg-pink-600 px-2 py-1 rounded text-white text-xs"
            >
              🔍 Detect
            </button>
            <button 
              onClick={testVoice}
              className="bg-blue-500 hover:bg-blue-600 px-2 py-1 rounded text-white text-xs"
            >
              🔊 Test
            </button>
          </div>
        </div>

        {/* Bottom Info Panel */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-6 z-10">
          <div className="text-center">
            <div className="text-white text-lg mb-2">
              Detecting: <span className="font-bold text-pink-400">People</span> & <span className="font-bold text-yellow-400">Vehicles</span>
            </div>
            
            {detections.length > 0 ? (
              <div className="flex flex-wrap justify-center gap-2 mt-3">
                {detections.map((detection, index) => {
                  const isVehicle = vehicleTypes.includes(detection.class);
                  return (
                    <div
                      key={index}
                      className={`backdrop-blur-sm text-white px-3 py-1 rounded-full text-sm font-medium ${
                        isVehicle ? 'bg-yellow-500/20' : 'bg-white/20'
                      }`}
                    >
                      {detection.class} - {Math.round(detection.score * 100)}%
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-gray-300 text-sm mt-2">
                👀 Scanning for objects and vehicles...
              </div>
            )}
            
            <div className="text-gray-400 text-xs mt-3">
              {voiceInitialized ? 
                '🎤 Say "Take me to train station" for navigation' : 
                '📱 Tap anywhere to enable voice assistant'
              }
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;