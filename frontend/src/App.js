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
  const [transcripts, setTranscripts] = useState([]);
  const [lastTranscriptId, setLastTranscriptId] = useState(0);
  const [isConnectedToOMI, setIsConnectedToOMI] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [navigationActive, setNavigationActive] = useState(false);
  const [currentRoute, setCurrentRoute] = useState(null);
  const [currentDirection, setCurrentDirection] = useState('');
  const [nextDirection, setNextDirection] = useState('');
  const [lastDirectionTime, setLastDirectionTime] = useState(0);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isChrome, setIsChrome] = useState(false);
  const [voiceRetryCount, setVoiceRetryCount] = useState(0);
  const [isElevenLabsEnabled, setIsElevenLabsEnabled] = useState(true);
  const [currentAudio, setCurrentAudio] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Vehicle types for detection
  const vehicleTypes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train'];
  const detectionTargets = ['person', 'chair', ...vehicleTypes];

  // ElevenLabs configuration
  const ELEVENLABS_API_KEY = 'sk_92ef1247b19b69721020876f6fec6bab973b593ec23176f1';
  const ELEVENLABS_VOICE_IDS = {
    female: 'EXAVITQu4vr4xnSDxMaL', // Bella - Natural, friendly female voice
    male: 'VR6AewLTigWG4xSOukaG' // Josh - Natural, clear male voice
  };

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

  // Poll for new transcripts from OMI backend
  const pollTranscripts = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/api/transcripts?after=${lastTranscriptId}`);
      const data = await response.json();
      
      if (data.transcripts && data.transcripts.length > 0) {
        setTranscripts(prev => [...prev, ...data.transcripts]);
        
        // Get the latest transcript ID
        const latestId = Math.max(...data.transcripts.map(t => t.id));
        setLastTranscriptId(latestId);
        
        // Process latest transcript
        const latestTranscript = data.transcripts[data.transcripts.length - 1];
        if (latestTranscript.is_user && latestTranscript.text) {
          console.log('OMI Voice command:', latestTranscript.text);
          await handleVoiceCommand(latestTranscript.text.toLowerCase());
        } else if (!latestTranscript.is_user && latestTranscript.has_audio && latestTranscript.audio_url) {
          // Play assistant response audio automatically
          console.log('Playing assistant response:', latestTranscript.text);
          await playAssistantResponse(latestTranscript.audio_url);
        }
        
        setIsConnectedToOMI(true);
      }
    } catch (error) {
      console.error('Error polling transcripts:', error);
      setIsConnectedToOMI(false);
    }
  };

  // Start polling for OMI transcripts
  useEffect(() => {
    const interval = setInterval(pollTranscripts, 1000); // Poll every second
    return () => clearInterval(interval);
  }, [lastTranscriptId]);

  // Enhanced conversational voice command handling
  const handleVoiceCommand = async (command) => {
    try {
      console.log('Processing voice command:', command);
      
      // Send to backend for conversational AI processing
      const response = await fetch('http://127.0.0.1:8000/api/conversation/command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: command,
          session_id: 'default',
          context: {
            current_location: userLocation,
            detected_objects: detections.map(d => d.class),
            navigation_active: navigationActive,
            current_direction: currentDirection
          }
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('AI Response:', data.response);
        
        // Speak the conversational response
        await speak(data.response);
        
        // Check if this was a navigation command and handle accordingly
        if (data.command_type === 'navigation') {
          const lowerCommand = command.toLowerCase();
          if (lowerCommand.includes('train') || lowerCommand.includes('station')) {
            setTimeout(() => findNearestTransit('train'), 2000);
          } else if (lowerCommand.includes('bus')) {
            setTimeout(() => findNearestTransit('bus'), 2000);
          } else if (lowerCommand.includes('subway') || lowerCommand.includes('metro')) {
            setTimeout(() => findNearestTransit('subway'), 2000);
          }
        }
      } else {
        // Fallback to basic responses if API fails
        await handleBasicVoiceCommand(command);
      }
    } catch (error) {
      console.error('Error processing voice command:', error);
      await handleBasicVoiceCommand(command);
    }
  };
  
  // Fallback basic voice command handling
  const handleBasicVoiceCommand = async (command) => {
    const lowerCommand = command.toLowerCase();
    
    if (lowerCommand.includes('hey') || lowerCommand.includes('hello')) {
      await speak("Hey there! I'm your vision assistant. How can I help you navigate today?");
    } else if (lowerCommand.includes('train') && lowerCommand.includes('station')) {
      await speak("I'll help you find the nearest train station. Let me search for options nearby.");
      await findNearestTransit('train');
    } else if (lowerCommand.includes('bus')) {
      await speak("Looking for bus stops in your area. One moment please.");
      await findNearestTransit('bus');
    } else if (lowerCommand.includes('subway') || lowerCommand.includes('metro')) {
      await speak("Searching for subway stations around you. Give me a second.");
      await findNearestTransit('subway');
    } else if (lowerCommand.includes('what') && (lowerCommand.includes('see') || lowerCommand.includes('detect'))) {
      const objectCount = detections.length;
      if (objectCount > 0) {
        const objects = detections.map(d => d.class).join(', ');
        await speak(`I can see ${objectCount} objects right now: ${objects}. Is there something specific you'd like me to help you with?`);
      } else {
        await speak("I'm currently scanning the area. I don't see any specific objects to highlight at the moment. Feel free to ask me anything else!");
      }
    } else if (lowerCommand.includes('help')) {
      await speak("I'm here to help! I can guide you to train stations, bus stops, or subway stations. I can also tell you what objects I see around you. What would you like to do?");
    } else {
      const responses = [
        "I didn't quite catch that. Could you try asking me to find a train station, bus stop, or tell you what I see?",
        "Hmm, I'm not sure what you meant. Try saying something like 'take me to the train station' or 'what do you see'?",
        "I'm still learning! Could you rephrase that? I'm great at finding transportation and describing what's around you."
      ];
      const randomResponse = responses[Math.floor(Math.random() * responses.length)];
      await speak(randomResponse);
    }
  };

  // Find nearest transit using Overpass API
  const findNearestTransit = async (type) => {
    if (!userLocation) {
      speak("I need to know your location to help you find transportation. Could you please enable location services, or let me know where you are?");
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
        // More conversational distance announcement
        const distanceText = distance > 1000 ? 
          `about ${(distance/1000).toFixed(1)} kilometers` : 
          `${Math.round(distance)} meters`;
        
        try {
          const response = await fetch('http://127.0.0.1:8000/api/conversation/process', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              message: `Found ${name}, ${distanceText} away. Getting walking directions`,
              session_id: 'default',
              context: {
                destination_name: name,
                distance: distance,
                getting_directions: true
              }
            })
          });
          
          if (response.ok) {
            const data = await response.json();
            speak(data.response);
          } else {
            speak(`Great! I found ${name}. It's ${distanceText} away. Let me get you the walking directions now.`);
          }
        } catch (error) {
          speak(`Perfect! I found ${name} - it's ${distanceText} from here. Getting your route now.`);
        }
        
        await getWalkingDirections(userLocation, { lat: nearest.lat, lng: nearest.lon });
      } else {
        // More conversational "not found" message
        try {
          const response = await fetch('http://127.0.0.1:8000/api/conversation/process', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              message: `No ${type} stations found in the nearby area`,
              session_id: 'default',
              context: {
                search_type: type,
                user_location: userLocation
              }
            })
          });
          
          if (response.ok) {
            const data = await response.json();
            speak(data.response);
          } else {
            speak(`I'm sorry, I couldn't find any ${type} stations in your immediate area. Would you like me to search for a different type of transportation?`);
          }
        } catch (error) {
          speak(`Hmm, I don't see any ${type} stations nearby. Let me know if you'd like me to look for buses or other transportation options instead.`);
        }
      }
    } catch (error) {
      console.error('Transit search error:', error);
      speak("I'm having trouble searching for transit stations at the moment. This might be a connectivity issue. Would you like me to try again, or is there something else I can help you with?");
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
        
        // More conversational route announcement
        const walkTime = duration === 1 ? '1 minute' : `${duration} minutes`;
        const walkDistance = distance > 1000 ? `${(distance/1000).toFixed(1)} kilometers` : `${distance} meters`;
        
        try {
          const response = await fetch('http://127.0.0.1:8000/api/conversation/process', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              message: `Walking route calculated: ${walkDistance}, approximately ${walkTime}`,
              session_id: 'default',
              context: {
                route_distance: distance,
                route_duration: duration,
                navigation_starting: true
              }
            })
          });
          
          if (response.ok) {
            const data = await response.json();
            speak(data.response);
          } else {
            speak(`Perfect! I've found your route. It's ${walkDistance} and should take about ${walkTime}. Let's get you there - I'll guide you step by step.`);
          }
        } catch (error) {
          speak(`Great! Your route is ready. It's ${walkDistance} and will take about ${walkTime}. Starting navigation now!`);
        }
        
        setCurrentRoute(route);
        setNavigationActive(true);
        setCurrentStepIndex(0);
        
        // Start with first direction
        if (route.legs && route.legs[0].steps && route.legs[0].steps.length > 0) {
          const steps = route.legs[0].steps;
          const firstStep = steps[0];
          const firstDirection = formatNavigationDirection(firstStep);
          setCurrentDirection(firstDirection);
          
          // Set next direction if available
          if (steps.length > 1) {
            const secondStep = steps[1];
            const secondDirection = formatNavigationDirection(secondStep);
            setNextDirection(secondDirection);
          }
          
          speak(firstDirection);
          setLastDirectionTime(Date.now());
          
          // Set up location tracking for turn-by-turn
          startLocationTracking(steps);
        }
      } else {
        speak("I'm having trouble calculating a walking route right now. This might be due to network issues or the area might not be well mapped. Would you like me to try a different destination?");
      }
    } catch (error) {
      console.error('Routing error:', error);
      speak("I'm experiencing some technical difficulties getting directions at the moment. This could be a network issue. Please try again in a moment, or let me know if there's anything else I can help you with.");
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
          
          // Update next direction to the step after next
          const stepAfterNext = nextStepIndex + 1;
          if (stepAfterNext < steps.length) {
            const futureDirection = formatNavigationDirection(steps[stepAfterNext]);
            setNextDirection(futureDirection);
          } else {
            setNextDirection('You will arrive at your destination');
          }
          
          // More conversational turn announcements
          const distanceText = distanceToNext > 100 ? 
            `in about ${Math.round(distanceToNext/10)*10} meters` :
            distanceToNext > 20 ? `in ${Math.round(distanceToNext)} meters` : 'coming up';
          
          speak(`${distanceText}, ${nextDirection}`);
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
        speak("Congratulations! You've arrived at your destination. I hope the journey went smoothly. Is there anything else I can help you with?");
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
    setNextDirection('');
    setCurrentRoute(null);
    setCurrentStepIndex(0);
    if (window.navigationWatchId) {
      navigator.geolocation.clearWatch(window.navigationWatchId);
    }
    speak("Navigation has been stopped. If you need directions again, just let me know where you'd like to go!");
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

  // Generate speech using ElevenLabs API
  const generateElevenLabsSpeech = async (text) => {
    try {
      const voiceId = ELEVENLABS_VOICE_IDS[selectedVoice];
      const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
        method: 'POST',
        headers: {
          'Accept': 'audio/mpeg',
          'Content-Type': 'application/json',
          'xi-api-key': ELEVENLABS_API_KEY
        },
        body: JSON.stringify({
          text: text,
          model_id: 'eleven_monolingual_v1',
          voice_settings: {
            stability: 0.5,
            similarity_boost: 0.75,
            style: 0.5,
            use_speaker_boost: true
          }
        })
      });

      if (!response.ok) {
        throw new Error(`ElevenLabs API error: ${response.status}`);
      }

      const audioBlob = await response.blob();
      return URL.createObjectURL(audioBlob);
    } catch (error) {
      console.error('ElevenLabs API error:', error);
      return null;
    }
  };

  // Play audio from URL
  const playAudio = (audioUrl) => {
    return new Promise((resolve, reject) => {
      // Stop any currently playing audio
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
      }

      const audio = new Audio(audioUrl);
      setCurrentAudio(audio);
      setIsSpeaking(true);

      audio.onended = () => {
        setIsSpeaking(false);
        setCurrentAudio(null);
        URL.revokeObjectURL(audioUrl);
        resolve();
      };

      audio.onerror = (error) => {
        setIsSpeaking(false);
        setCurrentAudio(null);
        URL.revokeObjectURL(audioUrl);
        reject(error);
      };

      audio.play().catch(reject);
    });
  };

  // Fallback browser speech synthesis
  const speakWithBrowserSynthesis = (text) => {
    try {
      if (!window.speechSynthesis) {
        console.log('Speech synthesis not available');
        return;
      }

      window.speechSynthesis.cancel();
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = selectedVoice === 'male' ? 0.7 : 0.8;
      utterance.pitch = selectedVoice === 'male' ? 0.7 : 1.1;
      utterance.volume = 1.0;
      utterance.lang = 'en-US';
      
      utterance.onstart = () => {
        console.log('🔊 Speaking (Browser):', text);
        setIsSpeaking(true);
      };
      
      utterance.onend = () => {
        console.log('🔊 Speech completed (Browser)');
        setIsSpeaking(false);
      };
      
      utterance.onerror = (e) => {
        console.error('Browser speech error:', e.error);
        setIsSpeaking(false);
      };

      window.speechSynthesis.speak(utterance);
    } catch (error) {
      console.error('Browser voice error:', error);
      setIsSpeaking(false);
    }
  };

  // Enhanced speak function with ElevenLabs integration
  const speak = async (text) => {
    try {
      console.log('🔊 Speaking:', text);
      
      // Stop any currently playing audio
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
      }
      
      if (isElevenLabsEnabled) {
        try {
          const audioUrl = await generateElevenLabsSpeech(text);
          if (audioUrl) {
            await playAudio(audioUrl);
            return;
          }
        } catch (error) {
          console.warn('ElevenLabs failed, falling back to browser synthesis:', error);
        }
      }
      
      // Fallback to browser speech synthesis
      speakWithBrowserSynthesis(text);
    } catch (error) {
      console.error('Voice error:', error);
      setIsSpeaking(false);
    }
  };

  // Start voice command listening
  const startListening = () => {
    if (recognition && !isListening) {
      recognition.start();
    }
  };

  // Voice system initialization
  const initializeVoice = () => {
    if (!voiceInitialized) {
      console.log('Initializing voice system...');
      
      const welcomeMessages = [
        "Hello! I'm your vision assistant. I'm here to help you navigate and describe what's around you. Just ask me anything!",
        "Hi there! Your conversational vision assistant is ready. I can help you find places or tell you what I see. How can I assist you?",
        "Welcome! I'm your AI companion for navigation and visual assistance. Feel free to ask me about directions or what's in your surroundings."
      ];
      const randomWelcome = welcomeMessages[Math.floor(Math.random() * welcomeMessages.length)];
      speak(randomWelcome);
      
      setVoiceInitialized(true);
      console.log('Voice system initialized with', isElevenLabsEnabled ? 'ElevenLabs' : 'browser synthesis');
    }
  };

  // Stop any currently playing speech
  const stopSpeech = () => {
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.currentTime = 0;
      setCurrentAudio(null);
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  };

  // Play assistant response from backend
  const playAssistantResponse = async (audioUrl) => {
    try {
      // Stop any currently playing audio
      stopSpeech();
      
      // Fetch audio from backend
      const fullUrl = audioUrl.startsWith('http') ? audioUrl : `http://127.0.0.1:8000${audioUrl}`;
      const response = await fetch(fullUrl);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch audio: ${response.status}`);
      }
      
      const audioBlob = await response.blob();
      const audioObjectUrl = URL.createObjectURL(audioBlob);
      
      await playAudio(audioObjectUrl);
    } catch (error) {
      console.error('Error playing assistant response:', error);
      // Fallback to text-to-speech if audio fails
      const transcript = transcripts.find(t => t.audio_url === audioUrl);
      if (transcript && transcript.text) {
        speak(transcript.text);
      }
    }
  };

  // Demo function to test assistant conversation
  const testAssistantConversation = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/demo/conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: 'Hello, can you help me navigate?'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Demo conversation created:', data);
      }
    } catch (error) {
      console.error('Error testing conversation:', error);
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

  // Enhanced conversational detection announcements
  const announceDetection = async (objectClass, confidence) => {
    const now = Date.now();
    const lastAnnounced = lastVoiceAnnouncement[objectClass] || 0;
    const cooldownPeriod = 30000; // 30 seconds

    if (now - lastAnnounced > cooldownPeriod) {
      try {
        // Send detection to backend for conversational announcement
        const response = await fetch('http://127.0.0.1:8000/api/conversation/process', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: `I detected a ${objectClass} with ${Math.round(confidence * 100)}% confidence`,
            session_id: 'default',
            context: {
              detected_object: objectClass,
              confidence: confidence,
              is_vehicle: vehicleTypes.includes(objectClass),
              total_detections: detections.length,
              user_location: userLocation
            }
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          speak(data.response);
        } else {
          // Fallback to more natural basic announcements
          const isVehicle = vehicleTypes.includes(objectClass);
          const naturalAnnouncements = {
            person: [
              "I notice someone nearby",
              "There's a person in view", 
              "I can see someone ahead"
            ],
            car: [
              "There's a car coming",
              "I see a vehicle approaching",
              "Car detected in the area"
            ],
            bus: [
              "I spot a bus nearby",
              "There's a bus in the vicinity",
              "Bus coming into view"
            ],
            chair: [
              "I see a chair here",
              "There's seating available",
              "Found a chair nearby"
            ]
          };
          
          const announcements = naturalAnnouncements[objectClass] || [
            `I can see a ${objectClass} nearby`,
            `There's a ${objectClass} in view`,
            `${objectClass} detected in the area`
          ];
          
          const randomAnnouncement = announcements[Math.floor(Math.random() * announcements.length)];
          speak(randomAnnouncement);
        }
      } catch (error) {
        console.error('Error with conversational detection:', error);
        // Simple fallback
        const isVehicle = vehicleTypes.includes(objectClass);
        const message = isVehicle ? 
          `I see a ${objectClass} vehicle nearby` : 
          `I notice a ${objectClass} in the area`;
        speak(message);
      }
      
      setLastVoiceAnnouncement(prev => ({
        ...prev,
        [objectClass]: now
      }));
      
      console.log(`🔊 Announced: ${objectClass}`);
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

  // Enhanced test voice function
  const testVoice = async () => {
    if (!voiceInitialized) {
      initializeVoice();
    } else {
      try {
        // Get a conversational test message from the AI
        const response = await fetch('http://127.0.0.1:8000/api/conversation/process', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: 'This is a voice system test',
            session_id: 'default',
            context: {
              voice_test: true,
              elevenlabs_enabled: isElevenLabsEnabled,
              voice_type: selectedVoice
            }
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          speak(data.response);
        } else {
          const testMessages = [
            `Hi there! Voice test successful. I'm using ${isElevenLabsEnabled ? 'ElevenLabs AI' : 'browser'} speech with the ${selectedVoice} voice. How can I help you today?`,
            `Hello! Your voice assistant is working perfectly. I'm ready to help you navigate or tell you about your surroundings. What would you like to do?`,
            `Voice test complete! I'm here and ready to assist you. Try asking me to find a train station or tell you what I see around you.`
          ];
          const randomMessage = testMessages[Math.floor(Math.random() * testMessages.length)];
          speak(randomMessage);
        }
      } catch (error) {
        console.error('Test voice error:', error);
        speak(`Voice test successful! I'm your conversational vision assistant, ready to help you navigate and explore your surroundings.`);
      }
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
      const boxColor = isVehicle ? 'border-amber-500 bg-amber-500' : 'border-rose-500 bg-rose-500';
      
      return (
        <div
          key={index}
          className={`absolute border-2 ${boxColor} bg-opacity-20 rounded-xl backdrop-blur-sm transition-all duration-300 hover:bg-opacity-30`}
          style={{
            left: x * scaleX,
            top: y * scaleY,
            width: width * scaleX,
            height: height * scaleY,
          }}
        >
          <div className={`${isVehicle ? 'bg-amber-600' : 'bg-rose-600'} text-white px-3 py-1 text-xs rounded-t-xl font-medium`}>
            {detection.class} ({Math.round(detection.score * 100)}%)
          </div>
        </div>
      );
    });
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center bg-black/10 backdrop-blur-sm p-4 rounded-md">
          <div className="animate-spin rounded-full h-6 w-6 border-b border-white/50 mx-auto mb-2"></div>
          <div className="text-white/70 text-xs">
            {!isModelLoaded ? 'Loading...' : 'Starting...'}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black relative">
      {/* Full-screen video */}
      <video
        ref={videoRef}
        className="w-full h-screen object-cover"
        muted
        playsInline
      />
      
      {/* Enhanced Vision Assistant Header */}
      <div className="absolute top-4 left-4 bg-black/40 backdrop-blur-md border border-white/20 text-white px-4 py-2 rounded-lg z-30 shadow-lg">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-white text-sm font-medium">Vision Assistant</span>
          </div>
          <div className="text-white/80 text-xs bg-green-500/20 px-2 py-1 rounded">
            LIVE
          </div>
          {isSpeaking && (
            <div className="text-green-400 text-xs bg-green-500/20 px-2 py-1 rounded animate-pulse">
              🔊 Speaking
            </div>
          )}
        </div>
      </div>

      {/* Enhanced Voice Controls Panel */}
      <div className="absolute top-20 left-4 bg-black/40 backdrop-blur-md border border-white/20 text-white p-4 rounded-lg z-30 shadow-lg">
        <div className="space-y-3">
          <div className="text-white/90 text-sm font-medium mb-2">Voice Controls</div>
          
          <div className="flex items-center space-x-3">
            <button 
              onClick={startListening}
              disabled={!recognition || isListening}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-all shadow-md ${
                isListening 
                  ? 'bg-red-500/80 hover:bg-red-500 text-white' 
                  : 'bg-blue-500/80 hover:bg-blue-500 text-white'
              }`}
            >
              <span>🎤</span>
              <span>{isListening ? 'Listening...' : 'Voice Command'}</span>
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <select 
                value={selectedVoice} 
                onChange={(e) => setSelectedVoice(e.target.value)}
                className="bg-white/10 border border-white/30 text-white text-xs px-2 py-1 rounded focus:outline-none focus:ring-1 focus:ring-blue-400"
              >
                <option value="female" className="bg-gray-800">♀ Bella</option>
                <option value="male" className="bg-gray-800">♂ Josh</option>
              </select>
              
              <label className="flex items-center space-x-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={isElevenLabsEnabled}
                  onChange={(e) => setIsElevenLabsEnabled(e.target.checked)}
                  className="w-3 h-3 text-blue-400 bg-white/10 border-white/30 rounded"
                />
                <span className="text-white/80 text-xs">Natural AI</span>
              </label>
            </div>
            
            {isSpeaking && (
              <button
                onClick={stopSpeech}
                className="text-red-400 hover:text-red-300 text-xs font-medium bg-red-500/20 px-2 py-1 rounded"
              >
                Stop
              </button>
            )}
          </div>
          
          <div className="text-xs text-white/70 bg-white/10 px-2 py-1 rounded">
            💡 Try: "Take me to the train station" or "What do you see?"
            {isElevenLabsEnabled && (
              <div className="text-green-400 mt-1">✨ Conversational AI Voice Active</div>
            )}
          </div>
        </div>
      </div>

      {/* Enhanced Status Panel */}
      <div className="absolute top-4 right-4 bg-black/40 backdrop-blur-md border border-white/20 text-white p-3 rounded-lg z-30 shadow-lg">
        <div className="space-y-2 text-sm">
          <div className="text-white/90 font-medium mb-2">System Status</div>
          
          <div className="flex items-center justify-between">
            <span className="text-white/80 text-xs">OMI Device:</span>
            <div className="flex items-center space-x-1">
              <div className={`w-2 h-2 rounded-full ${
                isConnectedToOMI ? 'bg-green-400' : 'bg-red-400'
              }`}></div>
              <span className={`text-xs ${
                isConnectedToOMI ? 'text-green-300' : 'text-red-300'
              }`}>
                {isConnectedToOMI ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-white/80 text-xs">Detections:</span>
            <span className="text-white/90 text-xs bg-white/10 px-2 py-1 rounded">
              {detections.length} objects
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-white/80 text-xs">Voice:</span>
            <span className={`text-xs px-2 py-1 rounded ${
              voiceInitialized 
                ? 'text-green-300 bg-green-500/20' 
                : 'text-yellow-300 bg-yellow-500/20'
            }`}>
              {voiceInitialized ? 'Ready' : 'Tap to enable'}
            </span>
          </div>
        </div>
      </div>
      
      {/* Minimal Action Buttons */}
      <div className="absolute bottom-3 right-3 flex gap-1 z-30">
        <button 
          onClick={runManualDetection}
          className="bg-black/10 hover:bg-black/20 backdrop-blur-sm text-white p-1 rounded transition-all"
          title="Detect"
        >
          <span className="text-xs">🔍</span>
        </button>
        <button 
          onClick={testVoice}
          disabled={isSpeaking}
          className={`backdrop-blur-sm text-white p-1 rounded transition-all ${
            isSpeaking 
              ? 'bg-gray-400/20 cursor-not-allowed' 
              : 'bg-black/10 hover:bg-black/20'
          }`}
          title="Test"
        >
          <span className="text-xs">🔊</span>
        </button>
      </div>
      
      {/* Detection Overlay */}
      <div className="absolute inset-0 pointer-events-none">
        {renderDetectionBoxes()}
      </div>

      {/* Ultra-minimal Navigation */}
      {navigationActive && currentDirection && (
        <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 bg-black/10 backdrop-blur-sm text-white px-3 py-1 rounded-md z-40 max-w-sm">
          <div className="flex items-center justify-between gap-2">
            <div className="flex-1 text-xs text-white/80">
              🧭 {currentDirection.substring(0, 35)}...
            </div>
            <button 
              onClick={stopNavigation}
              className="bg-red-400/60 hover:bg-red-400/80 px-1 py-0.5 rounded text-xs text-white transition-all"
            >
              ✕
            </button>
          </div>
        </div>
      )}



      
      {/* Enhanced OMI Transcripts Panel */}
      {transcripts.length > 0 && (
        <div className="absolute bottom-4 left-4 bg-black/40 backdrop-blur-md border border-white/20 text-white p-4 rounded-lg z-30 max-w-md shadow-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="text-white/90 text-sm font-medium">OMI Transcripts</div>
            <div className={`w-2 h-2 rounded-full ${
              isConnectedToOMI ? 'bg-green-400' : 'bg-red-400'
            }`}></div>
          </div>
          
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {transcripts.slice(-3).map((transcript, index) => (
              <div key={transcript.id} className={`p-3 rounded-lg border text-sm ${
                transcript.is_user 
                  ? 'bg-blue-500/20 border-blue-400/30 text-blue-200' 
                  : 'bg-white/10 border-white/20 text-white/90'
              }`}>
                <div className="flex items-start space-x-2">
                  <span className="text-xs">
                    {transcript.is_user ? '👤' : '🤖'}
                  </span>
                  <div className="flex-1">
                    <div className="font-medium text-xs mb-1">
                      {transcript.is_user ? 'You:' : 'Assistant:'}
                    </div>
                    <div className="text-xs leading-relaxed">
                      {transcript.text}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="mt-2 text-xs text-white/60 text-center">
            {isConnectedToOMI ? 
              '🟢 Device Connected - Listening for commands' : 
              '🔴 Device Disconnected'
            }
          </div>
        </div>
      )}

      {/* Ultra-minimal Detection */}
      {detections.length > 0 && (
        <div className="absolute bottom-12 left-3 bg-black/5 backdrop-blur-sm text-white px-2 py-1 rounded-md z-30">
          <div className="flex gap-1 text-xs text-white/60">
            {detections.slice(0, 2).map((detection, index) => {
              const isVehicle = vehicleTypes.includes(detection.class);
              return (
                <span
                  key={index}
                  className={isVehicle ? 'text-amber-300/80' : 'text-rose-300/80'}
                >
                  {detection.class.substring(0, 3)}
                </span>
              );
            })}
            {detections.length > 2 && <span>+{detections.length - 2}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;