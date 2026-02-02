import { GoogleGenAI, LiveServerMessage, Modality } from "@google/genai";
import { AppMode } from "../types";

const apiKey = process.env.API_KEY || '';
const getClient = () => new GoogleGenAI({ apiKey });

interface ChatParams {
  history: { role: string; parts: { text: string }[] }[];
  message: string;
  mode: AppMode;
  contextData?: string;
  selectedRegion?: any;
}

// 1. Text/Agent Chat Service
export const streamChatResponse = async function* ({ history, message, mode, contextData, selectedRegion }: ChatParams) {
  const ai = getClient();
  
  let modelName = 'gemini-3-pro-preview'; 
  
  // Highly specialized scientific instructions
  let systemInstruction = `You are CORE (Co.Research Engineer), an advanced AI computational biologist. 
  
  CORE OBJECTIVE: Optimize ligands for Intrinsically Disordered Proteins (IDPs) and identify Cryptic Pockets.
  
  METHODOLOGY:
  1. **Symbolic Regression**: When analyzing dynamics, propose mathematical equations (differential equations) that describe the ensemble motion.
  2. **Multi-Agent Simulation**: Assume you are orchestrating a fleet of agents using Evolutionary Algorithms and Reinforcement Learning to explore the conformational space.
  3. **Wet Lab Validation**: Always hypothesis check against standard biophysical constraints (Ramachandran plots, hydrophobicity).
  4. **RDKit/OpenMM**: Write your reasoning as if you are executing Python scripts using these libraries.
  
  VISUALIZATION PROTOCOL:
  When asked to simulate or visualize, you MUST provide a JSON object in a markdown block \`\`\`json:simulation ... \`\`\`. 
  This JSON must represent the protein nodes, ligand position, and importantly "cryptic pockets" (transient cavities).
  `;

  if (selectedRegion) {
    systemInstruction += `\n\nUSER ANNOTATION CONTEXT: The user has visually selected a specific region of the protein cloud. 
    Focus ONLY on these residues: ${JSON.stringify(selectedRegion)}. Explain the atomic interactions and potential cryptic pocket formation in this specific zone.`;
  }

  const tools: any[] = [];
  
  if (mode === AppMode.Ask) {
    systemInstruction += " Use deep reasoning. Access Google Search for latest PDB entries or AlphaFold predictions.";
    tools.push({ googleSearch: {} });
  }

  const config: any = {
    systemInstruction,
    tools,
  };

  // Enable thinking for deep scientific tasks
  if (mode === AppMode.Ask || mode === AppMode.Agent) {
      config.thinkingConfig = { thinkingBudget: 16000 };
  }

  const chat = ai.chats.create({
    model: modelName,
    config
  });

  if (contextData) {
    await chat.sendMessage({ message: `System Context: ${contextData}` });
  }

  const stream = await chat.sendMessageStream({ message });
  
  for await (const chunk of stream) {
    if (chunk.text) {
      yield { text: chunk.text };
    }
    if (chunk.candidates?.[0]?.groundingMetadata?.groundingChunks) {
      yield { grounding: chunk.candidates[0].groundingMetadata.groundingChunks };
    }
  }
};

// 2. Live API Service (The "Teacher")
export const connectLiveSession = async (
  onAudioData: (base64: string) => void,
  onTranscript: (text: string, isUser: boolean) => void
) => {
  const ai = getClient();
  
  // Audio Context setup for input (Microphone)
  const inputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  
  let currentSession: any = null;

  const sessionPromise = ai.live.connect({
    model: 'gemini-2.5-flash-native-audio-preview-12-2025',
    callbacks: {
      onopen: () => {
        console.log("Live Session Connected");
        // Start Input Stream
        const source = inputAudioContext.createMediaStreamSource(stream);
        const scriptProcessor = inputAudioContext.createScriptProcessor(4096, 1, 1);
        
        scriptProcessor.onaudioprocess = (e) => {
          const inputData = e.inputBuffer.getChannelData(0);
          const pcmBlob = createBlob(inputData);
          sessionPromise.then(session => {
             session.sendRealtimeInput({ media: pcmBlob });
          });
        };
        
        source.connect(scriptProcessor);
        scriptProcessor.connect(inputAudioContext.destination);
      },
      onmessage: (message: LiveServerMessage) => {
        // Handle Audio Output
        const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
        if (base64Audio) {
          onAudioData(base64Audio);
        }

        // Handle Transcription
        if (message.serverContent?.modelTurn?.parts[0]?.text) {
           onTranscript(message.serverContent.modelTurn.parts[0].text, false);
        }
      },
      onclose: () => console.log("Live Session Closed"),
      onerror: (err) => console.error("Live Session Error", err)
    },
    config: {
      responseModalities: [Modality.AUDIO],
      systemInstruction: "You are an expert Professor of Biophysics. You are teaching a student about IDP dynamics and ligand docking. Speak with authority but encourage curiosity. Explain concepts at an atomic resolution, referencing forces, entropy, and enthalpy.",
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Fenrir' } }
      }
    }
  });

  return sessionPromise;
};

// PCM Helper
function createBlob(data: Float32Array): any {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  // Simplified base64 encoding for brevity in this context
  const bytes = new Uint8Array(int16.buffer);
  let binary = '';
  for(let i=0; i<bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  const base64 = btoa(binary);

  return {
    data: base64,
    mimeType: 'audio/pcm;rate=16000'
  };
}