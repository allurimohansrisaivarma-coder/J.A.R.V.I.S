# J.A.R.V.I.S. — Desktop Intelligence Platform

A full-featured Windows desktop AI assistant with real-time voice conversations, screen understanding, file intelligence, long-term memory, and a translucent Iron Man HUD interface. Built on free-tier cloud APIs.

---

## Features

| Feature | Description |
|---|---|
| **Iron Man HUD UI** | Frameless, always-on-top pywebview window with a glowing holographic core, HUD-styled chat stream, and frosted glass aesthetic |
| **Hold-to-Speak Voice** | Hold the mic button or **Spacebar** to speak. JARVIS listens, transcribes (Groq Whisper), responds via LLM, and speaks back (Edge TTS) |
| **Widget Mode** | Collapse the full UI into a small floating core icon. Click it to expand back |
| **Live TTS Pipeline** | 3-stage async pipeline (buffer → download → play) begins with a short natural phrase and shows each phrase as its audio starts |
| **Screen Understanding** | Ask "what's on my screen?" and JARVIS captures + analyzes your display via Gemini Vision |
| **File Access** | JARVIS can see and read files from your Desktop, Documents, and Downloads folders |
| **Long-Term Memory** | LanceDB vector store + SQLite metadata for persistent conversational memory across sessions, powered by semantic HyDE query expansion |
| **Google Calendar** | Read upcoming events and create new ones (requires OAuth setup — see below) |
| **Smart Model Router** | Intent-based routing across Gemini Flash, Gemini Pro, and Groq Llama for cost-optimal inference |
| **Web Search** | Automatic DuckDuckGo search for real-time information when needed |
| **Barge-In** | Speak while JARVIS is talking to interrupt and take over |

---

## Quick Start

### Prerequisites
- **Python 3.11+** (tested on 3.11.9)
- **Windows 10/11** (uses pywebview with EdgeChromium, dxcam for screen capture, pywin32)
- API keys (both have generous free tiers):
  - [Google Gemini](https://aistudio.google.com/apikey) — primary reasoning, vision, and streaming
  - [Groq](https://console.groq.com/keys) — fast STT (Whisper) and lightweight LLM routing

### Installation

```bash
# Clone and enter the project
cd JARVIS

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install the project and all dependencies
pip install -e ".[dev]"

# Configure API keys
copy .env.example .env
# Edit .env with your API keys (see .env.example for details)

# Run JARVIS (Terminal mode)
jarvis

# Run JARVIS (Desktop UI mode)
jarvis --ui
```

### Running Modes

| Command | Mode | Description |
|---|---|---|
| `jarvis` | Terminal | Text-based chat in your terminal with colored output |
| `jarvis --voice` | Terminal + Voice | Terminal chat with voice conversation loop |
| `jarvis --ui` | Desktop HUD | Full Iron Man-style desktop application |
| `jarvis --version` | — | Print version and exit |

---

## Project Architecture

```
src/jarvis/
├── config/
│   └── settings.py          # Pydantic settings with YAML override support
├── context/
│   ├── engine.py             # Context orchestrator (runs all sources concurrently)
│   ├── base.py               # Abstract ContextSource interface
│   ├── memory_source.py      # Injects relevant memories into prompts
│   ├── web_source.py         # DuckDuckGo search for real-time info
│   ├── file_source.py        # Desktop/Documents/Downloads file scanning
│   └── screen_source.py      # Screenshot capture + change detection
├── core/
│   ├── session.py            # Session lifecycle, system prompt, tool hooks
│   └── conversation.py       # Sliding-window conversation buffer
├── llm/
│   ├── base.py               # Abstract LLMProvider + data models
│   ├── gemini.py             # Google Gemini provider (generate + stream)
│   ├── groq_provider.py      # Groq provider (Llama + Whisper)
│   └── router.py             # Intent-based model router with fallback
├── memory/
│   ├── manager.py            # MemoryManager (search + store via LanceDB)
│   ├── extractor.py          # Background memory extraction from conversations
│   ├── embeddings.py         # Text embeddings via Gemini embedding API
│   └── __init__.py           # SQLite schema init
├── tools/
│   └── calendar_tool.py      # Google Calendar OAuth + read/create events
├── ui/
│   ├── app.py                # pywebview window + JarvisAPI bridge
│   ├── server.py             # WebSocket server (bridges frontend ↔ backend)
│   └── static/
│       ├── index.html         # HUD layout
│       ├── styles.css         # Iron Man dark theme + widget mode
│       └── app.js             # WebSocket client, PTT, settings, widget toggle
├── voice/
│   ├── capture.py            # Microphone input + Silero VAD + PTT recording
│   ├── stt.py                # Speech-to-Text via Groq Whisper
│   ├── tts.py                # Text-to-Speech via Edge TTS (3-stage pipeline)
│   ├── manager.py            # Voice state machine (IDLE→RECORDING→THINKING→SPEAKING)
│   └── hotkey.py             # Global hotkey listener (pynput)
├── utils/
│   └── logging.py            # Structured JSON logging (structlog)
└── main.py                   # CLI entry point + argument parsing
```

---

## Model Router

JARVIS uses an **intent-based model router** that classifies each query and routes to the optimal provider:

| Tier | Provider | Use Case |
|---|---|---|
| FAST | Groq Llama 3.1 8B | Greetings, simple questions, follow-ups |
| STANDARD | Gemini Flash | General reasoning, coding, analysis, streaming |
| COMPLEX | Gemini Pro | Deep reasoning, large context, vision |

### Free Tier Budget (Daily)

| Resource | Limit | Estimated Use (200 interactions) | Headroom |
|---|---|---|---|
| Gemini Flash RPD | 1,500 | ~400 | 73% |
| Groq LLM RPD | 14,400 | ~300 | 98% |
| Gemini Pro RPD | 50 | ~5–10 | 80% |

---

## Voice System

### How it Works
1. **Hold Spacebar** (or hold the mic button in the UI) to start recording
2. Release to stop — audio is captured via `sounddevice` with Silero VAD
3. Audio is transcribed via **Groq Whisper** (fast, free-tier STT)
4. Transcript is sent to the LLM for response generation (streamed)
5. Response is spoken back via **Edge TTS** with a 3-stage async pipeline:
   - Stage 1: Buffer a short natural phrase, without waiting for a long sentence to finish
   - Stage 2: Download MP3 audio for each chunk in background
   - Stage 3: Play audio sequentially via pygame mixer
6. **Live transmission UI**: each phrase appears in chat as its audio begins, rather than ahead of the voice
7. **Barge-in**: use the mic button or configured global push-to-talk key while JARVIS is talking to interrupt playback

### Configuration (`config/jarvis.yaml`)
```yaml
voice:
  enabled: true
  push_to_talk_key: "ctrl+shift+j"    # Global hotkey
  tts_voice: "en-GB-RyanNeural"        # British male voice
  tts_rate: "+20%"                     # Speed adjustment
  tts_pitch: "-5Hz"                     # Lower, more authoritative delivery
  silence_duration: 0.55                 # VAD silence cutoff (seconds)
```

---

## Google Calendar Setup

JARVIS can read your upcoming events and create new ones via Google Calendar API. This requires a one-time OAuth 2.0 setup.

### Step 1: Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **"Create Project"** → name it "JARVIS" → click **Create**
3. Select the new project from the dropdown at the top

### Step 2: Enable the Calendar API
1. Go to **APIs & Services → Library**
2. Search for **"Google Calendar API"** → click it → click **Enable**
3. *(Optional for Gmail)* Search for **"Gmail API"** → click it → click **Enable**

### Step 3: Create OAuth 2.0 Credentials
1. Go to **APIs & Services → Credentials**
2. Click **"+ CREATE CREDENTIALS"** → **"OAuth client ID"**
3. If prompted, configure the **OAuth consent screen**:
   - User Type: **External** → Create
   - App name: "JARVIS", User support email: your email
   - Add your email to **Test users**
   - Click **Save and Continue** through all steps
4. Back in Credentials → **Create OAuth client ID**:
   - Application type: **Desktop app**
   - Name: "JARVIS Desktop"
   - Click **Create**
5. Click **"DOWNLOAD JSON"** → save the file

### Step 4: Place the Credentials
```bash
# Recommended: persistent user configuration outside the source tree
mkdir %APPDATA%\JARVIS
copy path\to\downloaded\credentials.json %APPDATA%\JARVIS\credentials.json

# Alternative: set JARVIS_GOOGLE_CREDENTIALS_PATH to the downloaded file's absolute path
```

### Step 5: First Run
The first time you ask JARVIS about your calendar or Gmail, it will open your browser for Google sign-in. Click **"Allow"** to grant access. Refreshable per-service tokens are saved in `%APPDATA%\JARVIS`, outside the source tree, so updates do not break access. Gmail creates and shows a draft before it sends anything; a later, explicit confirmation sends that exact draft.

**Try asking:**
- *"What's on my schedule today?"*
- *"Create a meeting tomorrow at 3 PM called Team Sync"*

---

## Desktop UI (HUD Mode)

The UI is a frameless, always-on-top pywebview window styled as an Iron Man HUD:

- **Top bar**: System status ticker + Settings gear + Collapse button
- **Central Core**: Animated holographic rings that change color based on state
- **Chat Stream**: Scrollable message area with JARVIS and user messages
- **Input Area**: Text input + Send button + Mic button (hold to speak)
- **Widget Mode**: Click the collapse button to shrink to a small floating core. Click the core to expand back.

### Core States

| State | Color | Meaning |
|---|---|---|
| IDLE | Cyan | Ready and waiting |
| LISTENING | Green | Microphone is active |
| RECORDING | Orange | Speech detected, capturing |
| THINKING | Yellow | Processing with LLM |
| SPEAKING | Blue | Playing TTS audio |

---

## Configuration

### Environment Variables (`.env`)
```env
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

### YAML Config (`config/jarvis.yaml`)
Override any default setting:
```yaml
llm:
  default_provider: gemini
  gemini:
    model_flash: "gemini-3.5-flash"
    model_pro: "gemini-3.5-flash"
    temperature: 0.7
  groq:
    model: "llama-3.1-8b-instant"

voice:
  enabled: true
  tts_voice: "en-GB-RyanNeural"
  silence_duration: 1.5

memory:
  enabled: true

logging:
  level: INFO
  format: json
```

---

## CLI Commands (Terminal Mode)

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/new` | Start a new conversation |
| `/history` | Show conversation history |
| `/voice` | Toggle voice mode on/off |
| `/quit` | Exit the application |

---

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=jarvis --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

---

## Dependencies

All dependencies are listed in `pyproject.toml`. Key packages:

| Package | Purpose |
|---|---|
| `google-genai` | Gemini LLM + Vision + Embeddings |
| `groq` | Groq Whisper STT + Llama LLM |
| `pywebview` | Desktop window (EdgeChromium backend) |
| `websockets` | Frontend ↔ Backend communication |
| `edge-tts` | Free Microsoft Text-to-Speech |
| `pygame` | Audio playback engine |
| `sounddevice` / `soundfile` | Microphone capture |
| `onnxruntime` | Silero VAD (voice activity detection) |
| `lancedb` | Vector store for long-term memory |
| `sqlalchemy` | SQLite metadata for memories |
| `mss` / `dxcam` / `Pillow` | Screen capture |
| `pynput` | Global hotkey listener |
| `google-api-python-client` | Google Calendar/Gmail API |
| `google-auth-oauthlib` | OAuth 2.0 authentication flow |

---

## License

This project is open-source and available under the [MIT License](LICENSE).
