# J.A.R.V.I.S. — Desktop Intelligence Platform

A Windows 10/11 desktop AI assistant with voice conversations, screen understanding, guarded local-file intelligence, local application launching, optional long-term memory, and an Iron Man-inspired HUD. Version **0.2.0** uses Groq for ordinary work and reserves Gemini for complex reasoning and vision so free-tier usage remains practical. Cloud availability and quotas still depend on your provider accounts.

---

## Features

| Feature | Description |
|---|---|
| **Iron Man HUD UI** | Frameless, always-on-top pywebview window with a glowing holographic core, HUD-styled chat stream, and frosted glass aesthetic |
| **Hold-to-Speak Voice** | Hold the mic button or **Spacebar** to speak. JARVIS listens, transcribes (Groq Whisper), responds via LLM, and speaks back (Edge TTS) |
| **Widget Mode** | Collapse into a draggable floating core; use its small ↗ button to restore the full HUD without accidental expansion while dragging |
| **Live TTS Pipeline** | 3-stage async pipeline (buffer → download → play) begins with a short natural phrase and shows each phrase as its audio starts |
| **Screen Understanding** | Ask "what's on my screen?" and JARVIS captures + analyzes your display via Gemini Vision |
| **File Access** | JARVIS can see and read files from your Desktop, Documents, and Downloads folders |
| **Long-Term Memory** | Optional LanceDB vector store + SQLite metadata for persistent conversational memory, powered by local sentence-transformer embeddings and semantic query expansion |
| **Google Calendar** | Read upcoming events and create new ones (requires OAuth setup — see below) |
| **Smart Model Router** | Intent-based routing across Gemini 3.7 Flash reasoning levels and Groq GPT-OSS, with provider fallback and health checks |
| **MCP Tools** | Live Web Search (DuckDuckGo), Weather, Browser integration, and a comprehensive World Monitor for global news and finance feeds |
| **World Monitor** | Full-window system, local-weather, live-headline, and non-interactive Google Calendar overview |
| **Windows App Launcher** | Opens installed desktop apps and Start Menu/PWA registrations locally without spending LLM tokens |
| **Barge-In** | Speak while JARVIS is talking to interrupt and take over |

---

## Quick Start

### Prerequisites
- **Python 3.11+** (tested on 3.11.9)
- **Windows 10/11** (uses pywebview with EdgeChromium, dxcam for screen capture, pywin32)
- API keys:
  - [Groq](https://console.groq.com/keys) — default chat/reasoning provider and Whisper speech-to-text
  - [Google Gemini](https://aistudio.google.com/apikey) — complex reasoning and screen/vision requests

### Installation

```powershell
# Clone and enter the project
cd JARVIS

# Create virtual environment
py -3.11 -m venv .venv

# Install directly through the virtual environment (activation is optional)
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Optional only when neither an installed Google Chrome nor another supported
# Chromium installation is available for browser automation
.\.venv\Scripts\playwright.exe install chromium

# Configure API keys
Copy-Item .env.example .env
# Edit .env with your API keys (see .env.example for details)

# Run JARVIS (Desktop UI mode)
.\.venv\Scripts\python.exe -m jarvis.main --ui
```

PowerShell activation, if desired, is `..\.venv\Scripts\Activate.ps1` only when the environment lives one directory above the project. For the documented `.venv` inside this repository, use `.\.venv\Scripts\Activate.ps1`. Calling the virtual environment's Python directly, as above, avoids execution-policy and path mistakes.

## Run the packaged Windows application

JARVIS is distributed as a **one-folder application** because its AI, audio, screen-capture, and MCP dependencies must remain beside the executable.

1. Download `JARVIS-Windows-x64-v0.2.0.zip` from the GitHub Release.
2. Extract the entire `JARVIS` folder. Do not copy `JARVIS.exe` out by itself.
3. Rename `.env.example` to `.env` inside the extracted folder and add your Groq and Gemini keys.
4. Double-click `JARVIS.exe`. A bundled executable opens the desktop UI automatically.

The laptop needs 64-bit Windows 10/11, Microsoft Edge WebView2 Runtime, internet access for cloud features, and microphone permission for voice input. Google OAuth credentials remain optional and belong in `%APPDATA%\JARVIS\credentials.json`.

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
│   ├── embeddings.py         # Local sentence-transformer embeddings (loaded lazily)
│   └── __init__.py           # SQLite schema init
├── tools/
│   ├── mcp_manager.py        # Central manager for loading and registering MCP servers
│   ├── mcp_ddg.py            # DuckDuckGo search MCP server
│   ├── mcp_weather.py        # Weather MCP server
│   ├── mcp_world_monitor.py  # Global news and financial feeds MCP server
│   ├── mcp_browser.py        # Web scraping and browser automation MCP server
│   ├── windows_launcher.py   # Safe Windows executable, Start Menu, and PWA launching
│   ├── calendar_tool.py      # Google Calendar OAuth + read/create events
│   └── gmail.py              # Gmail reading and drafting
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
| FAST | Groq `openai/gpt-oss-20b` | Greetings and lightweight follow-ups |
| STANDARD | Groq `openai/gpt-oss-20b` | Ordinary questions, coding, web-result summaries, Gmail, and Calendar |
| COMPLEX | Gemini `gemini-3.7-flash` (high thinking) | Explicit deep analysis, vision, and Gemini-native desktop tools |

The default policy is **Groq-first**: both FAST and STANDARD requests use Groq, while Gemini is reserved for requests classified as COMPLEX or requiring Gemini-native vision/function calling. Gemini remains an availability fallback if Groq fails. Providers are never invoked twice during one fallback chain. Set `llm.default_provider: gemini` only if you intentionally want ordinary requests to use Gemini again.

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
- **Widget Mode**: Click the collapse button to shrink to a small floating core. Drag anywhere on the core to move it and use the small **↗** button to restore the 1200×800 HUD.

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
  default_provider: groq  # recommended: preserves Gemini quota
  gemini:
    model_flash: "gemini-3.7-flash"
    model_pro: "gemini-3.7-flash"
    thinking_level_standard: low
    thinking_level_complex: high
    temperature: 0.7
  groq:
    model: "openai/gpt-oss-20b"

voice:
  enabled: true
  tts_voice: "en-GB-RyanNeural"
  silence_duration: 1.5
  stt_model: "whisper-large-v3-turbo"
  stt_language: en

memory:
  enabled: false  # opt in; the embedding model is downloaded on first use

logging:
  level: INFO
  format: json

system:
  port: 8741
  weather_city: "Dubai"  # recommended; blank uses IP-based automatic location
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

```powershell
# Run tests
.\.venv\Scripts\python.exe -m pytest

# Run tests with coverage
.\.venv\Scripts\python.exe -m pytest --cov=jarvis --cov-report=term-missing

# Lint
.\.venv\Scripts\python.exe -m ruff check src tests

# Type check
.\.venv\Scripts\python.exe -m mypy src
```

### Build the Windows release

```powershell
.\.venv\Scripts\python.exe build.py
```

The release build produces:

- `dist\JARVIS\JARVIS.exe` — the updated executable inside its required application folder
- `dist\JARVIS-Windows-x64-v0.2.0.zip` — upload this to a GitHub Release
- `dist\SHA256SUMS.txt` — integrity hashes for the executable and ZIP

`dist/`, user configuration, logs, tokens, and `.env` are intentionally ignored by Git. Push the source repository normally, then attach the generated ZIP and checksum file to the corresponding GitHub Release rather than committing the large binary bundle.

---

## Dependencies

All dependencies are listed in `pyproject.toml`. Key packages:

| Package | Purpose |
|---|---|
| `google-genai` | Gemini LLM, tool calling, and Vision |
| `groq` | Groq Whisper STT + GPT-OSS LLM |
| `sentence-transformers` | Local semantic embeddings for optional memory |
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
| `mcp` | Model Context Protocol server/client support |
| `ddgs` | DuckDuckGo web search tool |
| `httpx` | Async HTTP requests for news and feeds |

### Privacy and safety boundaries

- Local-file context is restricted to the configured project/user roots, resolves symlinks before access, bounds recursive scans, and never returns known secret files such as `.env`, OAuth tokens, or credential files.
- Generic questions are not automatically sent to web search. Search runs only for explicit or clearly time-sensitive requests.
- Live web retrieval uses one bounded MCP search per request, with a four-second search timeout and no repeated Gemini search loop.
- Set `system.weather_city` to avoid IP-based location lookup. When it is blank, World Monitor uses ipwho.is with ipapi.co fallback to approximate local weather before querying Open-Meteo.
- Gmail sends only a previously created draft after a separate explicit confirmation. Long-term-memory deletion likewise requires a preview followed by confirmation.
- API keys come from environment variables or `.env`; saving YAML configuration does not serialize them.
- Explicit app-launch commands are resolved locally against approved Windows executables and registered Start Menu/PWA shortcuts; they do not consume Groq or Gemini tokens.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
