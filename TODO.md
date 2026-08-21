# J.A.R.V.I.S. — TODO

## 🔴 Critical Fixes
- [x] **Voice/UI Event Loop Ownership**: UI session, MCP clients, WebSocket server, and voice callbacks now use one owned background event loop with thread-safe audio callback dispatch.
- [ ] **Transparent Window**: `pywebview` with `transparent=True` renders as a white/pale window on certain Windows graphics drivers (WebView2 compositor issue). Need to investigate alternative approaches: WinAPI `SetLayeredWindowAttributes`, or a custom Electron/Tauri shell.

## 🟡 Google Integrations
- [ ] **Google Drive Integration**: Allow JARVIS to search and read files from Google Drive
- [ ] **Google Calendar — Native Tool Calling**: Replace the keyword-matching calendar hook with proper Gemini function calling (tool use) so the LLM decides when to invoke calendar tools

## 🟢 UI / UX Improvements
- [ ] **Drag to Reposition**: Allow the user to drag the frameless window by holding the top bar
- [ ] **System Tray Icon**: Add a Windows system tray icon with right-click menu (Show/Hide, Settings, Quit)
- [ ] **Notification Toasts**: Show Windows toast notifications for JARVIS alerts when the window is minimized
- [ ] **Chat History Persistence**: Save and restore chat messages across app restarts
- [ ] **Markdown Rendering**: Upgrade from basic regex to a proper markdown renderer (marked.js or similar)
- [ ] **Theme Customization**: Let users pick accent colors (cyan, gold, red, green) via settings
- [ ] **Font Size Control**: Add font size slider in settings

## 🔵 Voice Improvements
- [ ] **Wake Word Detection**: Add "Hey JARVIS" wake word using a local model (Porcupine or OpenWakeWord)
- [ ] **Voice Selection UI**: Dropdown in settings to preview and select TTS voices
- [ ] **Streaming STT**: Replace batch Whisper transcription with real-time streaming STT for lower latency
- [ ] **Speaker Diarization**: Distinguish between different speakers in multi-person environments

## 🟣 Intelligence & Tools
- [x] **App Launcher**: Let JARVIS open applications ("open Chrome", "launch VS Code")
- [ ] **Clipboard Integration**: Read/write system clipboard on command
- [ ] **System Commands**: Execute shell commands ("create a folder called X on my desktop")
- [x] **Weather API**: Open-Meteo MCP integration for real-time weather queries
- [ ] **Spotify/Music Control**: Basic media playback control via system APIs
- [ ] **Reminder System**: Time-based reminders with Windows notifications
- [ ] **Multi-Monitor Screen Capture**: Support capturing specific monitors

## ⚪ Technical Debt
- [ ] **Coverage Expansion**: Existing router, voice, memory, settings, provider, context, and MCP tests pass; add deeper live session/UI integration coverage beyond the current unit suite
- [ ] **Error Recovery**: Improve graceful degradation when APIs are down (offline mode with cached responses)
- [x] **Memory Cleanup**: Confirmed semantic memory deletion tool and direct UI deletion
- [x] **Config Validation**: Typed Pydantic validation, invalid-YAML errors, and UI validation responses
- [ ] **Logging Dashboard**: Simple web page to view and filter structured logs
- [ ] **Windows Installer**: Create an MSI/MSIX installer for one-click setup
- [ ] **Auto-Update**: Check for and apply updates from a Git remote
