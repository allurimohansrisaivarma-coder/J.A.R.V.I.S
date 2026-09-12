# 0.2.7

- Prevent Markdown bullets from being sent to speech as empty audio requests.
- Retry Ryan synthesis up to four times and continue later phrases after an isolated failure.
- Use longer natural speech chunks to reduce service requests during daily debriefs.
- Route missed or repeated topic-news requests back to verified feeds instead of model generation.
- Correct fixed Google News fallback URLs so they return RSS directly without redirects.

# 0.2.6

- Recognize spoken "de brief" and "de-brief" variants and block invented continuation items.
- Add dated Google News RSS fallback when a direct publisher feed has insufficient fresh coverage.
- Extract text from bounded, non-encrypted PDFs in approved local folders.
- Resolve top-level Desktop file requests faster and keep Windows paths out of speech.

# 0.2.5

- Route daily debrief requests to fresh publisher feeds instead of language-model news generation.
- Require a publication timestamp within the past 24 hours; reject stale, future, and undated items.
- Show source links and explicit gaps without inventing scores, schedules, standings, or filler headlines.
- Handle personal-memory follow-ups separately and exclude prior assistant claims from live-web context.
- Retain the dark-blue minimal floating icon.

# 0.2.4

- Restore the HUD by clicking the reactor center; full click-through is optional.
- Suppress spoken URLs and citation markers, including links split across streamed chunks.
- Prioritize explicit requests to remember facts after long histories and restarts.

# 0.2.3

- Keep one selected voice per reply; retry without switching to the Windows default voice.
- Restore saved conversations and retrieve relevant older user statements.
- Ground web answers in retrieved sources and stop guessing when search fails.
- Add passive click-through floating orb and Ctrl+Alt+D interaction toggle.
- Show release version in the HUD and add a clear-history control.

# Changelog

## 0.2.2 — Arc reactor HUD

- Redesign the HUD with local SVG arc-reactor artwork, clearer typography, readable chat, and responsive panels.
- Replace the rectangular widget with a 112 px native circular window, hover controls, and independent remembered positions.
- Add desktop shortcuts for floating mode, screen reading, and stopping responses; preserve draft messages during screen requests.
- Reduce idle animation work, pause dashboard polling in compact mode, cap activity logs, and honor reduced-motion settings.
- Fix Space intercepting text entry in Settings and separate window dragging from interactive controls.

## 0.2.1 — Reliability repair

- Fix screenshot fallback dropping images and screenshot objects leaking into conversation history; preserve per-monitor readability.
- Add fast Groq Qwen vision routing with Gemini fallback, provider deadlines, and safe interruption handling.
- Replace duplicated bundled tool subprocesses with in-process tool registration and cached schemas.
- Add encrypted API-key setup, offline Windows speech fallback, actionable voice errors, and Stop/Screen controls.
- Recover from an occupied local port, suppress duplicate instances, authenticate the local UI connection, and keep preferences/logs in per-user storage.
- Add executable diagnostics and regression coverage; build into a new versioned directory to preserve existing installations.


## 0.2.0 — 2026-08-21

- Changed the default model policy to Groq-first for fast and standard requests, reserving Gemini for complex reasoning and vision.
- Added deterministic Windows application, Start Menu shortcut, and PWA launching without LLM usage.
- Added the full-window World Monitor with local weather, live headlines, system resources, and Calendar status.
- Added Gmail draft confirmation and hardened Calendar OAuth handling.
- Fixed desktop file URL startup, Gemini health checks, widget collapse/restore crashes, and widget drag expansion.
- Bounded MCP web retrieval and removed repeated search/tool loops.
- Improved voice interruption, push-to-talk, transcript logging, file safety, memory safety, provider fallback, and shutdown cleanup.
- Added Windows release packaging, version metadata, ZIP generation, and SHA-256 checksums.
- Fixed live F1/news retrieval, tomorrow-weather forecasts, affirmative search follow-ups, and empty provider responses.
- Added deterministic World Monitor aliases for Global Dashboard, Global Tab, World Desk, CPU, RAM, and system-stat requests.
- Added a Groq plain-text retry for erroneous internal tool calls, avoiding unnecessary Gemini fallback and quota usage.
- Added frozen startup crash reports and prevented private runtime configuration from entering release archives.

## 0.1.0

- Initial desktop assistant release.
