"""Jarvis CLI — Desktop Intelligence Platform.

Usage:
    jarvis              Start interactive chat
    jarvis --version    Show version
    jarvis --config     Path to config file
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Force UTF-8 encoding for Windows terminals
if sys.stdout is not None and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding is not None and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from jarvis import __version__
from jarvis.core.session import SessionManager
from jarvis.memory import MemoryManager, init_db
from jarvis.memory.embeddings import Embedder


# ANSI color codes
class _Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _supports_color() -> bool:
    """Check if the terminal supports ANSI color codes."""
    if sys.platform == "win32":
        # Windows Terminal and modern cmd.exe support ANSI
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


C = _Colors if _supports_color() else type("NoColor", (), {k: "" for k in vars(_Colors) if not k.startswith("_")})()


def print_banner(providers: list[str] | None = None) -> None:
    """Print the startup banner."""
    provider_lines = ""
    if providers:
        for i, p in enumerate(providers):
            prefix = "Providers:" if i == 0 else "          "
            provider_lines += f"│  {prefix} {p:<20} │\n"
    else:
        provider_lines = "│  Providers: (none loaded)        │\n"

    banner = f"""{C.CYAN}
╭──────────────────────────────────╮
│  JARVIS v{__version__:<24}│
│  Desktop Intelligence Platform   │
│                                  │
{provider_lines}│                                  │
│  Type /help for commands         │
╰──────────────────────────────────╯{C.RESET}"""
    print(banner)


def print_help() -> None:
    """Print available commands."""
    print(f"""
{C.YELLOW}Available commands:{C.RESET}
  /help       Show this help message
  /new        Start a new conversation
  /history    Show conversation history summary
  /remember <text>  Manually save a fact to long-term memory
  /recall <query>   Search your semantic memory
  /quit       Exit the application
  /exit       Exit the application
""")


async def async_main(args) -> int:
    """Async entry point for the Jarvis CLI."""
    session = SessionManager(config_path=args.config)

    try:
        await session.initialize()
    except ValueError as e:
        print(f"\n{C.RED}Configuration error: {e}{C.RESET}")
        print(f"{C.DIM}See .env.example for required API keys.{C.RESET}\n")
        return 1
    except Exception as e:
        print(f"\n{C.RED}Initialization failed: {e}{C.RESET}")
        return 1

    print_banner(session.available_providers)

    # Initialize memory if Gemini is available
    memory = None
    try:
        if session.settings.gemini_api_keys and session.settings.primary_gemini_key not in ("fallback", "env_or_placeholder"):
            init_db()
            embedder = Embedder(api_keys=session.settings.gemini_api_keys)
            memory = MemoryManager(embedder=embedder)
            print(f"{C.CYAN}Memory System: Online{C.RESET}")
    except Exception as e:
        print(f"{C.YELLOW}Memory System: Offline ({e}){C.RESET}")

    # Removed UI block from here since it's now handled in main()

    if args.voice:
        try:
            from jarvis.voice import VoiceManager, VoiceState
            from jarvis.voice.stt import STTProvider
            from jarvis.voice.tts import TTSProvider

            if not session.settings.groq_api_key:
                print(f"\n{C.RED}Voice mode requires GROQ_API_KEY for Whisper STT.{C.RESET}")
                return 1

            print(f"\n{C.YELLOW}[Initializing Voice Mode...]{C.RESET}")
            stt = STTProvider(api_key=session.settings.groq_api_key)
            
            # Use voice config from settings
            voice_cfg = session.settings.voice
            tts = TTSProvider(
                voice=voice_cfg.tts_voice,
                rate=voice_cfg.tts_rate,
                pitch=voice_cfg.tts_pitch,
            )

            manager = VoiceManager(session, stt, tts)

            def on_state_change(state: VoiceState):
                state_colors = {
                    VoiceState.IDLE: C.DIM,
                    VoiceState.LISTENING: C.CYAN,
                    VoiceState.RECORDING: C.RED,
                    VoiceState.THINKING: C.YELLOW,
                    VoiceState.SPEAKING: C.GREEN,
                }
                color = state_colors.get(state, C.RESET)
                print(f"\r{color}Status: {state.name:<10}{C.RESET} ", end="", flush=True)

            manager.on_state_change = on_state_change

            # Voice mode is deliberately push-to-talk only.  A held hotkey
            # interrupts speech first, then records until it is released.
            try:
                from jarvis.voice.hotkey import HotkeyListener
                hotkey = HotkeyListener(hotkey_combo=voice_cfg.push_to_talk_key)
                hotkey.start(asyncio.get_running_loop(), manager.hotkey_queue)
                print(f"{C.CYAN}Push-to-Talk active. Hold [{voice_cfg.push_to_talk_key.upper()}] to speak. Press Ctrl+C to exit.{C.RESET}\n")
                await manager.start_ptt_loop()
            except ImportError:
                print(f"{C.RED}Push-to-talk requires pynput. Install: pip install pynput{C.RESET}")
                return 1
        except KeyboardInterrupt:
            print()
        except Exception as e:
            print(f"\n{C.RED}Voice mode failed: {e}{C.RESET}")
        finally:
            await session.shutdown()
            print(f"\n{C.DIM}Goodbye.{C.RESET}")
        return 0

    try:
        while True:
            try:
                user_input = input(f"\n{C.GREEN}You:{C.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            # Handle commands
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/bye"):
                break
            elif cmd == "/help":
                print_help()
                continue
            elif cmd == "/new":
                session.conversation.end_conversation()
                session.conversation.new_conversation()
                print(f"{C.YELLOW}[Started a new conversation]{C.RESET}")
                continue
            elif cmd == "/history":
                history = session.conversation.get_history()
                if history:
                    print(f"{C.YELLOW}[{len(history)} conversation(s) in this session]{C.RESET}")
                    for i, conv in enumerate(history, 1):
                        msg_count = len(conv.user_messages)
                        print(f"  {i}. {msg_count} message(s) — {conv.created_at.strftime('%H:%M:%S')}")
                else:
                    print(f"{C.YELLOW}[No conversation history]{C.RESET}")
                continue
            elif cmd.startswith("/remember "):
                if not memory:
                    print(f"{C.RED}Memory system is offline.{C.RESET}")
                    continue
                fact = user_input[10:].strip()
                memory.store_fact(fact, source_context="Manual entry")
                print(f"{C.GREEN}[Stored in long-term memory]{C.RESET}")
                continue
            elif cmd.startswith("/recall "):
                if not memory:
                    print(f"{C.RED}Memory system is offline.{C.RESET}")
                    continue
                query = user_input[8:].strip()
                results = memory.semantic_search(query, limit=3)
                if not results:
                    print(f"{C.YELLOW}[No relevant memories found]{C.RESET}")
                else:
                    print(f"{C.CYAN}Memory recall for: '{query}'{C.RESET}")
                    for i, r in enumerate(results, 1):
                        print(f"  {i}. {r['content']} (score: {1.0 - r['distance']:.2f})")
                continue

            # Process input with streaming
            print(f"\n{C.BLUE}Jarvis:{C.RESET} ", end="", flush=True)

            try:
                async for chunk in session.process_input_stream(user_input):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"\n{C.RED}Error: {e}{C.RESET}")

    except Exception as e:
        print(f"\n{C.RED}Fatal error: {e}{C.RESET}")
    finally:
        await session.shutdown()
        print(f"\n{C.DIM}Goodbye.{C.RESET}")

    return 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Jarvis CLI — Desktop Intelligence Platform")
    parser.add_argument("--version", action="store_true", help="Show version")
    parser.add_argument("--config", type=Path, help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--voice", action="store_true", help="Start in voice mode")
    parser.add_argument("--ui", action="store_true", help="Launch the desktop UI")

    args = parser.parse_args()
    
    # If running as a bundled executable (double-clicked), default to UI mode
    import sys
    if getattr(sys, 'frozen', False) and not any(arg in sys.argv for arg in ['--ui', '--voice', '--version', '--help', '-h']):
        args.ui = True

    if args.version:
        print(f"Jarvis v{__version__}")
        sys.exit(0)

    if args.ui:
        session = SessionManager(config_path=args.config)
        try:
            asyncio.run(session.initialize())
        except Exception as e:
            print(f"\n{C.RED}Initialization failed: {e}{C.RESET}")
            import sys
            if getattr(sys, 'frozen', False):
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, f"JARVIS Initialization Failed:\n\n{e!s}\n\nPlease ensure your .env file is next to the JARVIS.exe or in the project root.", "JARVIS Error", 0x10)
            sys.exit(1)
            
        print_banner(session.available_providers)
        
        # Initialize memory if Gemini is available
        try:
            if session.settings.gemini_api_keys and session.settings.primary_gemini_key not in ("fallback", "env_or_placeholder"):
                init_db()
                embedder = Embedder(api_keys=session.settings.gemini_api_keys)
                MemoryManager(embedder=embedder)
                print(f"{C.CYAN}Memory System: Online{C.RESET}")
        except Exception as e:
            print(f"{C.YELLOW}Memory System: Offline ({e}){C.RESET}")
            
        # Initialize VoiceManager for Live Conversation if configured
        voice_manager = None
        if session.settings.groq_api_key:
            try:
                from jarvis.voice import VoiceManager
                from jarvis.voice.stt import STTProvider
                from jarvis.voice.tts import TTSProvider
                
                stt = STTProvider(api_key=session.settings.groq_api_key)
                voice_cfg = session.settings.voice
                tts = TTSProvider(
                    voice=voice_cfg.tts_voice,
                    rate=voice_cfg.tts_rate,
                    pitch=voice_cfg.tts_pitch,
                )
                voice_manager = VoiceManager(session, stt, tts)
                print(f"{C.CYAN}Voice System: Online{C.RESET}")
            except Exception as e:
                print(f"{C.YELLOW}Voice System: Offline ({e}){C.RESET}")
                import traceback
                print(traceback.format_exc())

        try:
            from jarvis.ui.app import launch_ui
            launch_ui(session, voice_manager=voice_manager)
        except ImportError as e:
            print(f"\n{C.RED}UI dependencies missing. Install: pip install pywebview websockets{C.RESET}")
            print(f"{C.DIM}Error: {e}{C.RESET}")
        except Exception as e:
            print(f"\n{C.RED}UI failed: {e}{C.RESET}")
        finally:
            asyncio.run(session.shutdown())
            print(f"\n{C.DIM}Goodbye.{C.RESET}")
        sys.exit(0)
    else:
        sys.exit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
