"""Jarvis CLI — Desktop Intelligence Platform.

Usage:
    jarvis              Start interactive chat
    jarvis --version    Show version
    jarvis --config     Path to config file
"""

import argparse
import asyncio
import sys
import traceback
from pathlib import Path
from typing import Any, cast

# Force UTF-8 encoding for Windows terminals
if (
    sys.stdout is not None
    and hasattr(sys.stdout, "encoding")
    and sys.stdout.encoding is not None
    and sys.stdout.encoding.lower() != "utf-8"
):
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")

from jarvis import __version__


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


C = (
    _Colors
    if _supports_color()
    else type("NoColor", (), {k: "" for k in vars(_Colors) if not k.startswith("_")})()
)


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


def _report_ui_startup_error(error: BaseException) -> None:
    """Persist frozen UI failures and show an actionable Windows error."""
    detail = "".join(traceback.format_exception(error)).strip()
    message = f"JARVIS failed to start:\n\n{error!s}"

    if getattr(sys, "frozen", False):
        try:
            from jarvis.config.settings import USER_DIR

            log_dir = USER_DIR / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "startup-error.log").write_text(detail + "\n", encoding="utf-8")
            message += f"\n\nDetails: {log_dir / 'startup-error.log'}"
        except Exception as report_error:
            if sys.stderr is not None:
                print(f"Unable to write JARVIS startup log: {report_error}", file=sys.stderr)

        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "JARVIS Error", 0x10)
        except Exception as dialog_error:
            if sys.stderr is not None:
                print(f"Unable to show JARVIS error dialog: {dialog_error}", file=sys.stderr)
        return

    print(f"\n{C.RED}UI failed: {error}{C.RESET}")


async def async_main(args) -> int:
    """Async entry point for the Jarvis CLI."""
    from jarvis.core.session import SessionManager

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

    settings = session.settings
    conversation = session.conversation
    assert settings is not None and conversation is not None

    print_banner(session.available_providers)

    memory = session.memory
    if memory:
        print(f"{C.CYAN}Memory System: Online{C.RESET}")

    # Removed UI block from here since it's now handled in main()

    if args.voice:
        try:
            from jarvis.voice import VoiceManager, VoiceState
            from jarvis.voice.stt import STTProvider
            from jarvis.voice.tts import TTSProvider

            if not settings.primary_groq_key:
                print(f"\n{C.RED}Voice mode requires GROQ_API_KEY for Whisper STT.{C.RESET}")
                return 1

            print(f"\n{C.YELLOW}[Initializing Voice Mode...]{C.RESET}")
            # Use voice config from settings
            voice_cfg = settings.voice
            stt = STTProvider(
                api_key=settings.primary_groq_key,
                model=voice_cfg.stt_model,
                language=voice_cfg.stt_language,
            )
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
                print(
                    f"{C.CYAN}Push-to-Talk active. Hold [{voice_cfg.push_to_talk_key.upper()}] to speak. Press Ctrl+C to exit.{C.RESET}\n"
                )
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
                conversation.end_conversation()
                conversation.new_conversation()
                print(f"{C.YELLOW}[Started a new conversation]{C.RESET}")
                continue
            elif cmd == "/history":
                history = conversation.get_history()
                if history:
                    print(f"{C.YELLOW}[{len(history)} conversation(s) in this session]{C.RESET}")
                    for i, conv in enumerate(history, 1):
                        msg_count = len(conv.user_messages)
                        print(
                            f"  {i}. {msg_count} message(s) — {conv.created_at.strftime('%H:%M:%S')}"
                        )
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
                    if isinstance(chunk, dict):
                        terminal_text = chunk.get("__terminal__")
                        if terminal_text:
                            print(f"\n{C.DIM}{terminal_text}{C.RESET}", end="", flush=True)
                    else:
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
    parser.add_argument("--mcp-server", help=argparse.SUPPRESS)
    parser.add_argument("--diagnose", type=Path, help="Write a JSON diagnostic report and exit")
    parser.add_argument("--smoke-test", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--online", action="store_true", help="Include synthetic AI checks in diagnostics"
    )

    args = parser.parse_args()

    if args.diagnose:
        from jarvis.diagnostics import diagnose

        sys.exit(asyncio.run(diagnose(args.diagnose, online=args.online)))

    if args.mcp_server:
        server_modules = {
            "mcp_ddg": "jarvis.tools.mcp_ddg",
            "mcp_browser": "jarvis.tools.mcp_browser",
            "mcp_weather": "jarvis.tools.mcp_weather",
            "mcp_world_monitor": "jarvis.tools.mcp_world_monitor",
        }
        module_name = server_modules.get(args.mcp_server)
        if module_name is None:
            parser.error(f"Unknown MCP server: {args.mcp_server}")
        import importlib

        module = importlib.import_module(module_name)
        module.mcp.run()
        return

    # If running as a bundled executable (double-clicked), default to UI mode
    if getattr(sys, "frozen", False) and not any(
        arg in sys.argv for arg in ["--ui", "--voice", "--version", "--help", "-h"]
    ):
        args.ui = True

    if args.version:
        print(f"Jarvis v{__version__}")
        sys.exit(0)

    if args.ui or args.smoke_test:
        instance = None
        if getattr(sys, "frozen", False) and not args.smoke_test:
            from jarvis.ui.instance import SingleInstance

            instance = SingleInstance()
            if instance.already_running:
                instance.focus_existing()
                instance.close()
                return
        from jarvis.core.session import SessionManager

        session = SessionManager(config_path=args.config)
        exit_code = 0
        try:
            from jarvis.ui.app import launch_ui

            launch_ui(session, smoke_report=args.smoke_test)
        except Exception as e:
            exit_code = 1
            if args.smoke_test:
                # Automated packaged checks must fail without a blocking Windows dialog.
                if sys.stderr is not None:
                    print(f"Native UI smoke test failed: {e}", file=sys.stderr)
            else:
                _report_ui_startup_error(e)
        finally:
            if instance:
                instance.close()
        print(f"\n{C.DIM}Goodbye.{C.RESET}")
        sys.exit(exit_code)
    else:
        sys.exit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
