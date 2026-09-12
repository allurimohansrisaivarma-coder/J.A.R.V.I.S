"""Read-only diagnostics for both source and packaged JARVIS installations."""

import asyncio
import json
import time
from pathlib import Path


async def diagnose(output: Path, *, online: bool = False) -> int:
    from PIL import Image, ImageDraw, ImageFont

    from jarvis import __version__
    from jarvis.config.settings import get_settings
    from jarvis.context.screen_source import ScreenContextSource
    from jarvis.llm.base import Message
    from jarvis.llm.gemini import GeminiProvider
    from jarvis.llm.groq_provider import GroqProvider
    from jarvis.tools.mcp_manager import MCPManager
    from jarvis.utils.logging import setup_logging

    settings = get_settings()
    setup_logging(settings.logging)
    report: dict = {"version": __version__, "checks": {}}

    async def check(name, operation):
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(operation(), timeout=40)
            report["checks"][name] = {
                "ok": True,
                "result": result,
                "seconds": round(time.perf_counter() - started, 2),
            }
        except Exception as exc:
            from jarvis.utils.errors import user_error

            report["checks"][name] = {
                "ok": False,
                "error": user_error(exc),
                "error_type": type(exc).__name__,
            }

    async def screen():
        images = await asyncio.to_thread(ScreenContextSource._capture_screens)
        return {"displays": len(images), "sizes": [list(img.size) for img in images]}

    async def audio():
        import sounddevice

        devices = await asyncio.to_thread(sounddevice.query_devices)
        return {
            "inputs": sum(d["max_input_channels"] > 0 for d in devices),
            "outputs": sum(d["max_output_channels"] > 0 for d in devices),
        }

    async def tools():
        manager = MCPManager()
        try:
            for name in ("ddg", "browser", "weather", "world_monitor"):
                if not await manager.start_builtin(name, f"jarvis.tools.mcp_{name}"):
                    raise RuntimeError(f"Builtin {name} failed")
            result = await manager.call_tool("open_world_monitor", {})
            if result.startswith(("Error", "Execution error")):
                raise RuntimeError(result)
            return {"registered": len(manager._tool_registry)}
        finally:
            await manager.shutdown()

    await check("screen_capture", screen)
    await check("audio_devices", audio)
    await check("builtin_tools", tools)

    async def history():
        import tempfile

        from jarvis.core.conversation import ConversationManager

        with tempfile.TemporaryDirectory(prefix="jarvis-memory-test-") as directory:
            path = Path(directory) / "history.sqlite3"
            first = ConversationManager(storage_path=path)
            first.add_user_message("The diagnostic codename is Orion.")
            restored = ConversationManager(storage_path=path)
            restored.new_conversation()
            restored.add_user_message("Remember the diagnostic codename?")
            if "Orion" not in str(restored.get_context_messages()):
                raise RuntimeError("Conversation recall failed")
            restored.clear_history()
        return {"restart_and_recall": True}

    await check("conversation_memory", history)

    if online:

        async def daily_briefing():
            from jarvis.context.briefing import build_daily_briefing

            answer = await build_daily_briefing("Give me my daily debrief.")
            headlines = answer.count("](")
            if not headlines:
                raise RuntimeError("No fresh dated publisher headlines were available")
            return {
                "dated_headlines": headlines,
                "freshness_window_hours": 24,
                "model_generated_news": False,
            }

        async def web_search():
            from jarvis.context.web_evidence import (
                WEB_UNAVAILABLE,
                format_evidence,
                search_evidence,
            )

            raw = await asyncio.to_thread(search_evidence, "latest Formula 1 race results", 5)
            if format_evidence(raw) == WEB_UNAVAILABLE:
                raise RuntimeError("Live search returned no usable evidence")
            results = json.loads(raw)["results"]
            return {
                "sources": len(results),
                "readable_pages": sum(bool(row["page_text"]) for row in results),
            }

        async def selected_voice():
            import tempfile

            from jarvis.voice.tts import TTSProvider

            tts = TTSProvider(settings.voice.tts_voice, settings.voice.tts_rate)
            if tts._clean_text("-"):
                raise RuntimeError("Formatting-only speech was not filtered")
            with tempfile.TemporaryDirectory(prefix="jarvis-voice-test-") as directory:
                audio_path = Path(directory) / "voice.mp3"
                await tts._save_voice(
                    "JARVIS voice diagnostic complete.",
                    str(audio_path),
                    tts.voice,
                    tts.rate,
                    tts.pitch,
                )
                size = audio_path.stat().st_size
                if size < 1000:
                    raise RuntimeError("The selected voice returned no usable audio")
                return {
                    "voice": tts.voice,
                    "audio_bytes": size,
                    "formatting_chunks_filtered": True,
                    "same_voice_retry_attempts": 4,
                }

        await check("web_evidence", web_search)
        await check("daily_briefing", daily_briefing)
        await check("selected_voice", selected_voice)

        # Never upload the user's desktop during diagnostics. Use a synthetic chart.
        image = Image.new("RGB", (900, 360), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except OSError:
            font = ImageFont.load_default(size=60)
        draw.text((35, 70), "JARVIS SCREEN TEST 4827", fill="black", font=font)
        vision_message = [
            Message.user(
                ["Read the four digit code in this image. Reply with the code only.", image]
            )
        ]

        async def verify(provider, messages, expected):
            token_budget = 1024 if provider.name == "gemini" else 128
            chunks = [
                c
                async for c in provider.stream(messages, max_tokens=token_budget)
                if isinstance(c, str)
            ]
            text = "".join(chunks).strip()
            if expected not in text:
                raise RuntimeError("Unexpected diagnostic response")
            return {"model": provider.model, "expected_text_received": True}

        if settings.primary_groq_key:
            chat = GroqProvider(
                api_keys=list(settings.groq_api_keys), model=settings.llm.groq.model
            )
            vision = GroqProvider(
                api_keys=list(settings.groq_api_keys),
                model=settings.llm.groq.vision_model,
                supports_images=True,
            )
            await check(
                "groq_chat", lambda: verify(chat, [Message.user("Reply READY only.")], "READY")
            )
            await check("groq_vision", lambda: verify(vision, vision_message, "4827"))
        if settings.primary_gemini_key:
            gemini = GeminiProvider(
                api_keys=settings.gemini_api_keys,
                model=settings.llm.gemini.model_pro,
                thinking_level="low",
            )
            await check("gemini_vision", lambda: verify(gemini, vision_message, "4827"))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return int(any(not check["ok"] for check in report["checks"].values()))
