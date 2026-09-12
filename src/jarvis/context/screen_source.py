"""Screen context source for capturing and analyzing the user's desktop."""

import asyncio
import re
import time

import structlog

try:
    from PIL import Image, ImageGrab

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imagehash

    HAS_VISION_DEPS = True
except ImportError:
    HAS_VISION_DEPS = False

from jarvis.context.base import ContextSource

logger = structlog.get_logger(__name__)


class ScreenContextSource(ContextSource):
    """Captures the screen to provide visual context with deduplication and OCR."""

    def __init__(self):
        super().__init__()
        self._last_hash = None

    @property
    def name(self) -> str:
        return "Screen Capture"

    async def can_handle(self, query: str) -> bool:
        """Determines if the query is asking about the screen."""
        if not HAS_PIL:
            return False

        query_lower = query.lower()
        # A request about Desktop files is a local-file operation, not a
        # request to analyse a screenshot just because it says "can you see".
        file_terms = {"file", "folder", "directory", "desktop", "document", "download", "path"}
        if any(term in query_lower for term in file_terms) and not any(
            term in query_lower for term in ("screen", "monitor", "display", "screenshot")
        ):
            return False
        visual_phrases = [
            "what am i looking at",
            "look at this",
            "what is this",
            "who is this",
            "where is this",
            "when is this",
            "how is this",
            "can you see",
            "do you see",
            "analyze this",
        ]
        explicit_visual_noun = bool(
            re.search(r"\b(?:screen|display|screenshot)\b", query_lower)
            or re.search(r"\b(?:my|the|this|that) monitor\b", query_lower)
        )
        return explicit_visual_noun or any(phrase in query_lower for phrase in visual_phrases)

    async def gather_context(self, query: str, **kwargs) -> tuple[str, list] | str:
        """Captures the active window and returns it as context with OCR."""
        logger.info("Executing screen capture", query=query)
        start_time = time.time()

        try:
            # Capture APIs are synchronous and can stall the chat event loop.
            images = await asyncio.to_thread(self._capture_screens)
            img = images[0]

            if img.mode != "RGB":
                img = img.convert("RGB")

            # 2. Resize if it's too large to save API bandwidth
            max_width, max_height = 1920, 1080
            if img.width > max_width or img.height > max_height:
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            # 4. Perceptual Hashing
            current_hash = None
            if HAS_VISION_DEPS:
                try:
                    current_hash = imagehash.average_hash(img)
                    if self._last_hash is not None and current_hash - self._last_hash < 3:
                        logger.info("screen_source.deduplicated", hash=str(current_hash))
                    else:
                        self._last_hash = current_hash
                except Exception as e:
                    logger.warning("screen_source.hashing_failed", error=str(e))

            latency = time.time() - start_time
            logger.info(
                "screen_source.processed",
                latency=round(latency, 2),
                size=img.size,
                hash=str(current_hash),
            )

            context_str = f"Screenshot captured for the user's screen question: {query}\n"

            context_str += (
                f"{len(images)} display(s) captured just now. Read visible text carefully; "
                "if text is too small or obscured, say so. Never infer unseen content.\n"
            )
            return (context_str, images)

        except Exception as e:
            logger.error("Screen capture failed", error=str(e))
            return (
                "Screen capture failed. I cannot see the screen right now. "
                "Unlock the Windows desktop and keep the target window visible, then retry. "
                "Never guess what is on the screen."
            )

    @staticmethod
    def _capture_screens():
        """Keep displays separate so small text survives image resizing."""
        try:
            import mss

            images = []
            with mss.mss() as capture:
                for monitor in capture.monitors[1:4]:
                    shot = capture.grab(monitor)
                    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                    img.thumbnail((1920, 1200), Image.Resampling.LANCZOS)
                    images.append(img)
            if images:
                return images
        except Exception as exc:
            logger.debug("Per-display capture failed", error=str(exc))
        img = ScreenContextSource._capture_screen().convert("RGB")
        img.thumbnail((1920, 1200), Image.Resampling.LANCZOS)
        return [img]

    @staticmethod
    def _capture_screen():
        """Capture all displays, falling back across the installed backends."""
        try:
            return ImageGrab.grab(all_screens=True)
        except Exception as first_error:
            logger.debug("ImageGrab all_screens failed", error=str(first_error))
        try:
            return ImageGrab.grab()
        except Exception as second_error:
            logger.debug("ImageGrab primary failed", error=str(second_error))
        try:
            import mss

            with mss.mss() as capture:
                monitor = capture.monitors[0]
                screenshot = capture.grab(monitor)
                return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        except Exception as third_error:
            logger.debug("mss capture failed", error=str(third_error))

        import dxcam

        camera = dxcam.create()
        frame = camera.grab()
        if frame is None:
            raise RuntimeError("All screen capture backends failed")
        return Image.fromarray(frame)
