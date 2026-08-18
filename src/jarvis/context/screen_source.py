"""Screen context source for capturing and analyzing the user's desktop."""

import time

import structlog

try:
    from PIL import Image, ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imagehash
    import win32gui
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
        keywords = [
            "screen", "monitor", "display", "what am i looking at", "look at this",
            "what is this", "who is this", "where is this", "when is this", "how is this",
            "can you see", "do you see", "analyze this"
        ]
        return any(k in query_lower for k in keywords)
        
    async def gather_context(self, query: str, **kwargs) -> tuple[str, list] | str:
        """Captures the active window and returns it as context with OCR."""
        logger.info("Executing screen capture", query=query)
        start_time = time.time()
        
        try:
            # 1. Capture the primary monitor
            img = None
            try:
                img = ImageGrab.grab(all_screens=True)
            except Exception as e1:
                logger.debug("ImageGrab all_screens failed, trying primary only", error=str(e1))
                try:
                    img = ImageGrab.grab()
                except Exception as e2:
                    logger.debug("ImageGrab primary failed, trying mss", error=str(e2))
                    try:
                        import mss
                        with mss.mss() as sct:
                            monitor = sct.monitors[0]
                            sct_img = sct.grab(monitor)
                            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    except Exception as e3:
                        logger.debug("ImageGrab mss failed, trying dxcam", error=str(e3))
                        import dxcam
                        camera = dxcam.create()
                        frame = camera.grab()
                        if frame is None:
                            raise RuntimeError("dxcam failed to grab frame")
                        img = Image.fromarray(frame)
            
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
            logger.info("screen_source.processed", latency=round(latency, 2), size=img.size, hash=str(current_hash))
            
            # 5. Targeted Prompt Injection
            context_str = f"[System Instruction: The user's specific query is '{query}'. Analyze the following screenshot of their screen to answer it accurately.]\n"
                
            return (context_str, [img])
            
        except Exception as e:
            logger.error("Screen capture failed", error=str(e))
            return "Screen capture failed. I cannot see the screen right now."
