"""Actionable, credential-free messages for recoverable failures."""


def user_error(error: BaseException, *, operation: str = "request") -> str:
    text = str(error).lower()
    if "no api keys" in text:
        return "Add a Groq or Gemini key in Settings, then reopen JARVIS to enable AI."
    if any(word in text for word in ("503", "high demand", "overloaded")):
        return "The AI services are busy. Please try again shortly."
    if any(word in text for word in ("401", "403", "api key", "authentication", "unauthorized")):
        return "The API key was rejected. Check your keys in Settings and try again."
    if any(word in text for word in ("429", "quota", "rate limit")):
        return "The AI provider's usage limit was reached. Please try again later."
    if any(word in text for word in ("503", "high demand", "overloaded")):
        return "The AI services are busy. Please try again shortly."
    if isinstance(error, TimeoutError) or any(
        word in text
        for word in ("connect", "network", "resolve", "getaddrinfo", "timed out", "timeout")
    ):
        return "Cannot reach the online service. Check your internet or VPN, then try again."
    if "microphone" in text or "input device" in text or "portaudio" in text:
        return "Microphone unavailable. Check Windows microphone permission and your input device."
    if "screenshot" in text or "vision" in text:
        return (
            "Screen reading is unavailable. Configure a Gemini or Groq key in Settings and retry."
        )
    return f"The {operation} failed. Please retry; details are in the JARVIS log."
