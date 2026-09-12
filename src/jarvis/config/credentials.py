"""Per-user Windows-encrypted credentials, separate from the release bundle."""

import json


def load_keys() -> dict:
    from jarvis.config.settings import USER_DIR

    path = USER_DIR / "provider-keys.dat"
    if not path.exists():
        return {}
    try:
        import win32crypt

        _, data = win32crypt.CryptUnprotectData(path.read_bytes(), None, None, None, 0)
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {}


def save_keys(groq_key: str, gemini_key: str) -> None:
    import win32crypt

    from jarvis.config.settings import USER_DIR

    keys = load_keys()
    for name, value in (("groq_api_keys", groq_key), ("gemini_api_keys", gemini_key)):
        if value.strip():
            parts = [key.strip() for key in value.split(",") if key.strip()]
            if len(parts) > 5 or any(any(c.isspace() for c in key) for key in parts):
                raise ValueError("Enter up to five comma-separated API keys without spaces.")
            keys[name] = parts
    if not keys:
        raise ValueError("Enter a Groq or Gemini API key.")
    USER_DIR.mkdir(parents=True, exist_ok=True)
    encrypted = win32crypt.CryptProtectData(
        json.dumps(keys).encode(), "JARVIS keys", None, None, None, 0
    )
    path = USER_DIR / "provider-keys.dat"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encrypted)
    temporary.replace(path)
