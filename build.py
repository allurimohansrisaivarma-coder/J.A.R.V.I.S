import os
import subprocess
from pathlib import Path

def main():
    print("Starting PyInstaller build for JARVIS...")
    
    # Base configuration
    entry_point = "src/jarvis/main.py"
    icon_path = "src/jarvis/ui/static/jarvis_icon.ico"
    
    import sys
    args = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",          # No console window
        "--name", "JARVIS",
        "--icon", icon_path,
        
        # UI Static files
        "--add-data", "src/jarvis/ui/static;jarvis/ui/static",
        
        # Config defaults
        "--add-data", "src/jarvis/config/defaults.yaml;jarvis/config",
        
        # Voice models
        "--add-data", "src/jarvis/voice/silero_vad.onnx;jarvis/voice",
        
        # Hidden imports for AI libraries that PyInstaller might miss
        "--hidden-import", "onnxruntime",
        "--hidden-import", "lancedb",
        "--hidden-import", "google.genai",
        "--hidden-import", "structlog",
        "--hidden-import", "webview",
        "--hidden-import", "websockets",
        "--hidden-import", "pynput",
        "--hidden-import", "sounddevice",
        "--hidden-import", "soundfile",
        "--hidden-import", "numpy",
        "--hidden-import", "PIL",
        "--hidden-import", "imagehash",
        "--hidden-import", "win32gui",
        "--hidden-import", "mss",
        "--hidden-import", "dxcam",
        "--hidden-import", "groq",
        
        entry_point
    ]
    
    # Run the build
    subprocess.run(args, check=True)
    print("Build complete! Executable is in the dist/ folder.")

if __name__ == "__main__":
    main()
