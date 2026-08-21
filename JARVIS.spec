# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src/jarvis/main.py'],
    pathex=[],
    binaries=[],
    datas=[('src/jarvis/ui/static', 'jarvis/ui/static'), ('src/jarvis/config/defaults.yaml', 'jarvis/config'), ('src/jarvis/voice/silero_vad.onnx', 'jarvis/voice')],
    hiddenimports=['onnxruntime', 'lancedb', 'google.genai', 'structlog', 'webview', 'websockets', 'pynput', 'sounddevice', 'soundfile', 'numpy', 'PIL', 'imagehash', 'win32gui', 'mss', 'dxcam', 'groq', 'sentence_transformers', 'mcp.server.fastmcp', 'jarvis.tools.mcp_ddg', 'jarvis.tools.mcp_browser', 'jarvis.tools.mcp_weather', 'jarvis.tools.mcp_world_monitor', 'jarvis.tools.windows_launcher', 'playwright.async_api'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='JARVIS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['src/jarvis/ui/static/jarvis_icon.ico'],
    version='version_info.txt',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='JARVIS',
)
