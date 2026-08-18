// ========================================
// J.A.R.V.I.S. HUD — Frontend Controller
// ========================================

let ws;
let reconnectTimeout = 1000;
let currentJarvisBubble = null;
let currentJarvisText = "";

// Voice state
let isRecording = false;

// Widget state
let isWidgetMode = false;

// ── DOM References ──
const aiCore          = document.getElementById("ai-core");
const coreStatusText  = document.getElementById("core-status-text");
const chatContainer   = document.getElementById("chat-container");
const messageInput    = document.getElementById("message-input");
const sendBtn         = document.getElementById("send-btn");
const micBtn          = document.getElementById("mic-btn");
const settingsBtn     = document.getElementById("settings-btn");
const settingsModal   = document.getElementById("settings-modal");
const closeSettingsBtn= document.getElementById("close-settings-btn");
const saveSettingsBtn = document.getElementById("save-settings-btn");
const settingHotkey   = document.getElementById("setting-hotkey");
const settingSilence  = document.getElementById("setting-silence");
const settingVoice    = document.getElementById("setting-voice");
const settingRate     = document.getElementById("setting-rate");
const minimizeBtn     = document.getElementById("minimize-btn");
const closeBtn        = document.getElementById("close-btn");


// ========================================
//  WebSocket
// ========================================

function connectWebSocket() {
    ws = new WebSocket("ws://localhost:8741");

    ws.onopen = () => {
        console.log("Connected to JARVIS HUD");
        reconnectTimeout = 1000;
        updateCoreState("IDLE");
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleMessage(data);
        } catch (e) {
            console.error("Failed to parse message", e);
        }
    };

    ws.onclose = () => {
        console.log("Disconnected from JARVIS HUD");
        updateCoreState("OFFLINE");
        setTimeout(connectWebSocket, reconnectTimeout);
        reconnectTimeout = Math.min(reconnectTimeout * 2, 10000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
    };
}


// ========================================
//  Message Handler
// ========================================

function handleMessage(data) {
    if (data.type === "chunk") {
        if (!currentJarvisBubble) {
            currentJarvisBubble = createMessageBubble("jarvis");
            currentJarvisText = "";
        }
        currentJarvisText += data.text;
        currentJarvisBubble.innerHTML =
            `<span class="hud-msg-prefix">></span> ` + renderMarkdown(currentJarvisText);
        scrollToBottom();

    } else if (data.type === "done") {
        currentJarvisBubble = null;
        currentJarvisText = "";

    } else if (data.type === "state") {
        updateCoreState(data.state);

    } else if (data.type === "audio_level") {
        if (aiCore.classList.contains("LISTENING") || aiCore.classList.contains("RECORDING")) {
            const center = aiCore.querySelector(".core-center");
            if (center) {
                const scale = 1.0 + (data.level * 15.0);
                const clamped = Math.min(Math.max(scale, 1.0), 1.8);
                center.style.transform = `scale(${clamped})`;
            }
        }

    } else if (data.type === "user_msg") {
        appendUserMessage(data.text);

    } else if (data.type === "settings_data") {
        settingHotkey.value = data.settings.push_to_talk_key || "ctrl+shift+j";
        settingSilence.value = data.settings.silence_duration || 0.5;
        settingVoice.value  = data.settings.tts_voice || "en-GB-RyanNeural";
        settingRate.value   = data.settings.tts_rate || "+20%";
    }
}


// ========================================
//  Core State
// ========================================

function updateCoreState(stateStr) {
    aiCore.className = `ai-core ${stateStr}`;
    coreStatusText.innerText = stateStr;
    
    if (stateStr === "RECORDING" || stateStr === "LISTENING") {
        micBtn.classList.add("active");
        micBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor"
             stroke-width="2" fill="currentColor" stroke-linecap="round"
             stroke-linejoin="round">
            <rect x="6" y="6" width="12" height="12"></rect>
        </svg>`;
    } else {
        micBtn.classList.remove("active");
        micBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor"
             stroke-width="2" fill="none" stroke-linecap="round"
             stroke-linejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="23"></line>
            <line x1="8" y1="23" x2="16" y2="23"></line>
        </svg>`;
    }
}


// ========================================
//  Chat Bubbles
// ========================================

function appendUserMessage(text) {
    const bubble = createMessageBubble("user");
    bubble.innerHTML = `<span class="hud-msg-prefix">USER></span> ` + escapeHtml(text);
    scrollToBottom();
}

function createMessageBubble(sender) {
    const div = document.createElement("div");
    div.className = `hud-message ${sender}`;
    chatContainer.appendChild(div);
    return div;
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}


// ========================================
//  Text Input — Send
// ========================================

function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    appendUserMessage(text);
    ws.send(JSON.stringify({ type: "chat", text: text }));
    messageInput.value = "";
    messageInput.style.height = "auto";
}

sendBtn.addEventListener("click", sendMessage);

messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea as user types
messageInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = this.scrollHeight + "px";
});


// ========================================
//  Hold-to-Speak (Mic Button + Spacebar)
// ========================================

function startRecording() {
    if (isRecording) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    isRecording = true;
    micBtn.classList.add("active");
    micBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor"
             stroke-width="2" fill="currentColor" stroke-linecap="round"
             stroke-linejoin="round">
            <rect x="6" y="6" width="12" height="12"></rect>
        </svg>`;
    ws.send(JSON.stringify({ type: "voice_start" }));
}

function stopRecording() {
    if (!isRecording) return;

    isRecording = false;
    micBtn.classList.remove("active");
    micBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor"
             stroke-width="2" fill="none" stroke-linecap="round"
             stroke-linejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="23"></line>
            <line x1="8" y1="23" x2="16" y2="23"></line>
        </svg>`;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "voice_stop" }));
    }
}

// Mic button: hold to speak
micBtn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    startRecording();
});
micBtn.addEventListener("pointerup", stopRecording);
micBtn.addEventListener("pointerleave", stopRecording);
micBtn.addEventListener("pointercancel", stopRecording);
micBtn.addEventListener("contextmenu", (e) => e.preventDefault());

// Spacebar: hold to speak (only when textarea is NOT focused)
document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && document.activeElement !== messageInput) {
        e.preventDefault();
        startRecording();
    }
});

document.addEventListener("keyup", (e) => {
    if (e.code === "Space" && document.activeElement !== messageInput) {
        e.preventDefault();
        stopRecording();
    }
});


// ========================================
//  Minimize / Widget Toggle
// ========================================

minimizeBtn.addEventListener("click", () => {
    if (isWidgetMode) return;
    isWidgetMode = true;
    document.body.classList.add("widget-mode");
    if (window.pywebview && window.pywebview.api) {
        window.pywebview.api.collapse_to_widget();
    }
});

aiCore.addEventListener("dblclick", () => {
    if (!isWidgetMode) return;
    isWidgetMode = false;
    document.body.classList.remove("widget-mode");
    if (window.pywebview && window.pywebview.api) {
        window.pywebview.api.expand_to_window();
    }
});

closeBtn.addEventListener("click", () => {
    if (window.pywebview && window.pywebview.api) {
        window.pywebview.api.close_app();
    }
});


// ========================================
//  Settings Modal
// ========================================

settingsBtn.addEventListener("click", () => {
    settingsModal.classList.remove("hidden");
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "get_settings" }));
    }
});

closeSettingsBtn.addEventListener("click", () => {
    settingsModal.classList.add("hidden");
});

settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) {
        settingsModal.classList.add("hidden");
    }
});

saveSettingsBtn.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: "update_settings",
            settings: {
                push_to_talk_key: settingHotkey.value,
                silence_duration: parseFloat(settingSilence.value),
                tts_voice: settingVoice.value,
                tts_rate: settingRate.value
            }
        }));
    }
    settingsModal.classList.add("hidden");
});


// ========================================
//  Utilities
// ========================================

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
    html = html.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
    html = html.replace(/`(.*?)`/g, "<code>$1</code>");
    return html;
}


// ========================================
//  Boot
// ========================================
connectWebSocket();
