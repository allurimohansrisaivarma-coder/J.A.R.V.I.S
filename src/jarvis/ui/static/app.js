// ========================================
// J.A.R.V.I.S. HUD — Frontend Controller
// ========================================

let ws;
let reconnectTimeout = 1000;
let reconnectTimer = null;
let dashboardInterval = null;
let currentJarvisBubble = null;
let currentJarvisText = "";
let currentJarvisMeta = "";
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
const settingsModal = document.getElementById("settings-modal");
const closeSettingsBtn = document.getElementById("close-settings-btn");
const saveSettingsBtn = document.getElementById("save-settings-btn");

const memoryFilter    = document.getElementById("memory-filter");
const refreshMemBtn   = document.getElementById("refresh-mem-btn");
const newMemoryInput  = document.getElementById("new-memory-input");
const addMemoryBtn    = document.getElementById("add-memory-btn");

const settingHotkey   = document.getElementById("setting-hotkey");
const settingSilence  = document.getElementById("setting-silence");
const settingVoice    = document.getElementById("setting-voice");
const settingRate     = document.getElementById("setting-rate");
const settingWeatherCity = document.getElementById("setting-weather-city");
const minimizeBtn     = document.getElementById("minimize-btn");
const widgetExpandBtn = document.getElementById("widget-expand-btn");
const closeBtn        = document.getElementById("close-btn");

// World Monitor DOM
const worldMonitor    = document.getElementById("world-monitor");
const closeMonitorBtn = document.getElementById("close-monitor-btn");
const wmRam           = document.getElementById("wm-ram");
const wmCpu           = document.getElementById("wm-cpu");
const wmWeather       = document.getElementById("wm-weather");
const wmNews          = document.getElementById("wm-news");
const wmSchedule      = document.getElementById("wm-schedule");


// ========================================
//  WebSocket
// ========================================

function connectWebSocket() {
    const queryParams = new URLSearchParams(window.location.search);
    const hashParams = new URLSearchParams(window.location.hash.slice(1));
    const port = queryParams.get("port") || hashParams.get("port") || "8741";
    ws = new WebSocket(`ws://127.0.0.1:${port}`);

    ws.onopen = () => {
        console.log("WebSocket connected");
        requestDashboardData(); // Fetch initial stats and memory
        reconnectTimeout = 1000;
        updateCoreState("IDLE");
        if (dashboardInterval) clearInterval(dashboardInterval);
        dashboardInterval = setInterval(requestDashboardData, 5000);
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
        if (dashboardInterval) {
            clearInterval(dashboardInterval);
            dashboardInterval = null;
        }
        reconnectTimer = setTimeout(connectWebSocket, reconnectTimeout);
        reconnectTimeout = Math.min(reconnectTimeout * 2, 10000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
    };
}


// ========================================
//  Terminal Log Functions
// ========================================

function appendTerminalLog(text) {
    const termContainer = document.getElementById("terminal-container");
    if (!termContainer) return;

    const line = document.createElement("div");
    line.className = "terminal-line";

    if (text.includes("Executing:")) {
        line.classList.add("command");
    } else if (text.includes("Finished:")) {
        line.classList.add("result");
    } else if (text.includes("Error:") || text.includes("Failed:")) {
        line.classList.add("error");
    }

    line.textContent = text;
    termContainer.appendChild(line);
    termContainer.scrollTop = termContainer.scrollHeight;
}

// ========================================
//  Input Handling
// ========================================

function handleMessage(data) {
    if (data.type === "chunk") {
        if (!currentJarvisBubble) {
            currentJarvisBubble = createMessageBubble("jarvis");
            currentJarvisText = "";
            currentJarvisMeta = "";
        }
        currentJarvisText += data.text;
        currentJarvisBubble.innerHTML =
            `<span class="hud-msg-prefix">></span> ` + renderMarkdown(currentJarvisText) + currentJarvisMeta;
        scrollToBottom();

    } else if (data.type === "done") {
        currentJarvisBubble = null;
        currentJarvisText = "";
        currentJarvisMeta = "";

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

    } else if (data.type === "terminal") {
        appendTerminalLog(data.text);
    } else if (data.type === "error") {
        const bubble = createMessageBubble("system");
        bubble.textContent = `SYSTEM> ${data.message || "Request failed"}`;
        scrollToBottom();
    } else if (data.type === "user_msg") {
        appendUserMessage(data.text);

    } else if (data.type === "settings_data") {
        settingHotkey.value = data.settings.push_to_talk_key || "ctrl+shift+j";
        settingSilence.value = data.settings.silence_duration || 0.5;
        settingVoice.value  = data.settings.tts_voice || "en-GB-RyanNeural";
        settingRate.value   = data.settings.tts_rate || "+20%";
        settingWeatherCity.value = data.settings.weather_city || "";
    } else if (data.type === "message_meta") {
        if (!currentJarvisBubble) {
            currentJarvisBubble = createMessageBubble("jarvis");
            currentJarvisText = "";
            currentJarvisMeta = "";
        }
        const provider = escapeHtml(String(data.meta.provider || "unknown"));
        const model = escapeHtml(String(data.meta.model || "unknown"));
        currentJarvisMeta = `<div class="model-footer">[Generated by ${provider} / ${model}]</div>`;
        currentJarvisBubble.innerHTML =
            `<span class="hud-msg-prefix">></span> ` + renderMarkdown(currentJarvisText) + currentJarvisMeta;
        scrollToBottom();
    } else if (data.type === "stats_data") {
        const up = data.stats.uptime;
        const hours = Math.floor(up / 3600).toString().padStart(2, '0');
        const mins = Math.floor((up % 3600) / 60).toString().padStart(2, '0');
        const secs = Math.floor(up % 60).toString().padStart(2, '0');
        document.getElementById("uptime-value").innerText = `${hours}:${mins}:${secs}`;
        document.getElementById("tokens-value").innerText = data.stats.total_tokens.toLocaleString();
    } else if (data.type === "memories_data") {
        const list = document.getElementById("memory-list");
        list.innerHTML = "";
        if (data.memories.length === 0) {
            list.innerHTML = `<div style="text-align: center; color: rgba(0, 255, 255, 0.4);">No memories found.</div>`;
        } else {
            data.memories.forEach(m => {
                const item = document.createElement("div");
                item.className = "memory-item";
                item.innerHTML = `
                    <div class="fact-text">${escapeHtml(m.fact)}</div>
                    <button class="delete-mem-btn" data-id="${m.id}" title="Delete">&times;</button>
                `;
                list.appendChild(item);
            });
            document.querySelectorAll(".delete-mem-btn").forEach(btn => {
                btn.addEventListener("click", (e) => {
                    const id = e.target.getAttribute("data-id");
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: "delete_memory", id: id }));
                        e.target.disabled = true;
                    }
                });
            });
        }
    } else if (data.type === "memory_added" || data.type === "memory_deleted") {
        // refresh memories
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "get_memories", filter: memoryFilter.value }));
        }
    } else if (data.type === "world_monitor_data") {
        updateWorldMonitor(data.data);
    } else if (data.type === "open_world_monitor") {
        worldMonitor.classList.remove("hidden");
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "get_world_monitor", refresh: true }));
        }
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

minimizeBtn.addEventListener("click", async () => {
    if (isWidgetMode) return;
    isWidgetMode = true;
    document.body.classList.add("widget-mode");
    try {
        if (!window.pywebview || !window.pywebview.api) throw new Error("Native UI bridge unavailable");
        const collapsed = await window.pywebview.api.collapse_to_widget();
        if (!collapsed) throw new Error("Native window rejected widget mode");
    } catch (error) {
        isWidgetMode = false;
        document.body.classList.remove("widget-mode");
        appendTerminalLog(`Failed: Widget mode unavailable (${error.message || error})`);
    }
});

widgetExpandBtn.addEventListener("click", async (event) => {
    if (!isWidgetMode) return;
    event.stopPropagation();
    try {
        if (!window.pywebview || !window.pywebview.api) throw new Error("Native UI bridge unavailable");
        const expanded = await window.pywebview.api.expand_to_window();
        if (!expanded) throw new Error("Native window rejected expansion");
        isWidgetMode = false;
        document.body.classList.remove("widget-mode");
    } catch (error) {
        appendTerminalLog(`Failed: Could not restore window (${error.message || error})`);
    }
});

document.getElementById('expand-btn').addEventListener('click', () => {
    if (window.pywebview) pywebview.api.toggle_fullscreen();
});

document.querySelectorAll('.close-app-btn').forEach(btn => {
    btn.addEventListener("click", () => {
        if (window.pywebview && window.pywebview.api) {
            window.pywebview.api.close_app();
        }
    });
});


// ========================================
//  Dashboard Memory & Stats
// ========================================

function requestDashboardData() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "get_stats" }));
        ws.send(JSON.stringify({ type: "get_memories", filter: memoryFilter.value }));
    }
}

refreshMemBtn.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "get_stats" }));
        ws.send(JSON.stringify({ type: "get_memories", filter: memoryFilter.value }));
    }
});

memoryFilter.addEventListener("change", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "get_memories", filter: memoryFilter.value }));
    }
});

addMemoryBtn.addEventListener("click", () => {
    const fact = newMemoryInput.value.trim();
    if (fact && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "add_memory", fact: fact }));
    }
});

newMemoryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        addMemoryBtn.click();
    }
});

// Update stats every 5 seconds
setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "get_stats" }));
        if (!worldMonitor.classList.contains("hidden")) {
            ws.send(JSON.stringify({ type: "get_world_monitor" }));
        }
    }
}, 5000);

// ========================================
//  World Monitor
// ========================================

closeMonitorBtn.addEventListener("click", () => {
    worldMonitor.classList.add("hidden");
});

function updateWorldMonitor(data) {
    if (data.ram) wmRam.innerText = data.ram + "%";
    if (data.cpu) wmCpu.innerText = data.cpu + "%";
    if (data.weather) wmWeather.innerHTML = data.weather;
    if (data.news) wmNews.innerHTML = data.news;
    if (data.schedule) wmSchedule.innerHTML = data.schedule;
}

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
                tts_rate: settingRate.value,
                weather_city: settingWeatherCity.value.trim()
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
