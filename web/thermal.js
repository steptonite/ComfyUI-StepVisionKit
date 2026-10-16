// Термо-сторож: плашка температури SoC у куті ComfyUI, налаштування (Settings → StepVisionKit →
// Термо-сторож) і тост після заморозки. Сервер — lex/thermal_guard.py: маршрути /svk/thermal/*
// і подія svk.thermal (cooling / ok / was_frozen). Джерело істини — thermal_config.json на сервері;
// UI лише показує його й пересилає зміни.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STATUS_ROUTE = "/svk/thermal/status";
const CONFIG_ROUTE = "/svk/thermal/config";
const POLL_MS = 5000;           // опитування статусу, як у брифі
const COOL_HOLD_MS = 6000;      // скільки тримати «Охолодження» після останньої події cooling
const TOAST_MS = 10000;
const CAT = "StepVisionKit";
const SUB = "Термо-сторож";

let server = null;              // остання відповідь сервера (та сама форма, що /status)
let coolingInfo = null;         // {t, resume_at, since, at} з останньої події cooling
let badge = null;

// ───────────────────────── мережа ─────────────────────────

async function getStatus() {
    try {
        const r = await api.fetchApi(STATUS_ROUTE);
        return r.ok ? await r.json() : null;
    } catch (_) {
        return null;
    }
}

async function postConfig(patch) {
    try {
        const r = await api.fetchApi(CONFIG_ROUTE, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patch),
        });
        const j = await r.json();
        if (!r.ok || !j.ok) {
            toast("Термо-сторож: " + (j.message || "конфіг не збережено"), true);
            return null;
        }
        server = j;
        return j;
    } catch (e) {
        toast("Термо-сторож: конфіг не збережено (" + e + ")", true);
        return null;
    }
}

// ───────────────────────── налаштування ─────────────────────────

// Значення налаштувань — відображення сервера. Зміна в UI → POST; повтор того самого значення
// не шлемо, тож синхронізація назад (reflect) не зациклюється.
function setSetting(id, v) {
    try {
        const s = app.extensionManager?.setting;
        if (s?.set) return s.set(id, v);
        return app.ui.settings.setSettingValue(id, v);
    } catch (_) {
        return undefined;
    }
}

function reflect(st) {
    setSetting("StepVisionKit.Thermal.Enabled", !!st.enabled);
    setSetting("StepVisionKit.Thermal.Profile", st.profile);
    setSetting("StepVisionKit.Thermal.PauseAt", st.pause_at);
    setSetting("StepVisionKit.Thermal.ResumeAt", st.resume_at);
    setSetting("StepVisionKit.Thermal.HardAt", st.hard_at);
}

const THRESHOLDS = { pause_at: 1, resume_at: 1, hard_at: 1 };

function pushSetting(key, v) {
    if (!server || server[key] === v) return;
    // зміна порогу вручну → профіль стає «свій», інакше сервер проігнорує число
    const patch = THRESHOLDS[key] ? { profile: "custom", [key]: v } : { [key]: v };
    postConfig(patch).then((j) => { if (j) { reflect(j); render(); } });
}

const SETTINGS = [
    {
        id: "StepVisionKit.Thermal.Enabled", name: "Пауза при перегріві SoC",
        category: [CAT, SUB, "Увімкнено"], type: "boolean", defaultValue: true,
        onChange: (v) => pushSetting("enabled", !!v),
    },
    {
        id: "StepVisionKit.Thermal.Profile", name: "Профіль",
        category: [CAT, SUB, "Профіль"], type: "combo", defaultValue: "stand",
        options: [
            { value: "stand", text: "з підставкою (95 / 85 / 100 °C)" },
            { value: "bare", text: "без підставки (90 / 80 / 97 °C)" },
            { value: "custom", text: "свій (пороги нижче)" },
        ],
        onChange: (v) => pushSetting("profile", v),
    },
    {
        id: "StepVisionKit.Thermal.PauseAt", name: "Пауза, °C",
        category: [CAT, SUB, "Пауза, °C"], type: "number", defaultValue: 95,
        attrs: { min: 30, max: 100, step: 1 },
        onChange: (v) => pushSetting("pause_at", Number(v)),
    },
    {
        id: "StepVisionKit.Thermal.ResumeAt", name: "Продовжити, °C",
        category: [CAT, SUB, "Продовжити, °C"], type: "number", defaultValue: 85,
        attrs: { min: 30, max: 99, step: 1 },
        onChange: (v) => pushSetting("resume_at", Number(v)),
    },
    {
        id: "StepVisionKit.Thermal.HardAt", name: "Жорстка межа, °C",
        category: [CAT, SUB, "Жорстка межа, °C"], type: "number", defaultValue: 100,
        attrs: { min: 30, max: 100, step: 1 },
        onChange: (v) => pushSetting("hard_at", Number(v)),
    },
];

// ───────────────────────── індикатор і тости ─────────────────────────

function ensureBadge() {
    if (badge) return badge;
    badge = document.createElement("div");
    badge.id = "svk-thermal-badge";
    badge.style.cssText = "position:fixed;right:10px;bottom:10px;z-index:9999;padding:4px 9px;"
        + "border-radius:6px;font:12px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;"
        + "background:rgba(24,24,27,.85);color:#8a8a93;border:1px solid #2e2e33;"
        + "pointer-events:none;user-select:none;display:none";
    document.body.appendChild(badge);
    return badge;
}

function toast(text, isErr = false, ms = TOAST_MS) {
    try {
        const d = document.createElement("div");
        const edge = isErr ? "#e0857c" : "#d9a441";
        d.style.cssText = "position:fixed;right:10px;bottom:48px;z-index:10000;max-width:360px;"
            + "padding:8px 12px;border-radius:8px;font:13px/1.35 system-ui,sans-serif;"
            + "background:" + (isErr ? "#3a1f1d" : "#2a2416") + ";color:#e8e8ea;"
            + "border:1px solid " + edge + ";box-shadow:0 4px 16px rgba(0,0,0,.35)";
        d.textContent = text;
        document.body.appendChild(d);
        setTimeout(() => d.remove(), ms);
    } catch (_) { /* тост — не критично */ }
}

function render() {
    const e = ensureBadge();
    if (!server) { e.style.display = "none"; return; }
    e.style.display = "block";
    if (!server.enabled) {
        e.style.color = "#8a8a93";
        e.textContent = "термо-сторож вимкнено";
        return;
    }
    const fresh = coolingInfo && Date.now() - coolingInfo.at < COOL_HOLD_MS;
    if (fresh) {
        e.style.color = "#d9a441";
        e.textContent = `Охолодження ${fmtT(coolingInfo.t)} → ${fmtT(coolingInfo.resume_at)} · ${Math.round(coolingInfo.since)} с`;
    } else if (server.state === "frozen") {
        e.style.color = "#e0857c";
        e.textContent = `ComfyUI заморожено · SoC ${fmtT(server.t)}`;
    } else if (server.reader_ok) {
        e.style.color = "#8a8a93";
        e.textContent = `SoC ${fmtT(server.t)}` + statsTail();
    } else {
        e.style.color = "#8a8a93";
        e.textContent = "SoC: датчик недоступний";
    }
}

// Скільки разів сторож уже спрацьовував за цей запуск ComfyUI (видно без консолі).
function statsTail() {
    const tot = server && server.total;
    if (!tot || (!tot.pauses && !tot.freezes)) return "";
    let s = ` · пауз ${tot.pauses}`;
    if (tot.freezes) s += ` · заморозок ${tot.freezes}`;
    return s;
}

function fmtT(t) {
    return (t === null || t === undefined) ? "—" : `${Math.round(t)}°C`;
}

function onEvent(d) {
    if (!d) return;
    if (d.state === "cooling") {
        coolingInfo = { t: d.t, resume_at: d.resume_at, since: d.since, at: Date.now() };
        render();
    } else if (d.state === "ok") {
        coolingInfo = null;
        render();
    } else if (d.state === "was_frozen") {
        const sec = Math.round(Number(d.seconds) || 0);
        const hot = (d.t !== null && d.t !== undefined) ? ` через ${fmtT(d.t)}` : "";
        toast(`ComfyUI був заморожений на ${sec} с${hot}. Задача продовжилась після охолодження.`);
    }
}

async function poll() {
    const st = await getStatus();
    if (st) { server = st; render(); }
}

// ───────────────────────── реєстрація ─────────────────────────

app.registerExtension({
    name: "StepVisionKit.Thermal",
    settings: SETTINGS,
    async setup() {
        try {
            api.addEventListener("svk.thermal", (ev) => onEvent(ev.detail));
        } catch (_) { /* без події індикатор працює з опитуванням */ }
        const st = await getStatus();
        if (st) { server = st; reflect(st); }
        render();
        setInterval(poll, POLL_MS);
    },
});
