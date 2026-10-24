// Термо-сторож: плашка температури SoC у куті ComfyUI, налаштування (Settings → StepVisionKit →
// Термо-сторож) і тост після заморозки. Сервер — lex/thermal_guard.py: маршрути /svk/thermal/*
// і подія svk.thermal (cooling / ok / was_frozen). Джерело істини — thermal_config.json на сервері;
// UI лише показує його й пересилає зміни.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { T } from "./svk_i18n.js";

const STATUS_ROUTE = "/svk/thermal/status";
const CONFIG_ROUTE = "/svk/thermal/config";
const POLL_MS = 5000;           // опитування статусу, як у брифі
const COOL_HOLD_MS = 6000;      // скільки тримати «Охолодження» після останньої події cooling
const TOAST_MS = 10000;
const CAT = "StepVisionKit";
const SUB = T("Thermal guard");

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
            toast(SUB + ": " + (j.message || T("config not saved")), true);
            return null;
        }
        server = j;
        return j;
    } catch (e) {
        toast(SUB + ": " + T("config not saved") + " (" + e + ")", true);
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

// Показ плашки — суто справа цього браузера: на сервер не йде, сторож працює й без неї.
const BADGE_SETTING = "StepVisionKit.Thermal.ShowBadge";

let showBadge = null;            // останнє значення з onChange (сховище оновлюється вже після нього)

function badgeWanted() {
    if (showBadge !== null) return showBadge;
    try {
        const s = app.extensionManager?.setting;
        const v = s?.get ? s.get(BADGE_SETTING) : app.ui.settings.getSettingValue(BADGE_SETTING);
        return v !== false;
    } catch (_) {
        return true;
    }
}

const SETTINGS = [
    {
        id: BADGE_SETTING, name: T("Show temperature badge"),
        category: [CAT, SUB, T("Badge")], type: "boolean", defaultValue: true,
        tooltip: T("Display only, in the screen corner. The overheat pause works without the badge too."),
        onChange: (v) => { showBadge = v !== false; render(); },
    },
    {
        id: "StepVisionKit.Thermal.Enabled", name: T("Pause on SoC overheat"),
        category: [CAT, SUB, T("Enabled")], type: "boolean", defaultValue: true,
        onChange: (v) => pushSetting("enabled", !!v),
    },
    {
        id: "StepVisionKit.Thermal.Profile", name: T("Profile"),
        category: [CAT, SUB, T("Profile")], type: "combo", defaultValue: "stand",
        options: [
            { value: "stand", text: T("with stand (95 / 85 / 100 °C)") },
            { value: "bare", text: T("without stand (90 / 80 / 97 °C)") },
            { value: "custom", text: T("custom (thresholds below)") },
        ],
        onChange: (v) => pushSetting("profile", v),
    },
    {
        id: "StepVisionKit.Thermal.PauseAt", name: T("Pause, °C"),
        category: [CAT, SUB, T("Pause, °C")], type: "number", defaultValue: 95,
        attrs: { min: 30, max: 100, step: 1 },
        onChange: (v) => pushSetting("pause_at", Number(v)),
    },
    {
        id: "StepVisionKit.Thermal.ResumeAt", name: T("Resume, °C"),
        category: [CAT, SUB, T("Resume, °C")], type: "number", defaultValue: 85,
        attrs: { min: 30, max: 99, step: 1 },
        onChange: (v) => pushSetting("resume_at", Number(v)),
    },
    {
        id: "StepVisionKit.Thermal.HardAt", name: T("Hard limit, °C"),
        category: [CAT, SUB, T("Hard limit, °C")], type: "number", defaultValue: 100,
        attrs: { min: 30, max: 100, step: 1 },
        onChange: (v) => pushSetting("hard_at", Number(v)),
    },
];

// ───────────────────────── індикатор і тости ─────────────────────────

function ensureBadge() {
    if (badge) return badge;
    badge = document.createElement("div");
    badge.id = "svk-thermal-badge";
    // За замовчуванням — над панеллю масштабу канви внизу праворуч, щоб не перекривати її кнопки.
    badge.style.cssText = "position:fixed;right:10px;bottom:72px;z-index:9999;padding:4px 9px;"
        + "border-radius:6px;font:12px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;"
        + "background:rgba(24,24,27,.85);color:#8a8a93;border:1px solid #2e2e33;"
        + "cursor:grab;touch-action:none;user-select:none;display:none";
    badge.title = T("Drag to move. Double-click to put it back.");
    document.body.appendChild(badge);
    makeDraggable(badge);
    return badge;
}

// Позиція плашки — лише зручність цього браузера (localStorage), сервер її не знає.
const POS_KEY = "svk.thermal.badgePos";

function placeBadge(e, pos) {
    if (!pos) {
        e.style.left = e.style.top = "";
        e.style.right = "10px"; e.style.bottom = "72px";
        return;
    }
    const r = e.getBoundingClientRect();
    const x = Math.max(0, Math.min(window.innerWidth - r.width, pos.x));
    const y = Math.max(0, Math.min(window.innerHeight - r.height, pos.y));
    e.style.right = e.style.bottom = "";
    e.style.left = x + "px"; e.style.top = y + "px";
}

function loadPos() {
    try { return JSON.parse(localStorage.getItem(POS_KEY) || "null"); } catch (_) { return null; }
}

function savePos(pos) {
    try {
        if (pos) localStorage.setItem(POS_KEY, JSON.stringify(pos));
        else localStorage.removeItem(POS_KEY);
    } catch (_) { /* приватне вікно — просто не запам'ятаємо */ }
}

function makeDraggable(e) {
    let drag = null;
    e.addEventListener("pointerdown", (ev) => {
        if (ev.button !== 0) return;
        const r = e.getBoundingClientRect();
        drag = { dx: ev.clientX - r.left, dy: ev.clientY - r.top, id: ev.pointerId };
        e.setPointerCapture(ev.pointerId);
        e.style.cursor = "grabbing";
        ev.preventDefault(); ev.stopPropagation();
    });
    e.addEventListener("pointermove", (ev) => {
        if (!drag || ev.pointerId !== drag.id) return;
        placeBadge(e, { x: ev.clientX - drag.dx, y: ev.clientY - drag.dy });
        ev.stopPropagation();
    });
    const end = (ev) => {
        if (!drag || ev.pointerId !== drag.id) return;
        drag = null;
        e.style.cursor = "grab";
        const r = e.getBoundingClientRect();
        savePos({ x: r.left, y: r.top });
    };
    e.addEventListener("pointerup", end);
    e.addEventListener("pointercancel", end);
    e.addEventListener("dblclick", (ev) => { savePos(null); placeBadge(e, null); ev.stopPropagation(); });
    window.addEventListener("resize", () => { const p = loadPos(); if (p) placeBadge(e, p); });
}

function toast(text, isErr = false, ms = TOAST_MS) {
    try {
        const d = document.createElement("div");
        const edge = isErr ? "#e0857c" : "#d9a441";
        // тост — над плашкою, хоч де її поставили
        const br = badge && badge.style.display !== "none" ? badge.getBoundingClientRect() : null;
        const bottom = br ? Math.max(8, window.innerHeight - br.top + 6) : 110;
        const right = br ? Math.max(8, window.innerWidth - br.right) : 10;
        d.style.cssText = `position:fixed;right:${right}px;bottom:${bottom}px;z-index:10000;max-width:360px;`
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
    if (!server || !badgeWanted()) { e.style.display = "none"; return; }
    e.style.display = "block";
    if (!e._svkPlaced) { e._svkPlaced = true; const p = loadPos(); if (p) placeBadge(e, p); }
    if (!server.enabled) {
        e.style.color = "#8a8a93";
        e.textContent = T("thermal guard off");
        return;
    }
    const fresh = coolingInfo && Date.now() - coolingInfo.at < COOL_HOLD_MS;
    if (fresh) {
        e.style.color = "#d9a441";
        e.textContent = T("Cooling {src} → {resume} · {s} s").replace("{src}", srcT(coolingInfo.src, coolingInfo.t)).replace("{resume}", fmtT(coolingInfo.resume_at)).replace("{s}", Math.round(coolingInfo.since));
    } else if (server.state === "frozen") {
        e.style.color = "#e0857c";
        e.textContent = T("ComfyUI frozen · {src}").replace("{src}", srcT(server.src, server.t));
    } else if (server.reader_ok) {
        e.style.color = "#8a8a93";
        e.textContent = srcT(server.src, server.t) + statsTail();
    } else {
        e.style.color = "#8a8a93";
        e.textContent = T("SoC: sensor unavailable");
    }
}

// Скільки разів сторож уже спрацьовував за цей запуск ComfyUI (видно без консолі).
function statsTail() {
    const tot = server && server.total;
    if (!tot || (!tot.pauses && !tot.freezes)) return "";
    let s = " " + T("· {n} pauses").replace("{n}", tot.pauses);
    if (tot.freezes) s += " " + T("· {n} freezes").replace("{n}", tot.freezes);
    return s;
}

function fmtT(t) {
    return (t === null || t === undefined) ? "—" : `${Math.round(t)}°C`;
}

// «GPU 78°C» / «CPU 71°C» / «SoC 62°C» — звідки взято найгарячіший сенсор (src з сервера).
function srcT(src, t) {
    return `${src || "SoC"} ${fmtT(t)}`;
}

function onEvent(d) {
    if (!d) return;
    if (d.state === "cooling") {
        coolingInfo = { t: d.t, src: d.src, resume_at: d.resume_at, since: d.since, at: Date.now() };
        render();
    } else if (d.state === "ok") {
        coolingInfo = null;
        render();
    } else if (d.state === "was_frozen") {
        const sec = Math.round(Number(d.seconds) || 0);
        const hot = (d.t !== null && d.t !== undefined) ? " " + T("due to {t}").replace("{t}", fmtT(d.t)) : "";
        toast(T("ComfyUI was frozen for {sec} s{hot}. The job resumed after cooling.").replace("{sec}", sec).replace("{hot}", hot));
    } else if (d.state === "api_hold") {
        toast(T("{src} — time to cool down, but partner node {node} is running. Not pausing, so the request is not cut off; the pause comes back as soon as it responds.").replace("{src}", srcT(d.src, d.t)).replace("{node}", d.node || ""), false, 9000);
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
