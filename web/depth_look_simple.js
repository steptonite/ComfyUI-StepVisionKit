// Depth Look · простий: картка замість «полотна налаштувань».
//   превʼю результату зверху -> стрічка плиток-луків (кожен пресет на ЦЬОМУ кадрі) -> чіпи ефектів -> «сила луку».
// Усі нативні віджети лишаються (їхні значення зберігаються у workflow і йдуть у Run), але сховані: картка їх лише читає/пише.
// Також тут: стрічка кадрів-превʼю відео-ноди і єдиний колір карток трьох нод. Стилі й токени - svk_ui.css.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ICON, presetLabel, control, morePanel, moreSummary, keepWheel, onWheel, fitToNode, scrollHint, stageLine, snapUI, restoreUI, migrateLegacyValues } from "./svk_ui.js";
import { T } from "./svk_i18n.js";

const SIMPLE = "DepthLookSimple";
const VIDEO = "DepthLookVideo";
const CSS_URL = new URL("./svk_ui.css", import.meta.url).href;
const PREVIEW_SIZE = 480;
const THUMB_SIZE = 192;
const NODE_COLOR = "#2a2a2f", NODE_BG = "#151518";      // = --svk-line / --svk-card

(function injectCss() {
    if (document.getElementById("svk-ui-css")) return;
    const l = document.createElement("link");
    l.id = "svk-ui-css"; l.rel = "stylesheet"; l.href = CSS_URL;
    document.head.appendChild(l);
})();

// ---------------------------------------------------------------- спільне
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const W = (node, name) => node.widgets?.find(w => w.name === name);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function hideWidget(w) {
    if (!w) return;
    w.hidden = true;
    w.options = Object.assign(w.options || {}, { hidden: true });
    w.computeSize = () => [0, -4];
    w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    if (w.element?.style) w.element.style.display = "none";
}
function markChanged() {
    try { app.graph?.setDirtyCanvas(true, true); } catch (_) {}
    try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState(); } catch (_) {}
}
function setW(node, name, v) {            // значення напряму, без callback (як у DepthLook: callback пресета скидає параметри)
    const w = W(node, name);
    if (w && w.value !== v) { w.value = v; return true; }
    return false;
}
const simpleURL = (node, tail) => api.apiURL("/depth_look/simple/" + encodeURIComponent(node.id) + "/" + tail);

// ---------------------------------------------------------------- картка Simple
function buildSimple(node) {
    const S = { node, info: null, token: 0, inflight: false, pending: false, blobURL: null, thumbToken: 0, dragged: false };
    const root = el("div", "svk-card svk-scroll");
    S.root = root;
    keepWheel(root);

    // кадр
    const frame = el("div", "svk-frame");
    S.frame = frame;
    S.imgOrig = el("img"); S.imgOrig.alt = ""; S.imgOrig.draggable = false;
    S.imgLook = el("img", "svk-look"); S.imgLook.alt = ""; S.imgLook.draggable = false;
    S.imgOrig.style.visibility = "hidden"; S.imgLook.style.visibility = "hidden";
    S.busy = el("div", "svk-busy");
    S.tag = el("div", "svk-tag"); S.tag.style.display = "none";      // тег лишився для сумісності стилів, підказка - у title кадру
    frame.title = T("hold the frame — original");
    S.empty = el("div", "svk-empty");
    frame.append(S.imgOrig, S.imgLook, S.busy, S.tag, S.empty);
    // утримуй кадр -> показати оригінал
    const hold = on => { S.imgLook.style.opacity = on ? 0 : String(mixValue(node)); };
    frame.addEventListener("pointerdown", e => { if (e.button === 0 && S.info?.node_frame) { hold(true); frame.setPointerCapture?.(e.pointerId); } });
    const release = () => hold(false);
    frame.addEventListener("pointerup", release); frame.addEventListener("pointercancel", release);

    // сила луку
    const mix = el("div", "svk-mix");
    mix.append(el("span", "svk-cap", "<b>" + T("Look strength") + "</b>"));
    mix.title = T("how much of the look is applied over the original; double-click — 100%");
    S.mixRange = el("input", "svk-range"); S.mixRange.type = "range"; S.mixRange.min = 0; S.mixRange.max = 1; S.mixRange.step = 0.01;
    S.mixNum = el("span", "svk-num", "100%");
    mix.append(S.mixRange, S.mixNum);
    S.mixRange.addEventListener("input", () => {
        const v = Number(S.mixRange.value);
        setW(node, "look_mix", v); markChanged(); paintMix(S);
    });
    S.mixRange.addEventListener("dblclick", () => { S.mixRange.value = 1; S.mixRange.dispatchEvent(new Event("input")); });

    // плитки
    const tilesCap = el("div", "svk-cap", "<b>" + T("Look") + "</b>");
    S.tiles = el("div", "svk-tiles");
    S.tiles.title = T("click to pick a look");

    // чіпи
    const chipsCap = el("div", "svk-cap");
    chipsCap.append(el("b", null, T("Effects")));
    S.reset = el("button", "svk-ibtn", ICON.reset); S.reset.type = "button"; S.reset.title = T("reset effects to the look"); S.reset.style.display = "none";
    S.reset.onclick = () => { resetChips(S); selectPreset(S, currentPreset(node), true); };
    chipsCap.append(S.reset);
    S.chips = el("div", "svk-chips");
    S.chips.title = T("tap — on/off · drag left/right or scroll — strength");

    // ⋯ ще
    S.more = el("details", "svk-more");
    S.more.append(moreSummary(4));
    const body = el("div", "body");
    S.more.append(body);
    const row = (label, ctl) => { body.append(el("span", null, label), ctl); };
    const selSize = el("select");
    for (const v of (W(node, "output_size")?.options?.values || [])) selSize.append(new Option(v, v));
    selSize.onchange = () => { setW(node, "output_size", selSize.value); markChanged(); };
    row(T("Output size"), selSize);
    const numMax = el("input"); numMax.type = "number"; numMax.min = 0; numMax.max = 16384; numMax.step = 8;
    numMax.onchange = () => { setW(node, "max_side", Math.max(0, Math.round(Number(numMax.value) || 0))); markChanged(); };
    row(T("Working resolution"), numMax);
    numMax.title = T("0 — compute directly at the output size. A smaller value saves memory on long clips.");
    const cbClean = el("input"); cbClean.type = "checkbox";
    cbClean.onchange = () => { setW(node, "clean_depth", cbClean.checked); markChanged(); };
    row(T("Clean depth"), cbClean);
    const cbNear = el("input"); cbNear.type = "checkbox";
    cbNear.onchange = () => { setW(node, "depth_light_is_near", cbNear.checked); markChanged(); };
    row(T("Light = near"), cbNear);
    S.more_ctl = { selSize, numMax, cbClean, cbNear };

    S.status = stageLine(node);
    root.append(frame, S.status, mix, tilesCap, S.tiles, chipsCap, S.chips, S.more);
    S.hint = scrollHint(root);
    return S;
}

function mixValue(node) { const v = Number(W(node, "look_mix")?.value); return Number.isFinite(v) ? clamp(v, 0, 1) : 1; }
function currentPreset(node) { return W(node, "preset")?.value; }
function paintMix(S) {
    const v = mixValue(S.node);
    S.mixRange.value = v;
    S.mixRange.style.setProperty("--p", (v * 100) + "%");
    S.mixNum.textContent = Math.round(v * 100) + "%";
    S.imgLook.style.opacity = String(v);
}

// ---- стан чіпів = нативні віджети (_on/_power) + extra_json (ефекти, додані поверх пресету)
function getExtra(node) { try { const j = JSON.parse(W(node, "extra_json")?.value || "{}"); return j && typeof j === "object" ? j : {}; } catch (_) { return {}; } }
function setExtra(node, o) { setW(node, "extra_json", JSON.stringify(o)); }
function presetPowers(S) { return (S.info?.presets || []).find(p => p.name === currentPreset(S.node))?.powers || {}; }
function chipState(S, g) {
    const node = S.node;
    const on = !!W(node, g.key + "_on")?.value;
    const mul = Number(W(node, g.key + "_power")?.value ?? 1);
    const base = presetPowers(S)[g.key] || 0;
    const extra = getExtra(node)[g.key] || 0;
    const present = base > 0 || extra > 0;
    return { on, mul, present, active: present && on && mul > 0 };
}
function resetChips(S) {
    for (const g of S.info?.groups || []) { setW(S.node, g.key + "_on", true); setW(S.node, g.key + "_power", 1.0); }
    setExtra(S.node, {});
    markChanged(); paintChips(S);
}
function isModified(S) {
    for (const g of S.info?.groups || []) {
        const c = chipState(S, g);
        if (!c.on || Math.abs(c.mul - 1) > 1e-6) return true;
    }
    return Object.keys(getExtra(S.node)).length > 0;
}

function buildChips(S) {
    S.chips.textContent = "";
    S.chipEls = {};
    for (const g of S.info.groups) {
        const b = el("button", "svk-chip");
        b.type = "button";
        b.title = g.name + " — " + g.hint + "\n" + T("tap — on/off · drag left/right or use the wheel — strength");
        b.innerHTML = (g.icon || "") + '<span class="val"></span><span class="lb"></span>';
        b.querySelector(".lb").textContent = g.label;
        wireChip(S, g, b);
        S.chips.append(b);
        S.chipEls[g.key] = b;
    }
    paintChips(S);
}
function paintChips(S) {
    for (const g of S.info?.groups || []) {
        const b = S.chipEls?.[g.key]; if (!b) continue;
        const c = chipState(S, g);
        b.classList.toggle("is-absent", !c.present);
        b.classList.toggle("is-off", c.present && !c.active);
        b.style.setProperty("--fill", clamp(c.mul / 2, 0, 1) * 100 + "%");
        b.querySelector(".val").textContent = c.active ? "×" + c.mul.toFixed(c.mul >= 1 ? 1 : 2).replace(/\.?0+$/, "") : (c.present ? T("off") : "+");
    }
    const mod = isModified(S);
    S.reset.style.display = mod ? "" : "none";
    for (const t of S.tiles.children) t.classList.toggle("is-mod", mod);
}
function wireChip(S, g, b) {
    const node = S.node;
    let st = null;
    const apply = (mul, silent) => {
        mul = Math.round(clamp(mul, 0, 2) * 20) / 20;
        if (Math.abs(mul - 1) < 0.04) mul = 1;
        setW(node, g.key + "_power", mul);
        if (mul > 0) setW(node, g.key + "_on", true);
        paintChips(S); markChanged(); schedulePreview(S);
    };
    const ensurePresent = () => {            // ефекту нема в пресеті - додаємо з типовою силою (extra_json)
        const c = chipState(S, g);
        if (c.present) return;
        const ex = getExtra(node); ex[g.key] = g.add; setExtra(node, ex);
        setW(node, g.key + "_on", true); setW(node, g.key + "_power", 1.0);
    };
    b.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        const c = chipState(S, g);
        st = { x: e.clientX, mul: c.mul || 1, moved: false, w: Math.max(40, b.getBoundingClientRect().width), absent: !c.present };
        b.setPointerCapture?.(e.pointerId);
    });
    b.addEventListener("pointermove", e => {
        if (!st) return;
        const dx = e.clientX - st.x;
        if (!st.moved && Math.abs(dx) < 5) return;
        if (!st.moved) { st.moved = true; b.classList.add("is-drag"); if (st.absent) { ensurePresent(); st.absent = false; } }
        apply(st.mul + (dx / st.w) * 2);       // ширина чіпа = 2.0 сили
    });
    const end = () => {
        if (!st) return;
        const was = st; st = null;
        b.classList.remove("is-drag");
        if (was.moved) return;
        // тап
        const c = chipState(S, g);
        if (!c.present) { ensurePresent(); }
        else if (c.active) setW(node, g.key + "_on", false);
        else { setW(node, g.key + "_on", true); if (Number(W(node, g.key + "_power")?.value) <= 0) setW(node, g.key + "_power", 1.0); }
        paintChips(S); markChanged(); schedulePreview(S);
    };
    b.addEventListener("pointerup", end);
    b.addEventListener("pointercancel", () => { st = null; b.classList.remove("is-drag"); });
    onWheel(b, e => {
        e.preventDefault();
        const c = chipState(S, g);
        if (!c.present) ensurePresent();
        apply((Number(W(node, g.key + "_power")?.value) || 0) + (e.deltaY < 0 ? 0.1 : -0.1));
    });
}

// ---- плитки
function buildTiles(S) {
    S.tiles.textContent = "";
    S.tileEls = {};
    for (const p of S.info.presets) {
        const t = el("button", "svk-tile"); t.type = "button"; t.title = p.name;
        t.innerHTML = '<div class="th"><img alt="" draggable="false"><span class="dot" title="' + T("effects edited manually") + '"></span></div><div class="nm"></div>';
        t.querySelector(".nm").textContent = presetLabel(p.name); t.title = presetLabel(p.name) + " · " + p.name;
        t.onclick = () => selectPreset(S, p.name, false);
        S.tiles.append(t);
        S.tileEls[p.name] = t;
    }
    paintTiles(S);
    loadThumbs(S);
}
function paintTiles(S) {
    const cur = currentPreset(S.node);
    for (const [n, t] of Object.entries(S.tileEls || {})) t.classList.toggle("is-sel", n === cur);
}
async function loadThumbs(S) {
    const my = ++S.thumbToken;
    if (!S.info?.node_frame) return;
    const names = Object.keys(S.tileEls);
    const cur = currentPreset(S.node);
    names.sort((a, b) => (a === cur ? -1 : b === cur ? 1 : 0));       // обраний лук - першим
    for (const n of names) {
        if (my !== S.thumbToken) return;
        const img = S.tileEls[n]?.querySelector("img"); if (!img) continue;
        await new Promise(res => {
            img.onload = () => { img.classList.add("ok"); img.parentElement.classList.add("done"); res(); };
            img.onerror = () => { img.parentElement.classList.add("done"); res(); };
            img.src = simpleURL(S.node, "thumb?preset=" + encodeURIComponent(n) + "&size=" + THUMB_SIZE + "&v=" + (S.info.stamp || 0));
        });
        await new Promise(r => setTimeout(r, 30));                    // не душимо Мак: плитки йдуть по одній
    }
}
function selectPreset(S, name, keepChips) {
    const node = S.node;
    if (!keepChips) { resetChips(S); }
    setW(node, "preset", name);
    paintTiles(S); paintChips(S); markChanged(); schedulePreview(S);
}

// ---- превʼю: один запит у польоті, береться останній стан
let _timer = new WeakMap();
function schedulePreview(S) {
    clearTimeout(_timer.get(S));
    _timer.set(S, setTimeout(() => doPreview(S), 140));
}
function collectBody(S) {
    const sw = {};
    for (const g of S.info.groups) sw[g.key] = [!!W(S.node, g.key + "_on")?.value, Number(W(S.node, g.key + "_power")?.value ?? 1)];
    return { preset: currentPreset(S.node), switches: sw, extra: getExtra(S.node), size: PREVIEW_SIZE, frame: 0 };
}
async function doPreview(S) {
    if (!S.info?.node_frame) return;
    if (S.inflight) { S.pending = true; return; }
    S.inflight = true; S.pending = false;
    S.frame.classList.add("is-busy");
    try {
        const r = await fetch(simpleURL(S.node, "render"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(collectBody(S)) });
        if (r.status === 409) {
            const j = await r.json().catch(() => ({}));
            if (j.busy) return;                            // іде фінальний рендер: лишаємо старе превʼю, після Run картка оновиться сама
            S.info.node_frame = false; showEmpty(S); return;
        }
        if (!r.ok) { let m = ""; try { m = (await r.json()).err; } catch (_) {} showEmpty(S, m || T("error {s}").replace("{s}", r.status)); return; }
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        if (S.blobURL) URL.revokeObjectURL(S.blobURL);
        S.blobURL = url;
        S.imgLook.src = url;
        S.imgLook.style.visibility = "visible";
        S.empty.style.display = "none"; S.tag.style.display = "";
    } catch (e) {
        showEmpty(S, String(e.message || e));
    } finally {
        S.inflight = false;
        S.frame.classList.remove("is-busy");
        if (S.pending) schedulePreview(S);
    }
}
function showEmpty(S, err) {
    S.empty.style.display = ""; S.tag.style.display = "none";
    S.imgLook.style.visibility = "hidden"; S.imgOrig.style.visibility = "hidden";
    S.empty.className = "svk-empty" + (err ? " still" : "");
    S.empty.textContent = "";
    if (err) {
        const p = el("p", "svk-err"); p.textContent = err; S.empty.append(p);
        return;
    }
    const b = el("button", "svk-btn primary", ICON.play + " " + T("Run"));
    b.style.cssText = "display:inline-flex;align-items:center;gap:6px";
    b.title = T("Connect image and depth and run the graph once — after that the frame and look tiles update without Run");
    b.onclick = () => { try { app.queuePrompt(0, 1); } catch (e) { console.warn("[DepthLook]", e); } };
    S.empty.append(b);
}

async function loadInfo(S) {
    const node = S.node;
    if (node.id == null || node.id < 0) return;
    const my = ++S.token;
    let info;
    try {
        const r = await fetch(simpleURL(node, "info"));
        info = await r.json();
    } catch (e) { showEmpty(S, T("server did not respond: {e}").replace("{e}", e.message || e)); return; }
    if (my !== S.token) return;
    const first = !S.info;
    const stampChanged = !S.info || S.info.stamp !== info.stamp;
    S.info = info;
    if (first) { buildChips(S); }
    if (first || stampChanged || !S.tileEls || Object.keys(S.tileEls).length !== info.presets.length) buildTiles(S);
    syncFromWidgets(S, true);
    if (!info.node_frame) { showEmpty(S); return; }
    if (stampChanged) {
        S.imgOrig.src = simpleURL(node, "original?size=" + PREVIEW_SIZE + "&v=" + (info.stamp || 0));
        S.imgOrig.style.visibility = "visible";
    }
    if (!S.imgLook.src) S.empty.className = "svk-empty";      // скелет-шиммер, поки перший кадр рахується
    schedulePreview(S);
}
function syncFromWidgets(S, noPreview) {
    const node = S.node;
    const names = (S.info?.presets || []).map(p => p.name);
    const cur = currentPreset(node);
    if (names.length && !names.includes(cur)) setW(node, "preset", names[0]);   // пресет зі workflow зник з диска
    paintMix(S); paintTiles(S); if (S.info) paintChips(S);
    const c = S.more_ctl;
    c.selSize.value = W(node, "output_size")?.value ?? c.selSize.value;
    c.numMax.value = W(node, "max_side")?.value ?? 0;
    c.cbClean.checked = !!W(node, "clean_depth")?.value;
    c.cbNear.checked = !!W(node, "depth_light_is_near")?.value;
}

function setupSimple(node) {
    if (node.__svk) return;
    const S = buildSimple(node);
    node.__svk = S;
    const MIN_H = 420;
    const dw = node.addDOMWidget("svk_simple", "svk_simple", S.root, { serialize: false, hideOnZoom: false, getMinHeight: () => MIN_H });
    for (const w of [...node.widgets]) if (w !== dw && w.name !== "svk_simple") hideWidget(w);
    showEmpty(S);
    node.setSize([340, MIN_H + 60]);
    fitToNode(node, S.root);
    node.color = NODE_COLOR; node.bgcolor = NODE_BG;
    syncFromWidgets(S);
}
function startSimple(node) {
    const S = node.__svk;
    if (S && !S.started && node.id != null && node.id >= 0) { S.started = true; loadInfo(S); }
}

// ---------------------------------------------------------------- стрічка відео-ноди
// на ноді: превʼю кадрів, fps, превʼю/фінал, «Рух кольору» (режим словами) і «Рух ефектів» - усі ефекти на видноті.
// Тонкі параметри показуються лише тоді, коли діють (look.py): у вибраному режимі руху кольору, при ввімкненому пульсі чи зерні.
// Під «ще» - тільки технічне: старт, прорідження превʼю, розміри.
const GRAD_MODES = [[0, T("Off")], [1, T("Follows camera")], [2, T("Drift")], [3, T("Flow")], [4, T("Through depth")], [5, T("Wave")]];
const GRAD_SUB = {                 // які параметри діють у кожному режимі grad_mode
    1: ["grad_offset_x", "grad_offset_y"],
    2: ["grad_speed", "grad_angle", "grad_phase", "grad_loop"],
    3: ["grad_speed", "grad_phase", "grad_loop"],
    4: ["grad_speed", "grad_wavelength", "grad_phase", "grad_loop"],
    5: ["grad_speed", "grad_angle", "grad_wavelength", "grad_amp", "grad_phase", "grad_loop"],
};
const GRAD_LABELS = {
    grad_speed: T("Speed, cycles/s"), grad_angle: T("Direction, °"), grad_wavelength: T("Wavelength"), grad_amp: T("Wave height"),
    grad_phase: T("Start phase"), grad_loop: T("Seamless loop, s"), grad_offset_x: T("Shift X"), grad_offset_y: T("Shift Y"),
};
const FX = [                       // [параметр, підпис, підпараметри, коли показувати підпараметри]
    ["warp_speed", T("Breathing wave")],
    ["glow_pulse", T("Light pulse"), [["pulse_hz", T("Pulse rate, Hz")]]],
    ["haze_drift", T("Drifting haze")],
    ["trail_wobble", T("Trail wobble, °")],
    ["grain_animate", T("Live grain"), [["grain_fps", T("Grain, frames/s")]]],
];
// «Рух по глибині» (depth_look/depth_motion.py): [сила, підпис, [[параметр, підпис, "c" = hex-колір | {names} = вибір]...]]
const DM = [
    ["dm_scan", T("Depth scanner"), [["dm_scan_speed", T("Speed, sweeps/s")], ["dm_scan_width", T("Width")], ["dm_scan_color", T("Colour"), "c"]]],
    ["dm_par", T("2.5D camera (parallax)"), [["dm_par_speed", T("Speed, loops/s")], ["dm_par_focus", T("Still plane")],
        ["dm_par_shape", T("Path"), { circle: T("Circle"), figure8: T("Figure-8") }]]],
    ["dm_focus", T("Focus pull"), [["dm_focus_speed", T("Speed, sweeps/s")], ["dm_focus_range", T("Sharp zone width")]]],
    ["dm_reveal", T("Reveal / dissolve"), [["dm_reveal_speed", T("Speed, cycles/s")], ["dm_reveal_soft", T("Edge softness")],
        ["dm_reveal_mode", T("Mode"), { appear_near: T("Appear: near first"), appear_far: T("Appear: far first"), loop_near: T("Loop: near first"), loop_far: T("Loop: far first") }]]],
    ["dm_iso", T("Depth contour lines"), [["dm_iso_count", T("Lines count")], ["dm_iso_speed", T("Run speed, lines/s")], ["dm_iso_width", T("Thickness, px")], ["dm_iso_color", T("Colour"), "c"]]],
    ["dm_echo", T("Echo copies"), [["dm_echo_count", T("Copies count")], ["dm_echo_speed", T("Wave speed, waves/s")], ["dm_echo_scale", T("Growth")], ["dm_echo_thresh", T("Near threshold")]]],
    ["dm_light", T("Relief light"), [["dm_light_speed", T("Rotation, turns/s")], ["dm_light_height", T("Light height")], ["dm_light_relief", T("Relief depth")], ["dm_light_spec", T("Highlight")]]],
];
// рядок із вибором кольору, що пише hex прямо в нативний STRING-віджет
function colorControl(node, name, label, onChange) {
    const row = el("label", "svk-field");
    const inp = el("input"); inp.type = "color";
    const norm = v => { const s = String(v || "").trim(); return /^#[0-9a-fA-F]{6}$/.test(s) ? s.toLowerCase() : "#ffffff"; };
    inp.oninput = () => { setW(node, name, inp.value); markChanged(); onChange?.(); };
    row.append(el("span", "t", label), inp);
    const sync = () => { inp.value = norm(W(node, name)?.value); };
    sync();
    return { row, sync };
}
const VIDEO_MORE = [["start_time", T("Start, s")], ["preview_every", T("Every Nth frame in preview")],
                    ["output_size", T("Output size")], ["max_side", T("Working resolution")],
                    ["hdr", T("HDR input"), { names: { "SDR or HLG/PQ as encoded": T("SDR / HLG·PQ as is"), "linear (keeps highlights)": T("linear, keep highlights") } }]];
// «Render final»: свіжий одноразовий токен + запуск; бекенд зрахує превʼю, потім фінал і пропустить виходи далі
async function renderFinal(node) {
    setW(node, "render_token", String(Date.now()));
    const hasFinal = !!node.outputs?.find(o => o.name === "final")?.links?.length;
    const q = W(node, "quality");
    if (!hasFinal && q && Array.isArray(node.__svkv?.specs?.quality?.[0])) setW(node, "quality", node.__svkv.specs.quality[0][1]);   // у image підключено - він і має нести фінал
    markChanged();
    node.__svkv?.sync?.();
    await runVideo(node);
    setW(node, "render_token", "");          // токен уже в завданні; у графі не лишаємо - збережений граф сам фінал не запустить
    markChanged();
}
// верхній рядок відео-ноди: fps + превʼю/фінал. Контроли пишуть прямо у нативні віджети ноди, тож їх можна показати де завгодно
function buildTop(node, specs, onChange) {
    const bar = el("div", "svk-bar");
    const ctl = [];
    let fpsRow = null, fpsTip = "";
    if (specs.fps) { const c = control(node, "fps", specs.fps, "fps", { number: true, onChange }); c.row.style.flex = "1"; fpsRow = c.row; fpsTip = c.row.title; bar.append(c.row); ctl.push(c); }
    if (specs.quality) {
        const vals = specs.quality[0] || [];
        const c = control(node, "quality", specs.quality, "", { seg: { [vals[0]]: T("preview"), [vals[1]]: T("final") }, tip: specs.quality[1]?.tooltip, onChange });
        bar.append(c.row); ctl.push(c);
    }
    // гейт рендеру: превʼю йде завжди, фінал - за кнопкою (одноразовий токен) або з галочкою auto
    const gate = el("div", "svk-bar"); gate.style.cssText = "flex-basis:100%;justify-content:flex-end;align-items:center;gap:8px";
    const rb = el("button", "svk-btn primary", ICON.play + " " + T("Render final")); rb.type = "button";
    rb.style.cssText = "display:inline-flex;align-items:center;gap:6px;flex:none;padding:3px 12px;font-size:12px";   // компактна, справа - не на всю ширину
    rb.title = T("Renders the final once with the current settings and passes it to the connected nodes (Video Combine / Save). A normal run only renders the preview.");
    rb.onclick = () => renderFinal(node);
    // авто-превʼю: властивість ноди, не віджет (widgets_values позиційні - новий віджет зсунув би старі графи)
    const ap = el("label", "svk-num"); ap.style.cssText = "display:inline-flex;align-items:center;gap:4px;margin-right:auto;cursor:pointer";
    ap.title = T("Re-render the preview by itself 1.5 s after you stop changing settings");
    const apc = el("input"); apc.type = "checkbox"; apc.checked = autoOn(node);
    apc.onchange = () => {
        node.properties = node.properties || {}; node.properties.svk_auto_preview = apc.checked; markChanged();
        const V = node.__svkv; if (V) { V.autoBlock = null; paintStale(V); tryAuto(V); }
    };
    ap.append(apc, document.createTextNode(T("Auto preview")));
    ctl.push({ sync: () => { apc.checked = autoOn(node); } });
    gate.append(ap, rb);
    if (specs.auto_render && W(node, "auto_render")) {
        const c = control(node, "auto_render", specs.auto_render, T("Auto"), { onChange });
        c.row.style.flex = "none"; gate.append(c.row); ctl.push(c);
    }
    bar.style.flexWrap = "wrap"; bar.append(gate);
    const paintFps = () => {
        const linked = fpsLinked(node);
        if (fpsRow) { fpsRow.style.opacity = linked ? ".4" : ""; fpsRow.style.pointerEvents = linked ? "none" : ""; fpsRow.title = linked ? T("fps comes from the connected input") : fpsTip; }
    };
    return { bar, ctl, fpsRow, fpsTip, paintFps };
}
// ---- іменовані пресети руху/анімації відео-ноди (сервер: depth_look/video_presets.py). Зберігається все, що є в нативних віджетах
// (сервер лишає лише ключі руху й анімації; fps, розмір, якість, службові поля не входять).
const VP_URL = "/depth_look/video_presets";
async function vpCall(path, body) {
    const r = await api.fetchApi(VP_URL + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.err || ("HTTP " + r.status));
    return j;
}
function coerceToSpec(name, spec, v) {          // значення пресета -> допустиме для віджета (діапазон / список), інакше undefined
    if (!spec) return undefined;
    const [t, o = {}] = spec;
    const opts = Array.isArray(t) ? t : t === "COMBO" ? (o.options || []) : null;
    if (opts) return opts.includes(v) ? v : undefined;
    if (t === "INT" || t === "FLOAT") {
        if (typeof v !== "number" || !isFinite(v)) return undefined;
        if (o.min != null) v = Math.max(o.min, v);
        if (o.max != null) v = Math.min(o.max, v);
        return t === "INT" ? Math.round(v) : v;
    }
    if (t === "BOOLEAN") return typeof v === "boolean" ? v : undefined;
    if (t === "STRING") return typeof v === "string" && (!/color$/.test(name) || /^#[0-9a-fA-F]{6}$/.test(v)) ? v : undefined;
    return undefined;
}
function presetRow(node, specs, onLoaded) {
    const row = el("div", "svk-bar svk-field");
    const sel = el("select"); sel.style.cssText = "flex:1;min-width:0;width:auto";
    sel.title = T("Saved motion & animation settings — pick one to load it");
    const save = el("button", "svk-btn", T("Save as…")); save.type = "button";
    save.style.cssText = "flex:none;padding:3px 10px;font-size:12px";
    save.title = T("Save the current motion & animation settings under a name");
    const del = el("button", "svk-ibtn", "✕"); del.type = "button"; del.title = T("Delete this preset");
    row.append(sel, save, del);
    let list = [];
    const cur = () => list.find(p => p.name === sel.value);
    const paint = () => {
        const off = !cur() || cur().locked;
        del.disabled = off; del.style.opacity = off ? ".35" : ""; del.style.pointerEvents = off ? "none" : "";
        del.title = cur()?.locked ? T("Built-in preset — cannot be deleted") : T("Delete this preset");
    };
    const fill = (pick) => {
        sel.textContent = "";
        sel.append(new Option(T("— preset —"), ""));
        for (const p of list) sel.append(new Option(p.name, p.name));
        sel.value = list.some(p => p.name === pick) ? pick : "";
        paint();
    };
    const refresh = async (pick) => {
        try { list = (await vpCall("")).presets || []; } catch (e) { console.warn("[DepthLook] presets:", e); list = []; }
        fill(pick ?? sel.value);
    };
    sel.onchange = async () => {
        paint();
        if (!sel.value) return;
        try {
            const { values } = await vpCall("/" + encodeURIComponent(sel.value));
            for (const [k, v] of Object.entries(values || {})) {
                if (!W(node, k)) continue;
                const c = coerceToSpec(k, specs[k], v);
                if (c !== undefined) setW(node, k, c);
            }
            markChanged();
            onLoaded?.();
        } catch (e) { console.warn("[DepthLook] preset load:", e); window.alert(T("Could not load the preset") + ": " + e.message); }
    };
    save.onclick = async () => {
        const name = (window.prompt(T("Preset name (letters, digits, _ - .)"), cur() && !cur().locked ? cur().name : "") || "").trim();
        if (!name) return;
        const exists = list.some(p => p.name.toLowerCase() === name.replace(/ /g, "_").toLowerCase());
        if (exists && !window.confirm(T("Overwrite preset") + " «" + name + "»?")) return;
        const values = {};
        for (const w of node.widgets || []) if (["number", "string", "boolean"].includes(typeof w.value) && specs[w.name]) values[w.name] = w.value;
        try { const j = await vpCall("/save", { name, values }); list = j.presets || list; fill(j.name); }
        catch (e) { console.warn("[DepthLook] preset save:", e); window.alert(T("Could not save the preset") + ": " + e.message); }
    };
    del.onclick = async () => {
        const p = cur(); if (!p || p.locked) return;
        if (!window.confirm(T("Delete preset") + " «" + p.name + "»?")) return;
        try { const j = await vpCall("/delete", { name: p.name }); list = j.presets || []; fill(""); }
        catch (e) { console.warn("[DepthLook] preset delete:", e); window.alert(T("Could not delete the preset") + ": " + e.message); }
    };
    refresh("");
    return { row, refresh };
}
// рух кольору + рух ефектів + «ще»: один збірник і для картки в ноді, і для панелі повного екрана
function buildMotion(node, specs, onChange) {
    const ctl = [];
    const add = (parent, name, label, opts = {}) => {
        if (!specs[name] || !W(node, name)) return null;
        const c = control(node, name, specs[name], label, { ...opts, onChange });
        parent.append(c.row); ctl.push(c); return c;
    };
    // рух кольору: режим словами замість «0 static · 1 camera…», далі лише те, що в ньому діє
    const col = el("div", "svk-sec");
    col.append(el("div", "svk-cap", "<b>" + T("Color motion") + "</b>"));
    const modeRow = el("label", "svk-field"); modeRow.title = specs.grad_mode?.[1]?.tooltip || "";
    const modeSel = el("select");
    for (const [v, lb] of GRAD_MODES) modeSel.append(new Option(lb, v));
    modeSel.onchange = () => { setW(node, "grad_mode", Number(modeSel.value)); markChanged(); onChange?.(); };
    modeRow.append(el("span", "t", T("Mode")), modeSel);
    col.append(modeRow);
    const amt = add(col, "grad_amount", T("Strength"));
    const sub = el("div", "svk-sub"); col.append(sub);
    const subCtl = {};
    for (const name of new Set(Object.values(GRAD_SUB).flat())) { const c = add(sub, name, GRAD_LABELS[name] || name); if (c) subCtl[name] = c; }
    // рух ефектів: кожен ефект - один повзунок; уточнення - під ним, коли ефект увімкнено
    const fx = el("div", "svk-sec");
    fx.append(el("div", "svk-cap", "<b>" + T("Effect motion") + "</b>"));
    const fxSub = [];
    for (const [name, label, subs] of FX) {
        const c = add(fx, name, label);
        if (!c || !subs) continue;
        const box = el("div", "svk-sub"); fx.append(box);
        for (const [sn, sl] of subs) add(box, sn, sl);
        fxSub.push([name, box]);
    }
    // рух по глибині: сила - повзунок; уточнення з'являються, коли сила > 0
    const dm = el("div", "svk-sec");
    dm.append(el("div", "svk-cap", "<b>" + T("Depth motion") + "</b>"));
    const dmSub = [];
    for (const [name, label, subs] of DM) {
        const c = add(dm, name, label);
        if (!c) continue;
        const box = el("div", "svk-sub"); dm.append(box);
        for (const [sn, sl, kind] of subs) {
            if (kind === "c") {
                if (!specs[sn] || !W(node, sn)) continue;
                const cc = colorControl(node, sn, sl, onChange); box.append(cc.row); ctl.push(cc);
            } else add(box, sn, sl, kind ? { names: kind } : {});
        }
        dmSub.push([name, box]);
    }
    const more = morePanel(node, specs, VIDEO_MORE);
    const pr = presetRow(node, specs, () => { sync(); onChange?.(); });
    const refresh = () => {
        const mode = Number(W(node, "grad_mode")?.value || 0);
        modeSel.value = String(mode);
        if (amt) amt.row.style.display = mode ? "" : "none";
        const on = new Set(GRAD_SUB[mode] || []);
        for (const [n, c] of Object.entries(subCtl)) c.row.style.display = on.has(n) ? "" : "none";
        sub.classList.toggle("is-hidden", !on.size);
        for (const [n, box] of fxSub) box.classList.toggle("is-hidden", !Number(W(node, n)?.value));
        for (const [n, box] of dmSub) box.classList.toggle("is-hidden", !Number(W(node, n)?.value));
    };
    const sync = () => { ctl.forEach(c => c.sync()); more.sync(); refresh(); };
    refresh();
    return { els: [pr.row, col, fx, dm, more], more, ctl, refresh, sync };
}
function setupVideo(node, specs) {
    if (node.__svkv) return;
    const V = { node, meta: null, sel: 0, timer: null };
    node.__svkv = V;
    const root = el("div", "svk-card svk-scroll");
    keepWheel(root);
    const cap = el("div", "svk-cap");
    V.capL = el("b", null, T("Frames"));
    V.capR = el("span", "svk-num", "");
    cap.append(V.capL, V.capR);
    const view = el("div", "svk-frame svk-viewer");
    V.view = view;
    V.img = el("img"); V.img.alt = ""; V.img.draggable = false; V.img.style.visibility = "hidden";
    // рамка під пропорції кадру (вертикальне 9:16 і широке), а не завжди 16:9; висоту стримує max-height картки
    V.img.addEventListener("load", () => { if (V.img.naturalWidth) view.style.aspectRatio = String(clamp(V.img.naturalWidth / V.img.naturalHeight, 0.5, 2.4)); });
    V.empty = el("div", "svk-empty still");
    const run = el("button", "svk-btn primary", ICON.play + " " + T("Preview"));
    run.style.cssText = "display:inline-flex;align-items:center;gap:6px";
    run.title = T("The frame strip appears after a preview run: set the mode to “preview” and run the graph");
    run.onclick = () => { setW(node, "quality", (specs.quality?.[0] || [])[0] ?? W(node, "quality")?.value); markChanged(); runVideo(node); };
    V.empty.append(run);
    V.play = el("button", "svk-ibtn", ICON.play); V.play.type = "button"; V.play.title = T("play the strip");
    V.play.style.cssText = "position:absolute;right:6px;bottom:6px;display:none";
    V.badge = el("div", "svk-stale"); V.badge.style.display = "none";
    V.badgeTxt = el("span");
    V.badgeBtn = el("button", "svk-btn primary", T("Update")); V.badgeBtn.type = "button";
    V.badgeBtn.title = T("Render the preview with the current settings now");
    V.badgeBtn.onclick = () => runVideo(node);
    V.badge.append(V.badgeTxt, V.badgeBtn);
    view.append(V.img, V.empty, V.play, el("div", "svk-busy"), V.badge);
    V.strip = el("div", "svk-strip");
    V.status = stageLine(node);
    const refresh = () => { V.refresh?.(); videoDirty(V); };      // зміна параметра користувачем: превʼю застаріло
    // fps + превʼю/фінал; повзунки руху й «ще» - ті самі збірники, що й у повному екрані (openVideoFull)
    const top = buildTop(node, specs, refresh);
    const bar = top.bar, ctl = top.ctl;
    V.fpsRow = top.fpsRow; V.fpsTip = top.fpsTip;
    V.fullBtn = el("button", "svk-ibtn", ICON.full); V.fullBtn.type = "button"; V.fullBtn.title = T("full screen (Esc — back)");
    V.fullBtn.onclick = () => openVideoFull(V);
    bar.append(V.fullBtn);
    V.specs = specs;
    const M = buildMotion(node, specs, refresh);
    V.more = M.more;
    root.append(cap, view, V.strip, V.status, bar, ...M.els);
    V.hint = scrollHint(root);
    V.refresh = () => {
        top.paintFps();                                    // fps прийшов із сокета - поле не діє
        M.refresh();
        V.hint?.();
    };
    V.sync = () => { ctl.forEach(c => c.sync()); M.sync(); V.refresh(); };
    V.play.onclick = () => togglePlay(V);
    node.addDOMWidget("svk_strip", "svk_strip", root, { serialize: false, hideOnZoom: false, getMinHeight: () => 380 });
    for (const w of [...node.widgets]) if (w.name !== "svk_strip") hideWidget(w);
    node.color = NODE_COLOR; node.bgcolor = NODE_BG;
    node.setSize([360, 640]);
    fitToNode(node, root);
    V.refresh();
}
// значення з'їхали (поле hdr переїжджало між версіями): невалідне поле → дефолт, щоб граф запускався
function sanitizeWidgets(node, input) {
    const spec = { ...(input?.required || {}), ...(input?.optional || {}) };
    for (const w of node.widgets || []) {
        const s = spec[w.name]; if (!s) continue;
        const [t, o = {}] = s;
        const opts = Array.isArray(t) ? t : t === "COMBO" ? (o.options || []) : null;
        let ok = true;
        if (opts) ok = opts.includes(w.value);
        else if (t === "INT" || t === "FLOAT") ok = typeof w.value === "number" && isFinite(w.value) && !(o.min != null && w.value < o.min) && !(o.max != null && w.value > o.max);
        else if (t === "BOOLEAN") ok = typeof w.value === "boolean";
        else if (t === "STRING") ok = typeof w.value === "string" && (!/^dm_.*color/.test(w.name) || /^#[0-9a-fA-F]{6}$/.test(w.value));
        if (!ok) { w.value = "default" in o ? o.default : opts?.[0]; console.warn("[DepthLook] reset shifted value:", w.name); }
    }
    // зсув на одну позицію лишає валідні числа: у силі ефектів лягають дефолти швидкостей → вмикаються 5 ефектів, яких не обирали
    const val = k => node.widgets?.find(x => x.name === k)?.value;
    const sig = [["dm_scan", 0.25], ["dm_par", 0.3], ["dm_focus", 0.15], ["dm_reveal", 0.2], ["dm_light", 0.15]];
    if (sig.every(([k, v]) => Math.abs(val(k) - v) < 1e-6)) {
        for (const w of node.widgets || []) {
            const o = spec[w.name]?.[1];
            if (w.name.startsWith("dm_") && o && "default" in o) w.value = o.default;
        }
        console.warn("[DepthLook] depth-motion values were shifted by an old graph — reset to defaults (all effects off)");
    }
}
// вихід final підключено → прев'ю-запуск рахує ще й фінал (бекенд не бачить, куди йдуть виходи)
function syncFinalOut(node) {
    const w = node.widgets?.find(x => x.name === "final_out");
    if (w) w.value = !!node.outputs?.find(o => o.name === "final")?.links?.length;
}
const fpsLinked = node => node.inputs?.some(i => i.name === "fps_in" && i.link != null);
// ---- превʼю за параметрами: зміна → плашка «кадри старі» і (авто) перерахунок після 1.5 с тиші; черга Comfy зайнята - чекаємо її.
// ver - скільки разів змінювали параметри, sentVer - з якою версією пішов прогін, shownVer - якій версії відповідають кадри
let QUEUE_LEFT = 0;
const AUTO_DELAY = 1500;
const autoOn = node => node.properties?.svk_auto_preview !== false;
const isStale = V => (V.ver || 0) !== (V.shownVer || 0);
const canAuto = V => autoOn(V.node) && !!V.meta?.n && !W(V.node, "auto_render")?.value && V.autoBlock !== (V.ver || 0);
async function runVideo(node) {            // усі запуски з відео-ноди йдуть сюди
    const V = node.__svkv;
    if (V) { V.sentVer = V.ver || 0; clearTimeout(V.autoT); V.autoT = null; V.waiting = true; paintStale(V); }
    try { await app.queuePrompt(0, 1); } catch (e) { console.warn("[DepthLook]", e); if (V) { V.waiting = false; paintStale(V); } }
}
function videoDirty(V) {
    V.ver = (V.ver || 0) + 1;
    clearTimeout(V.autoT); V.autoT = null;
    if (canAuto(V)) V.autoT = setTimeout(() => { V.autoT = null; tryAuto(V); }, AUTO_DELAY);
    paintStale(V);
}
function tryAuto(V) {
    if (!isStale(V) || !canAuto(V) || V.autoT || V.waiting) return;
    if (QUEUE_LEFT > 0) { paintStale(V); return; }          // повернемось на події status, коли черга спорожніє
    runVideo(V.node);
}
function paintStale(V) {
    if (!V.badge) return;
    const stale = isStale(V), sent = V.waiting && V.sentVer === (V.ver || 0);
    V.view.classList.toggle("is-busy", !!V.waiting);
    V.view.classList.toggle("is-stale", stale && !!V.meta?.n);
    const show = stale && !!V.meta?.n;
    V.badge.style.display = show ? "" : "none";
    V.badgeTxt.textContent = sent ? T("Rendering the preview with the new settings…")
        : canAuto(V) ? (QUEUE_LEFT > 0 ? T("Settings changed — the preview updates when the queue is free") : T("Settings changed — updating the preview…"))
        : T("Settings changed — these frames are old");
    V.badgeBtn.style.display = sent ? "none" : "";
    V.full?.paintStale?.(show ? V.badgeTxt.textContent : "");
}
function videoNodes() { return (app.graph?._nodes || []).filter(n => n.__svkv); }
function stripURL(node, i, stamp) { return api.apiURL("/depth_look/strip/" + encodeURIComponent(node.id) + "/" + i + "?v=" + (stamp || 0)); }
function showShot(V, i) {
    V.sel = i;
    V.img.src = stripURL(V.node, i, V.meta.stamp);
    V.img.style.visibility = "visible"; V.empty.style.display = "none"; V.play.style.display = "inline-flex";
    [...V.strip.children].forEach((c, k) => c.classList.toggle("is-sel", k === i));
    V.full?.show(i);
}
function togglePlay(V) {
    if (V.timer) { clearInterval(V.timer); V.timer = null; V.play.innerHTML = ICON.play; V.play.classList.remove("is-on"); V.full?.paintPlay(); return; }
    const fps = clamp(V.meta?.play_fps || 8, 1, 12);
    V.play.innerHTML = ICON.pause; V.play.classList.add("is-on");
    V.timer = setInterval(() => showShot(V, (V.sel + 1) % V.meta.n), 1000 / fps);
    V.full?.paintPlay();
}
// повний екран відео-ноди: оверлей поверх графа (Esc - назад). Зліва кадр і стрічка, справа ті самі повзунки анімації, що на ноді:
// контроли пишуть у ті самі нативні віджети, тож крутити можна, не виходячи, а «Запустити» ставить прогін у чергу Comfy.
function openVideoFull(V) {
    if (V.full) return;
    const node = V.node, specs = V.specs || {};
    const savedSize = node.size ? [node.size[0], node.size[1]] : null;
    const savedUI = snapUI(node);
    const ov = el("div");
    ov.style.cssText = "position:fixed;inset:0;z-index:100000;background:#0e0e10;display:flex;flex-direction:column;color:#ccc;font:13px system-ui";
    const bar = el("div");
    bar.style.cssText = "display:flex;align-items:center;gap:12px;padding:6px 12px;background:#18181b";
    const play = el("button", "svk-ibtn", ICON.play); play.type = "button"; play.title = T("play the strip");
    const cnt = el("span", "svk-num");
    const stat = el("span", "svk-num"); stat.style.cssText = "margin-left:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0";
    const runBtn = el("button", "svk-btn primary", ICON.play + " " + T("Run")); runBtn.type = "button";
    runBtn.style.cssText = "display:inline-flex;align-items:center;gap:6px;margin-left:auto";
    runBtn.title = T("queue a run with the current settings (a run already in progress is not changed)");
    const back = el("button", null, T("✕ back to graph (Esc)"));
    back.style.cssText = "padding:5px 14px;border:1px solid #444;border-radius:6px;background:#26262b;color:#eee;cursor:pointer";
    const staleEl = el("span", "svk-num"); staleEl.style.cssText = "color:var(--svk-accent);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0";
    bar.append(el("span", null, T("Depth Look · video node #") + node.id), play, cnt, stat, staleEl, runBtn, back);
    const body = el("div"); body.style.cssText = "flex:1;min-height:0;display:flex";
    const left = el("div"); left.style.cssText = "flex:1;min-width:0;display:flex;flex-direction:column";
    const stage = el("div"); stage.style.cssText = "flex:1;min-height:0;position:relative";
    const img = el("img"); img.alt = ""; img.draggable = false;
    img.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block";
    const none = el("div", null, T("No frames yet — press Run, the preview strip appears when it finishes"));
    none.style.cssText = "position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:#888";
    stage.append(img, none);
    const strip = el("div", "svk-strip"); strip.style.cssText += ";padding:8px 12px;justify-content:center";
    left.append(stage, strip);
    // права панель: ті самі збірники, що й на ноді; зміна тут одразу видна й на ноді (V.sync), і навпаки після виходу
    const panel = el("div", "svk-card svk-scroll");
    panel.style.cssText = "flex:none;width:360px;height:100%;max-height:none;border-left:1px solid #2a2a2f;padding:10px 10px 12px;background:#151518";
    keepWheel(panel);
    const onChange = () => { V.sync(); M.refresh(); top.paintFps(); videoDirty(V); };
    const top = buildTop(node, specs, onChange);
    const M = buildMotion(node, specs, onChange);
    panel.append(top.bar, ...M.els);
    body.append(left, panel);
    ov.append(bar, body);
    document.body.appendChild(ov);
    top.paintFps();
    // рядок стану ноди (етап / кадр k з N) дзеркалимо в шапку, не створюючи другий stageLine (він перехопив би події в ноди)
    const st = V.status.querySelector(".st"), tm = V.status.querySelector(".tm");
    const paintStat = () => { stat.textContent = V.status.style.display === "none" ? "" : ((st?.textContent || "") + " " + (tm?.textContent || "")).trim(); };
    const mo = new MutationObserver(paintStat);
    mo.observe(V.status, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ["style"] });
    paintStat();
    const esc = e => {
        const tag = (e.target?.tagName || "").toUpperCase();
        const typing = tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
        if (e.key === "Escape") { e.stopPropagation(); close(); }
        else if (typing) return;                          // стрілки/пробіл у полях панелі - не гортання кадрів
        else if (!V.meta?.n) return;
        else if (e.key === "ArrowRight") showShot(V, (V.sel + 1) % V.meta.n);
        else if (e.key === "ArrowLeft") showShot(V, (V.sel - 1 + V.meta.n) % V.meta.n);
        else if (e.key === " ") { e.preventDefault(); togglePlay(V); }
    };
    function close() {
        window.removeEventListener("keydown", esc, true);
        mo.disconnect();
        ov.remove();
        V.full = null;
        V.sync();                                        // значення, змінені в оверлеї, - на картку ноди
        if (savedSize && node.size && (node.size[0] !== savedSize[0] || node.size[1] !== savedSize[1])) node.setSize(savedSize);
        restoreUI(node, savedUI);
        setTimeout(() => restoreUI(node, savedUI), 250);
        markChanged();
    }
    function fillStrip() {                               // (пере)будувати плитки: після нового прогону їх кількість інша
        strip.textContent = "";
        const m = V.meta;
        none.style.display = m?.n ? "none" : "flex"; img.style.display = m?.n ? "" : "none";
        play.style.visibility = m?.n ? "" : "hidden";
        if (!m?.n) { cnt.textContent = ""; return; }
        for (let i = 0; i < m.n; i++) {
            const b = el("button", "svk-shot"); b.type = "button"; b.style.width = "88px";
            b.innerHTML = '<div class="th"><img alt="" draggable="false"></div><div class="nm"></div>';
            const th = b.querySelector(".th"); th.style.cssText = "width:88px;height:88px";
            b.querySelector("img").src = stripURL(node, i, m.stamp);
            b.querySelector(".nm").textContent = "#" + (m.frames?.[i] ?? i);
            b.onclick = () => showShot(V, i);
            strip.append(b);
        }
    }
    V.full = {
        close,
        show(i) {
            if (!V.meta?.n) return;
            img.src = stripURL(node, i, V.meta.stamp); cnt.textContent = (i + 1) + " / " + V.meta.n;
            [...strip.children].forEach((c, k) => c.classList.toggle("is-sel", k === i));
        },
        reload() { fillStrip(); this.show(Math.min(V.sel, Math.max(0, (V.meta?.n || 1) - 1))); this.paintPlay(); },   // прийшла нова стрічка
        paintPlay() { play.innerHTML = V.timer ? ICON.pause : ICON.play; play.classList.toggle("is-on", !!V.timer); },
        paintStale(t) { staleEl.textContent = t; staleEl.title = t; },
    };
    play.onclick = () => togglePlay(V);
    runBtn.onclick = () => { markChanged(); runVideo(node); };
    back.onclick = close;
    window.addEventListener("keydown", esc, true);
    V.full.reload();
    paintStale(V);
}
function applyStrip(node, meta) {
    const V = node.__svkv; if (!V || !meta || !meta.n) return;
    if (V.timer) togglePlay(V);
    V.meta = meta;
    V.capR.textContent = meta.n + " / " + meta.total;
    V.capR.title = T("showing {n} of {total} frames · every {every}th · play at {fps} fps").replace("{n}", meta.n).replace("{total}", meta.total).replace("{every}", meta.every).replace("{fps}", Math.round(meta.play_fps * 100) / 100);
    V.strip.textContent = "";
    for (let i = 0; i < meta.n; i++) {
        const b = el("button", "svk-shot"); b.type = "button";
        b.innerHTML = '<div class="th"><img alt="" draggable="false"></div><div class="nm"></div>';
        b.querySelector("img").src = stripURL(node, i, meta.stamp);
        b.querySelector(".nm").textContent = "#" + (meta.frames?.[i] ?? i);
        b.onclick = () => showShot(V, i);
        V.strip.append(b);
    }
    showShot(V, 0);
    V.full?.reload();                                  // повний екран лишається відкритим і підхоплює новий результат
    // кадри від нашого прогону - версії, з якою він пішов; від чужого (кнопка Queue Comfy, перезавантаження) - поточних віджетів
    V.shownVer = V.waiting ? V.sentVer : (V.ver || 0);
    V.waiting = false;
    paintStale(V);
    tryAuto(V);                                        // поки рахувалось, параметри знову міняли
}
async function loadStripMeta(node) {          // після перезавантаження сторінки: стрічка з диска
    try {
        const r = await fetch(api.apiURL("/depth_look/strip/" + encodeURIComponent(node.id) + "/meta"));
        if (r.ok) applyStrip(node, await r.json());
    } catch (_) {}
}

// ---------------------------------------------------------------- реєстрація
app.registerExtension({
    name: "StepVisionKit.DepthLookSimple",
    setup() {
        api.addEventListener("executed", ev => {
            const id = ev.detail?.node, out = ev.detail?.output || {};
            const node = app.graph?.getNodeById?.(Number(id)) ?? app.graph?._nodes_by_id?.[id];
            if (!node) return;
            if (out.depth_look_simple && node.__svk) loadInfo(node.__svk);
            if (out.depth_look_strip && node.__svkv) applyStrip(node, out.depth_look_strip[0]);
        });
        // превʼю готове раніше за фінал: бекенд шле стрічку одразу, не чекаючи завершення ноди
        api.addEventListener("svk.strip", ev => {
            const id = ev.detail?.node;
            const node = app.graph?.getNodeById?.(Number(id)) ?? app.graph?._nodes_by_id?.[id];
            if (node?.__svkv && ev.detail?.strip) applyStrip(node, ev.detail.strip);
        });
        // черга спорожніла: прогін без нової стрічки (нода з кешу) - кадри й так відповідають тим параметрам; далі - відкладене авто-превʼю
        api.addEventListener("status", ev => {
            const left = ev.detail?.exec_info?.queue_remaining;
            if (typeof left !== "number") return;
            QUEUE_LEFT = left;
            for (const n of videoNodes()) {
                const V = n.__svkv;
                if (!left && V.waiting) { V.waiting = false; V.shownVer = V.sentVer; }
                paintStale(V); if (!left) tryAuto(V);
            }
        });
        // збій/скасування: кадри лишаються старі, але сам не перезапускаємо, доки параметр не змінять (інакше коло помилок)
        const stop = () => { for (const n of videoNodes()) { const V = n.__svkv; if (V.waiting) { V.waiting = false; V.autoBlock = V.ver || 0; paintStale(V); } } };
        api.addEventListener("execution_error", stop);
        api.addEventListener("execution_interrupted", stop);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name;
        if (name !== SIMPLE && name !== VIDEO && name !== "DepthLook") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            this.color = NODE_COLOR; this.bgcolor = NODE_BG;            // один вигляд для трьох нод
            if (name === SIMPLE) setupSimple(this);
            if (name === VIDEO) setupVideo(this, { ...(nodeData.input?.optional || {}), ...(nodeData.input?.required || {}) });
            return r;
        };
        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const r = onAdded?.apply(this, arguments);
            if (name === SIMPLE) startSimple(this);
            if (name === VIDEO && this.__svkv && !this.__svkv.started) { this.__svkv.started = true; loadStripMeta(this); }
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            migrateLegacyValues(this);
            this.color = NODE_COLOR; this.bgcolor = NODE_BG;
            // верхній вихід - превʼю (26.10.2026); старі графи зберегли назву «image»
            const o0 = (name === VIDEO || name === "DepthLook") ? this.outputs?.[0] : null;
            if (o0 && o0.name === "image") { o0.name = "preview"; o0.label = "preview"; }
            if (name === SIMPLE && this.__svk) {
                for (const w of this.widgets || []) if (w.name !== "svk_simple") hideWidget(w);
                syncFromWidgets(this.__svk);
                if (this.__svk.info) schedulePreview(this.__svk);
                startSimple(this);
            }
            if (name === VIDEO) {
                sanitizeWidgets(this, nodeData.input);
                setW(this, "render_token", "");                // токен кнопки не переживає збереження графа
            }
            if (name === VIDEO && this.__svkv) {
                for (const w of this.widgets || []) if (w.name !== "svk_strip") hideWidget(w);
                this.__svkv.sync();
                syncFinalOut(this);
                if (!this.__svkv.started) { this.__svkv.started = true; loadStripMeta(this); }
            }
            return r;
        };
        const onConn = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConn?.apply(this, arguments);
            if (name === VIDEO) { syncFinalOut(this); this.__svkv?.refresh?.(); }
            return r;
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this.__svk?.blobURL) URL.revokeObjectURL(this.__svk.blobURL);
            if (this.__svk) this.__svk.thumbToken++;
            if (this.__svkv?.timer) clearInterval(this.__svkv.timer);
            this.__svkv?.full?.close();
            return onRemoved?.apply(this, arguments);
        };
    },
});
