// Depth Look · простий: картка замість «полотна налаштувань».
//   превʼю результату зверху -> стрічка плиток-луків (кожен пресет на ЦЬОМУ кадрі) -> чіпи ефектів -> «сила луку».
// Усі нативні віджети лишаються (їхні значення зберігаються у workflow і йдуть у Run), але сховані: картка їх лише читає/пише.
// Також тут: стрічка кадрів-превʼю відео-ноди і єдиний колір карток трьох нод. Стилі й токени - svk_ui.css.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ICON, presetLabel, control, morePanel, keepWheel, onWheel } from "./svk_ui.js";

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
    frame.title = "утримуй кадр — оригінал";
    S.empty = el("div", "svk-empty");
    frame.append(S.imgOrig, S.imgLook, S.busy, S.tag, S.empty);
    // утримуй кадр -> показати оригінал
    const hold = on => { S.imgLook.style.opacity = on ? 0 : String(mixValue(node)); };
    frame.addEventListener("pointerdown", e => { if (e.button === 0 && S.info?.node_frame) { hold(true); frame.setPointerCapture?.(e.pointerId); } });
    const release = () => hold(false);
    frame.addEventListener("pointerup", release); frame.addEventListener("pointercancel", release);

    // сила луку
    const mix = el("div", "svk-mix");
    mix.append(el("span", "svk-cap", "<b>Сила луку</b>"));
    mix.title = "наскільки лук накладено на оригінал; подвійний клік — 100%";
    S.mixRange = el("input", "svk-range"); S.mixRange.type = "range"; S.mixRange.min = 0; S.mixRange.max = 1; S.mixRange.step = 0.01;
    S.mixNum = el("span", "svk-num", "100%");
    mix.append(S.mixRange, S.mixNum);
    S.mixRange.addEventListener("input", () => {
        const v = Number(S.mixRange.value);
        setW(node, "look_mix", v); markChanged(); paintMix(S);
    });
    S.mixRange.addEventListener("dblclick", () => { S.mixRange.value = 1; S.mixRange.dispatchEvent(new Event("input")); });

    // плитки
    const tilesCap = el("div", "svk-cap", "<b>Лук</b>");
    S.tiles = el("div", "svk-tiles");
    S.tiles.title = "клік — обрати лук";

    // чіпи
    const chipsCap = el("div", "svk-cap");
    chipsCap.append(el("b", null, "Ефекти"));
    S.reset = el("button", "svk-ibtn", ICON.reset); S.reset.type = "button"; S.reset.title = "скинути ефекти до луку"; S.reset.style.display = "none";
    S.reset.onclick = () => { resetChips(S); selectPreset(S, currentPreset(node), true); };
    chipsCap.append(S.reset);
    S.chips = el("div", "svk-chips");
    S.chips.title = "тап — вкл/викл · тягни вліво-вправо або крути колесо — сила";

    // ⋯ ще
    S.more = el("details", "svk-more");
    S.more.append(el("summary", null, ICON.more + "<span>ще</span>"));
    const body = el("div", "body");
    S.more.append(body);
    const row = (label, ctl) => { body.append(el("span", null, label), ctl); };
    const selSize = el("select");
    for (const v of (W(node, "output_size")?.options?.values || [])) selSize.append(new Option(v, v));
    selSize.onchange = () => { setW(node, "output_size", selSize.value); markChanged(); };
    row("Фінальний розмір", selSize);
    const numMax = el("input"); numMax.type = "number"; numMax.min = 0; numMax.max = 16384; numMax.step = 8;
    numMax.onchange = () => { setW(node, "max_side", Math.max(0, Math.round(Number(numMax.value) || 0))); markChanged(); };
    row("Робоча роздільність", numMax);
    numMax.title = "0 — рахувати одразу у фінальному розмірі. Менше значення рятує памʼять на довгих кліпах.";
    const cbClean = el("input"); cbClean.type = "checkbox";
    cbClean.onchange = () => { setW(node, "clean_depth", cbClean.checked); markChanged(); };
    row("Чистити глибину", cbClean);
    const cbNear = el("input"); cbNear.type = "checkbox";
    cbNear.onchange = () => { setW(node, "depth_light_is_near", cbNear.checked); markChanged(); };
    row("Світле — близько", cbNear);
    S.more_ctl = { selSize, numMax, cbClean, cbNear };

    root.append(frame, mix, tilesCap, S.tiles, chipsCap, S.chips, S.more);
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
        b.title = g.name + " — " + g.hint + "\nтап — вкл/викл · перетягни вліво-вправо або крути колесо — сила";
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
        b.querySelector(".val").textContent = c.active ? "×" + c.mul.toFixed(c.mul >= 1 ? 1 : 2).replace(/\.?0+$/, "") : (c.present ? "вимк" : "+");
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
        t.innerHTML = '<div class="th"><img alt="" draggable="false"><span class="dot" title="ефекти змінено вручну"></span></div><div class="nm"></div>';
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
        if (r.status === 409) { S.info.node_frame = false; showEmpty(S); return; }
        if (!r.ok) { let m = ""; try { m = (await r.json()).err; } catch (_) {} showEmpty(S, m || ("помилка " + r.status)); return; }
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
    const b = el("button", "svk-btn primary", ICON.play + " Запустити");
    b.style.cssText = "display:inline-flex;align-items:center;gap:6px";
    b.title = "Підключи image і depth та запусти граф один раз — далі кадр і плитки луків оновлюються без Run";
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
    } catch (e) { showEmpty(S, "сервер не відповів: " + (e.message || e)); return; }
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
    node.color = NODE_COLOR; node.bgcolor = NODE_BG;
    syncFromWidgets(S);
}
function startSimple(node) {
    const S = node.__svk;
    if (S && !S.started && node.id != null && node.id >= 0) { S.started = true; loadInfo(S); }
}

// ---------------------------------------------------------------- стрічка відео-ноди
// на ноді: превʼю кадрів, fps, превʼю/фінал і 4 керування рухом; решта ~15 параметрів анімації - під «ще».
const VIDEO_MAIN = [
    ["grad_amount", "Рух кольору"], ["grad_speed", "Швидкість руху"], ["glow_pulse", "Пульс світла"], ["grain_animate", "Живе зерно"],
];
const VIDEO_MORE_FIRST = [["start_time", "Старт, с"], ["preview_every", "Кожен N-й кадр у превʼю"],
                          ["output_size", "Фінальний розмір"], ["max_side", "Робоча роздільність"]];
function setupVideo(node, specs) {
    if (node.__svkv) return;
    const V = { node, meta: null, sel: 0, timer: null };
    node.__svkv = V;
    const root = el("div", "svk-card svk-scroll");
    root.style.setProperty("--svk-frame-h", "150px");
    keepWheel(root);
    const cap = el("div", "svk-cap");
    V.capL = el("b", null, "Кадри");
    V.capR = el("span", "svk-num", "");
    cap.append(V.capL, V.capR);
    const view = el("div", "svk-frame svk-viewer");
    V.img = el("img"); V.img.alt = ""; V.img.draggable = false; V.img.style.visibility = "hidden";
    V.empty = el("div", "svk-empty still");
    const run = el("button", "svk-btn primary", ICON.play + " Превʼю");
    run.style.cssText = "display:inline-flex;align-items:center;gap:6px";
    run.title = "Стрічка кадрів зʼявиться після превʼю-прогону: режим «превʼю», запусти граф";
    run.onclick = () => { setW(node, "quality", (specs.quality?.[0] || [])[0] ?? W(node, "quality")?.value); markChanged(); try { app.queuePrompt(0, 1); } catch (e) { console.warn("[DepthLook]", e); } };
    V.empty.append(run);
    V.play = el("button", "svk-ibtn", ICON.play); V.play.type = "button"; V.play.title = "грати стрічку";
    V.play.style.cssText = "position:absolute;right:6px;bottom:6px;display:none";
    view.append(V.img, V.empty, V.play);
    V.strip = el("div", "svk-strip");
    // керування
    const ctl = [];
    const bar = el("div", "svk-bar");
    if (specs.fps) { const c = control(node, "fps", specs.fps, "fps", { number: true }); c.row.style.flex = "1"; bar.append(c.row); ctl.push(c); }
    if (specs.quality) {
        const vals = specs.quality[0] || [];
        const c = control(node, "quality", specs.quality, "", { seg: { [vals[0]]: "превʼю", [vals[1]]: "фінал" }, tip: specs.quality[1]?.tooltip });
        bar.append(c.row); ctl.push(c);
    }
    const main = el("div", "body-col"); main.style.cssText = "display:flex;flex-direction:column;gap:9px";
    for (const [name, label] of VIDEO_MAIN) {
        if (!specs[name]) continue;
        const c = control(node, name, specs[name], label);
        main.append(c.row); ctl.push(c);
    }
    const shown = new Set(["image", "depth", "grade", "fps", "quality", ...VIDEO_MAIN.map(x => x[0])]);
    const rest = [...VIDEO_MORE_FIRST];
    for (const name of Object.keys(specs)) {
        if (shown.has(name) || rest.some(x => x[0] === name)) continue;
        const tip = specs[name]?.[1]?.tooltip || name;
        rest.push([name, tip.includes(" · ") ? tip.split(" · ").slice(1).join(" · ").replace(/\s*\(.*\)$/, "") : name]);
    }
    V.more = morePanel(node, specs, rest);
    root.append(cap, view, V.strip, bar, main, V.more);
    V.sync = () => { ctl.forEach(c => c.sync()); V.more.sync(); };
    V.play.onclick = () => togglePlay(V);
    node.addDOMWidget("svk_strip", "svk_strip", root, { serialize: false, hideOnZoom: false, getMinHeight: () => 380 });
    for (const w of [...node.widgets]) if (w.name !== "svk_strip") hideWidget(w);
    node.color = NODE_COLOR; node.bgcolor = NODE_BG;
    node.setSize([360, 460]);
}
function stripURL(node, i, stamp) { return api.apiURL("/depth_look/strip/" + encodeURIComponent(node.id) + "/" + i + "?v=" + (stamp || 0)); }
function showShot(V, i) {
    V.sel = i;
    V.img.src = stripURL(V.node, i, V.meta.stamp);
    V.img.style.visibility = "visible"; V.empty.style.display = "none"; V.play.style.display = "inline-flex";
    [...V.strip.children].forEach((c, k) => c.classList.toggle("is-sel", k === i));
}
function togglePlay(V) {
    if (V.timer) { clearInterval(V.timer); V.timer = null; V.play.innerHTML = ICON.play; V.play.classList.remove("is-on"); return; }
    const fps = clamp(V.meta?.play_fps || 8, 1, 12);
    V.play.innerHTML = ICON.pause; V.play.classList.add("is-on");
    V.timer = setInterval(() => showShot(V, (V.sel + 1) % V.meta.n), 1000 / fps);
}
function applyStrip(node, meta) {
    const V = node.__svkv; if (!V || !meta || !meta.n) return;
    if (V.timer) togglePlay(V);
    V.meta = meta;
    V.capR.textContent = meta.n + " / " + meta.total;
    V.capR.title = "показано " + meta.n + " з " + meta.total + " кадрів · кожен " + meta.every + "-й · грати з " + (Math.round(meta.play_fps * 100) / 100) + " fps";
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
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name;
        if (name !== SIMPLE && name !== VIDEO && name !== "DepthLook") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            this.color = NODE_COLOR; this.bgcolor = NODE_BG;            // один вигляд для трьох нод
            if (name === SIMPLE) setupSimple(this);
            if (name === VIDEO) setupVideo(this, nodeData.input?.required || {});
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
            this.color = NODE_COLOR; this.bgcolor = NODE_BG;
            if (name === SIMPLE && this.__svk) {
                for (const w of this.widgets || []) if (w.name !== "svk_simple") hideWidget(w);
                syncFromWidgets(this.__svk);
                if (this.__svk.info) schedulePreview(this.__svk);
                startSimple(this);
            }
            if (name === VIDEO && this.__svkv) {
                for (const w of this.widgets || []) if (w.name !== "svk_strip") hideWidget(w);
                this.__svkv.sync();
                if (!this.__svkv.started) { this.__svkv.started = true; loadStripMeta(this); }
            }
            return r;
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this.__svk?.blobURL) URL.revokeObjectURL(this.__svk.blobURL);
            if (this.__svk) this.__svk.thumbToken++;
            if (this.__svkv?.timer) clearInterval(this.__svkv.timer);
            return onRemoved?.apply(this, arguments);
        };
    },
});
