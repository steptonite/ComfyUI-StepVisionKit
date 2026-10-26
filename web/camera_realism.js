// Camera Realism: картка ноди SVKCameraRealism. Згори - живе превʼю (анімація шматка відео, A/B з оригіналом) і крупняк 1:1/2:1/4:1
// («до | після», веди по кадру, клік - закріпити). Далі профіль камери (плитки) + власні пресети, сила, seed, «Власне зерно», під «ще» - групи.
// Нативні віджети лишаються (їхні значення йдуть у workflow і в Run), але сховані: картка лише читає й пише їх.
// Вибір профілю заповнює повзунки його значеннями; рух повзунка перемикає профіль на Custom (з тими самими значеннями).
// Правила: CONVENTIONS_UI.md - жодних абзаців на ноді, пояснення тільки в title.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { el, W, control, moreSummary, keepWheel, fitToNode, scrollHint, hideWidget, markChanged, ICON } from "./svk_ui.js";
import { T } from "./svk_i18n.js";

const NODE = "SVKCameraRealism";
const CUSTOM = "Custom";
const CSS_URL = new URL("./svk_ui.css", import.meta.url).href;
const NODE_COLOR = "#2a2a2f", NODE_BG = "#151518";      // = --svk-line / --svk-card
const MIN_H = 600;
const PV_SIZE = 512, FPS = 24, DEBOUNCE = 420, LOUPE_THROTTLE = 120, ZOOMS = [1, 2, 4];

(function injectCss() {
    if (document.getElementById("svk-ui-css")) return;
    const l = document.createElement("link");
    l.id = "svk-ui-css"; l.rel = "stylesheet"; l.href = CSS_URL;
    document.head.appendChild(l);
})();

// групи «ще»: [назва групи, [[ключ віджета, підпис], ...]]; ключі = realism/params.py SPEC (гейт ui_widgets_gate звіряє)
const MORE = [
    [T("Sensor"), [["noise_luma", T("Luma noise")], ["noise_chroma", T("Colour noise")], ["noise_size", T("Grain size")],
                   ["iso", "ISO"], ["fixed_pattern", T("Fixed pattern")]]],
    [T("Bayer"), [["bayer_amount", T("Bayer mosaic")]]],
    [T("Camera processing"), [["isp_denoise", T("Noise reduction")], ["isp_sharpen", T("Sharpening")], ["isp_sharpen_radius", T("Halo width")],
                              ["isp_blocks", T("Block compression")]]],
    [T("Optics"), [["vignette", T("Vignette")], ["distortion", T("Distortion")], ["chroma_ab", T("Colour fringes")],
                   ["corner_softness", T("Soft corners")], ["veil", T("Veiling glare")]]],
    [T("Depth"), [["depth_haze", T("Aerial haze")], ["depth_detail", T("Near detail")]]],
    [T("Tone & colour"), [["highlight_rolloff", T("Highlight rolloff")], ["highlight_clip", T("Highlight clip")], ["blacks", T("Blacks")],
                          ["warmth", T("Warmth")], ["saturation", T("Saturation")], ["skin_hue", T("Skin hue")]]],
    [T("Film"), [["film_grain", T("Film grain")], ["halation", T("Halation")], ["gate_weave", T("Gate weave")], ["dust", T("Dust & scratches")]]],
    [T("Phone"), [["smart_hdr", "Smart HDR"], ["deep_fusion", "Deep Fusion"], ["shadow_smear", T("Shadow smear")],
                  ["portrait_bokeh", T("Portrait bokeh")], ["ghost_flare", T("Ghost flare")]]],
    [T("Old video"), [["line_skip", T("Line skipping")], ["chroma_blotch", T("Colour blotches")], ["banding", T("Banding")]]],
    [T("Video"), [["shake", T("Hand shake")], ["shake_speed", T("Shake tempo")], ["flicker", T("Exposure flicker")],
                  ["rolling_shutter", T("Rolling shutter")], ["motion_blur", T("Motion blur")], ["frame_offset", T("First frame number")]]],
];
// власне зерно: поверх будь-якого профілю, рух повзунка НЕ перемикає на Custom (Python бере їх завжди)
const OWN = [["og_amount", T("Amount")], ["og_size", T("Grain size")], ["og_clump", T("Clumping")], ["og_softness", T("Softness")],
             ["og_colour", T("Colour")], ["og_blue", T("Blue layer size")], ["og_shadows", T("In shadows")], ["og_mids", T("In mid-tones")],
             ["og_highlights", T("In highlights")], ["og_animate", T("Changes per frame")], ["og_replace", T("Replace camera noise")]];
const KEYS = [...MORE.flatMap(g => g[1].map(x => x[0])), ...OWN.map(x => x[0])].filter(k => k !== "frame_offset");
const PROFILE_LABEL = { "ARRI Alexa 35": "ARRI Alexa 35", "Cinema digital": T("Cinema digital"), "Mirrorless": T("Mirrorless"),
                        "iPhone": "iPhone", "DSLR 2012": "DSLR 2012", "16mm film": T("16mm film"), "Custom": T("Custom") };

function setW(node, name, v) {            // значення напряму, без callback
    const w = W(node, name);
    if (w && w.value !== v) { w.value = v; return true; }
    return false;
}

function paintChips(S) {
    const cur = W(S.node, "profile")?.value;
    for (const [name, b] of S.chips) {
        const on = S.userActive ? name === S.userActive : name === cur;
        b.classList.toggle("is-off", !on);
        b.style.setProperty("--fill", on ? "100%" : "0%");
    }
}
function syncAll(S) { S.ctl.forEach(c => c.sync()); paintChips(S); }

// значення профілю -> повзунки (лише відображення: Python бере профіль, а не повзунки, поки профіль не Custom)
function fillFromProfile(S, name) {
    const vals = S.profiles?.[name];
    if (!vals) return;
    for (const [k, v] of Object.entries(vals)) setW(S.node, k, v);
}
function pickProfile(S, name) {
    S.userActive = null;
    setW(S.node, "profile", name);
    if (name !== CUSTOM) fillFromProfile(S, name);
    syncAll(S); markChanged(); schedule(S);
}
function toCustom(S) {
    S.userActive = null;
    if (W(S.node, "profile")?.value !== CUSTOM) setW(S.node, "profile", CUSTOM);
    paintChips(S);
    schedule(S);
}

// ---------------------------------------------------------------- власні пресети (realism/presets.py, user-тека Comfy)
async function postJSON(path, body) {
    const r = await api.fetchApi("/svk_realism" + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    let j = {}; try { j = await r.json(); } catch (_) {}
    return [r.status, j];
}
function snapshot(S) {
    const v = {};
    for (const k of KEYS) { const w = W(S.node, k); if (w && typeof w.value === "number") v[k] = w.value; }
    return v;
}
async function loadPresets(S) {
    try { const r = await api.fetchApi("/svk_realism/presets"); if (r.ok) S.user = (await r.json()).presets || {}; } catch (_) {}
    paintUser(S);
}
function paintUser(S) {
    for (const n of [...S.chips.keys()]) if (S.userChips.has(n)) { S.chips.get(n).remove(); S.chips.delete(n); }
    S.userChips.clear();
    for (const name of Object.keys(S.user || {})) {
        const b = el("button", "svk-chip is-user");
        b.type = "button"; b.style.cursor = "pointer";
        b.append(el("span", "lb", name));
        const x = el("span", "x", "×");
        x.title = T("delete (click twice)");
        b.title = name;
        b.onclick = ev => {
            if (ev.target === x) {
                if (!b.classList.contains("is-arm")) { b.classList.add("is-arm"); x.textContent = "?"; setTimeout(() => { b.classList.remove("is-arm"); x.textContent = "×"; }, 2500); return; }
                delPreset(S, name); return;
            }
            pickUser(S, name);
        };
        b.append(x);
        S.chipsBox.insertBefore(b, S.saveChip);
        S.chips.set(name, b); S.userChips.add(name);
    }
    paintChips(S);
}
function pickUser(S, name) {
    const vals = S.user?.[name];
    if (!vals) return;
    setW(S.node, "profile", CUSTOM);
    for (const [k, v] of Object.entries(vals)) setW(S.node, k, v);
    S.userActive = name;
    syncAll(S); markChanged(); schedule(S);
}
async function savePreset(S) {
    const name = S.nameIn.value.trim();
    if (!name) { S.nameIn.focus(); return; }
    const [code, j] = await postJSON("/presets/save", { name, values: { ...snapshot(S), strength: W(S.node, "strength")?.value } });
    if (code !== 200) { S.nameIn.title = j.err || String(code); S.nameIn.classList.add("is-bad"); return; }
    S.nameIn.classList.remove("is-bad"); S.nameIn.value = ""; S.saveRow.style.display = "none";
    S.user = j.presets || {}; S.userActive = j.name; paintUser(S);
}
async function delPreset(S, name) {
    const [code, j] = await postJSON("/presets/delete", { name });
    if (code === 200) { S.user = j.presets || {}; if (S.userActive === name) S.userActive = null; paintUser(S); }
}

// ---------------------------------------------------------------- превʼю: рендер на сервері, анімація шматка, крупняк
const jpg = b => "data:image/jpeg;base64," + b, png = b => "data:image/png;base64," + b;
function body(S) {
    return { profile: W(S.node, "profile")?.value, strength: W(S.node, "strength")?.value, seed: W(S.node, "seed")?.value, values: snapshot(S) };
}
function schedule(S, ms = DEBOUNCE) {
    clearTimeout(S.tm);
    S.tm = setTimeout(() => render(S), ms);
}
function showEmpty(S, on, msg) {
    S.empty.style.display = on ? "" : "none";
    if (msg) S.empty.querySelector("p").textContent = msg;
    S.close.style.display = on ? "none" : "";
    S.bar.style.display = on ? "none" : "";
}
async function render(S) {
    if (S.busy) { S.again = true; return; }
    S.busy = true; S.again = false; S.view.classList.add("is-busy");
    try {
        const [code, j] = await postJSON("/" + encodeURIComponent(S.uid()) + "/render", { ...body(S), size: PV_SIZE, orig: !S.orig });
        if (code === 409) { S.frames = []; S.orig = null; showEmpty(S, true, T("Run the node once — then the preview follows every change")); return; }
        if (code !== 200) { showEmpty(S, true, j.err || T("error {s}").replace("{s}", code)); return; }
        if (S.stamp !== j.stamp) { S.stamp = j.stamp; if (!j.orig) { S.orig = null; S.again = true; } S.lo = null; }
        if (j.orig) S.orig = j.orig.map(jpg);
        S.frames = j.frames.map(jpg);
        S.fi %= S.frames.length;
        S.view.dataset.w = j.w; S.view.dataset.h = j.h;
        S.full = { w: j.fw, h: j.fh };
        showEmpty(S, false);
        paintFrame(S); setPlay(S, S.playing);
        if (S.pos) loupe(S);
    } catch (e) {
        showEmpty(S, true, String(e));
    } finally {
        S.busy = false; S.view.classList.remove("is-busy");
        if (S.again) render(S);
    }
}
function paintFrame(S) {
    const src = (S.ab && S.orig ? S.orig : S.frames)[S.fi];
    if (src && S.img.getAttribute("src") !== src) S.img.src = src;
    S.tag.textContent = S.ab ? T("original") : T("camera");
    S.fc.textContent = S.frames.length > 1 ? (S.fi + 1) + " / " + S.frames.length : "";
    if (S.lo) {
        const k = S.lo.proc.length > 1 ? S.ti % S.lo.proc.length : 0;
        S.cOrig.src = S.lo.orig[k]; S.cProc.src = S.lo.proc[k];
    }
}
function setPlay(S, on) {
    S.playing = on;
    S.play.innerHTML = on ? ICON.pause : ICON.play;
    const anim = S.frames.length > 1 || (S.lo?.proc.length > 1);
    S.play.style.display = anim ? "" : "none";
    clearInterval(S.timer); S.timer = null;
    if (on && anim) S.timer = setInterval(() => {
        if (!S.root.isConnected) { clearInterval(S.timer); return; }
        if (S.frames.length > 1) S.fi = (S.fi + 1) % S.frames.length;
        S.ti++;
        paintFrame(S);
    }, 1000 / FPS);
}
// де в блоці превʼю лежить сам кадр (object-fit: contain)
function frameRect(S) {
    const r = S.view.getBoundingClientRect(), w = +S.view.dataset.w || 1, h = +S.view.dataset.h || 1;
    const k = Math.min(r.width / w, r.height / h);
    return { left: r.left + (r.width - w * k) / 2, top: r.top + (r.height - h * k) / 2, w: w * k, h: h * k, ox: (r.width - w * k) / 2, oy: (r.height - h * k) / 2 };
}
function boxPx(S) { return Math.max(96, Math.min(260, Math.floor((S.close.clientWidth - 6) / 2) || 150)); }
function posFromEvent(S, ev) {
    const f = frameRect(S);
    S.pos = { x: Math.min(1, Math.max(0, (ev.clientX - f.left) / f.w)), y: Math.min(1, Math.max(0, (ev.clientY - f.top) / f.h)) };
}
function paintMark(S) {
    if (!S.pos || !S.lo) { S.mark.style.display = "none"; return; }
    const f = frameRect(S), full = S.lo.fw;
    const k = f.w / full.w;
    Object.assign(S.mark.style, { display: "block", left: (f.ox + S.lo.x0 * k) + "px", top: (f.oy + S.lo.y0 * k) + "px",
                                  width: (S.lo.w * k) + "px", height: (S.lo.h * k) + "px" });
    S.mark.classList.toggle("is-pin", !!S.pinned);
}
async function loupe(S) {
    if (!S.pos || !S.frames.length) return;
    const now = Date.now();
    if (S.lbusy || now - (S.lt || 0) < LOUPE_THROTTLE) { S.lagain = true; clearTimeout(S.ltm); S.ltm = setTimeout(() => loupe(S), LOUPE_THROTTLE); return; }
    S.lbusy = true; S.lagain = false; S.lt = now;
    const box = boxPx(S), crop = Math.round(box / S.zoom);
    const anim = S.frames.length > 1;
    try {
        const [code, j] = await postJSON("/" + encodeURIComponent(S.uid()) + "/loupe",
            { ...body(S), x: S.pos.x, y: S.pos.y, crop, frame: anim && !S.playing ? S.fi : 0, count: anim ? 8 : 1 });
        if (code !== 200) return;
        S.lo = { orig: j.orig.map(png), proc: j.proc.map(png), x0: j.x0, y0: j.y0, w: j.w, h: j.h, fw: S.full || { w: 1, h: 1 } };
        for (const im of [S.cOrig, S.cProc]) { im.style.width = (j.w * S.zoom) + "px"; im.style.height = (j.h * S.zoom) + "px"; }
        S.ti = 0; paintFrame(S); paintMark(S); setPlay(S, S.playing);
    } finally {
        S.lbusy = false;
    }
}
function buildPreview(S) {
    const view = el("div", "svk-rl-view");
    view.title = T("Close-up: move over the frame, click to pin the spot");
    const img = el("img"); img.alt = "";
    const mark = el("div", "svk-rl-mark"), tag = el("div", "svk-rl-tag");
    const empty = el("div", "svk-empty still", "<p></p>");
    view.append(img, mark, tag, empty);
    view.onmousemove = ev => { if (S.pinned || !S.frames.length) return; posFromEvent(S, ev); loupe(S); };
    view.onclick = ev => { if (!S.frames.length) return; posFromEvent(S, ev); S.pinned = !S.pinned; loupe(S); };

    const bar = el("div", "svk-rl-bar");
    const play = el("button", "svk-btn ghost"); play.type = "button"; play.title = T("play / pause");
    play.onclick = () => { setPlay(S, !S.playing); if (S.pos) loupe(S); };
    const fc = el("span", "fc");
    const ab = el("button", "svk-btn ghost", "A/B"); ab.type = "button"; ab.title = T("show the original (A/B)");
    ab.onclick = () => { S.ab = !S.ab; ab.classList.toggle("is-on", S.ab); paintFrame(S); };
    const zoom = el("div", "svk-rl-zoom"); zoom.title = T("close-up zoom: 1:1 = one frame pixel per screen pixel");
    for (const z of ZOOMS) {
        const b = el("button", z === S.zoom ? "is-on" : null, z + ":1"); b.type = "button";
        b.onclick = () => { S.zoom = z; [...zoom.children].forEach(c => c.classList.toggle("is-on", c === b)); if (S.pos) loupe(S); };
        zoom.append(b);
    }
    bar.append(play, fc, el("span", "sp"), ab, zoom);

    const close = el("div", "svk-rl-close");
    close.title = T("left: original, right: camera; 1:1 crop of the full-size frame");
    const bo = el("div", "bx"), bp = el("div", "bx");
    const cOrig = el("img"), cProc = el("img");
    cOrig.alt = cProc.alt = "";
    bo.append(cOrig, el("div", "svk-rl-tag", T("original"))); bp.append(cProc, el("div", "svk-rl-tag", T("camera")));
    close.append(bo, bp);
    Object.assign(S, { view, img, mark, tag, empty, bar, play, fc, close, cOrig, cProc });
    return [view, bar, close];
}

function build(node, specs) {
    const S = { node, ctl: [], chips: new Map(), userChips: new Set(), user: {}, userActive: null,
                profiles: specs.profile?.[1]?.svk_profiles || null,
                frames: [], orig: null, fi: 0, ti: 0, playing: true, ab: false, zoom: 1, pos: null, pinned: false, lo: null };
    S.uid = () => S.execId || String(node.id);
    const root = el("div", "svk-card svk-scroll");
    S.root = root;
    keepWheel(root);
    const pv = buildPreview(S);

    const cap = el("div", "svk-cap", "<b>" + T("Camera") + "</b>");
    const chips = el("div", "svk-chips");
    chips.title = T("pick a camera look; Custom uses the sliders under «more»");
    for (const name of (specs.profile?.[0] || [])) {
        const b = el("button", "svk-chip");
        b.type = "button"; b.style.cursor = "pointer";
        b.append(el("span", "lb", PROFILE_LABEL[name] || name));
        b.title = PROFILE_LABEL[name] || name;
        b.onclick = () => pickProfile(S, name);
        chips.append(b); S.chips.set(name, b);
    }
    S.chipsBox = chips;
    const save = el("button", "svk-chip is-absent");
    save.type = "button"; save.title = T("save the current settings as your own preset");
    save.append(el("span", "lb", "+ " + T("Save mine")));
    save.onclick = () => { const on = S.saveRow.style.display === "none"; S.saveRow.style.display = on ? "" : "none"; if (on) S.nameIn.focus(); };
    chips.append(save); S.saveChip = save;
    const saveRow = el("div", "svk-rl-save"); saveRow.style.display = "none";
    const nameIn = el("input"); nameIn.type = "text"; nameIn.maxLength = 48; nameIn.placeholder = T("preset name");
    nameIn.onkeydown = ev => { ev.stopPropagation(); if (ev.key === "Enter") savePreset(S); if (ev.key === "Escape") saveRow.style.display = "none"; };
    const ok = el("button", "svk-btn primary", T("Save")); ok.type = "button"; ok.onclick = () => savePreset(S);
    saveRow.append(nameIn, ok);
    Object.assign(S, { saveRow, nameIn });

    const strength = control(node, "strength", specs.strength, T("Strength"), { fmt: v => Math.round(v * 100) + "%", onChange: () => schedule(S) });
    const seed = control(node, "seed", specs.seed, T("Seed"), { number: true, onChange: () => schedule(S) });

    const own = el("details", "svk-more");
    const ownBody = el("div", "body body-col"), ownBox = el("div", "svk-sub");
    for (const [name, label] of OWN) {
        if (!specs[name] || !W(node, name)) continue;
        const c = control(node, name, specs[name], label, { onChange: () => schedule(S) });
        ownBox.append(c.row); S.ctl.push(c);
    }
    ownBody.append(ownBox);
    const os = el("summary", null, "<span>" + T("Own grain") + "</span>" + '<span class="chev">' + ICON.chev + "</span>");
    os.title = T("your own grain on top of any profile: size, clumping, softness, colour, response by tones, per-frame change");
    own.append(os, ownBody);
    S.ctl.push(strength, seed);

    const more = el("details", "svk-more");
    const body = el("div", "body body-col");
    let count = 0;
    for (const [group, items] of MORE) {
        const hd = el("div", "svk-cap", "<b>" + group + "</b>");
        const box = el("div", "svk-sub");
        for (const [name, label] of items) {
            if (!specs[name] || !W(node, name)) continue;
            const c = control(node, name, specs[name], label, { onChange: () => toCustom(S) });
            box.append(c.row); S.ctl.push(c); count++;
        }
        body.append(hd, box);
    }
    more.append(moreSummary(count), body);

    root.append(...pv, cap, chips, saveRow, strength.row, seed.row, own, more);
    S.hint = scrollHint(root);
    return S;
}

function setup(node, specs) {
    if (node.__svkRealism) return;
    const S = build(node, specs);
    node.__svkRealism = S;
    const dw = node.addDOMWidget("svk_realism", "svk_realism", S.root, { serialize: false, hideOnZoom: false, getMinHeight: () => MIN_H });
    for (const w of [...node.widgets]) if (w !== dw && w.name !== "svk_realism") hideWidget(w);
    const ctl = W(node, "control_after_generate");         // seed тут детермінований: після Run не міняється
    if (ctl) ctl.value = "fixed";
    node.setSize([360, MIN_H + 140]);
    fitToNode(node, S.root);
    node.color = NODE_COLOR; node.bgcolor = NODE_BG;
    syncAll(S);
    showEmpty(S, true, T("Run the node once — then the preview follows every change"));
    loadPresets(S);
    schedule(S, 600);              // кадри могли лишитись у памʼяті сервера від попереднього Run
}

app.registerExtension({
    name: "StepVisionKit.CameraRealism",
    setup() {
        // після Run: нові кадри в памʼяті сервера -> перемалювати превʼю (оригінали теж)
        api.addEventListener("executed", ev => {
            const out = ev.detail?.output?.svk_realism;
            if (!out) return;
            const id = ev.detail.display_node ?? ev.detail.node;
            const node = app.graph?.getNodeById?.(Number(id)) ?? app.graph?._nodes_by_id?.[id];
            const S = node?.__svkRealism;
            if (!S) return;
            S.execId = String(ev.detail.node);
            S.orig = null; S.lo = null;
            schedule(S, 0);
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;
        const specs = { ...(nodeData.input?.optional || {}), ...(nodeData.input?.required || {}) };
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            setup(this, specs);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const S = this.__svkRealism;
            if (S) {
                for (const w of this.widgets || []) if (w.name !== "svk_realism") hideWidget(w);
                const p = W(this, "profile")?.value;
                if (p && p !== CUSTOM) fillFromProfile(S, p);       // збережені значення повзунків могли відстати від профілю
                syncAll(S);
            }
            this.color = NODE_COLOR; this.bgcolor = NODE_BG;
            return r;
        };
    },
});
