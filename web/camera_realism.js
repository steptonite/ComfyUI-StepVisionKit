// Camera Realism: картка ноди SVKCameraRealism. На видноті: профіль камери (плитки), сила, seed; усе решта - під «ще», по групах.
// Нативні віджети лишаються (їхні значення йдуть у workflow і в Run), але сховані: картка лише читає й пише їх.
// Вибір профілю заповнює повзунки його значеннями; рух повзунка перемикає профіль на Custom (з тими самими значеннями).
// Правила: CONVENTIONS_UI.md - жодних абзаців на ноді, пояснення тільки в title.
import { app } from "../../scripts/app.js";
import { el, W, control, moreSummary, keepWheel, fitToNode, scrollHint, hideWidget, markChanged } from "./svk_ui.js";
import { T } from "./svk_i18n.js";

const NODE = "SVKCameraRealism";
const CUSTOM = "Custom";
const CSS_URL = new URL("./svk_ui.css", import.meta.url).href;
const NODE_COLOR = "#2a2a2f", NODE_BG = "#151518";      // = --svk-line / --svk-card
const MIN_H = 300;

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
        const on = name === cur;
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
    setW(S.node, "profile", name);
    if (name !== CUSTOM) fillFromProfile(S, name);
    syncAll(S); markChanged();
}
function toCustom(S) {
    if (W(S.node, "profile")?.value !== CUSTOM) { setW(S.node, "profile", CUSTOM); paintChips(S); }
}

function build(node, specs) {
    const S = { node, ctl: [], chips: new Map(), profiles: specs.profile?.[1]?.svk_profiles || null };
    const root = el("div", "svk-card svk-scroll");
    S.root = root;
    keepWheel(root);

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

    const strength = control(node, "strength", specs.strength, T("Strength"), { fmt: v => Math.round(v * 100) + "%" });
    const seed = control(node, "seed", specs.seed, T("Seed"), { number: true });
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

    root.append(cap, chips, strength.row, seed.row, more);
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
    node.setSize([340, MIN_H + 110]);
    fitToNode(node, S.root);
    node.color = NODE_COLOR; node.bgcolor = NODE_BG;
    syncAll(S);
}

app.registerExtension({
    name: "StepVisionKit.CameraRealism",
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
