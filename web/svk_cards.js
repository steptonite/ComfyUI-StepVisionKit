// Однакова картка для нод без власного UI (Detect, HUD, IO, Depth Curve, потік глибини): вигляд і логіка як у Depth Look, правила - CONVENTIONS_UI.md.
// Нативні віджети сховані, картка читає й пише їх через control(). Головні (MAIN) - на видноті, усі інші входи-віджети
// самі йдуть під «ще», тож новий вхід у Python не губиться. Імена й значення віджетів не міняються: старі workflow відкриваються як були.
import { app } from "../../scripts/app.js";
import { el, W, control, morePanel, keepWheel, fitToNode, scrollHint, hideWidget } from "./svk_ui.js";
import { api } from "../../scripts/api.js";
import { T, LANG, DICT } from "./svk_i18n.js";
import { yoloPreview, PREVIEW_H } from "./yolo_preview.js";
import { videoSection } from "./depth_grade_video.js";
import { frameSection } from "./depth_grade_live.js";

const CSS_URL = new URL("./svk_ui.css", import.meta.url).href;
const NODE_COLOR = "#2a2a2f", NODE_BG = "#151518";      // = --svk-line / --svk-card
const START_W = 340, MIN_W = 300, ROW_H = 40, CARD_PAD = 24;

(function injectCss() {
    if (document.getElementById("svk-ui-css")) return;
    const l = document.createElement("link");
    l.id = "svk-ui-css"; l.rel = "stylesheet"; l.href = CSS_URL;
    document.head.appendChild(l);
})();

// ключ ноди -> головні віджети в порядку картки: вибір -> 3–4 повзунки; решта - під «ще»
const MAIN = {
    LexYOLODetect: ["model_name", "conf", "class_filter", "select", "max_boxes"],
    LexBoxesToBBOX: ["mode", "index", "pad"],
    LexHUDBoxes: ["style", "color", "thickness", "show_label"],
    LexHUDSprite: ["mode", "margin_pct", "label", "tint", "tint_strength"],
    LexDepthCurve: ["source", "mode", "near_bias", "gamma", "clarity"],
    VKDetect: ["model_name", "confidence", "classes", "max_objects", "track"],
    VKSelectBox: ["mode", "index", "class_filter", "pad"],
    VKBoxesToBBOX: ["format_in", "on_missing"],
    VKDrawBoxes: ["style", "color", "thickness", "fill_opacity", "show_class"],
    VKCutByMask: ["background", "color", "feather", "grow"],
    VKOverlayOnBoxes: ["fit", "anchor", "scale", "opacity"],
    SVKLoadRAW: ["file", "white_balance", "exposure", "highlights", "output_curve"],
    SVKLoadVideoPro: ["file", "start_frame", "frame_count", "output_curve"],
    SVKLoadImageSequence: ["folder", "pattern", "start_frame", "frame_count"],
    SVKSaveVideoHDR: ["filename_prefix", "format", "fps", "transfer"],
    SVKVideoDepthStream: ["file", "frame_count", "resolution", "batch"],
    SVKDepthGrade: ["depth_is", "look", "foreground_pop", "midtones", "deflicker"],
    SVKRawDepthSave: null,           // null - усі віджети на видноті (їх ≤3)
    SVKDepthReceiver: null,
};
// превʼю над керуваннями: кадр із боксами останнього Run; вужче MIN_W кнопки наближення й рядок «на виходи» не влазять (як Camera Realism, 0f10619)
const PREVIEW = { LexYOLODetect: yoloPreview, VKDetect: yoloPreview };
// розгортка під головними керуваннями: відео-превʼю Depth Grade (окрема нода View · відео лишається для старих workflow)
const EXTRA = { SVKDepthGrade: [frameSection, videoSection] };
// віджети, які картка веде сама (вибір кліком у превʼю): сховані, у «ще» не йдуть, але в workflow зберігаються
const OWN = { LexYOLODetect: ["picked"] };
// лише колір ноди: власний UI або превʼю картинок під віджетами, картку поверх них не ставимо; Live - стара, лишається як була
const COLOR_ONLY = ["SVKSaveImageHDR", "SVKSaveDepth16", "SVKRawDepthLoad", "SVKDepthGradeLive", "SVKDepthGradeView", "SVKDepthGradeVideoView"];
const COLOR_FIELDS = new Set(["color", "tint", "key_color"]);
// англійські підписи, де імʼя віджета не читається; решта - імʼя без «_»
const EN = { conf: "Confidence", imgsz: "Image size", model_name: "Model", class_filter: "Classes", format_in: "Box format",
             on_miss: "If no box", on_missing: "If no box", pad: "Padding", every_nth: "Every Nth frame", hevc_crf: "HEVC quality",
             max_nits: "Max nits", exr_compression: "EXR compression", wb_red: "WB red", wb_blue: "WB blue", ram_budget_gb: "RAM budget, GB",
             sky_to_far: "Sky = far", pre_scale: "Pre-scale", margin_pct: "Margin, %", margin_px: "Margin, px", label_conf: "Label confidence",
             emit_box_mask: "Box mask output", nine_slice: "Nine-slice corners", tint_strength: "Tint strength", make_preview: "Preview output" };

function label(key, name) {
    if (LANG === "uk") { const l = DICT.nodes?.[key]?.inputs?.[name]?.label; if (l) return l; }
    return EN[name] || (name.charAt(0).toUpperCase() + name.slice(1)).replace(/_/g, " ");
}
// onChange: власний callback віджета, як при зміні нативного (Depth Grade на ньому перемальовує View)
function opts(node, name, spec) {
    const vals = Array.isArray(spec?.[0]) ? spec[0] : null;
    const o = { onChange: v => { W(node, name)?.callback?.(v, app.canvas, node); node.__svkPreview?.draw(); } };
    if (COLOR_FIELDS.has(name)) o.color = true;
    else if (vals && vals.length <= 3 && vals.every(v => String(v).length <= 12)) o.seg = Object.fromEntries(vals.map(v => [v, T(String(v))]));
    return o;
}

function build(node, key, specs) {
    const names = (node.widgets || []).map(w => w.name).filter(n => specs[n] && !OWN[key]?.includes(n));
    const main = (MAIN[key] || names).filter(n => names.includes(n));
    const root = el("div", "svk-card svk-scroll");
    keepWheel(root);
    const prev = PREVIEW[key]?.(node, key);
    if (prev) { node.__svkPreview = prev; root.append(prev.root); }
    const ctl = [];
    for (const n of main) {
        const lb = label(key, n), o = opts(node, n, specs[n]);
        const c = control(node, n, specs[n], lb, o);
        if (o.seg) {                       // сегмент без підпису не читається - підпис ліворуч, як у полів
            const r = el("div", "svk-field"); r.title = c.row.title;
            r.append(el("span", "t", lb), c.row); root.append(r);
        } else root.append(c.row);
        ctl.push(c);
    }
    const parts = (EXTRA[key] || []).map(f => f(node));
    parts.forEach(p => root.append(p.root));
    const extra = parts.length ? {
        restore: () => parts.forEach(p => p.restore()),
        extraH: () => parts.reduce((s, p) => s + p.extraH(), 0),
        dispose: () => parts.forEach(p => p.dispose()),
    } : null;
    root.extra = extra;
    const rest = names.filter(n => !main.includes(n));
    const more = rest.length ? morePanel(node, specs, rest.map(n => { const { seg, ...o } = opts(node, n, specs[n]); return [n, label(key, n), o]; })) : null;
    if (more) root.append(more);
    scrollHint(root);
    root.sync = () => { ctl.forEach(c => c.sync()); more?.sync(); prev?.draw(); };
    const minH = (prev ? PREVIEW_H + ROW_H + 20 : 0) + main.length * ROW_H + (more ? ROW_H : 0) + parts.length * ROW_H + CARD_PAD;
    return { root, minH: () => minH + (extra?.extraH() || 0) };
}

function hideRepresented(node, specs) {
    for (const w of node.widgets || []) if (w.name !== "svk_card" && specs[w.name]) hideWidget(w);
}

app.registerExtension({
    name: "StepVisionKit.Cards",
    setup() {
        // Nodes 2.0 не викликає onExecuted, тож превʼю бере вивід із події виконання
        api.addEventListener("executed", (ev) => {
            const p = ev.detail?.output?.svk_yolo?.[0];
            if (!p) return;
            const id = String(ev.detail.node ?? "").split(":").pop();
            app.graph?.getNodeById?.(/^\d+$/.test(id) ? +id : id)?.__svkPreview?.set(p);
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const key = nodeData?.name;
        const card = Object.hasOwn(MAIN, key);
        if (!card && !COLOR_ONLY.includes(key)) return;
        const specs = { ...(nodeData.input?.optional || {}), ...(nodeData.input?.required || {}) };
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            this.color = NODE_COLOR; this.bgcolor = NODE_BG;
            if (!card) return r;
            const { root, minH } = build(this, key, specs);
            this.__svkCard = root;
            this.addDOMWidget("svk_card", "svk_card", root, { serialize: false, hideOnZoom: false, getMinHeight: minH });
            hideRepresented(this, specs);
            const changed = this.onWidgetChanged;     // віджет змінили не з картки (скидання, пресет у Depth Grade View) -> оновити картку
            this.onWidgetChanged = function () { const r = changed?.apply(this, arguments); root.sync(); return r; };
            if (PREVIEW[key]) {
                const computeSize = this.computeSize;
                this.computeSize = function () { const r = computeSize.apply(this, arguments); r[0] = Math.max(r[0], MIN_W); return r; };
            }
            const sz = this.computeSize();
            this.setSize([Math.max(START_W, sz[0]), sz[1]]);
            fitToNode(this, root);
            const removed = this.onRemoved;
            this.onRemoved = function () { root.extra?.dispose(); return removed?.apply(this, arguments); };
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            this.color = NODE_COLOR; this.bgcolor = NODE_BG;
            if (this.__svkCard) { hideRepresented(this, specs); this.__svkCard.sync(); this.__svkCard.extra?.restore(); }  // значення з workflow -> картка
            if (this.__svkCard && PREVIEW[key] && this.size[0] < MIN_W) this.setSize([MIN_W, this.size[1]]);     // воркфлоу зі стягнутою вужче нодою
            return r;
        };
    },
});
