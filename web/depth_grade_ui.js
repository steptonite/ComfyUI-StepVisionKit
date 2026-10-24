// Спільне для обох View Depth Grade: тема, класи кнопок і чіпів, іконки, пресети кривої,
// реєстр відкритих панелей і пошук джерела за входом depth. Тонова математика — у depth_grade_math.js.
import { app } from "../../scripts/app.js";
import { T } from "./svk_i18n.js";

export const NODE = "SVKDepthGrade";
export const LIVE = "SVKDepthGradeLive";
export const VIEW = "SVKDepthGradeView";
export const VIDEO = "SVKDepthGradeVideoView";
export const SOURCES = [NODE, LIVE];

export const TONE_KEYS = ["depth_is", "look", "foreground_pop", "blacks", "whites", "midtones",
    "foreground_ceiling", "ceiling_from", "hero_plane", "hero_pop",
    "trim_far", "trim_near", "flip", "levels_scope"];

// усі віджети, що йдуть у маршрут відео-прев'ю (імена як у Depth Grade)
export const PARAM_KEYS = ["depth_is", "look", "foreground_pop", "blacks", "whites", "midtones",
    "foreground_ceiling", "ceiling_from", "hero_plane", "hero_pop", "clarity", "texture",
    "smooth_banding", "banding_edge", "trim_far", "trim_near", "flip", "levels_scope",
    "deflicker", "motion_guard"];

export const C = {
    bg: "#18181b", panel: "#1f1f23", panelHi: "#28282e", line: "#2e2e33", text: "#d4d4d8", dim: "#8a8a93",
    hist: "rgba(161,161,170,0.22)", accent: "#6fbf9f", accentDim: "rgba(111,191,159,0.35)", err: "#e0857c",
};
export const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";
export const GAP = 6;       // проміжок між блоками, px
export const CTRL_H = 22;   // висота всіх контролів, px

// ── стилі: один раз на сторінку, щоб обидві View виглядали однаково
const CSS = `
.svk-row{display:flex;align-items:center;gap:8px;height:${CTRL_H}px;white-space:nowrap;overflow-x:auto;scrollbar-width:none}
.svk-row::-webkit-scrollbar,.svk-chips::-webkit-scrollbar{display:none}
.svk-seg{display:inline-flex;gap:2px;flex:none}
.svk-end{display:inline-flex;gap:4px;margin-left:auto;flex:none}
.svk-lbl{font:500 11px/1 system-ui,sans-serif;color:${C.dim};flex:none}
.svk-btn{box-sizing:border-box;height:${CTRL_H}px;padding:0 8px;display:inline-flex;align-items:center;justify-content:center;gap:4px;flex:none;border-radius:6px;border:1px solid ${C.line};background:${C.panel};color:${C.text};font:500 11px/1 system-ui,sans-serif;white-space:nowrap;cursor:pointer;user-select:none;transition:transform .12s cubic-bezier(.16,1,.3,1),background-color .15s,border-color .15s,color .15s}
.svk-btn:hover{background:${C.panelHi}}
.svk-btn:active{transform:translateY(1px)}
.svk-btn:disabled{opacity:.45;cursor:default;transform:none;background:${C.panel}}
.svk-btn.svk-on{border-color:${C.accent};color:${C.accent}}
.svk-ico{width:${CTRL_H}px;padding:0}
.svk-ico svg{width:13px;height:13px;display:block}
.svk-sel{box-sizing:border-box;height:${CTRL_H}px;padding:0 4px 0 6px;border-radius:6px;border:1px solid ${C.line};background:${C.panel};color:${C.text};font:500 11px/1 system-ui,sans-serif;flex:none;cursor:pointer}
.svk-chips{display:flex;gap:4px;overflow-x:auto;scrollbar-width:none;white-space:nowrap}
.svk-num{font:11px ${MONO};color:${C.dim};white-space:nowrap}
.svk-busy{height:2px;border-radius:1px;background:${C.line};overflow:hidden}
.svk-busy i{display:block;width:30%;height:100%;background:${C.accent};animation:svk-slide 1.1s cubic-bezier(.4,0,.2,1) infinite}
@keyframes svk-slide{from{transform:translateX(-100%)}to{transform:translateX(340%)}}
@media (prefers-reduced-motion:reduce){.svk-busy i{animation-duration:3s}}
`;

export function ensureStyles() {
    if (document.getElementById("svk-dg-ui")) return;
    const s = document.createElement("style");
    s.id = "svk-dg-ui";
    s.textContent = CSS;
    document.head.append(s);
}

// ── DOM-хелпери
export function h(tag, css = "", text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text != null) e.textContent = text;
    return e;
}

export function place(el, x, y, w, hh) {
    el.style.left = x + "px";
    el.style.top = y + "px";
    el.style.width = w + "px";
    el.style.height = hh + "px";
}

// міняє текст лише тоді, коли він справді інший (поллінг викликає часто)
export function setText(el, t) {
    if (el.textContent !== t) el.textContent = t;
}

const ICONS = {
    play: '<path d="M7 5.2v9.6L15 10z"/>',
    pause: '<path d="M7 5v10M13 5v10"/>',
    split: '<rect x="3.5" y="4.5" width="13" height="11" rx="1.5"/><path d="M10 4.5v11"/>',
    curve: '<path d="M3.5 16.5C8 16.5 8.5 3.5 16.5 3.5"/>',
    reset: '<path d="M4.2 10a5.8 5.8 0 1 0 1.7-4.1"/><path d="M4.2 3.2v3h3"/>',
};

function svgOf(name) {
    return `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] ?? ""}</svg>`;
}

// іконка-кнопка 22×22 з підказкою
export function icon(name, title, onclick = null) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "svk-btn svk-ico";
    b.title = title;
    b.setAttribute("aria-label", title);
    b.innerHTML = svgOf(name);
    if (onclick) b.onclick = onclick;
    return b;
}

export function setIcon(b, name) {
    b.innerHTML = svgOf(name);
}

// текстова кнопка-чіп
export function btn(text, title = "", onclick = null) {
    const b = h("button", "", text);
    b.type = "button";
    b.className = "svk-btn";
    if (title) b.title = title;
    if (onclick) b.onclick = onclick;
    return b;
}

// ── пресети кривої: клік ставить значення у віджети Depth Grade; ключів, яких нема в пресеті, не чіпаємо
export const PRESETS = [
    { id: "neutral", label: T("Neutral"), hint: T("Original look: base volume, no level shift, no hero, no ceiling."),
        values: { foreground_pop: 0.45, blacks: 0, whites: 0, midtones: 0, hero_pop: 0, hero_plane: 0.5,
            foreground_ceiling: 1.0, ceiling_from: 0.75 } },
    { id: "volume", label: T("Volume"), hint: T("Stronger volume: near areas separate more sharply, midtones slightly brighter."),
        values: { foreground_pop: 0.65, midtones: 0.15 } },
    { id: "soft", label: T("Soft"), hint: T("Weaker volume with no hero S-curve: soft transition between planes."),
        values: { foreground_pop: 0.25, midtones: 0.1, hero_pop: 0 } },
    { id: "hero", label: T("Hero"), hint: T("Stronger S-curve around the hero; the hero position stays put."),
        values: { hero_pop: 6 } },
    { id: "press", label: T("Press foreground"), hint: T("Lower foreground ceiling: near areas brighten less, not up to white."),
        values: { foreground_ceiling: 0.8, ceiling_from: 0.65 } },
    { id: "deep", label: T("Deep background"), hint: T("Darker background: black point raised, midtones lowered."),
        values: { blacks: 0.06, midtones: -0.25 } },
];

export function applyPreset(node, preset) {
    for (const [name, value] of Object.entries(preset.values)) {
        const w = widgetOf(node, name);
        if (w && w.value !== value) resetWidget(node, w, value);
    }
    scheduleViews();
}

export function presetActive(node, preset) {
    return Object.entries(preset.values).every(([name, value]) => {
        const w = widgetOf(node, name);
        return !!w && Math.abs(Number(w.value) - value) < 1e-4;
    });
}

// рядок чіпів; getNode() → нода Depth Grade (джерело) або null
export function presetRow(getNode) {
    const el = h("div");
    el.className = "svk-chips";
    const chips = PRESETS.map((p) => {
        const b = btn(p.label, p.hint, () => {
            const A = getNode();
            if (A) applyPreset(A, p);
            refresh();
        });
        el.append(b);
        return { p, b };
    });
    function refresh() {
        const A = getNode();
        for (const { p, b } of chips) {
            b.disabled = !A;
            b.classList.toggle("svk-on", !!A && presetActive(A, p));
        }
    }
    return { el, refresh };
}

// ── реєстр відкритих панелей: після зміни тонів перемалювати всі
const PANELS = new Set();
export function registerPanel(p) { PANELS.add(p); }
export function unregisterPanel(p) { PANELS.delete(p); }
export function scheduleViews() { for (const p of PANELS) p.schedule(); }

// ── ноди й віджети
export function typeOf(node) { return node?.comfyClass ?? node?.type ?? ""; }
export function isSource(node) { return SOURCES.includes(typeOf(node)); }

export function nodeById(id) {
    const s = String(id ?? "");
    const last = s.includes(":") ? s.split(":").pop() : s;   // підграфи: "uuid:innerId"
    const key = /^\d+$/.test(last) ? +last : last;
    return app.graph?.getNodeById?.(key) ?? null;
}

export function widgetOf(node, name) { return node?.widgets?.find((w) => w.name === name); }

// від входу depth іде (reroute…) до ноди Depth Grade; повертає її або null
export function findSource(node) {
    const g = node?.graph ?? app.graph;
    let cur = node;
    for (let step = 0; step < 9; step++) {
        const inp = cur?.inputs?.[0];
        if (inp?.link == null || !g) return null;
        const L = g.links?.get?.(inp.link) ?? g.links?.[inp.link];
        if (!L) return null;
        const origin = g.getNodeById?.(L.origin_id);
        if (!origin) return null;
        if (isSource(origin)) return origin;
        if (/reroute/i.test(typeOf(origin))) { cur = origin; continue; }
        return null;
    }
    return null;
}

export function toneSig(A) { return JSON.stringify(TONE_KEYS.map((k) => widgetOf(A, k)?.value)); }

// значення всіх віджетів для маршруту відео-прев'ю
export function paramValues(A) {
    const o = {};
    for (const k of PARAM_KEYS) {
        const w = widgetOf(A, k);
        if (w) o[k] = w.value;
    }
    return o;
}

// сирий payload превью: з події виконання, з onExecuted або (після перезавантаження) з історії виводу
export function rawOf(A) {
    if (A.__svkRaw) return A.__svkRaw;
    try {
        const r = app.nodeOutputs?.[String(A.id)]?.svk_depth_preview?.[0];
        if (r) A.__svkRaw = r;
        return r ?? null;
    } catch (_) { return null; }
}

// стандартні значення з опису ноди — для «Скинути»
let DEFAULTS = {};
export function getDefaults() { return DEFAULTS; }
export function setDefaults(d) { DEFAULTS = d; }

export function readDefaults(nodeData) {
    const d = {};
    for (const [name, spec] of Object.entries(nodeData?.input?.required || {})) {
        const def = spec?.[1]?.default ?? (Array.isArray(spec?.[0]) ? spec[0][0] : undefined);
        if (def !== undefined) d[name] = def;
    }
    return d;
}

export function resetWidget(node, w, value) {
    const old = w.value;
    w.value = value;
    w.callback?.(value, app.canvas, node);
    node.onWidgetChanged?.(w.name, value, old, w);
    app.graph?.setDirtyCanvas(true, true);
}

// запис повзунка з межами й округленням, як у Python-ноді; перемальовує граф
export function setWidgetValue(node, name, value) {
    const w = widgetOf(node, name);
    if (!node || !w) return;
    const o = w.options || {};
    if (typeof value === "number") {
        if (o.min != null) value = Math.max(o.min, value);
        if (o.max != null) value = Math.min(o.max, value);
        const step = o.round || o.step2 || 0.01;
        value = Math.round(value / step) * step;
        value = +value.toFixed(3);
    }
    if (w.value === value) return;
    const old = w.value;
    w.value = value;
    w.callback?.(value, app.canvas, node);
    node.onWidgetChanged?.(name, value, old, w);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas(true, true);
}

export function resetAllOf(node) {
    for (const w of node.widgets || []) {
        if (w.name in DEFAULTS && w.value !== DEFAULTS[w.name]) resetWidget(node, w, DEFAULTS[w.name]);
    }
    scheduleViews();
}
