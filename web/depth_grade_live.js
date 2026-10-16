import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as M from "./depth_grade_math.js";

// Depth Grade: дві ноди.
//  • SVKDepthGrade (нода A) — повні повзунки, «Скинути все», меню правої кнопки. Після Run
//    віддає мініатюру сирого депсу (svk_depth_preview) — її читає нода B.
//  • SVKDepthGradeView (нода B) — превью і інтерактив: під'єднується входом depth до виходу
//    depth ноди A; тягаєш повзунки A — тут миттєво видно зміну, без перегенерації.
//  • SVKDepthGradeLive — застаріла, поводиться як A.
// Тонова математика — в depth_grade_math.js; тут лише UI.
// Розкладка детермінована (без scrollHeight): висоти рахуємо з розмірів ноди.

const NODE = "SVKDepthGrade";
const LIVE = "SVKDepthGradeLive";
const VIEW = "SVKDepthGradeView";
const SOURCES = [NODE, LIVE];
const TONE_KEYS = ["depth_is", "look", "foreground_pop", "blacks", "whites", "midtones",
    "foreground_ceiling", "ceiling_from", "hero_plane", "hero_pop",
    "trim_far", "trim_near", "flip", "levels_scope"];

const C = {
    bg: "#18181b", panel: "#1f1f23", line: "#2e2e33", text: "#d4d4d8", dim: "#8a8a93",
    hist: "rgba(161,161,170,0.22)", accent: "#6fbf9f", accentDim: "rgba(111,191,159,0.35)",
};
const MONO = "ui-monospace, SFMono-Regular, Menlo, monospace";

const TOOLS = [
    { id: "hero", label: "Герой", hint: "клікни на головний об'єкт — його об'єм підсилиться" },
    { id: "dim", label: "Пригасити", hint: "клікни на те, що спереду заважає, — воно й усе ближче стане сірим" },
    { id: "black", label: "Чорне", hint: "клікни на фон — він і все далі піде в чорне" },
    { id: "white", label: "Біле", hint: "клікни на близьке — воно й усе ближче стане білим" },
];

const MAXH = 440;        // максимальна висота превью, px
const GAP = 6;           // проміжок між блоками, px
const SIDE_W = 210;      // ширина бічної колонки в портреті
const ROW_GAP = 8;       // проміжок між превью і бічною колонкою
const CURVE_H = 150;     // висота полотна кривих
const LEGEND_COL = 48;   // легенда під кривими (колонка)
const LEGEND_ROW = 124;  // легенда під кривими (бічна колонка)
const START_W = 460;     // стартова ширина ноди B

const VIEWS = new Set();
let DEFAULTS = {};       // стандартні значення тонових і решти повзунків ноди A

const BTN = `font:500 11px/1 system-ui,sans-serif;color:${C.text};background:${C.panel};
    border:1px solid ${C.line};border-radius:6px;padding:5px 7px;cursor:pointer;
    transition:transform .12s cubic-bezier(.16,1,.3,1),border-color .15s;user-select:none`;

function h(tag, css = "", text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text != null) e.textContent = text;
    return e;
}

function place(el, x, y, w, hh) {
    el.style.left = x + "px";
    el.style.top = y + "px";
    el.style.width = w + "px";
    el.style.height = hh + "px";
}

function scheduleViews() { for (const v of VIEWS) v.schedule(); }

function typeOf(node) { return node?.comfyClass ?? node?.type ?? ""; }
function isSource(node) { return SOURCES.includes(typeOf(node)); }

function nodeById(id) {
    const s = String(id ?? "");
    const last = s.includes(":") ? s.split(":").pop() : s;   // підграфи: "uuid:innerId"
    const key = /^\d+$/.test(last) ? +last : last;
    return app.graph?.getNodeById?.(key) ?? null;
}

function widgetOf(node, name) { return node?.widgets?.find((w) => w.name === name); }

function toneSig(A) { return JSON.stringify(TONE_KEYS.map((k) => widgetOf(A, k)?.value)); }

// сирий payload превью: з події виконання, з onExecuted або (після перезавантаження) з історії виводу
function rawOf(A) {
    if (A.__svkRaw) return A.__svkRaw;
    try {
        const r = app.nodeOutputs?.[String(A.id)]?.svk_depth_preview?.[0];
        if (r) A.__svkRaw = r;
        return r ?? null;
    } catch (_) { return null; }
}

// декодування ліниве, з кешем за ідентичністю payload (новий Run = новий об'єкт)
function prevOf(A) {
    const raw = rawOf(A);
    if (!raw?.keys) return null;
    if (A.__svkRawDec !== raw) {
        try {
            A.__svkPrev = M.decodePreview(raw);
        } catch (_) {
            A.__svkPrev = null;
        }
        A.__svkRawDec = raw;
    }
    return A.__svkPrev ?? null;
}

// стандартні значення з опису ноди — для «Скинути»
function readDefaults(nodeData) {
    const d = {};
    for (const [name, spec] of Object.entries(nodeData?.input?.required || {})) {
        const def = spec?.[1]?.default ?? (Array.isArray(spec?.[0]) ? spec[0][0] : undefined);
        if (def !== undefined) d[name] = def;
    }
    return d;
}

function resetWidget(node, w, value) {
    const old = w.value;
    w.value = value;
    w.callback?.(value, app.canvas, node);
    node.onWidgetChanged?.(w.name, value, old, w);
    app.graph?.setDirtyCanvas(true, true);
}

function resetAllOf(node) {
    for (const w of node.widgets || []) {
        if (w.name in DEFAULTS && w.value !== DEFAULTS[w.name]) resetWidget(node, w, DEFAULTS[w.name]);
    }
    scheduleViews();
}

class ViewPanel {
    constructor(node) {
        this.node = node;
        this.A = null;           // нода-джерело (SVKDepthGrade)
        this.raw = undefined;    // ідентичність payload, з якого зроблено prev
        this.W = -1;             // остання ширина ноди B
        this.sig = "";           // остання сигнатура тонових значень A
        this.prev = null;
        this.last = null;        // результат M.grade для піпеток і кривих
        this.hs = null;          // маркери на осі кривих
        this.tool = null;
        this.compare = false;
        this.drag = null;
        this.hover = null;
        this.ki = 0;
        this.raf = 0;
        this.showCurve = false;  // «Криві» за замовчуванням згорнуті
        this.total = 160;        // висота DOM-віджета, px

        this.root = h("div", `position:relative;display:block;width:100%;box-sizing:border-box;
            color:${C.text};font:12px system-ui,sans-serif`);
        // не тягати вузол замість маркера; колесо не чіпаємо — полотно має зумитись
        for (const ev of ["pointerdown", "pointerup", "click", "dblclick"])
            this.root.addEventListener(ev, (e) => e.stopPropagation());

        // ── кнопки
        this.bar = h("div", "display:flex;gap:4px;flex-wrap:wrap;align-items:center");
        this.toolBtns = {};
        for (const t of TOOLS) {
            const b = h("button", BTN, t.label);
            b.title = t.hint;
            b.onclick = () => this.setTool(this.tool === t.id ? null : t.id);
            b.onpointerdown = () => (b.style.transform = "scale(0.97)");
            b.onpointerup = b.onpointerleave = () => (b.style.transform = "");
            this.toolBtns[t.id] = b;
            this.bar.append(b);
        }
        this.cmp = h("button", BTN + ";margin-left:auto", "До / після");
        this.cmp.title = "тримай — покаже вхід без грейду";
        const on = () => { this.compare = true; this.schedule(); };
        const off = () => { this.compare = false; this.schedule(); };
        this.cmp.onpointerdown = on; this.cmp.onpointerup = off; this.cmp.onpointerleave = off;
        this.curveBtn = h("button", BTN, "Криві ▸");
        this.curveBtn.title = "гістограма, крива і маркери Ч / Б / Г / С";
        this.curveBtn.onclick = () => this.toggleCurve();
        this.resetBtn = h("button", BTN, "↺");
        this.resetBtn.title = "скинути всі повзунки Depth Grade до стандартних";
        this.resetBtn.onclick = () => { if (this.A) resetAllOf(this.A); };
        this.bar.append(this.cmp, this.curveBtn, this.resetBtn);

        // ── тіло: усі блоки абсолютно позиціоновані, висоту тіла рахує layout()
        this.body = h("div", "position:relative;margin-top:6px;width:100%;box-sizing:border-box");

        this.empty = h("div", `position:absolute;display:none;box-sizing:border-box;padding:0 14px;
            align-items:center;justify-content:center;text-align:center;color:${C.dim};line-height:1.5;
            border:1px dashed ${C.line};border-radius:8px`);

        this.stage = h("div", `position:absolute;display:none;overflow:hidden;border-radius:8px;
            background:${C.bg};box-shadow:0 0 0 1px ${C.line}`);
        this.view = h("canvas", "position:absolute;left:0;top:0;display:block;image-rendering:auto");
        this.readout = h("div", `position:absolute;left:6px;bottom:6px;padding:3px 6px;border-radius:5px;
            background:rgba(24,24,27,.82);font:11px ${MONO};color:${C.text};pointer-events:none;display:none`);
        this.stage.append(this.view, this.readout);
        this.view.addEventListener("pointermove", (e) => this.onHover(e));
        this.view.addEventListener("pointerleave", () => { this.hover = null; this.readout.style.display = "none"; this.schedule(); });
        this.view.addEventListener("click", (e) => this.onPick(e));

        this.scrub = h("div", "position:absolute;display:none;align-items:center;gap:8px");
        this.scrubIn = h("input", `flex:1;min-width:0;accent-color:${C.accent}`);
        this.scrubIn.type = "range"; this.scrubIn.min = 0; this.scrubIn.step = 1; this.scrubIn.value = 0;
        this.scrubLbl = h("span", `font:11px ${MONO};color:${C.dim};white-space:nowrap`);
        this.scrubIn.oninput = () => { this.ki = +this.scrubIn.value; this.schedule(); };
        this.scrub.append(this.scrubIn, this.scrubLbl);

        this.curve = h("canvas", `position:absolute;display:none;border-radius:8px;background:${C.bg};
            box-shadow:0 0 0 1px ${C.line};touch-action:none;cursor:default`);
        this.curve.addEventListener("pointerdown", (e) => this.onCurveDown(e));
        this.curve.addEventListener("pointermove", (e) => this.onCurveMove(e));
        this.curve.addEventListener("pointerup", (e) => this.onCurveUp(e));
        this.curve.addEventListener("pointercancel", (e) => this.onCurveUp(e));
        this.curve.addEventListener("dblclick", (e) => this.onCurveDbl(e));

        this.legend = h("div", `position:absolute;display:none;overflow:hidden;color:${C.dim};
            font-size:11px;line-height:1.4`);
        this.legend.innerHTML =
            "Тягни: Ч / Б — чорне й біле · <b style='color:" + C.accent + "'>Г</b> — герой (вгору — сильніше) · " +
            "С — звідки пригашувати, кружок справа — наскільки · порожнє місце вгору/вниз — середні тони. " +
            "Подвійний клік — скинути. Clarity, texture, смуги й deflicker — після Run.";

        this.body.append(this.empty, this.stage, this.scrub, this.curve, this.legend);
        this.root.append(this.bar, this.body);

        this.timer = setInterval(() => this.tick(), 120);
        VIEWS.add(this);
    }

    dispose() {
        clearInterval(this.timer);
        VIEWS.delete(this);
        if (this.raf) cancelAnimationFrame(this.raf);
        this.raf = 0;
    }

    // ── джерело: вхід depth → (reroute…) → нода A
    source() {
        const g = this.node.graph ?? app.graph;
        let cur = this.node;
        for (let step = 0; step < 9; step++) {
            const inp = cur.inputs?.[0];
            if (inp?.link == null || !g) return null;
            const L = g.links?.get?.(inp.link) ?? g.links?.[inp.link];
            if (!L) return null;
            const origin = g.getNodeById?.(L.origin_id);
            if (!origin) return null;
            const t = typeOf(origin);
            if (isSource(origin)) return origin;
            if (/reroute/i.test(t)) { cur = origin; continue; }
            return null;
        }
        return null;
    }

    // ── повзунки ноди A як джерело правди
    widget(name) { return widgetOf(this.A, name); }

    values() {
        const o = {};
        for (const k of TONE_KEYS) o[k] = this.widget(k)?.value;
        return o;
    }

    set(name, value) {
        const A = this.A;
        const w = this.widget(name);
        if (!A || !w) return;
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
        w.callback?.(value, app.canvas, A);
        A.onWidgetChanged?.(name, value, old, w);
        A.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas(true, true);
    }

    reset(names) {
        for (const n of names) if (n in DEFAULTS) this.set(n, DEFAULTS[n]);
        this.schedule();
    }

    setTool(id) {
        this.tool = id;
        for (const [k, b] of Object.entries(this.toolBtns)) {
            const act = k === id;
            b.style.borderColor = act ? C.accent : C.line;
            b.style.color = act ? C.accent : C.text;
        }
        this.view.style.cursor = id ? "crosshair" : "default";
    }

    toggleCurve() {
        this.showCurve = !this.showCurve;
        this.layout();
        this.schedule();
    }

    // ── стан → розкладка: виклик лише при зміні джерела, payload або ширини ноди
    tick() {
        const A = this.source();
        const raw = A ? rawOf(A) : null;
        const W = this.node.size?.[0] ?? 0;
        if (A !== this.A || raw !== this.raw || W !== this.W) {
            this.A = A;
            this.raw = raw;
            this.W = W;
            this.setPrev(A ? prevOf(A) : null);
            this.layout();
            this.sig = A ? toneSig(A) : "";
            this.schedule();
            return;
        }
        const sig = A ? toneSig(A) : "";
        if (sig !== this.sig) {
            this.sig = sig;
            this.schedule();
        }
    }

    setPrev(p) {
        this.prev = p;
        this.last = null;
        this.hs = null;
        if (!p) return;
        const nk = p.keys.length;
        this.ki = Math.min(this.ki, nk - 1);
        this.scrubIn.max = nk - 1;
        this.scrubIn.value = this.ki;
        this.view.width = p.w;
        this.view.height = p.h;
    }

    // ── розкладка: квадрат/альбом — колонка; портрет з кривими або кадрами — рядок
    layout() {
        const node = this.node;
        const W = Math.max(200, (node.size?.[0] || START_W) - 16);
        const barH = this.bar.offsetHeight || 28;
        const p = this.prev;
        const nk = p ? p.keys.length : 0;
        const ar = p ? p.w / p.h : 16 / 9;
        const curves = !!p && this.showCurve;
        const scrub = !!p && nk > 1;
        const row = !!p && ar < 0.95 && (this.showCurve || nk > 1);

        this.empty.textContent = this.A
            ? "Натисни Run один раз. Далі тягай повзунки Depth Grade — тут видно одразу, без перегенерації. Clarity, texture, смуги й deflicker видно лише після Run."
            : "Під'єднай вхід depth до виходу depth ноди Depth Grade";
        this.empty.style.display = p ? "none" : "flex";
        this.stage.style.display = p ? "block" : "none";
        this.scrub.style.display = scrub ? "flex" : "none";
        this.curve.style.display = curves ? "block" : "none";
        this.legend.style.display = curves ? "block" : "none";
        this.curveBtn.textContent = `Криві ${this.showCurve ? "▾" : "▸"}`;

        let bodyH;
        if (!p) {
            place(this.empty, 0, 0, W, 120);
            bodyH = 120;
        } else if (!row) {
            let imgW = W, imgH = W / ar;
            if (imgH > MAXH) { imgH = MAXH; imgW = imgH * ar; }
            place(this.stage, Math.round((W - imgW) / 2), 0, imgW, imgH);
            this.sizeView(imgW, imgH);
            let y = imgH + GAP;
            if (scrub) { place(this.scrub, 0, y, W, 24); y += 24 + GAP; }
            if (curves) {
                place(this.curve, 0, y, W, CURVE_H); y += CURVE_H + GAP;
                place(this.legend, 0, y, W, LEGEND_COL); y += LEGEND_COL + GAP;
            }
            bodyH = y - GAP;
        } else {
            const rowW = W - SIDE_W - ROW_GAP;
            let imgW = rowW, imgH = rowW / ar;
            if (imgH > MAXH) { imgH = MAXH; imgW = imgH * ar; }
            place(this.stage, Math.round((rowW - imgW) / 2), 0, imgW, imgH);
            this.sizeView(imgW, imgH);
            const x0 = W - SIDE_W;
            let y = 0;
            if (scrub) { place(this.scrub, x0, y, SIDE_W, 24); y += 24 + GAP; }
            if (curves) {
                place(this.curve, x0, y, SIDE_W, CURVE_H); y += CURVE_H + GAP;
                place(this.legend, x0, y, SIDE_W, LEGEND_ROW); y += LEGEND_ROW + GAP;
            }
            bodyH = Math.max(imgH, y - GAP);
        }

        this.body.style.height = bodyH + "px";
        this.total = barH + GAP + bodyH + 8;
        this.root.style.height = this.total + "px";
        const sz = node.computeSize?.();
        if (sz) node.setSize([node.size[0], sz[1]]);
        app.graph?.setDirtyCanvas(true, true);
    }

    sizeView(w, hh) {
        this.view.style.width = w + "px";
        this.view.style.height = hh + "px";
    }

    schedule() {
        if (this.raf) return;
        this.raf = requestAnimationFrame(() => { this.raf = 0; this.draw(); });
    }

    // ── малювання
    draw() {
        const A = this.A, p = this.prev;
        if (!A || !p) { this.last = null; this.hs = null; return; }
        const w = this.values();
        const ki = Math.min(this.ki, p.keys.length - 1);
        const g = M.grade(p, w, ki);
        const key = p.keys[ki];
        if (p.keys.length > 1) this.scrubLbl.textContent = `кадр ${key.i + 1} з ${p.frames}`;
        this.last = g;
        const { w: W, h: H } = p;
        const ctx = this.view.getContext("2d");
        const img = ctx.createImageData(W, H);
        const src = this.compare ? null : g.out;
        let xmin = Infinity, xmax = -Infinity;
        if (this.compare) for (const v of key.x) { if (v < xmin) xmin = v; if (v > xmax) xmax = v; }
        for (let i = 0; i < W * H; i++) {
            const v = src ? src[i] : (key.x[i] - xmin) / Math.max(xmax - xmin, 1e-9);
            const b = v * 255;
            img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = b;
            img.data[4 * i + 3] = 255;
        }
        // підсвітити план під курсором — що саме зачепить піпетка
        if (this.hover && this.tool && !this.compare) {
            const v0 = this.hover.v;
            for (let i = 0; i < W * H; i++) {
                if (Math.abs(g.v[i] - v0) < 0.02 || ((this.tool === "dim" || this.tool === "white") && g.v[i] > v0)
                    || (this.tool === "black" && g.v[i] < v0)) {
                    img.data[4 * i] = img.data[4 * i] * 0.55 + 111 * 0.45;
                    img.data[4 * i + 1] = img.data[4 * i + 1] * 0.55 + 191 * 0.45;
                    img.data[4 * i + 2] = img.data[4 * i + 2] * 0.55 + 159 * 0.45;
                }
            }
        }
        ctx.putImageData(img, 0, 0);
        if (this.showCurve) this.drawCurve(g);
    }

    geom() {
        const dpr = window.devicePixelRatio || 1;
        const cw = this.curve.clientWidth || 300, ch = this.curve.clientHeight || CURVE_H;
        if (this.curve.width !== Math.round(cw * dpr)) this.curve.width = Math.round(cw * dpr);
        if (this.curve.height !== Math.round(ch * dpr)) this.curve.height = Math.round(ch * dpr);
        const pad = { l: 10, r: 34, t: 10, b: 24 };
        return { dpr, cw, ch, pad, x0: pad.l, x1: cw - pad.r, y0: ch - pad.b, y1: pad.t };
    }

    // позиції маркерів у координатах осі v (вхідний тон після об'єму)
    handles(P) {
        const lv = (v) => M.levels(v, P);
        const hv = (v) => M.hero(M.levels(v, P), P);
        return [
            { id: "black", label: "Ч", v: P.bp },
            { id: "white", label: "Б", v: P.wp },
            { id: "hero", label: "Г", v: M.invert(lv, P.heroP), on: P.heroS > 0, accent: true },
            { id: "from", label: "С", v: M.invert(hv, P.k), on: P.c < 1 },
        ];
    }

    drawCurve(g) {
        const G = this.geom();
        const ctx = this.curve.getContext("2d");
        ctx.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
        ctx.clearRect(0, 0, G.cw, G.ch);
        const X = (v) => G.x0 + v * (G.x1 - G.x0);
        const Y = (o) => G.y0 - o * (G.y0 - G.y1);

        ctx.strokeStyle = C.line; ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const t = i / 4;
            ctx.beginPath(); ctx.moveTo(X(t), Y(0)); ctx.lineTo(X(t), Y(1)); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(X(0), Y(t)); ctx.lineTo(X(1), Y(t)); ctx.stroke();
        }
        ctx.fillStyle = C.dim; ctx.font = `10px system-ui,sans-serif`;
        ctx.textBaseline = "alphabetic";
        ctx.fillText("далеко", X(0) + 3, G.y1 + 11);
        ctx.textAlign = "right"; ctx.fillText("близько", X(1) - 3, G.y1 + 11); ctx.textAlign = "left";
        if (!g) return;

        // гістограма планів (вхід після об'єму): горб = план сцени
        const bins = 96, hist = new Float32Array(bins);
        for (const v of g.v) hist[Math.min(bins - 1, (v * bins) | 0)]++;
        let hm = 0;
        for (let i = 1; i < bins - 1; i++) hm = Math.max(hm, hist[i]);   // крайні біни — обрізане, не масштабуємо по них
        ctx.fillStyle = C.hist;
        for (let i = 0; i < bins; i++) {
            const hh = Math.min(1, Math.sqrt(hist[i] / Math.max(hm, 1)));
            ctx.fillRect(X(i / bins), Y(hh * 0.92), (G.x1 - G.x0) / bins - 0.5, Y(0) - Y(hh * 0.92));
        }

        // діагональ-орієнтир і сама крива
        ctx.strokeStyle = C.line; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.beginPath();
        const L = g.lut.length - 1;
        for (let i = 0; i <= 200; i++) {
            const v = i / 200, o = g.lut[Math.round(v * L)];
            i ? ctx.lineTo(X(v), Y(o)) : ctx.moveTo(X(v), Y(o));
        }
        ctx.stroke();

        // маркери на осі
        const P = g.P;
        this.hs = this.handles(P);
        ctx.font = `600 10px system-ui,sans-serif`; ctx.textAlign = "center";
        for (const m of this.hs) {
            const x = X(m.v), active = m.on !== false, hot = this.drag?.id === m.id;
            ctx.fillStyle = m.accent ? (active ? C.accent : C.accentDim) : (active ? C.text : C.dim);
            ctx.globalAlpha = active || hot ? 1 : 0.55;
            ctx.beginPath(); ctx.moveTo(x, G.y0 + 1); ctx.lineTo(x - 5, G.y0 + 9); ctx.lineTo(x + 5, G.y0 + 9); ctx.closePath(); ctx.fill();
            ctx.fillText(m.label, x, G.y0 + 20);
            if (hot) { ctx.strokeStyle = ctx.fillStyle; ctx.beginPath(); ctx.moveTo(x, G.y1); ctx.lineTo(x, G.y0); ctx.stroke(); }
        }
        ctx.globalAlpha = 1;

        // стеля: кружок на правому краї
        const cy = Y(P.c), xr = G.x1 + 14;
        ctx.strokeStyle = C.line; ctx.beginPath(); ctx.moveTo(xr, Y(0)); ctx.lineTo(xr, Y(1)); ctx.stroke();
        ctx.fillStyle = P.c < 1 ? C.text : C.dim;
        ctx.beginPath(); ctx.arc(xr, cy, 5, 0, Math.PI * 2); ctx.fill();
        ctx.font = `10px ${MONO}`; ctx.fillText(P.c.toFixed(2), xr, Math.min(G.y0 + 20, cy + 16));
        if (P.c < 1) {
            ctx.strokeStyle = C.dim; ctx.setLineDash([2, 3]);
            ctx.beginPath(); ctx.moveTo(X(0), cy); ctx.lineTo(G.x1, cy); ctx.stroke(); ctx.setLineDash([]);
        }

        if (this.drag?.tip) {
            ctx.font = `11px ${MONO}`; ctx.textAlign = "center"; ctx.fillStyle = C.text;
            ctx.fillText(this.drag.tip, (G.x0 + G.x1) / 2, G.y1 + 26);
        }
        ctx.textAlign = "left";
    }

    // ── мишка на кривих
    pos(e) {
        const r = this.curve.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top };
    }

    onCurveDown(e) {
        if (!this.last) return;
        const G = this.geom(), p = this.pos(e);
        const X = (v) => G.x0 + v * (G.x1 - G.x0);
        let pick = null;
        if (Math.abs(p.x - (G.x1 + 14)) < 12) pick = { id: "ceil" };
        else {
            let best = 14;
            for (const m of this.hs || []) {
                const d = Math.abs(p.x - X(m.v)) + (p.y < G.y0 - 30 ? 20 : 0);
                if (d < best) { best = d; pick = { id: m.id }; }
            }
        }
        if (!pick && p.x > G.x0 && p.x < G.x1 && p.y > G.y1 && p.y < G.y0) pick = { id: "mid" };
        if (!pick) return;
        e.stopPropagation();
        pick.y = p.y;
        pick.mid = this.widget("midtones")?.value || 0;
        pick.pop = this.widget("hero_pop")?.value || 0;
        this.drag = pick;
        this.curve.setPointerCapture(e.pointerId);
        e.preventDefault();
        this.schedule();
    }

    onCurveMove(e) {
        const G = this.geom(), p = this.pos(e);
        if (!this.drag) {
            const near = Math.abs(p.x - (G.x1 + 14)) < 12 ||
                (this.hs || []).some((m) => Math.abs(p.x - (G.x0 + m.v * (G.x1 - G.x0))) < 10 && p.y > G.y0 - 30);
            this.curve.style.cursor = near ? "grab" : "default";
            return;
        }
        this.curve.style.cursor = "grabbing";
        const v = Math.min(1, Math.max(0, (p.x - G.x0) / (G.x1 - G.x0)));
        const o = Math.min(1, Math.max(0, (G.y0 - p.y) / (G.y0 - G.y1)));
        const P = this.last.P;
        const d = this.drag;
        if (d.id === "black") { this.set("blacks", Math.min(v, P.wp - 0.02)); d.tip = `чорне від ${v.toFixed(2)}`; }
        if (d.id === "white") { this.set("whites", 1 - Math.max(v, P.bp + 0.02)); d.tip = `біле від ${v.toFixed(2)}`; }
        if (d.id === "hero") {
            this.set("hero_plane", M.levels(v, P));
            const pop = Math.min(14, Math.max(0, d.pop + (d.y - p.y) / 6));
            this.set("hero_pop", pop);
            d.tip = `герой ${M.levels(v, P).toFixed(2)} · сила ${pop.toFixed(1)}`;
        }
        if (d.id === "from") {
            const k = M.hero(M.levels(v, P), P);
            this.set("ceiling_from", k);
            if ((this.widget("foreground_ceiling")?.value ?? 1) >= 1) this.set("foreground_ceiling", k + 0.3 * (1 - k));
            d.tip = `пригашувати від ${k.toFixed(2)}`;
        }
        if (d.id === "ceil") { this.set("foreground_ceiling", o); d.tip = `стеля ${o.toFixed(2)}`; }
        if (d.id === "mid") {
            const m = Math.min(2, Math.max(-2, d.mid + (d.y - p.y) / 60));
            this.set("midtones", m);
            d.tip = `середні тони ${m >= 0 ? "+" : ""}${m.toFixed(2)}`;
        }
        this.schedule();
    }

    onCurveDbl(e) {
        const G = this.geom(), p = this.pos(e);
        const X = (v) => G.x0 + v * (G.x1 - G.x0);
        const groups = { black: ["blacks"], white: ["whites"], hero: ["hero_plane", "hero_pop"],
            from: ["ceiling_from", "foreground_ceiling"] };
        if (Math.abs(p.x - (G.x1 + 14)) < 12) return this.reset(["foreground_ceiling", "ceiling_from"]);
        for (const m of this.hs || []) if (Math.abs(p.x - X(m.v)) < 12 && p.y > G.y0 - 30) return this.reset(groups[m.id]);
        this.reset(["midtones"]);
    }

    onCurveUp(e) {
        if (!this.drag) return;
        this.drag = null;
        try { this.curve.releasePointerCapture(e.pointerId); } catch (_) { /* уже відпущено */ }
        this.schedule();
    }

    // ── піпетки на прев'ю
    sample(e) {
        if (!this.last || !this.prev) return null;
        const r = this.view.getBoundingClientRect();
        const x = Math.floor(((e.clientX - r.left) / r.width) * this.prev.w);
        const y = Math.floor(((e.clientY - r.top) / r.height) * this.prev.h);
        if (x < 0 || y < 0 || x >= this.prev.w || y >= this.prev.h) return null;
        const i = y * this.prev.w + x;
        return { v: this.last.v[i], o: this.last.out[i] };
    }

    onHover(e) {
        const s = this.sample(e);
        if (!s) return;
        this.hover = s;
        this.readout.style.display = "";
        this.readout.textContent = `тон ${s.v.toFixed(2)} → ${s.o.toFixed(2)}`;
        if (this.tool) this.schedule();
    }

    onPick(e) {
        const s = this.sample(e);
        if (!s || !this.tool) return;
        const P = this.last.P;
        if (this.tool === "hero") {
            this.set("hero_plane", M.levels(s.v, P));
            if ((this.widget("hero_pop")?.value || 0) < 1) this.set("hero_pop", 6);
        }
        if (this.tool === "dim") {
            // клікнуте стає ~80% своєї яскравості, усе ближче — не світліше за нього;
            // далі кружком справа можна відпустити чи притиснути сильніше
            const t0 = M.hero(M.levels(s.v, P), P);
            const k = Math.max(0.05, 0.75 * t0);
            this.set("ceiling_from", k);
            this.set("foreground_ceiling", Math.max(k + 0.02, 0.8 * t0));
        }
        if (this.tool === "black") this.set("blacks", Math.min(s.v, P.wp - 0.02));
        if (this.tool === "white") this.set("whites", 1 - Math.max(s.v, P.bp + 0.02));
        this.schedule();
    }
}

function registerSource(nodeType) {
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onCreated?.apply(this, arguments);
        const rb = this.addWidget("button", "↺ Скинути все до стандартних", null, () => resetAllOf(this));
        rb.serialize = false;
        // будь-яка зміна тонового повзунка — перемалювати всі ноди-перегляди
        for (const w of this.widgets || []) {
            if (!TONE_KEYS.includes(w.name)) continue;
            const cb = w.callback;
            w.callback = function () {
                const out = cb?.apply(this, arguments);
                scheduleViews();
                return out;
            };
        }
        return r;
    };

    // Run: зберігаємо сирий payload превью на самій ноді
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        const p = message?.svk_depth_preview?.[0];
        if (p) { this.__svkRaw = p; scheduleViews(); }
    };

    // права кнопка по ноді: скинути все або один змінений повзунок
    const extra = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
        const r = extra?.apply(this, arguments);
        const changed = (this.widgets || []).filter((w) => w.name in DEFAULTS && w.value !== DEFAULTS[w.name]);
        options.push(null, { content: "↺ Скинути все до стандартних", callback: () => resetAllOf(this) });
        for (const w of changed) {
            const cur = typeof w.value === "number" ? +w.value.toFixed(3) : w.value;
            options.push({ content: `↺ ${w.name}: ${cur} → ${DEFAULTS[w.name]}`,
                callback: () => resetWidget(this, w, DEFAULTS[w.name]) });
        }
        return r;
    };
}

function registerView(nodeType) {
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onCreated?.apply(this, arguments);
        const panel = new ViewPanel(this);
        this.__svkView = panel;
        this.addDOMWidget("view", "svk_depth_view", panel.root, {
            serialize: false,
            hideOnZoom: false,
            getMinHeight: () => panel.total,
        });
        this.setSize([Math.max(this.size[0], START_W), this.size[1]]);
        panel.layout();
        return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        this.__svkView?.schedule();
        return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
        this.__svkView?.dispose();
        this.__svkView = null;
        return onRemoved?.apply(this, arguments);
    };
}

app.registerExtension({
    name: "StepVisionKit.DepthGrade",
    setup() {
        // Nodes 2.0 не викликає onExecuted — беремо превью з події виконання
        api.addEventListener("executed", (ev) => {
            try {
                const out = ev.detail?.output?.svk_depth_preview?.[0];
                if (!out) return;
                const node = nodeById(ev.detail.node);
                if (node && isSource(node)) { node.__svkRaw = out; scheduleViews(); }
            } catch (_) { /* превью — не критично */ }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name;
        if (name === NODE || (name === LIVE && !Object.keys(DEFAULTS).length)) DEFAULTS = readDefaults(nodeData);
        if (SOURCES.includes(name)) registerSource(nodeType);
        else if (name === VIEW) registerView(nodeType);
    },
});
