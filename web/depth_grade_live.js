import { app } from "../../scripts/app.js";
import * as M from "./depth_grade_math.js";

// Живий пульт для Depth Grade: прев'ю з піпетками + «Криві» з маркерами.
//
// Нащо (16.10.2026, Льоша): «не повзунки, а щось інтерактивне; крутити — і щоб
// генерація заново не йшла». Після ОДНОГО Run нода віддає сюди мініатюру сирого
// депсу, далі тон рахується в браузері (depth_grade_math.js) — миттєво.
// Повзунки лишаються джерелом правди: маркери й піпетки пишуть у них, тому воркфлоу
// зберігається як завжди, а без цього js нода повністю робоча.

const NODE = "SVKDepthGrade";
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

function h(tag, css = "", text) {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text != null) e.textContent = text;
    return e;
}

const BTN = `font:500 11px/1 system-ui,sans-serif;color:${C.text};background:${C.panel};
    border:1px solid ${C.line};border-radius:6px;padding:6px 9px;cursor:pointer;
    transition:transform .12s cubic-bezier(.16,1,.3,1),border-color .15s;user-select:none`;

class LivePanel {
    constructor(node) {
        this.node = node;
        this.prev = null;
        this.tool = null;
        this.compare = false;
        this.drag = null;
        this.hover = null;
        this.raf = 0;

        this.root = h("div", `display:flex;flex-direction:column;gap:6px;padding:6px 2px 2px;
            box-sizing:border-box;width:100%;color:${C.text};font:12px system-ui,sans-serif`);
        this.root.addEventListener("pointerdown", (e) => e.stopPropagation());   // не тягати вузол замість маркера
        this.root.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });

        // ── інструменти
        const bar = h("div", "display:flex;gap:4px;flex-wrap:wrap;align-items:center");
        this.toolBtns = {};
        for (const t of TOOLS) {
            const b = h("button", BTN, t.label);
            b.title = t.hint;
            b.onclick = () => this.setTool(this.tool === t.id ? null : t.id);
            b.onpointerdown = () => (b.style.transform = "scale(0.97)");
            b.onpointerup = b.onpointerleave = () => (b.style.transform = "");
            this.toolBtns[t.id] = b;
            bar.append(b);
        }
        const cmp = h("button", BTN + ";margin-left:auto", "До / після");
        cmp.title = "тримай — покаже вхід без грейду";
        const on = () => { this.compare = true; this.schedule(); };
        const off = () => { this.compare = false; this.schedule(); };
        cmp.onpointerdown = on; cmp.onpointerup = off; cmp.onpointerleave = off;
        bar.append(cmp);

        // ── прев'ю
        this.stage = h("div", `position:relative;border-radius:8px;overflow:hidden;background:${C.bg};
            border:1px solid ${C.line}`);
        this.view = h("canvas", "display:block;width:100%;height:auto;image-rendering:auto");
        this.empty = h("div", `padding:28px 14px;text-align:center;color:${C.dim};line-height:1.5;
            border:1px dashed ${C.line};border-radius:8px`,
            "Натисни Run один раз. Далі все крутиться наживо — без перегенерації.");
        this.readout = h("div", `position:absolute;left:6px;bottom:6px;padding:3px 6px;border-radius:5px;
            background:rgba(24,24,27,.82);font:11px ${MONO};color:${C.text};pointer-events:none;display:none`);
        this.stage.append(this.view, this.readout);
        this.stage.style.display = "none";
        this.view.addEventListener("pointermove", (e) => this.onHover(e));
        this.view.addEventListener("pointerleave", () => { this.hover = null; this.readout.style.display = "none"; this.schedule(); });
        this.view.addEventListener("click", (e) => this.onPick(e));

        // ── криві
        this.curve = h("canvas", `display:block;width:100%;height:150px;border-radius:8px;
            background:${C.bg};border:1px solid ${C.line};touch-action:none;cursor:default`);
        this.curve.addEventListener("pointerdown", (e) => this.onCurveDown(e));
        this.curve.addEventListener("pointermove", (e) => this.onCurveMove(e));
        this.curve.addEventListener("pointerup", (e) => this.onCurveUp(e));
        this.curve.addEventListener("pointercancel", (e) => this.onCurveUp(e));

        this.legend = h("div", `color:${C.dim};font-size:11px;line-height:1.45`);
        this.legend.innerHTML =
            "Ч / Б — що піде в чорне й біле · <b style='color:" + C.accent + "'>Г</b> — герой (тягни вгору — сильніше) · " +
            "С — звідки пригашувати, повзунок справа — наскільки. Clarity, texture, смуги й deflicker видно після Run.";

        this.root.append(bar, this.empty, this.stage, this.curve, this.legend);
        new ResizeObserver(() => this.schedule()).observe(this.curve);
    }

    // ── віджети як джерело правди
    widget(name) { return this.node.widgets?.find((w) => w.name === name); }

    values() {
        const o = {};
        for (const k of TONE_KEYS) o[k] = this.widget(k)?.value;
        return o;
    }

    set(name, value) {
        const w = this.widget(name);
        if (!w) return;
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
        w.callback?.(value, app.canvas, this.node);
        this.node.onWidgetChanged?.(name, value, old, w);
        this.node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas(true, true);
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

    load(payload) {
        this.prev = M.decodePreview(payload);
        this.view.width = this.prev.w;
        this.view.height = this.prev.h;
        this.empty.style.display = "none";
        this.stage.style.display = "";
        this.resizeNode();
        this.schedule();
    }

    resizeNode() {
        const n = this.node;
        const want = n.computeSize?.();
        if (want && n.size[1] < want[1]) n.setSize([Math.max(n.size[0], want[0]), want[1]]);
        app.graph?.setDirtyCanvas(true, true);
    }

    minHeight() {
        const wpx = Math.max(240, (this.node.size?.[0] || 340) - 20);
        const img = this.prev ? (wpx * this.prev.h) / this.prev.w : 80;
        return 34 + Math.min(img, 520) + 160 + 44;
    }

    schedule() {
        if (this.raf) return;
        this.raf = requestAnimationFrame(() => { this.raf = 0; this.draw(); });
    }

    // ── малювання
    draw() {
        if (!this.prev) { this.drawCurve(null); return; }
        const w = this.values();
        const g = M.grade(this.prev, w);
        this.last = g;
        const { w: W, h: H } = this.prev;
        const ctx = this.view.getContext("2d");
        const img = ctx.createImageData(W, H);
        const src = this.compare ? null : g.out;
        let xmin = Infinity, xmax = -Infinity;
        if (this.compare) for (const v of this.prev.x) { if (v < xmin) xmin = v; if (v > xmax) xmax = v; }
        for (let i = 0; i < W * H; i++) {
            const v = src ? src[i] : (this.prev.x[i] - xmin) / Math.max(xmax - xmin, 1e-9);
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
        this.drawCurve(g);
    }

    geom() {
        const dpr = window.devicePixelRatio || 1;
        const cw = this.curve.clientWidth || 300, ch = this.curve.clientHeight || 150;
        if (this.curve.width !== Math.round(cw * dpr)) this.curve.width = Math.round(cw * dpr);
        if (this.curve.height !== Math.round(ch * dpr)) this.curve.height = Math.round(ch * dpr);
        const pad = { l: 10, r: 34, t: 10, b: 24 };
        return { dpr, cw, ch, pad, x0: pad.l, x1: cw - pad.r, y0: ch - pad.b, y1: pad.t };
    }

    // позиції маркерів у координатах осі v (вхідний тон після об'єму)
    handles(P) {
        const lv = (v) => M.levels(v, P);
        const hv = (v) => M.hero(M.levels(v, P), P);
        const H = [
            { id: "black", label: "Ч", v: P.bp },
            { id: "white", label: "Б", v: P.wp },
            { id: "hero", label: "Г", v: M.invert(lv, P.heroP), on: P.heroS > 0, accent: true },
            { id: "from", label: "С", v: M.invert(hv, P.k), on: P.c < 1 },
        ];
        return H;
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

        // стеля: повзунок на правому краї
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
        if (!pick) return;
        pick.y = p.y;
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
        this.schedule();
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
            const k = Math.max(0.05, M.hero(M.levels(s.v, P), P) - 0.04);
            this.set("ceiling_from", k);
            this.set("foreground_ceiling", k + 0.3 * (1 - k));
        }
        if (this.tool === "black") this.set("blacks", Math.min(s.v, P.wp - 0.02));
        if (this.tool === "white") this.set("whites", 1 - Math.max(s.v, P.bp + 0.02));
        this.schedule();
    }
}

app.registerExtension({
    name: "StepVisionKit.DepthGradeLive",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const panel = new LivePanel(this);
            this.__svkLive = panel;
            this.addDOMWidget("live", "svk_depth_live", panel.root, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => panel.minHeight(),
            });
            // будь-яка зміна тонового повзунка — перемалювати прев'ю
            for (const w of this.widgets || []) {
                if (!TONE_KEYS.includes(w.name)) continue;
                const cb = w.callback;
                w.callback = function () {
                    const out = cb?.apply(this, arguments);
                    panel.schedule();
                    return out;
                };
            }
            this.setSize([Math.max(this.size[0], 360), this.computeSize()[1]]);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const p = message?.svk_depth_preview?.[0];
            if (p && this.__svkLive) this.__svkLive.load(p);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            this.__svkLive?.schedule();
            return r;
        };
    },
});
