import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as M from "./depth_grade_math.js";
import {
    NODE, LIVE, VIEW, SOURCES, TONE_KEYS, C, MONO, GAP, CTRL_H, ensureStyles, h, place, setText, icon, btn,
    presetRow, isSource, nodeById, widgetOf, toneSig, rawOf, findSource, readDefaults, getDefaults, setDefaults,
    resetWidget, resetAllOf, scheduleViews, registerPanel, unregisterPanel, setWidgetValue,
} from "./depth_grade_ui.js";
import { CurvePanel, CURVE_H, LEGEND_COL, LEGEND_ROW, SIDE_W, ROW_GAP } from "./depth_grade_curve.js";
import { T } from "./svk_i18n.js";

// Depth Grade: ноди і спільна тонова математика.
//  • SVKDepthGrade (нода A) — повні повзунки, «Скинути все», меню правої кнопки. Після Run
//    віддає мініатюру сирого депсу (svk_depth_preview).
//  • SVKDepthGradeView (нода B, «кадр / калібрування») — піпетки, криві, До/після. Під'єднується
//    входом depth до виходу depth ноди A (наскрізний вихід, як у SaveImage); тягаєш повзунки A —
//    тут миттєво видно зміну, без перегенерації.
//  • SVKDepthGradeLive — застаріла, поводиться як A.
// Відео-прев'ю (живе, без Run) — окрема нода: depth_grade_video.js.
// Тонова математика — depth_grade_math.js; спільні кнопки, пресети, реєстр — depth_grade_ui.js.
// Розкладка від розмірів САМОГО root (ResizeObserver): картинка вписується contain,
// висота ноди не залежить від вмісту — немає петлі «вміст росте → нода росте».

const AUTO_MAXH = 480;   // одноразове авто-розміру ноди: картинка не вища за це, px
const MIN_VIEW_H = 160;  // мінімальна висота картинки (для getMinHeight), px
const START_W = 460;     // стартова ширина ноди B
const BAR_H = 2 * CTRL_H + 6;   // два рядки панелі + проміжок

const TOOLS = [
    { id: "hero", label: T("Hero"), hint: T("click the main subject — its volume gets boosted") },
    { id: "dim", label: T("Dim"), hint: T("click what is in the way up front — it and everything nearer turns gray") },
    { id: "black", label: T("Black"), hint: T("click the background — it and everything farther goes to black") },
    { id: "white", label: T("White"), hint: T("click something near — it and everything nearer turns white") },
];

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

class ViewPanel {
    constructor(node) {
        ensureStyles();
        this.node = node;
        this.A = null;           // нода-джерело (SVKDepthGrade)
        this.raw = undefined;    // ідентичність payload, з якого зроблено prev
        this.sig = "";           // остання сигнатура тонових значень A
        this.prev = null;
        this.last = null;        // результат M.grade для піпеток і кривих
        this.tool = null;
        this.compare = false;
        this.hover = null;
        this.ki = 0;
        this.raf = 0;
        this.showCurve = false;  // «Криві» за замовчуванням згорнуті
        this.disposed = false;

        this.root = h("div", `position:relative;display:block;width:100%;height:100%;box-sizing:border-box;
            overflow:hidden;color:${C.text};font:12px system-ui,sans-serif`);
        // не тягати вузол замість маркера; колесо не чіпаємо — полотно має зумитись
        for (const ev of ["pointerdown", "click", "dblclick"])
            this.root.addEventListener(ev, (e) => e.stopPropagation());

        // ── рядок 1: піпетки ліворуч, «До/після», «Криві», «Скинути» — іконками праворуч
        const r1 = h("div");
        r1.className = "svk-row";
        const pick = h("div");
        pick.className = "svk-seg";
        const lbl = h("span", "", T("Eyedropper"));
        lbl.className = "svk-lbl";
        this.toolBtns = {};
        for (const t of TOOLS) {
            const b = btn(t.label, t.hint, () => this.setTool(this.tool === t.id ? null : t.id));
            this.toolBtns[t.id] = b;
            pick.append(b);
        }
        const end = h("div");
        end.className = "svk-end";
        this.cmp = icon("split", T("Before / after: hold the button to show the input without the grade"));
        const on = () => { this.compare = true; this.schedule(); };
        const off = () => { this.compare = false; this.schedule(); };
        this.cmp.onpointerdown = on;
        this.cmp.onpointerup = off;
        this.cmp.onpointerleave = off;
        this.curveBtn = icon("curve", T("Curves: histogram, curve and B / W / H / F markers"), () => this.toggleCurve());
        this.resetBtn = icon("reset", T("Reset all Depth Grade sliders to defaults"),
            () => { if (this.A) resetAllOf(this.A); });
        end.append(this.cmp, this.curveBtn, this.resetBtn);
        r1.append(lbl, pick, end);

        // ── рядок 2: пресети кривої
        this.presets = presetRow(() => this.A);
        this.bar = h("div", "display:flex;flex-direction:column;gap:6px;flex:none");
        this.bar.append(r1, this.presets.el);

        // ── тіло: усі блоки абсолютно позиціоновані; розкладку рахує layout() від розмірів root
        this.body = h("div", `position:absolute;left:0;right:0;bottom:0;top:${BAR_H + GAP}px;box-sizing:border-box`);

        this.empty = h("div", `position:absolute;display:none;box-sizing:border-box;padding:0 14px;
            align-items:center;justify-content:center;text-align:center;color:${C.dim};
            font:11px system-ui,sans-serif;line-height:1.5;border:1px dashed ${C.line};border-radius:8px`);
        this.emptyTxt = h("span", "max-width:32ch");
        this.empty.append(this.emptyTxt);

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
        this.scrubIn.type = "range";
        this.scrubIn.min = 0;
        this.scrubIn.step = 1;
        this.scrubIn.value = 0;
        this.scrubLbl = h("span", `font:11px ${MONO};color:${C.dim};white-space:nowrap`);
        this.scrubIn.oninput = () => { this.ki = +this.scrubIn.value; this.schedule(); };
        this.scrub.append(this.scrubIn, this.scrubLbl);

        // криві: полотно і легенда живуть у CurvePanel; розкладку тут рахує layout()
        this.curves = new CurvePanel(this);
        this.curve = this.curves.curve;
        this.legend = this.curves.legend;

        this.body.append(this.empty, this.stage, this.scrub, this.curve, this.legend);
        this.root.append(this.bar, this.body);

        this.ro = new ResizeObserver(() => this.layout());
        this.ro.observe(this.root);
        this.timer = setInterval(() => this.tick(), 120);
        registerPanel(this);
        this.presets.refresh();
    }

    dispose() {
        this.disposed = true;
        clearInterval(this.timer);
        this.ro?.disconnect();
        unregisterPanel(this);
        if (this.raf) cancelAnimationFrame(this.raf);
        this.raf = 0;
    }

    // ── джерело: вхід depth → (reroute…) → нода A
    source() { return findSource(this.node); }

    // ── повзунки ноди A як джерело правди
    widget(name) { return widgetOf(this.A, name); }

    values() {
        const o = {};
        for (const k of TONE_KEYS) o[k] = this.widget(k)?.value;
        return o;
    }

    set(name, value) {
        if (this.A) setWidgetValue(this.A, name, value);
    }

    reset(names) {
        for (const n of names) if (n in getDefaults()) this.set(n, getDefaults()[n]);
        this.schedule();
    }

    setTool(id) {
        this.tool = id;
        for (const [k, b] of Object.entries(this.toolBtns)) b.classList.toggle("svk-on", k === id);
        this.view.style.cursor = id ? "crosshair" : "default";
    }

    toggleCurve() {
        this.showCurve = !this.showCurve;
        this.curveBtn.classList.toggle("svk-on", this.showCurve);
        this.layout();
        this.schedule();
    }

    // кількість кроків повзунка ключових кадрів
    scrubCount() { return this.prev ? this.prev.keys.length : 0; }

    syncScrub() {
        if (this.prev) {
            this.scrubIn.max = this.prev.keys.length - 1;
            this.scrubIn.value = this.ki;
        }
    }

    // розміри картинки
    media() { return this.prev ? { w: this.prev.w, h: this.prev.h } : null; }

    tick() {
        const A = this.source();
        const raw = A ? rawOf(A) : null;
        if (A !== this.A || raw !== this.raw) {
            this.A = A;
            this.raw = raw;
            this.setPrev(A ? prevOf(A) : null);
            if (this.prev) this.autoSize(this.prev);
            this.layout();
            this.sig = A ? toneSig(A) : "";
            this.presets.refresh();
            this.schedule();
        } else {
            const sig = A ? toneSig(A) : "";
            if (sig !== this.sig) {
                this.sig = sig;
                this.presets.refresh();
                this.schedule();
            }
        }
    }

    setPrev(p) {
        this.prev = p;
        this.last = null;
        this.curves.hs = null;
        if (!p) return;
        this.ki = Math.min(this.ki, p.keys.length - 1);
        this.syncScrub();
        this.view.width = p.w;
        this.view.height = p.h;
    }

    // мінімальна висота блоку — фіксована (бар + картинка + рядок + криві), не від вмісту
    minH() {
        const scrub = this.scrubCount() > 1;
        return (this.bar.offsetHeight || BAR_H) + GAP + MIN_VIEW_H
            + (scrub ? 24 + GAP : 0)
            + (this.showCurve ? CURVE_H + GAP + LEGEND_COL + GAP : 0) + 8;
    }

    // одноразово (на кожну пропорцію кадру, у самій ноді): картинка вписується в ≤ AUTO_MAXH
    autoSize(p) {
        const node = this.node;
        const ar = p.w / p.h;
        const key = Math.round(ar * 100) / 100;
        node.properties ??= {};
        if (node.properties.svkSizedAr === key) return;
        const W = this.root.clientWidth, H = this.root.clientHeight;
        if (!W || !H) return;                 // ще не на екрані — спробуємо на наступному payload
        node.properties.svkSizedAr = key;
        const nk = p.keys.length;
        const row = ar < 0.95 && (this.showCurve || nk > 1);
        let body;
        if (row) {
            const imgH = Math.min(AUTO_MAXH, (W - SIDE_W - ROW_GAP) / ar);
            const side = (nk > 1 ? 24 + GAP : 0) + (this.showCurve ? CURVE_H + GAP + LEGEND_ROW : 0);
            body = Math.max(imgH, side);
        } else {
            const imgH = Math.min(AUTO_MAXH, W / ar);
            body = imgH + (nk > 1 ? GAP + 24 : 0) + (this.showCurve ? GAP + CURVE_H + GAP + LEGEND_COL : 0);
        }
        const wantH = (this.bar.offsetHeight || BAR_H) + GAP + body + 8;
        const delta = wantH - H;
        if (Math.abs(delta) > 2) node.setSize([node.size[0], node.size[1] + delta]);
        app.graph?.setDirtyCanvas(true, true);
    }

    // ── розкладка: картинка contain у доступному прямокутнику root; root сам не росте
    layout() {
        const root = this.root;
        const W = root.clientWidth || Math.max(200, (this.node.size?.[0] || START_W) - 16);
        const H = root.clientHeight || this.minH();
        const barH = this.bar.offsetHeight || BAR_H;
        const top = barH + GAP;
        this.body.style.top = top + "px";
        const BH = Math.max(120, H - top);

        const m = this.media();
        const n = this.scrubCount();
        const scrub = !!m && n > 1;
        const ar = m ? m.w / m.h : 16 / 9;
        const curves = !!m && this.showCurve;
        const row = !!m && ar < 0.95 && (this.showCurve || scrub);   // портрет + криві/кадри — збоку

        setText(this.emptyTxt, this.A
            ? T("Press Run once. Then drag the Depth Grade sliders — the result shows here instantly, no regeneration. Clarity, texture, banding and deflicker only show after Run.")
            : T("Connect the depth input to the depth output of the Depth Grade node"));
        this.empty.style.display = m ? "none" : "flex";
        this.stage.style.display = m ? "block" : "none";
        this.scrub.style.display = scrub ? "flex" : "none";
        this.curve.style.display = curves ? "block" : "none";
        this.legend.style.display = curves ? "block" : "none";

        if (!m) {
            place(this.empty, 0, 0, W, BH);
        } else if (!row) {
            const under = [];
            if (scrub) under.push(24);
            if (curves) under.push(CURVE_H, LEGEND_COL);
            const underH = under.reduce((a, b) => a + b, 0) + GAP * under.length;
            const avail = Math.max(60, BH - underH);
            const imgH = Math.min(avail, W / ar), imgW = imgH * ar;
            place(this.stage, Math.round((W - imgW) / 2), 0, imgW, imgH);
            this.sizeView(imgW, imgH);
            let y = imgH + GAP;
            if (scrub) { place(this.scrub, 0, y, W, 24); y += 24 + GAP; }
            if (curves) {
                place(this.curve, 0, y, W, CURVE_H); y += CURVE_H + GAP;
                place(this.legend, 0, y, W, LEGEND_COL);
            }
        } else {
            const areaW = W - SIDE_W - ROW_GAP;
            const imgH = Math.min(BH, areaW / ar), imgW = imgH * ar;
            place(this.stage, Math.round((areaW - imgW) / 2), 0, imgW, imgH);
            this.sizeView(imgW, imgH);
            const x0 = W - SIDE_W;
            let y = 0;
            if (scrub) { place(this.scrub, x0, y, SIDE_W, 24); y += 24 + GAP; }
            if (curves) {
                place(this.curve, x0, y, SIDE_W, CURVE_H); y += CURVE_H + GAP;
                place(this.legend, x0, y, SIDE_W, LEGEND_ROW);
            }
        }
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
        if (!A || !p) { this.last = null; this.curves.hs = null; return; }
        const w = this.values();
        const ki = Math.min(this.ki, p.keys.length - 1);
        const g = M.grade(p, w, ki);
        const key = p.keys[ki];
        if (p.keys.length > 1) setText(this.scrubLbl, T("frame {i} of {n}").replace("{i}", key.i + 1).replace("{n}", p.frames));
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
        if (this.showCurve) this.curves.draw(g);
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
        this.readout.textContent = T("tone {v} → {o}").replace("{v}", s.v.toFixed(2)).replace("{o}", s.o.toFixed(2));
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
        const rb = this.addWidget("button", T("↺ Reset all to defaults"), null, () => resetAllOf(this));
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

    // Run: зберігаємо сирий payload превью і сигнатуру тонів, з якими його зроблено
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        const p = message?.svk_depth_preview?.[0];
        if (p) { this.__svkRaw = p; this.__svkRunSig = toneSig(this); scheduleViews(); }
    };

    // права кнопка по ноді: скинути все або один змінений повзунок
    const extra = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
        const r = extra?.apply(this, arguments);
        const defaults = getDefaults();
        const changed = (this.widgets || []).filter((w) => w.name in defaults && w.value !== defaults[w.name]);
        options.push(null, { content: T("↺ Reset all to defaults"), callback: () => resetAllOf(this) });
        for (const w of changed) {
            const cur = typeof w.value === "number" ? +w.value.toFixed(3) : w.value;
            options.push({ content: `↺ ${w.name}: ${cur} → ${defaults[w.name]}`,
                callback: () => resetWidget(this, w, defaults[w.name]) });
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
            getMinHeight: () => panel.minH(),
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
        // Nodes 2.0 не викликає onExecuted — беремо виводи з події виконання
        api.addEventListener("executed", (ev) => {
            try {
                const node = nodeById(ev.detail?.node);
                if (!node || !isSource(node)) return;
                const p = ev.detail.output?.svk_depth_preview?.[0];
                if (p) { node.__svkRaw = p; node.__svkRunSig = toneSig(node); scheduleViews(); }
            } catch (_) { /* превью — не критично */ }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name;
        if (name === NODE || (name === LIVE && !Object.keys(getDefaults()).length)) setDefaults(readDefaults(nodeData));
        if (SOURCES.includes(name)) registerSource(nodeType);
        else if (name === VIEW) registerView(nodeType);
    },
});
