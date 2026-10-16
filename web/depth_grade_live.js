import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as M from "./depth_grade_math.js";

// Depth Grade: дві ноди.
//  • SVKDepthGrade (нода A) — повні повзунки, «Скинути все», меню правої кнопки. Після Run
//    віддає мініатюру сирого депсу (svk_depth_preview) — її читає нода B.
//  • SVKDepthGradeView (нода B) — превью і інтерактив: під'єднується входом depth до виходу
//    depth ноди A; тягаєш повзунки A — тут миттєво видно зміну, без перегенерації.
//    Режим «Відео» — кадри результату з deflicker (svk_frames) з відтворенням по колу.
//  • SVKDepthGradeLive — застаріла, поводиться як A.
// Тонова математика — в depth_grade_math.js; тут лише UI.
// Розкладка від розмірів САМОГО root (ResizeObserver): картинка вписується contain,
// висота ноди не залежить від вмісту — немає петлі «вміст росте → нода росте».

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

const AUTO_MAXH = 480;   // одноразове авто-розміру ноди: картинка не вища за це, px
const MIN_VIEW_H = 160;  // мінімальна висота картинки (для getMinHeight), px
const GAP = 6;           // проміжок між блоками, px
const SIDE_W = 210;      // ширина бічної колонки в портреті
const ROW_GAP = 8;       // проміжок між превью і бічною колонкою
const CURVE_H = 150;     // висота полотна кривих
const LEGEND_COL = 46;   // легенда під кривими (колонка)
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
        this.mode = "cal";       // "cal" — калібрування, "vid" — відео з deflicker
        this.frames = [];        // Image кадрів результату
        this.framesKey = null;   // ідентичність набору кадрів (імена файлів)
        this.loadTok = 0;
        this.fps = 24;
        this.stride = 1;
        this.vi = 0;             // поточний кадр відео
        this.playing = false;
        this.playRaf = 0;
        this.lastT = null;
        this.acc = 0;
        this.disposed = false;

        this.root = h("div", `position:relative;display:block;width:100%;height:100%;box-sizing:border-box;
            overflow:hidden;color:${C.text};font:12px system-ui,sans-serif`);
        // не тягати вузол замість маркера; колесо не чіпаємо — полотно має зумитись
        for (const ev of ["pointerdown", "pointerup", "click", "dblclick"])
            this.root.addEventListener(ev, (e) => e.stopPropagation());

        // ── кнопки
        this.bar = h("div", "display:flex;gap:4px;flex-wrap:wrap;align-items:center");
        this.modeCal = h("button", BTN, "Калібрування");
        this.modeCal.title = "ключові кадри: повзунки, піпетки, криві — тягаєш і бачиш одразу";
        this.modeCal.onclick = () => this.setMode("cal");
        this.modeVid = h("button", BTN, "Відео");
        this.modeVid.onclick = () => this.setMode("vid");
        this.bar.append(this.modeCal, this.modeVid);
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

        // ── тіло: усі блоки абсолютно позиціоновані; розкладку рахує layout() від розмірів root
        this.body = h("div", `position:absolute;left:0;right:0;bottom:0;top:${34 + GAP}px;box-sizing:border-box`);

        this.empty = h("div", `position:absolute;display:none;box-sizing:border-box;padding:0 14px;
            align-items:center;justify-content:center;text-align:center;color:${C.dim};line-height:1.5;
            border:1px dashed ${C.line};border-radius:8px`);

        this.stage = h("div", `position:absolute;display:none;overflow:hidden;border-radius:8px;
            background:${C.bg};box-shadow:0 0 0 1px ${C.line}`);
        this.view = h("canvas", "position:absolute;left:0;top:0;display:block;image-rendering:auto");
        this.readout = h("div", `position:absolute;left:6px;bottom:6px;padding:3px 6px;border-radius:5px;
            background:rgba(24,24,27,.82);font:11px ${MONO};color:${C.text};pointer-events:none;display:none`);
        this.stale = h("div", `position:absolute;display:none;left:0;right:0;top:0;padding:6px 8px;
            box-sizing:border-box;background:rgba(24,24,27,.8);color:${C.text};font:11px system-ui,sans-serif;
            text-align:center;pointer-events:none`);
        this.stale.textContent = "Повзунки змінено — натисни Run, щоб побачити це у відео";
        this.stage.append(this.view, this.readout, this.stale);
        this.view.addEventListener("pointermove", (e) => this.onHover(e));
        this.view.addEventListener("pointerleave", () => { this.hover = null; this.readout.style.display = "none"; this.schedule(); });
        this.view.addEventListener("click", (e) => this.onPick(e));

        this.scrub = h("div", "position:absolute;display:none;align-items:center;gap:8px");
        this.playBtn = h("button", BTN + ";min-width:64px;text-align:center", "Пуск");
        this.playBtn.title = "відтворити кадри по колу";
        this.playBtn.style.display = "none";
        this.playBtn.onclick = () => this.togglePlay();
        this.scrubIn = h("input", `flex:1;min-width:0;accent-color:${C.accent}`);
        this.scrubIn.type = "range"; this.scrubIn.min = 0; this.scrubIn.step = 1; this.scrubIn.value = 0;
        this.scrubLbl = h("span", `font:11px ${MONO};color:${C.dim};white-space:nowrap`);
        this.scrubIn.oninput = () => {
            if (this.mode === "vid") { this.pause(); this.vi = +this.scrubIn.value; this.drawVideo(); }
            else { this.ki = +this.scrubIn.value; this.schedule(); }
        };
        this.scrub.append(this.playBtn, this.scrubIn, this.scrubLbl);

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

        this.ro = new ResizeObserver(() => this.layout());
        this.ro.observe(this.root);
        this.timer = setInterval(() => this.tick(), 120);
        VIEWS.add(this);
        this.updateModeUI();
    }

    dispose() {
        this.disposed = true;
        clearInterval(this.timer);
        this.ro?.disconnect();
        VIEWS.delete(this);
        if (this.raf) cancelAnimationFrame(this.raf);
        this.raf = 0;
        this.pause();
        this.frames = [];
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
        if (id && this.mode === "vid") this.setMode("cal");   // піпетка — це калібрування
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

    // ── режими: калібрування (ключові кадри) / відео (кадри результату)
    setMode(m) {
        if (m === "vid" && !this.frames.length) return;
        this.mode = m;
        this.pause();
        this.updateModeUI();
        this.syncScrub();
        if (m === "vid") {
            this.setTool(null);
            this.drawVideo();
        } else if (this.prev) {
            this.view.width = this.prev.w;
            this.view.height = this.prev.h;
            this.draw();
        }
        this.layout();
        this.updateStale();
    }

    updateModeUI() {
        const vid = this.mode === "vid";
        const has = this.frames.length > 0;
        this.modeCal.style.borderColor = vid ? C.line : C.accent;
        this.modeCal.style.color = vid ? C.text : C.accent;
        this.modeVid.style.borderColor = vid ? C.accent : C.line;
        this.modeVid.style.color = vid ? C.accent : (has ? C.text : C.dim);
        this.modeVid.disabled = !has;
        this.modeVid.style.opacity = has ? "1" : "0.45";
        this.modeVid.style.cursor = has ? "pointer" : "default";
        this.modeVid.title = has ? "відтворити результат з deflicker по колу"
            : "натисни Run — тут з'явиться результат з deflicker";
        for (const b of Object.values(this.toolBtns)) b.style.opacity = vid ? "0.45" : "1";
        this.cmp.disabled = vid;
        this.cmp.style.opacity = vid ? "0.45" : "1";
        this.playBtn.style.display = vid ? "" : "none";
        this.playBtn.textContent = this.playing ? "Пауза" : "Пуск";
    }

    // кількість кроків повзунка в поточному режимі
    scrubCount() { return this.mode === "vid" ? this.frames.length : (this.prev ? this.prev.keys.length : 0); }

    syncScrub() {
        if (this.mode === "vid") {
            this.scrubIn.max = Math.max(0, this.frames.length - 1);
            this.scrubIn.value = this.vi;
        } else if (this.prev) {
            this.scrubIn.max = this.prev.keys.length - 1;
            this.scrubIn.value = this.ki;
        }
    }

    // розміри картинки поточного режиму
    media() {
        if (this.mode === "vid" && this.frames.length) {
            const f = this.frames[0];
            return { w: f.naturalWidth, h: f.naturalHeight };
        }
        if (this.mode === "cal" && this.prev) return { w: this.prev.w, h: this.prev.h };
        return null;
    }

    togglePlay() { if (this.playing) this.pause(); else this.play(); }

    play() {
        if (this.mode !== "vid" || this.frames.length < 2) return;
        this.playing = true;
        this.lastT = null;
        this.acc = 0;
        this.updateModeUI();
        this.playRaf = requestAnimationFrame((t) => this.loopPlay(t));
    }

    pause() {
        this.playing = false;
        if (this.playRaf) cancelAnimationFrame(this.playRaf);
        this.playRaf = 0;
        if (this.playBtn) this.updateModeUI();
    }

    // відтворення по колу: таймінг за fps, не setInterval
    loopPlay(now) {
        this.playRaf = 0;
        if (!this.playing || this.disposed) return;
        if (this.lastT == null) this.lastT = now;
        const dt = Math.min(250, now - this.lastT);   // фон/пауза вкладки не має стрибати кадрами
        this.lastT = now;
        this.acc += dt;
        const frameMs = 1000 / this.fps;
        if (this.acc >= frameMs) {
            const adv = Math.floor(this.acc / frameMs);
            this.acc -= adv * frameMs;
            this.vi = (this.vi + adv) % this.frames.length;
            this.drawVideo();
        }
        this.playRaf = requestAnimationFrame((t) => this.loopPlay(t));
    }

    // набір кадрів з виходу Run ноди B: завантажуємо й перемикаємось у «Відео» на першому кадрі
    setFrames(out) {
        const list = out?.svk_frames;
        if (!Array.isArray(list) || !list.length) return;
        const key = list.map((f) => f.filename).join("|");
        if (key === this.framesKey) return;
        this.framesKey = key;
        this.fps = Math.max(1, +out.svk_fps?.[0] || 24);
        this.stride = Math.max(1, +out.svk_stride?.[0] || 1);
        const tok = ++this.loadTok;
        const urls = list.map((f) => api.apiURL(
            `/view?filename=${encodeURIComponent(f.filename)}&subfolder=${encodeURIComponent(f.subfolder ?? "")}` +
            `&type=${encodeURIComponent(f.type ?? "temp")}`));
        Promise.all(urls.map((u) => new Promise((ok) => {
            const im = new Image();
            im.onload = () => ok(im);
            im.onerror = () => ok(null);
            im.src = u;
        }))).then((imgs) => {
            if (this.disposed || tok !== this.loadTok) return;
            const good = imgs.filter(Boolean);
            if (!good.length) return;
            this.frames = good;
            this.vi = 0;
            this.setMode("vid");   // на паузі, на першому кадрі
        });
    }

    // після перезавантаження сторінки — кадри з історії виводу, якщо вони ще є
    restoreFrames() {
        try {
            const out = app.nodeOutputs?.[String(this.node.id)];
            if (out?.svk_frames) this.setFrames(out);
        } catch (_) { /* історія — не критично */ }
    }

    // зміна тонів після Run: у відео ця зміна ще не видна — показуємо плашку
    updateStale() {
        const A = this.A;
        const on = this.mode === "vid" && !!A && !!A.__svkRunSig && A.__svkRunSig !== toneSig(A);
        this.stale.style.display = on ? "" : "none";
    }

    // ── стан → розкладка: виклик лише при зміні джерела або payload
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
            this.schedule();
        } else {
            const sig = A ? toneSig(A) : "";
            if (sig !== this.sig) {
                this.sig = sig;
                this.schedule();
            }
        }
        if (this.framesKey === null) this.restoreFrames();
        this.updateStale();
    }

    setPrev(p) {
        this.prev = p;
        this.last = null;
        this.hs = null;
        if (!p) return;
        this.ki = Math.min(this.ki, p.keys.length - 1);
        this.syncScrub();
        if (this.mode !== "vid") {
            this.view.width = p.w;
            this.view.height = p.h;
        }
    }

    // мінімальна висота блоку — фіксована (бар + картинка + рядок + криві), не від вмісту
    minH() {
        const scrub = this.scrubCount() > 1;
        return (this.bar.offsetHeight || 28) + GAP + MIN_VIEW_H
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
        const wantH = (this.bar.offsetHeight || 28) + GAP + body + 8;
        const delta = wantH - H;
        if (Math.abs(delta) > 2) node.setSize([node.size[0], node.size[1] + delta]);
        app.graph?.setDirtyCanvas(true, true);
    }

    // ── розкладка: картинка contain у доступному прямокутнику root; root сам не росте
    layout() {
        const root = this.root;
        const W = root.clientWidth || Math.max(200, (this.node.size?.[0] || START_W) - 16);
        const H = root.clientHeight || this.minH();
        const barH = this.bar.offsetHeight || 28;
        const top = barH + GAP;
        this.body.style.top = top + "px";
        const BH = Math.max(120, H - top);

        const m = this.media();
        const n = this.scrubCount();
        const scrub = !!m && n > 1;
        const ar = m ? m.w / m.h : 16 / 9;
        const curves = !!m && this.showCurve;
        const row = !!m && ar < 0.95 && (this.showCurve || scrub);   // портрет + криві/кадри — збоку

        this.empty.textContent = this.A
            ? "Натисни Run один раз. Далі тягай повзунки Depth Grade — тут видно одразу, без перегенерації. Clarity, texture, смуги й deflicker видно лише після Run."
            : "Під'єднай вхід depth до виходу depth ноди Depth Grade";
        this.empty.style.display = m ? "none" : "flex";
        this.stage.style.display = m ? "block" : "none";
        this.scrub.style.display = scrub ? "flex" : "none";
        this.curve.style.display = curves ? "block" : "none";
        this.legend.style.display = curves ? "block" : "none";
        this.curveBtn.textContent = `Криві ${this.showCurve ? "▾" : "▸"}`;

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
    drawVideo() {
        const n = this.frames.length;
        if (!n) return;
        this.vi = Math.min(this.vi, n - 1);
        const im = this.frames[this.vi];
        const W = im.naturalWidth, H = im.naturalHeight;
        if (this.view.width !== W || this.view.height !== H) {
            this.view.width = W;
            this.view.height = H;
        }
        this.view.getContext("2d").drawImage(im, 0, 0, W, H);
        this.scrubIn.value = this.vi;
        this.scrubLbl.textContent = `кадр ${this.vi + 1} з ${n}` + (this.stride > 1 ? ` · кожен ${this.stride}-й` : "");
        this.hover = null;
        this.readout.style.display = "none";
    }

    draw() {
        if (this.mode === "vid") {
            this.drawVideo();
            const p = this.prev;
            if (this.showCurve && this.A && p) {
                const g = M.grade(p, this.values(), Math.min(this.ki, p.keys.length - 1));
                this.last = g;
                this.drawCurve(g);
            }
            return;
        }
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
        if (this.mode === "vid") this.setMode("cal");   // тяга кривої — калібрування
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
        if (this.mode === "vid") this.setMode("cal");
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
        if (this.mode === "vid") return;
        const s = this.sample(e);
        if (!s) return;
        this.hover = s;
        this.readout.style.display = "";
        this.readout.textContent = `тон ${s.v.toFixed(2)} → ${s.o.toFixed(2)}`;
        if (this.tool) this.schedule();
    }

    onPick(e) {
        if (this.mode === "vid") return;
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
            getMinHeight: () => panel.minH(),
        });
        this.setSize([Math.max(this.size[0], START_W), this.size[1]]);
        panel.layout();
        return r;
    };

    // Run ноди B: кадри результату (deflicker) → режим «Відео»
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (message?.svk_frames) this.__svkView?.setFrames(message);
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
                const d = ev.detail;
                const node = nodeById(d?.node);
                if (!node) return;
                const out = d.output;
                if (isSource(node)) {
                    const p = out?.svk_depth_preview?.[0];
                    if (p) { node.__svkRaw = p; node.__svkRunSig = toneSig(node); scheduleViews(); }
                } else if (typeOf(node) === VIEW) {
                    node.__svkView?.setFrames(out);
                }
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
