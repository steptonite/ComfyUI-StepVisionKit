// Depth Grade View · відео (живе прев'ю).
// Під'єднай вхід depth до виходу depth ноди Depth Grade і натисни Run один раз. Далі кожна зміна
// повзунків Depth Grade сама перераховує фрагмент кліпу на сервері (POST /svk/depth_grade/preview)
// і показує його по колу — без Run. Порядок: зміна → 250 мс тиші → запит; відповідь зі старішим
// gen ігнорується. Вигляд і спільні кнопки — з depth_grade_ui.js.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    VIDEO, C, GAP, CTRL_H, MONO, ensureStyles, h, place, setText, icon, setIcon,
    presetRow, findSource, rawOf, paramValues, widgetOf, registerPanel, unregisterPanel,
} from "./depth_grade_ui.js";

const ROUTE = "/svk/depth_grade/preview";
const SIZES = [256, 384, 512, 768];
const FRAGMENTS = [24, 48, 96];
const DEFAULT_SIZE = 384;
const DEFAULT_FRAGMENT = 48;
const DEBOUNCE_MS = 250;
const POLL_MS = 120;
const MIN_PIC_H = 160;   // мінімальна висота картинки (для getMinHeight), px
const AUTO_MAXH = 480;   // одноразове авто-розміру ноди: картинка не вища за це, px
const START_W = 460;
const TRACK_H = 16;      // доріжка кліпу, px
const STATUS_H = 40;     // рядок статусу (смужка + текст + підказка), px
const BAR_H = 2 * CTRL_H + 6;
const ERR = C.err;
const PLAY_TITLE = "Відтворити фрагмент по колу (швидкість — fps ноди)";
const HINT = "Clarity / texture / смуги на малому прев'ю видно слабше — точно покаже Run.";
const EMPTY_NO_SRC = "Під'єднай depth від ноди Depth Grade";
const EMPTY_NO_RUN = "Натисни Run один раз — далі прев'ю відео рахується саме, без Run.";

function clampStart(s, total, count) {
    return Math.max(0, Math.min(Math.floor(s) || 0, Math.max(0, total - count)));
}

// base64 JPEG → Image; у списку null, якщо кадр не декодувався
function loadAll(list) {
    return Promise.all((list || []).map((b64) => new Promise((ok) => {
        const im = new Image();
        im.onload = () => ok(im);
        im.onerror = () => ok(null);
        im.src = "data:image/jpeg;base64," + b64;
    })));
}

function makeSelect(opts, title, onChange) {
    const s = document.createElement("select");
    s.className = "svk-sel";
    s.title = title;
    for (const [v, t] of opts) {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = t;
        s.append(o);
    }
    s.onchange = () => onChange(s.value);
    return s;
}

// підказка про clarity/texture/смуги — лише коли вони ввімкнені
function hintNeeded(A) {
    if (!A) return false;
    return ["clarity", "texture", "smooth_banding"].some((k) => Number(widgetOf(A, k)?.value) > 0);
}

class VideoPanel {
    constructor(node) {
        ensureStyles();
        this.node = node;
        this.A = null;           // нода Depth Grade за входом depth
        this.raw = null;         // payload превью (cache_key, frames_total, …) — ідентичність
        this.rawSeq = 0;         // лічильник появ payload (входить у підпис)
        this.sig = "";           // підпис усього, що впливає на кадри
        this.reqSig = "";        // підпис останнього поставленого в чергу запиту
        this.gen = Date.now();   // лічильник запитів: монотонний і між перезавантаженнями сторінки
        this.appliedGen = 0;     // останній застосований gen
        this.busy = false;
        this.noCache = false;    // сервер: кеш цього Run вже немає
        this.errMsg = "";
        this.meta = null;        // { w, h, start, count, total, ms } останнього застосованого набору
        this.after = [];         // Image[] результату
        this.before = [];        // Image[] сирого депсу (для До/після)
        this.vi = 0;             // індекс кадру у фрагменті
        this.wantVi = null;      // куди стати після переходу до нового фрагмента
        this.playing = false;
        this.playRaf = 0;
        this.lastT = null;
        this.acc = 0;
        this.compare = false;
        this.dragSplit = false;
        this.dragTrack = false;
        this.iconState = "";
        this.dbT = 0;
        this.timer = 0;
        this.disposed = false;

        this.root = h("div", `position:relative;display:flex;flex-direction:column;gap:${GAP}px;width:100%;height:100%;box-sizing:border-box;overflow:hidden;color:${C.text};font:12px system-ui,sans-serif`);
        // не тягати вузол замість маркера; колесо не чіпаємо
        for (const ev of ["pointerdown", "pointerup", "click", "dblclick"])
            this.root.addEventListener(ev, (e) => e.stopPropagation());

        // ── рядок 1: Пуск/Пауза і лічильник ліворуч; До/після, розмір, фрагмент праворуч
        const r1 = h("div");
        r1.className = "svk-row";
        this.playBtn = icon("play", PLAY_TITLE, () => this.togglePlay());
        this.counter = h("span");
        this.counter.className = "svk-num";
        const end = h("div");
        end.className = "svk-end";
        this.cmp = icon("split", "До / після: ліва частина — вхід без грейду; лінію можна тягнути", () => this.toggleCompare());
        this.sizeSel = makeSelect(SIZES.map((s) => [String(s), `${s} px`]), "Розмір прев'ю (довша сторона)",
            (v) => { this.setProp("svkPreviewSize", +v); this.queue(); });
        this.fragSel = makeSelect(FRAGMENTS.map((n) => [String(n), `${n} кадрів`]), "Скільки кадрів у фрагменті",
            (v) => { this.setProp("svkFragment", +v); this.queue(); });
        end.append(this.cmp, this.sizeSel, this.fragSel);
        r1.append(this.playBtn, this.counter, end);

        // ── рядок 2: пресети кривої
        this.presets = presetRow(() => this.A);
        this.bar = h("div", "display:flex;flex-direction:column;gap:6px;flex:none");
        this.bar.append(r1, this.presets.el);

        // ── картинка: вписується contain у доступну зону
        this.pic = h("div", "position:relative;flex:1 1 auto;min-height:0");
        this.emptyEl = h("div", `position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;padding:0 14px;box-sizing:border-box;color:${C.dim};font:11px system-ui,sans-serif;line-height:1.5;border:1px dashed ${C.line};border-radius:8px`);
        this.emptyTxt = h("span", "max-width:32ch");
        this.emptyEl.append(this.emptyTxt);
        this.stage = h("div", `position:absolute;display:none;overflow:hidden;border-radius:8px;background:${C.bg};box-shadow:0 0 0 1px ${C.line}`);
        this.canvas = h("canvas", "position:absolute;left:0;top:0;display:block");
        this.divider = h("div", "position:absolute;top:0;bottom:0;width:14px;margin-left:-7px;cursor:ew-resize;touch-action:none;display:none");
        this.divider.append(h("div", `position:absolute;top:0;bottom:0;left:6px;width:2px;background:${C.accent}`));
        this.stage.append(this.canvas, this.divider);
        this.pic.append(this.emptyEl, this.stage);
        this.divider.addEventListener("pointerdown", (e) => {
            this.dragSplit = true;
            this.divider.setPointerCapture(e.pointerId);
            e.preventDefault();
        });
        this.divider.addEventListener("pointermove", (e) => this.onSplitMove(e));
        const endSplit = (e) => {
            if (!this.dragSplit) return;
            this.dragSplit = false;
            try { this.divider.releasePointerCapture(e.pointerId); } catch (_) { /* уже відпущено */ }
        };
        this.divider.addEventListener("pointerup", endSplit);
        this.divider.addEventListener("pointercancel", endSplit);

        // ── доріжка всього кліпу: підсвічений фрагмент, поточний кадр
        this.trackWrap = h("div", `position:relative;flex:none;height:${TRACK_H}px;cursor:pointer;touch-action:none;display:none`);
        const base = h("div", `position:absolute;left:0;right:0;top:6px;height:4px;border-radius:2px;background:${C.line}`);
        this.trackSeg = h("div", `position:absolute;top:4px;height:8px;border-radius:3px;background:${C.accentDim};box-shadow:0 0 0 1px ${C.accent}`);
        this.trackHead = h("div", `position:absolute;top:0;width:2px;height:${TRACK_H}px;margin-left:-1px;border-radius:1px;background:${C.text}`);
        this.trackWrap.append(base, this.trackSeg, this.trackHead);
        this.trackWrap.addEventListener("pointerdown", (e) => this.onTrackDown(e));
        this.trackWrap.addEventListener("pointermove", (e) => this.onTrackMove(e));
        this.trackWrap.addEventListener("pointerup", (e) => this.onTrackUp(e));
        this.trackWrap.addEventListener("pointercancel", (e) => this.onTrackUp(e));

        // ── статус: смужка обчислення, рядок стану, підказка
        this.status = h("div", `flex:none;min-height:${STATUS_H}px;display:flex;flex-direction:column;gap:3px;box-sizing:border-box`);
        this.busyEl = h("div");
        this.busyEl.className = "svk-busy";
        this.busyEl.append(h("i"));
        this.statusLine = h("div", `font:10px ${MONO};color:${C.dim};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-height:12px`);
        this.hintLine = h("div", `font:10px system-ui,sans-serif;color:${C.dim};line-height:1.35;display:none`, HINT);
        this.status.append(this.busyEl, this.statusLine, this.hintLine);

        this.root.append(this.bar, this.pic, this.trackWrap, this.status);

        this.ro = new ResizeObserver(() => this.layout());
        this.ro.observe(this.pic);
        this.timer = setInterval(() => this.tick(), POLL_MS);
        registerPanel(this);
        this.paint();
    }

    dispose() {
        this.disposed = true;
        clearInterval(this.timer);
        clearTimeout(this.dbT);
        this.ro?.disconnect();
        unregisterPanel(this);
        this.pause();
        this.after = [];
        this.before = [];
    }

    // ── налаштування, що живуть у properties ноди (зберігаються у граф)
    prop(name, def) {
        this.node.properties ??= {};
        return this.node.properties[name] ?? def;
    }

    setProp(name, v) {
        this.node.properties ??= {};
        this.node.properties[name] = v;
    }

    size() {
        const s = +this.prop("svkPreviewSize", DEFAULT_SIZE);
        return SIZES.includes(s) ? s : DEFAULT_SIZE;
    }

    count() {
        const n = +this.prop("svkFragment", DEFAULT_FRAGMENT);
        return FRAGMENTS.includes(n) ? n : DEFAULT_FRAGMENT;
    }

    start() { return Math.max(0, Math.floor(+this.prop("svkStart", 0)) || 0); }

    splitPos() { return Math.min(1, Math.max(0, +this.prop("svkSplit", 0.5))); }

    fps() {
        const v = +widgetOf(this.node, "fps")?.value;
        return v > 0 ? Math.min(120, v) : 24;
    }

    // ── опитування: джерело, payload, підпис → запит (після паузи)
    schedule() { this.tick(); }

    tick() {
        const A = findSource(this.node);
        const raw = A ? rawOf(A) : null;
        if (raw !== this.raw) {
            this.raw = raw;
            if (raw) this.rawSeq++;
            this.noCache = false;
            this.errMsg = "";
        }
        this.A = A;
        const ready = !!A && !!raw?.cache_key;
        const sig = ready ? this.signature() : "";
        if (sig !== this.sig) {
            this.sig = sig;
            if (ready && sig !== this.reqSig) {
                this.reqSig = sig;
                this.queue();
            }
        }
        this.presets.refresh();
        this.paint();
    }

    signature() {
        return JSON.stringify([this.rawSeq, this.size(), this.start(), this.count(), paramValues(this.A)]);
    }

    queue() {
        clearTimeout(this.dbT);
        this.dbT = setTimeout(() => { this.dbT = 0; this.request(); }, DEBOUNCE_MS);
    }

    // ── запит до маршруту; відповідь зі старішим gen ігнорується
    async request() {
        const A = this.A, raw = this.raw;
        if (this.disposed || !A || !raw?.cache_key) return;
        const gen = ++this.gen;
        const body = {
            key: String(raw.cache_key),
            gen,
            size: this.size(),
            start: this.start(),
            count: this.count(),
            params: paramValues(A),
        };
        this.busy = true;
        this.paint();
        try {
            const res = await api.fetchApi(ROUTE, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            if (!res.ok) throw new Error(`сервер відповів ${res.status}`);
            const j = await res.json();
            if (this.disposed || gen < this.appliedGen) return;
            if (!j.ok) {
                if (j.reason === "superseded") return;
                if (j.reason === "no_cache") this.noCache = true;
                else this.errMsg = j.message || "помилка обчислення";
                return;
            }
            await this.apply(j, gen);
        } catch (e) {
            if (!this.disposed && gen >= this.appliedGen) this.errMsg = String(e?.message || e);
        } finally {
            if (gen === this.gen) this.busy = false;
            this.paint();
        }
    }

    async apply(j, gen) {
        const [after, before] = await Promise.all([loadAll(j.after), loadAll(j.before)]);
        if (this.disposed || gen < this.appliedGen) return;
        if (!after.length || after.some((im) => !im)) throw new Error("кадри не декодувалися");
        this.appliedGen = gen;
        this.after = after;
        this.before = before;
        this.meta = {
            w: +j.w,
            h: +j.h,
            start: +j.start || 0,
            count: after.length,
            total: +j.frames_total || 0,
            ms: +j.ms || 0,
        };
        this.vi = this.wantVi != null ? this.wantVi : Math.min(this.vi, after.length - 1);
        this.wantVi = null;
        this.vi = Math.min(Math.max(0, this.vi), after.length - 1);
        this.noCache = false;
        this.errMsg = "";
        this.autoSize();
        this.layout();
        this.draw();
    }

    // одноразово (на кожну пропорцію кадру, у самій ноді): картинка вписується в ≤ AUTO_MAXH
    autoSize() {
        const m = this.meta;
        if (!m || !m.w || !m.h) return;
        const node = this.node;
        const ar = m.w / m.h;
        const key = Math.round(ar * 100) / 100;
        node.properties ??= {};
        if (node.properties.svkVidAr === key) return;
        const W = this.root.clientWidth, H = this.root.clientHeight;
        if (!W || !H) return;   // ще не на екрані — спробуємо на наступному результаті
        node.properties.svkVidAr = key;
        const pic = Math.min(AUTO_MAXH, W / ar);
        const want = (this.bar.offsetHeight || BAR_H) + GAP + pic + GAP + TRACK_H + GAP + STATUS_H + 8;
        const delta = want - H;
        if (Math.abs(delta) > 2) node.setSize([node.size[0], node.size[1] + delta]);
        app.graph?.setDirtyCanvas(true, true);
    }

    // мінімальна висота блоку — фіксована, не від вмісту
    minH() {
        return (this.bar.offsetHeight || BAR_H) + GAP + MIN_PIC_H + GAP + TRACK_H + GAP + STATUS_H + 8;
    }

    // ── геометрія: картинка contain у зоні pic
    layout() {
        const pw = this.pic.clientWidth, ph = this.pic.clientHeight;
        const m = this.meta;
        if (!m || !this.after.length || !pw || !ph) return;
        const ar = m.w / m.h;
        let W = pw, H = pw / ar;
        if (H > ph) { H = ph; W = ph * ar; }
        place(this.stage, Math.round((pw - W) / 2), Math.round((ph - H) / 2), Math.round(W), Math.round(H));
        this.canvas.style.width = Math.round(W) + "px";
        this.canvas.style.height = Math.round(H) + "px";
        this.placeDivider();
    }

    placeDivider() {
        const on = !!(this.compare && this.after.length && this.before[this.vi]);
        this.divider.style.display = on ? "block" : "none";
        this.divider.style.left = (this.splitPos() * 100) + "%";
    }

    // ── малювання кадру: зображення, До/після, лічильник і доріжка
    draw() {
        const n = this.after.length, m = this.meta;
        if (!n || !m) return;
        this.vi = Math.min(Math.max(0, this.vi), n - 1);
        const W = m.w, H = m.h;
        if (this.canvas.width !== W || this.canvas.height !== H) {
            this.canvas.width = W;
            this.canvas.height = H;
        }
        const ctx = this.canvas.getContext("2d");
        ctx.drawImage(this.after[this.vi], 0, 0, W, H);
        const bf = this.before[this.vi];
        if (this.compare && bf) {
            const xs = Math.round(this.splitPos() * W);
            if (xs > 0) ctx.drawImage(bf, 0, 0, xs, H, 0, 0, xs, H);
        }
        this.placeDivider();
        this.paintFrame();
    }

    paintFrame() {
        const m = this.meta, n = this.after.length;
        if (!m || !n) { setText(this.counter, ""); return; }
        setText(this.counter, m.total
            ? `кадр ${m.start + this.vi + 1} / ${m.total}`
            : `кадр ${this.vi + 1} / ${n}`);
        this.placeTrack();
    }

    placeTrack() {
        const m = this.meta, total = m?.total || 0;
        if (!total) return;
        const c = Math.min(this.count(), total);
        const s = clampStart(this.start(), total, c);
        this.trackSeg.style.left = (s / total * 100) + "%";
        this.trackSeg.style.width = (c / total * 100) + "%";
        this.trackHead.style.left = ((m.start + this.vi + 0.5) / total * 100) + "%";
    }

    // ── доріжка: клік/перетяг усередині фрагмента — кадр; поза ним — новий фрагмент
    frameAt(e) {
        const total = this.meta?.total || 0;
        const r = this.trackWrap.getBoundingClientRect();
        const t = Math.min(0.999999, Math.max(0, (e.clientX - r.left) / Math.max(1, r.width)));
        return Math.floor(t * total);
    }

    onTrackDown(e) {
        if (!this.meta?.total) return;
        this.dragTrack = true;
        this.trackWrap.setPointerCapture(e.pointerId);
        e.preventDefault();
        this.onTrackAt(e);
    }

    onTrackMove(e) {
        if (this.dragTrack) this.onTrackAt(e);
    }

    onTrackUp(e) {
        if (!this.dragTrack) return;
        this.dragTrack = false;
        try { this.trackWrap.releasePointerCapture(e.pointerId); } catch (_) { /* уже відпущено */ }
    }

    onTrackAt(e) {
        const m = this.meta;
        if (!m?.total || !this.after.length) return;
        const f = this.frameAt(e);
        if (f >= m.start && f < m.start + m.count) {       // усередині показаного фрагмента — перехід до кадру
            this.pause();
            this.vi = f - m.start;
            this.draw();
            return;
        }
        // поза фрагментом: новий фрагмент центрується навколо точки
        const c = Math.min(this.count(), m.total);
        const s = clampStart(f - Math.floor(c / 2), m.total, c);
        this.wantVi = f - s;
        this.setProp("svkStart", s);
        this.queue();
        this.placeTrack();
    }

    // ── До / після: лінію можна тягнути (позиція зберігається в svkSplit)
    toggleCompare() {
        this.compare = !this.compare;
        this.draw();
        this.paint();
    }

    onSplitMove(e) {
        if (!this.dragSplit) return;
        const r = this.stage.getBoundingClientRect();
        const v = Math.min(1, Math.max(0, (e.clientX - r.left) / Math.max(1, r.width)));
        this.setProp("svkSplit", Math.round(v * 1000) / 1000);
        this.draw();
    }

    // ── відтворення по колу: таймінг за fps, не setInterval
    togglePlay() {
        if (this.playing) this.pause();
        else this.play();
    }

    play() {
        if (this.after.length < 2) return;
        this.playing = true;
        this.lastT = null;
        this.acc = 0;
        this.paint();
        this.playRaf = requestAnimationFrame((t) => this.loop(t));
    }

    pause() {
        this.playing = false;
        if (this.playRaf) cancelAnimationFrame(this.playRaf);
        this.playRaf = 0;
        this.paint();
    }

    loop(now) {
        this.playRaf = 0;
        if (!this.playing || this.disposed) return;
        if (this.lastT == null) this.lastT = now;
        const dt = Math.min(250, now - this.lastT);   // фон/пауза вкладки не має стрибати кадрами
        this.lastT = now;
        this.acc += dt;
        const frameMs = 1000 / this.fps();
        if (this.acc >= frameMs) {
            const adv = Math.floor(this.acc / frameMs);
            this.acc -= adv * frameMs;
            this.vi = (this.vi + adv) % this.after.length;
            this.draw();
        }
        this.playRaf = requestAnimationFrame((t) => this.loop(t));
    }

    // ── стан → текст, кнопки, видимість; викликається з опитування і з подій
    paint() {
        const A = this.A, n = this.after.length, has = n > 0 && !!this.meta;

        // порожній стан по центру картинки
        let empty = "";
        if (!has) {
            if (!A) empty = EMPTY_NO_SRC;
            else if (!this.raw?.cache_key || this.noCache) empty = EMPTY_NO_RUN;
            else if (this.errMsg) empty = this.errMsg;
            else empty = "Рахую прев'ю…";
        }
        setText(this.emptyTxt, empty);
        this.emptyTxt.style.color = !has && this.errMsg ? ERR : C.dim;
        this.emptyEl.style.display = has ? "none" : "flex";
        this.stage.style.display = has ? "block" : "none";

        // рядок статусу
        let st = "";
        if (this.errMsg) st = this.errMsg;
        else if (this.noCache) st = "Кеш застарів — натисни Run";
        else if (has) {
            const m = this.meta;
            st = `${Math.max(m.w, m.h)} px · ${n} кадрів · ${(m.ms / 1000).toFixed(2).replace(".", ",")} с`;
        }
        if (this.busy) st = st ? `${st} · рахую…` : "рахую…";
        setText(this.statusLine, st);
        this.statusLine.style.color = this.errMsg ? ERR : C.dim;
        this.busyEl.style.visibility = this.busy ? "visible" : "hidden";
        this.hintLine.style.display = has && hintNeeded(A) ? "" : "none";
        this.trackWrap.style.display = has && this.meta.total ? "block" : "none";

        // кнопки і селектори
        const playState = this.playing ? "pause" : "play";
        if (this.iconState !== playState) {
            setIcon(this.playBtn, playState);
            this.iconState = playState;
        }
        this.playBtn.title = this.playing ? "Пауза" : PLAY_TITLE;
        this.playBtn.disabled = n < 2;
        this.cmp.disabled = !has;
        this.cmp.classList.toggle("svk-on", this.compare && has);
        if (this.sizeSel.value !== String(this.size())) this.sizeSel.value = String(this.size());
        if (this.fragSel.value !== String(this.count())) this.fragSel.value = String(this.count());
        if (!has) setText(this.counter, "");
    }
}

function registerVideo(nodeType) {
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onCreated?.apply(this, arguments);
        const panel = new VideoPanel(this);
        this.__svkVideo = panel;
        this.addDOMWidget("video", "svk_depth_video", panel.root, {
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
        this.__svkVideo?.schedule();
        return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
        this.__svkVideo?.dispose();
        this.__svkVideo = null;
        return onRemoved?.apply(this, arguments);
    };
}

app.registerExtension({
    name: "StepVisionKit.DepthGradeVideo",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === VIDEO) registerVideo(nodeType);
    },
});
