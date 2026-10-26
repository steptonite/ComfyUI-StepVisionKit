// Depth Grade View · відео. Два режими перемикача «Статика | Програвання».
//  • Статика (за замовчуванням): один вибраний кадр. Тон рахується в браузері миттєво на кожну зміну
//    повзунка (M.grade, як у View кадр). Сервер смикаємо лише при зміні кадру чи розміру
//    (POST /svk/depth_grade/frame). Clarity, texture, banding і deflicker тут не видно.
//  • Програвання: фрагмент кліпу рахує сервер (POST /svk/depth_grade/preview) — Пуск/Пауза, fps,
//    До/після з сервера. Зміна → 250 мс тиші → запит; відповідь зі старішим gen ігнорується.
// Вигляд і спільні кнопки — з depth_grade_ui.js, тонова математика — з depth_grade_math.js.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as M from "./depth_grade_math.js";
import { T, LANG } from "./svk_i18n.js";
import {
    VIDEO, C, GAP, CTRL_H, MONO, TONE_KEYS, ensureStyles, h, place, setText, icon, btn, setIcon,
    presetRow, findSource, rawOf, paramValues, widgetOf, toneSig, registerPanel, unregisterPanel,
    getDefaults, setWidgetValue,
} from "./depth_grade_ui.js";
import { CurvePanel, CURVE_H, LEGEND_COL, LEGEND_ROW, SIDE_W, ROW_GAP } from "./depth_grade_curve.js";

const ROUTE = "/svk/depth_grade/preview";
const FRAME_ROUTE = "/svk/depth_grade/frame";
const SIZES = [256, 384, 512, 768];
const FRAGMENTS = [24, 48, 96];
const DEFAULT_SIZE = 384;
const DEFAULT_FRAGMENT = 48;
const DEBOUNCE_MS = 250;
const STILL_DEBOUNCE_MS = 150;
const POLL_MS = 120;
const MIN_PIC_H = 160;   // мінімальна висота картинки (для getMinHeight), px
const AUTO_MAXH = 480;   // одноразове авто-розміру ноди: картинка не вища за це, px
const START_W = 460;
const TRACK_H = 16;      // доріжка кліпу, px
const STATUS_H = 40;     // рядок статусу (смужка + текст + підказка), px
const BAR_H = 2 * CTRL_H + 6;
const ERR = C.err;
const PLAY_TITLE = T("Loop the clip fragment (speed = node fps)");
const HINT = T("Clarity / texture / banding look weaker in a small preview — Run shows them exactly.");
const STILL_HINT = T("Still shows the tone: levels, curve, hero, ceiling. Clarity, texture, smooth banding and deflicker are visible in Playback.");
const EMPTY_NO_SRC = T("Connect depth from the Depth Grade node");
const EMPTY_NO_RUN = T("Press Run once — after that the preview computes itself, no Run needed.");

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
    s.className = "dg-sel";
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

// у Статиці підказка показується й тоді, коли ввімкнений deflicker (він не видно без Програвання)
function hintOn(A, still) {
    return hintNeeded(A) || (still && Number(widgetOf(A, "deflicker")?.value) > 0);
}

// тонові повзунки Depth Grade у формі, яку читає M.grade
function toneValues(A) {
    const o = {};
    for (const k of TONE_KEYS) o[k] = widgetOf(A, k)?.value;
    return o;
}

class VideoPanel {
    constructor(node, opts = {}) {
        ensureStyles();
        this.node = node;
        this.embedded = !!opts.embedded;   // у картці самої Depth Grade: джерело - ця нода, розмір ноди не чіпаємо
        this.A = null;           // нода Depth Grade за входом depth
        this.raw = null;         // payload превью (cache_key, frames_total, …) — ідентичність
        this.rawSeq = 0;         // лічильник появ payload (входить у підпис)
        this.sig = "";           // Програвання: підпис усього, що впливає на фрагмент
        this.reqSig = "";        // Програвання: підпис останнього поставленого в чергу запиту
        this.gen = Date.now();   // лічильник запитів фрагмента: монотонний і між перезавантаженнями сторінки
        this.appliedGen = 0;     // останній застосований gen фрагмента
        this.busy = false;       // Програвання: запит фрагмента в роботі
        this.stillGen = 0;       // Статика: лічильник запитів кадру
        this.stillBusy = false;  // Статика: запит кадру в роботі
        this.noCache = false;    // сервер: кеш цього Run вже немає
        this.errMsg = "";
        this.meta = null;        // { w, h, start, count, total, ms } того, що зараз на екрані
        this.playMeta = null;    // meta фрагмента Програвання
        this.stillMeta = null;   // meta кадру Статики
        this.after = [];         // Програвання: Image[] результату
        this.before = [];        // Програвання: Image[] сирого депсу
        this.still = null;       // Статика: розкодований кадр (M.decodePreview), один ключ
        this.last = null;        // Статика: останній результат M.grade (для кривих)
        this.showCurve = false;  // «Криві» за замовчуванням згорнуті (як у фото-View)
        this.curveDirty = false; // кривим потрібна перемальовка Статики без зміни тонів
        this.stillSig = "";      // підпис (rawSeq, розмір, кадр) завантаженого кадру
        this.stillReqSig = "";   // підпис останнього кадру, поставленого в чергу
        this.vi = 0;             // індекс кадру у фрагменті (у Статиці завжди 0)
        this.wantVi = null;      // куди стати після переходу до нового фрагмента
        this.playing = false;
        this.playRaf = 0;
        this.lastT = null;
        this.acc = 0;
        this.compare = false;
        this.dragSplit = false;
        this.dragTrack = false;
        this.iconState = "";
        this.toneKey = "";       // підпис тонів (і ноди) — зміна → presets і перемальовка Статики
        this.paintSig = "";      // підпис стану для перемальовки DOM
        this.metaSeq = 0;        // змінюється з кожним новим результатом
        this.dbT = 0;
        this.stillT = 0;
        this.timer = 0;
        this.disposed = false;

        this.root = h("div", `position:relative;display:flex;flex-direction:column;gap:${GAP}px;width:100%;height:100%;box-sizing:border-box;overflow:hidden;color:${C.text};font:12px system-ui,sans-serif`);
        // не тягати вузол замість маркера; колесо не чіпаємо
        for (const ev of ["pointerdown", "click", "dblclick"])
            this.root.addEventListener(ev, (e) => e.stopPropagation());

        // ── рядок 1: перемикач режимів, Пуск/Пауза і лічильник; До/після, розмір, фрагмент праворуч
        const r1 = h("div");
        r1.className = "dg-row";
        this.modeStill = btn(T("Still"), T("One frame: the tone is computed instantly in the browser, no request per slider"),
            () => this.setMode("still"));
        this.modePlay = btn(T("Playback"), T("The server computes a clip fragment: Play/Pause, speed = node fps"),
            () => this.setMode("play"));
        const modeSeg = h("div");
        modeSeg.className = "dg-seg";
        modeSeg.append(this.modeStill, this.modePlay);
        this.playBtn = icon("play", PLAY_TITLE, () => this.togglePlay());
        this.counter = h("span");
        this.counter.className = "dg-num";
        const end = h("div");
        end.className = "dg-end";
        this.cmp = icon("split", T("Before / after: the left part is the input without the grade; the line can be dragged"), () => this.toggleCompare());
        this.sizeSel = makeSelect(SIZES.map((s) => [String(s), `${s} px`]), T("Preview size (long side)"),
            (v) => { this.setProp("svkPreviewSize", +v); this.queue(); });
        this.fragSel = makeSelect(FRAGMENTS.map((n) => [String(n), T("{n} frames").replace("{n}", n)]), T("How many frames in the fragment"),
            (v) => { this.setProp("svkFragment", +v); this.queue(); });
        this.curveBtn = icon("curve", T("Curves: histogram, curve and B / W / H / F markers (Still mode)"), () => this.toggleCurve());
        end.append(this.cmp, this.curveBtn, this.sizeSel, this.fragSel);
        r1.append(modeSeg, this.playBtn, this.counter, end);

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
        // криві: полотно і легенда живуть у pic, розкладка — у layout(); хост — цей VideoPanel
        const panel = this;
        this.curves = new CurvePanel({
            widget: (name) => widgetOf(panel.A, name),
            set: (name, value) => setWidgetValue(panel.A, name, value),
            reset: (names) => {
                const d = getDefaults();
                for (const n of names) if (n in d) setWidgetValue(panel.A, n, d[n]);
                panel.curveRedraw();
            },
            schedule: () => panel.curveRedraw(),
            get last() { return panel.last; },
        });
        this.curve = this.curves.curve;
        this.legend = this.curves.legend;
        this.pic.append(this.emptyEl, this.stage, this.curve, this.legend);
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

        // ── доріжка всього кліпу: у Статиці клік = кадр; у Програванні — підсвічений фрагмент і поточний кадр
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
        this.busyEl.className = "dg-busy";
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

    // відкладене опитування (після зміни тонів або завантаження конфігу); кілька викликів = один кадр
    schedule() {
        if (this.schedRaf || this.disposed) return;
        this.schedRaf = requestAnimationFrame(() => {
            this.schedRaf = 0;
            if (!this.disposed) this.tick();
        });
    }

    dispose() {
        this.disposed = true;
        clearInterval(this.timer);
        if (this.schedRaf) cancelAnimationFrame(this.schedRaf);
        this.schedRaf = 0;
        clearTimeout(this.dbT);
        clearTimeout(this.stillT);
        this.ro?.disconnect();
        unregisterPanel(this);
        this.pause();
        this.after = [];
        this.before = [];
        this.still = null;
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

    mode() { return this.prop("svkVidMode", "still") === "play" ? "play" : "still"; }

    isStill() { return this.mode() === "still"; }

    size() {
        const s = +this.prop("svkPreviewSize", DEFAULT_SIZE);
        return SIZES.includes(s) ? s : DEFAULT_SIZE;
    }

    count() {
        const n = +this.prop("svkFragment", DEFAULT_FRAGMENT);
        return FRAGMENTS.includes(n) ? n : DEFAULT_FRAGMENT;
    }

    start() { return Math.max(0, Math.floor(+this.prop("svkStart", 0)) || 0); }

    frame() { return Math.max(0, Math.floor(+this.prop("svkFrame", 0)) || 0); }

    splitPos() { return Math.min(1, Math.max(0, +this.prop("svkSplit", 0.5))); }

    fps() {
        const v = +widgetOf(this.node, "fps")?.value;
        return v > 0 ? Math.min(120, v) : 24;
    }

    // загальна кількість кадрів кліпу (з payload або з того, що вже показано)
    total() { return +this.raw?.frames_total || this.stillMeta?.total || this.playMeta?.total || 0; }

    busyNow() { return this.isStill() ? this.stillBusy : this.busy; }

    hasImage() { return !!this.meta && (this.isStill() ? !!this.still : this.after.length > 0); }

    // кадр, що зараз на екрані (або вибраний) — переноситься при перемиканні режиму
    curFrame() {
        if (this.meta && this.hasImage()) return this.meta.start + (this.isStill() ? 0 : this.vi);
        return this.frame();
    }

    // ── опитування: джерело, payload, тони → режим; перемальовка DOM лише при зміні підпису
    tick() {
        const A = this.embedded ? this.node : findSource(this.node);
        const raw = A ? rawOf(A) : null;
        if (raw !== this.raw) {
            this.raw = raw;
            if (raw) this.rawSeq++;
            this.noCache = false;
            this.errMsg = "";
        }
        this.A = A;
        const ready = !!A && !!raw?.cache_key;
        const tk = A ? `${A.id}|${toneSig(A)}` : "";
        if (tk !== this.toneKey || this.curveDirty) {
            this.curveDirty = false;
            this.toneKey = tk;
            this.presets.refresh();
            if (this.isStill() && this.still) this.drawStill();
        }
        if (this.isStill()) this.tickStill(ready);
        else this.tickPlay(ready);
        const pk = this.paintKey();
        if (pk !== this.paintSig) {
            this.paintSig = pk;
            this.paint();
        }
    }

    paintKey() {
        return JSON.stringify([this.mode(), !!this.A, !!this.raw?.cache_key, this.noCache, this.errMsg,
            this.busyNow(), this.playing, this.compare, this.hasImage(), this.size(), this.count(),
            this.meta?.total || 0, this.metaSeq, this.after.length, hintOn(this.A, this.isStill())]);
    }

    // Статика: запит кадру лише при зміні кадру/розміру/рівня; тони → миттєвий перерахунок у браузері
    tickStill(ready) {
        if (!ready) return;
        const fs = this.frameSig();
        if (fs === this.stillReqSig) return;
        if (fs === this.stillSig && this.still) {
            this.stillReqSig = fs;
            this.drawStill();
            return;
        }
        this.queueStill();
    }

    // Програвання: як раніше — підпис фрагмента → запит
    tickPlay(ready) {
        const sig = ready ? this.signature() : "";
        if (sig !== this.sig) {
            this.sig = sig;
            if (ready && sig !== this.reqSig) {
                this.reqSig = sig;
                this.queue();
            }
        }
    }

    signature() {
        return JSON.stringify([this.rawSeq, this.size(), this.start(), this.count(), paramValues(this.A)]);
    }

    frameSig() { return JSON.stringify([this.rawSeq, this.size(), this.frame()]); }

    queue() {
        if (this.isStill()) { this.queueStill(); return; }
        clearTimeout(this.dbT);
        this.dbT = setTimeout(() => { this.dbT = 0; this.request(); }, DEBOUNCE_MS);
    }

    // доріжка тягнеться: якщо таймер уже чекає, він візьме актуальний кадр (оновлення раз на 150 мс)
    queueStill() {
        this.stillReqSig = this.frameSig();
        if (this.stillT) return;
        this.stillT = setTimeout(() => { this.stillT = 0; this.requestStill(); }, STILL_DEBOUNCE_MS);
    }

    // ── Статика: один кадр; відповідь зі старішим gen ігнорується
    async requestStill() {
        const A = this.A, raw = this.raw;
        if (this.disposed || !this.isStill() || !A || !raw?.cache_key) return;
        const sig = this.frameSig();
        const gen = ++this.stillGen;
        this.stillBusy = true;
        this.paint();
        try {
            const res = await api.fetchApi(FRAME_ROUTE, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ key: String(raw.cache_key), i: this.frame(), size: this.size() }),
            });
            if (!res.ok) throw new Error(T("server replied {s}").replace("{s}", res.status));
            const j = await res.json();
            if (this.disposed || gen !== this.stillGen || !this.isStill()) return;
            if (!j.ok) {
                if (j.reason === "no_cache") this.noCache = true;
                else this.errMsg = j.message || T("computation error");
                return;
            }
            const s = M.decodePreview(j);
            this.still = s;
            this.stillSig = sig;
            this.stillMeta = { w: s.w, h: s.h, start: s.keys[0].i, count: 1, total: s.frames, ms: 0 };
            this.meta = this.stillMeta;
            this.vi = 0;
            this.noCache = false;
            this.errMsg = "";
            this.metaSeq++;
            this.autoSize();
            this.drawStill();
        } catch (e) {
            if (!this.disposed && gen === this.stillGen && this.isStill()) this.errMsg = String(e?.message || e);
        } finally {
            if (gen === this.stillGen) this.stillBusy = false;
            this.paint();
        }
    }

    // ── Програвання: один запит фрагмента; відповідь зі старішим gen ігнорується
    async request() {
        const A = this.A, raw = this.raw;
        if (this.disposed || this.isStill() || !A || !raw?.cache_key) return;
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
            if (!res.ok) throw new Error(T("server replied {s}").replace("{s}", res.status));
            const j = await res.json();
            if (this.disposed || this.isStill() || gen < this.appliedGen) return;
            if (!j.ok) {
                if (j.reason === "superseded") return;
                if (j.reason === "no_cache") this.noCache = true;
                else this.errMsg = j.message || T("computation error");
                return;
            }
            await this.apply(j, gen);
        } catch (e) {
            if (!this.disposed && !this.isStill() && gen >= this.appliedGen) this.errMsg = String(e?.message || e);
        } finally {
            if (gen === this.gen) this.busy = false;
            this.paint();
        }
    }

    async apply(j, gen) {
        const [after, before] = await Promise.all([loadAll(j.after), loadAll(j.before)]);
        if (this.disposed || this.isStill() || gen < this.appliedGen) return;
        if (!after.length || after.some((im) => !im)) throw new Error(T("frames failed to decode"));
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
        this.playMeta = this.meta;
        this.metaSeq++;
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
        if (this.embedded || !m || !m.w || !m.h) return;
        const node = this.node;
        const ar = m.w / m.h;
        const key = Math.round(ar * 100) / 100;
        node.properties ??= {};
        if (node.properties.svkVidAr === key) return;
        const W = this.root.clientWidth, H = this.root.clientHeight;
        if (!W || !H) return;   // ще не на екрані — спробуємо на наступному результаті
        node.properties.svkVidAr = key;
        // криві відкриті — місце під них: збоку (портрет) або під кадром (ландшафт), як у фото-View
        const curves = this.curvesOn();
        const body = curves && ar < 0.95
            ? Math.max(Math.min(AUTO_MAXH, (W - SIDE_W - ROW_GAP) / ar), CURVE_H + GAP + LEGEND_ROW)
            : Math.min(AUTO_MAXH, W / ar) + (curves ? GAP + CURVE_H + GAP + LEGEND_COL : 0);
        const want = (this.bar.offsetHeight || BAR_H) + GAP + body + GAP + TRACK_H + GAP + STATUS_H + 8;
        const delta = want - H;
        if (Math.abs(delta) > 2) node.setSize([node.size[0], node.size[1] + delta]);
        app.graph?.setDirtyCanvas(true, true);
    }

    // мінімальна висота блоку — фіксована, не від вмісту
    minH() {
        return (this.bar.offsetHeight || BAR_H) + GAP + MIN_PIC_H + GAP + TRACK_H + GAP + STATUS_H + 8
            + (this.isStill() && this.showCurve ? CURVE_H + GAP + LEGEND_COL + GAP : 0);
    }

    // криві показані лише в Статиці, коли кадр є
    curvesOn() { return this.isStill() && this.showCurve && this.hasImage(); }

    // «Криві»: перемикач у Статиці; кадр перемальовується (з кривими або без них)
    toggleCurve() {
        this.showCurve = !this.showCurve;
        this.layout();
        if (this.isStill()) this.drawStill();
        this.paint();
    }

    // кривим потрібна перемальовка навіть без зміни тонів (маркер підсвічується, тягнеться)
    curveRedraw() {
        this.curveDirty = true;
        this.schedule();
    }

    // ── геометрія: картинка contain у зоні pic; криві — під кадром (ландшафт) або збоку (портрет)
    layout() {
        const pw = this.pic.clientWidth, ph = this.pic.clientHeight;
        const m = this.meta;
        const curves = this.curvesOn();
        this.curve.style.display = curves ? "block" : "none";
        this.legend.style.display = curves ? "block" : "none";
        if (!m || !this.hasImage() || !pw || !ph) return;
        const ar = m.w / m.h;
        let W, H, x, y;
        if (curves && ar < 0.95) {
            // портрет: кадр ліворуч, колонка кривих праворуч
            const areaW = pw - SIDE_W - ROW_GAP;
            H = Math.min(ph, areaW / ar); W = H * ar;
            x = Math.round((areaW - W) / 2);
            y = Math.round((ph - H) / 2);
            const y0 = Math.round((ph - (CURVE_H + GAP + LEGEND_ROW)) / 2);
            place(this.curve, pw - SIDE_W, y0, SIDE_W, CURVE_H);
            place(this.legend, pw - SIDE_W, y0 + CURVE_H + GAP, SIDE_W, LEGEND_ROW);
        } else {
            // ландшафт: кадр, під ним криві й легенда; разом центруються по висоті
            const under = curves ? GAP + CURVE_H + GAP + LEGEND_COL : 0;
            const avail = under ? Math.max(60, ph - under) : ph;
            H = Math.min(avail, pw / ar); W = H * ar;
            const top = Math.round((ph - H - under) / 2);
            x = Math.round((pw - W) / 2);
            y = top;
            if (curves) {
                place(this.curve, 0, top + H + GAP, pw, CURVE_H);
                place(this.legend, 0, top + H + GAP + CURVE_H + GAP, pw, LEGEND_COL);
            }
        }
        place(this.stage, x, y, Math.round(W), Math.round(H));
        this.canvas.style.width = Math.round(W) + "px";
        this.canvas.style.height = Math.round(H) + "px";
        this.placeDivider();
    }

    placeDivider() {
        const on = !!(this.compare && this.hasImage() && (this.isStill() || this.before[this.vi]));
        this.divider.style.display = on ? "block" : "none";
        this.divider.style.left = (this.splitPos() * 100) + "%";
    }

    // ── малювання: Статика або Програвання
    draw() { return this.isStill() ? this.drawStill() : this.drawPlay(); }

    // Статика: тон рахує M.grade на вибраному кадрі; «до» — сирий кадр, нормалізований як у View кадр
    drawStill() {
        const s = this.still, A = this.A, m = this.stillMeta;
        if (!s || !A || !m) return;
        this.meta = m;
        const g = M.grade(s, toneValues(A), 0);
        this.last = g;
        const key = s.keys[0];
        const W = s.w, H = s.h, n = W * H;
        if (this.canvas.width !== W || this.canvas.height !== H) {
            this.canvas.width = W;
            this.canvas.height = H;
        }
        const split = this.compare ? Math.round(this.splitPos() * W) : 0;
        let xmin = Infinity, xmax = -Infinity;
        if (split > 0) for (const v of key.x) { if (v < xmin) xmin = v; if (v > xmax) xmax = v; }
        const span = Math.max(xmax - xmin, 1e-9);
        const ctx = this.canvas.getContext("2d");
        const img = ctx.createImageData(W, H);
        for (let i = 0; i < n; i++) {
            const v = (i % W) < split ? (key.x[i] - xmin) / span : g.out[i];
            const b = v * 255;
            img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = b;
            img.data[4 * i + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
        this.layout();
        if (this.curvesOn()) this.curves.draw(g);   // після layout: полотно вже має свій розмір
        this.paintFrame();
    }

    // Програвання: кадр фрагмента з сервера
    drawPlay() {
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
        const m = this.meta;
        if (!m || !this.hasImage()) { setText(this.counter, ""); return; }
        const n = this.isStill() ? 1 : this.after.length;
        setText(this.counter, m.total
            ? T("frame {i} of {n}").replace("{i}", m.start + this.vi + 1).replace("{n}", m.total)
            : T("frame {i} of {n}").replace("{i}", this.vi + 1).replace("{n}", n));
        this.placeTrack();
    }

    placeTrack() {
        const m = this.meta, total = m?.total || 0;
        if (!total) return;
        this.trackHead.style.left = ((m.start + this.vi + 0.5) / total * 100) + "%";
        if (this.isStill()) { this.trackSeg.style.display = "none"; return; }
        this.trackSeg.style.display = "block";
        const c = Math.min(this.count(), total);
        const s = clampStart(this.start(), total, c);
        this.trackSeg.style.left = (s / total * 100) + "%";
        this.trackSeg.style.width = (c / total * 100) + "%";
    }

    // ── доріжка: Статика — будь-яка точка = кадр; Програвання — усередині фрагмента кадр, поза ним новий фрагмент
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
        if (!m?.total) return;
        const f = this.frameAt(e);
        if (this.isStill()) {
            this.setProp("svkFrame", f);
            this.queueStill();
            this.placeTrack();
            return;
        }
        if (!this.after.length) return;
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

    // ── перемикання режиму: вибраний кадр переходить із Програвання у Статику і назад
    setMode(m) {
        if (this.disposed || m === this.mode()) return;
        const frame = this.curFrame();
        this.pause();
        clearTimeout(this.dbT);
        this.dbT = 0;
        clearTimeout(this.stillT);
        this.stillT = 0;
        this.errMsg = "";
        this.noCache = false;
        this.setProp("svkVidMode", m);
        this.setProp("svkFrame", frame);
        const A = this.A, raw = this.raw;
        const ready = !!A && !!raw?.cache_key;
        if (m === "still") {
            this.meta = this.still ? this.stillMeta : null;
            this.metaSeq++;
            this.stillReqSig = "";
            this.tick();
        } else {
            this.meta = this.after.length ? this.playMeta : null;
            this.last = null;   // криві у Програванні не показуємо
            this.metaSeq++;
            if (ready) {
                // один запит фрагмента зі start так, щоб вибраний кадр був у ньому
                const total = this.total();
                const c = total ? Math.min(this.count(), total) : this.count();
                const s = clampStart(frame - Math.floor(c / 2), total, c);
                this.setProp("svkStart", s);
                this.wantVi = frame - s;
                this.sig = this.signature();
                this.reqSig = this.sig;
                this.queue();
            }
        }
        this.layout();
        this.paintSig = "";
        this.paint();
    }

    // ── Статика: доріжка й розмір міняють кадр — запит тільки тут
    // ── Програвання: Пуск/Пауза — таймінг за fps, не setInterval
    togglePlay() {
        if (this.playing) this.pause();
        else this.play();
    }

    play() {
        if (this.isStill() || this.after.length < 2) return;
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

    // ── стан → текст, кнопки, видимість; викликається з опитування і з подій
    paint() {
        this.presets.refresh();
        const A = this.A, still = this.isStill(), n = this.after.length, has = this.hasImage();
        const busy = this.busyNow();

        // порожній стан по центру картинки
        let empty = "";
        if (!has) {
            if (!A) empty = EMPTY_NO_SRC;
            else if (!this.raw?.cache_key || this.noCache) empty = EMPTY_NO_RUN;
            else if (this.errMsg) empty = this.errMsg;
            else empty = T("Computing preview…");
        }
        setText(this.emptyTxt, empty);
        this.emptyTxt.style.color = !has && this.errMsg ? ERR : C.dim;
        this.emptyEl.style.display = has ? "none" : "flex";
        this.stage.style.display = has ? "block" : "none";

        // рядок статусу
        let st = "";
        if (this.errMsg) st = this.errMsg;
        else if (this.noCache) st = T("Cache is stale — press Run");
        else if (has) {
            const m = this.meta;
            const px = `${Math.max(m.w, m.h)} px`;
            st = still
                ? T("{px} · still").replace("{px}", px)
                : T("{px} · {n} frames · {s} s").replace("{px}", px).replace("{n}", n).replace("{s}", LANG === "uk" ? (m.ms / 1000).toFixed(2).replace(".", ",") : (m.ms / 1000).toFixed(2));
        }
        if (busy) st = st ? `${st} · ${T("computing…")}` : T("computing…");
        setText(this.statusLine, st);
        this.statusLine.style.color = this.errMsg ? ERR : C.dim;
        this.busyEl.style.visibility = busy ? "visible" : "hidden";
        setText(this.hintLine, still ? STILL_HINT : HINT);
        this.hintLine.style.display = has && hintOn(A, still) ? "" : "none";
        this.trackWrap.style.display = has && this.meta.total ? "block" : "none";

        // режим, кнопки і селектори
        this.modeStill.classList.toggle("dg-on", still);
        this.modePlay.classList.toggle("dg-on", !still);
        this.playBtn.style.display = still ? "none" : "";
        this.fragSel.style.display = still ? "none" : "";
        const playState = this.playing ? "pause" : "play";
        if (this.iconState !== playState) {
            setIcon(this.playBtn, playState);
            this.iconState = playState;
        }
        this.playBtn.title = this.playing ? T("Pause") : PLAY_TITLE;
        this.playBtn.disabled = n < 2;
        this.cmp.disabled = !has;
        this.curveBtn.style.display = still ? "" : "none";
        this.curveBtn.classList.toggle("dg-on", still && this.showCurve);
        this.cmp.classList.toggle("dg-on", this.compare && has);
        if (this.sizeSel.value !== String(this.size())) this.sizeSel.value = String(this.size());
        if (this.fragSel.value !== String(this.count())) this.fragSel.value = String(this.count());
        if (!has) setText(this.counter, "");
    }
}

// Відео-превʼю в картці Depth Grade - та сама панель, що в ноді View · відео, без окремої ноди (fps 24).
// Панель живе, лише поки розгортка відкрита: закрив - опитування й запити зупиняються.
export const EMBED_H = 380;
export function videoSection(node) {
    const det = document.createElement("details");
    det.className = "svk-more svk-dg-video";
    const sum = document.createElement("summary");
    sum.innerHTML = `<span>${T("Video preview")}</span>`;
    sum.title = T("Plays a piece of the clip with the current sliders. Run the node once, then it recomputes by itself");
    const host = h("div", `height:${EMBED_H}px;margin-top:${GAP}px`);
    det.append(sum, host);
    let panel = null, restoring = false;
    const open = () => { if (panel) return; panel = new VideoPanel(node, { embedded: true }); host.append(panel.root); panel.layout(); };
    const close = () => { panel?.dispose(); panel?.root.remove(); panel = null; };
    det.addEventListener("toggle", () => {
        det.open ? open() : close();
        node.properties ??= {};
        node.properties.svkVideoOpen = det.open;
        if (restoring) { restoring = false; return; }          // розмір ноди з workflow вже враховує відкриту панель
        node.setSize([node.size[0], Math.max(80, node.size[1] + (det.open ? EMBED_H : -EMBED_H))]);
        node.setDirtyCanvas?.(true, true);
    });
    const restore = () => { if (node.properties?.svkVideoOpen && !det.open) { restoring = true; det.open = true; } };
    restore();
    return { root: det, restore, extraH: () => (det.open ? EMBED_H : 0), dispose: close };
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
