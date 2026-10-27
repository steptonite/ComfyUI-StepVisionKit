// Спільні дрібниці UI нод StepVisionKit: іконки, підписи луків, приховування нативних віджетів,
// контроли «нативний віджет -> DOM» (повзунок, вибір, перемикач) і панель «⋯ ще».
// Правило паку (CONVENTIONS_UI.md): на самій ноді - жодних абзаців; пояснення живуть у title (підказка на наведенні).
import { app } from "../../scripts/app.js";
import { T } from "./svk_i18n.js";
import { api } from "../../scripts/api.js";

export const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
export const W = (node, name) => node.widgets?.find(w => w.name === name);
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function hideWidget(w) {
    if (!w) return;
    w.hidden = true;
    w.options = Object.assign(w.options || {}, { hidden: true });
    w.computeSize = () => [0, -4];
    w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    if (w.element?.style) w.element.style.display = "none";
}
// Знімок того, що бачить користувач на ноді (назва вже в мові паку, підписи, сховані віджети). Повноекранний режим нічого з цього
// не міняє, але фронтенд міг за цей час перемалювати ноду з англійськими базовими підписами - тож після виходу повертаємо знімок.
export function snapUI(node) {
    return { title: node.title, labels: (node.widgets || []).map(w => [w, w.label, !!w.hidden]) };
}
export function restoreUI(node, s, after) {
    if (!s) return;
    if (s.title != null && node.title !== s.title) node.title = s.title;
    for (const [w, label, hidden] of s.labels) {
        if (w.label !== label) w.label = label;
        if (hidden && !w.hidden) hideWidget(w);
    }
    try { after?.(); } catch (_) {}
    try { app.graph?.setDirtyCanvas(true, true); } catch (_) {}
}
export function markChanged() {
    try { app.graph?.setDirtyCanvas(true, true); } catch (_) {}
    try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState(); } catch (_) {}
}
// Колесо над карткою гортає картку, а не масштабує полотно Comfy.
// Nodes 2.0: обгортка Vue-ноди ловить колесо в capture-фазі й віддає полотну раніше, ніж воно дійде до картки,
// тож перехоплюємо ще вище - на window - і віддаємо адресату самі. Ctrl/щипок лишаються масштабом полотна.
const WHEEL = new WeakMap();
let wheelHooked = false;
export function onWheel(elm, fn) {
    WHEEL.set(elm, fn);
    if (wheelHooked) return;
    wheelHooked = true;
    window.addEventListener("wheel", e => {
        if (e.ctrlKey || e.metaKey) return;
        for (let t = e.target; t && t !== document; t = t.parentElement) {
            const f = WHEEL.get(t);
            if (f) { e.stopPropagation(); f(e); return; }
        }
    }, { capture: true, passive: false });
}
export function keepWheel(e) { onWheel(e, () => {}); }     // рідне гортання картки, без масштабу полотна

// іконки 16px, штрих (як lucide); колір - currentColor
const svg = d => '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + d + "</svg>";
export const ICON = {
    full: svg('<path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="m21 3-7 7"/><path d="m3 21 7-7"/>'),
    play: svg('<path d="M7 4v16l13-8z"/>'),
    pause: svg('<path d="M8 4v16M16 4v16"/>'),
    more: svg('<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>'),
    reset: svg('<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>'),
    chev: svg('<path d="m6 9 6 6 6-6"/>'),
    down: svg('<path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>'),
};

// заголовок «ще» однаковий на всіх нодах: іконка · «ще» · скільки сховано · стрілка праворуч
export function moreSummary(n) {
    const s = el("summary", null, ICON.more + "<span>" + T("more") + "</span>" + (n ? '<span class="n">' + n + "</span>" : "") + '<span class="chev">' + ICON.chev + "</span>");
    s.title = T("{n} more settings").replace("{n}", n);
    return s;
}

// людські назви луків замість імен файлів; новий пресет без запису тут - імʼя файлу без префіксів
const PRESET_LABELS = { "look_v1": T("Look v1"), "test_grade": T("Test"), "00_original": T("Original") };
export function presetLabel(name) {
    return PRESET_LABELS[name] || String(name).replace(/^\d+_/, "").replace(/^look_/, "").replace(/_/g, " ");
}

// значення combo до перекладу (жовтень 2026) -> нові; сервер приймає старі теж (depth_look/core.py LEGACY)
const LEGACY_VALUES = { "як оригінал": "original", "налаштування (швидко, 720)": "tune (fast, 720)", "повний рендер": "full render",
                        "превʼю (швидко)": "preview (fast)", "фінал": "final", "00_оригінал": "00_original" };
export function migrateLegacyValues(node) {
    for (const w of node.widgets || []) if (typeof w.value === "string" && Object.hasOwn(LEGACY_VALUES, w.value)) w.value = LEGACY_VALUES[w.value];
    // старе імʼя пресета з суфіксом (<база>_...) -> базовий пресет, як look_params.preset_alias
    const p = W(node, "preset"), opts = p?.options?.values;
    if (p && Array.isArray(opts) && !opts.includes(p.value)) {
        const base = opts.filter(o => String(p.value).startsWith(o + "_")).sort((a, b) => b.length - a.length)[0];
        if (base) p.value = base;
    }
}

// клік по числу - вписати значення вручну (27.10: вписувати можна всюди).
// Enter/вихід із поля - прийняти, Esc - скасувати; apply отримує число в межах [lo, hi] (scale - у тих одиницях, що на екрані, напр. % = 100)
export function typeIn(val, apply, lo, hi, scale = 1) {
    val.style.cursor = "text";
    val.addEventListener("click", ev => {
        ev.stopPropagation();
        if (!val.isConnected) return;
        const inp = el("input", "svk-num"); inp.type = "text"; inp.inputMode = "decimal";
        inp.value = val.textContent.replace(/[^\d.,\-]/g, "");
        Object.assign(inp.style, { width: Math.max(4, val.textContent.length + 2) + "ch", textAlign: "right", padding: "0 2px",
                                   background: "transparent", border: "1px solid currentColor", borderRadius: "3px", outline: "none" });
        let done = false;
        const finish = ok => {
            if (done) return; done = true;
            const x = Number(inp.value.trim().replace(",", ".")) / scale;
            inp.replaceWith(val);
            if (ok && inp.value.trim() !== "" && Number.isFinite(x)) apply(clamp(x, lo, hi));
        };
        inp.onkeydown = e => { e.stopPropagation(); if (e.key === "Enter") finish(true); else if (e.key === "Escape") finish(false); };
        inp.onblur = () => finish(true);
        for (const k of ["pointerdown", "mousedown", "dblclick", "click"]) inp.addEventListener(k, e => e.stopPropagation());
        val.replaceWith(inp); inp.focus(); inp.select();
    });
}

// ---- контроли, привʼязані до нативного віджета (його значення йде у workflow і в Run)
// spec - рядок INPUT_TYPES з object_info: [тип | список, {min,max,step,tooltip,default}]
export function control(node, name, spec, label, opts = {}) {
    const [type, o = {}] = spec || [];
    const tip = (opts.tip || o.tooltip || "") + (o.default !== undefined && !Array.isArray(type) && type !== "STRING" ?"\n" + T("double-click the label for the default value") : "");
    const get = () => W(node, name)?.value;
    const set = v => { const w = W(node, name); if (w && w.value !== v) { w.value = v; markChanged(); opts.onChange?.(v); } };
    let row, sync;
    if (Array.isArray(type) && opts.seg) {                         // короткий вибір - сегменти
        row = el("div", "svk-seg"); row.title = tip;
        const btns = type.map(v => {
            const b = el("button", null); b.type = "button"; b.textContent = opts.seg[v] || v; b.title = v;
            b.onclick = () => { set(v); sync(); };
            row.append(b); return [v, b];
        });
        sync = () => { const cur = get(); for (const [v, b] of btns) b.classList.toggle("is-on", v === cur); };
    } else if (Array.isArray(type)) {
        row = el("label", "svk-field"); row.title = tip;
        const sel = el("select");
        for (const v of type) sel.append(new Option(opts.names?.[v] || v, v));
        sel.onchange = () => set(sel.value);
        row.append(el("span", "t", label), sel);
        sync = () => { sel.value = get(); };
    } else if (type === "STRING") {                                // текст; для hex-кольору (opts.color) - ще й пікер поруч
        row = el("label", "svk-field svk-text"); row.title = tip;
        const txt = el("input"); txt.type = "text"; txt.spellcheck = false;
        txt.onchange = () => set(txt.value);
        txt.onkeydown = ev => ev.stopPropagation();               // гарячі клавіші Comfy не перехоплюють набір
        row.append(el("span", "t", label), txt);
        let pick = null;
        if (opts.color) {
            pick = el("input"); pick.type = "color";
            pick.oninput = () => { txt.value = pick.value; set(pick.value); };
            row.append(pick);
        }
        sync = () => { const v = get() ?? ""; txt.value = v; if (pick && /^#[0-9a-f]{6}$/i.test(v)) pick.value = v; };
    } else if (type === "BOOLEAN" || (type === "INT" && o.min === 0 && o.max === 1)) {
        const asInt = type === "INT";
        row = el("label", "svk-field"); row.title = tip;
        const cb = el("input"); cb.type = "checkbox";
        cb.onchange = () => set(asInt ? (cb.checked ? 1 : 0) : cb.checked);
        row.append(el("span", "t", label), cb);
        sync = () => { cb.checked = !!get(); };
    } else {
        const lo = Number(o.min ?? 0), hi = Number(o.max ?? 1), step = Number(o.step ?? (type === "INT" ? 1 : 0.01));
        const isInt = type === "INT";
        const dec = isInt ? 0 : Math.min(3, (String(step).split(".")[1] || "").length || 2);
        if ((hi - lo) / step > 4000 || opts.number) {             // великий діапазон (px, fps) - числом, повзунок тут лише заважає
            row = el("label", "svk-field"); row.title = tip;
            const num = el("input"); num.type = "number"; num.min = lo; num.max = hi; num.step = step;
            num.onchange = () => { let v = clamp(Number(num.value) || 0, lo, hi); if (isInt) v = Math.round(v); set(v); sync(); };
            row.append(el("span", "t", label), num);
            sync = () => { num.value = get() ?? ""; };
        } else {
            row = el("div", "svk-sl"); row.title = tip + "\n" + T("click the number to type a value");
            const hd = el("div", "hd"); const t = el("span", "t", label); const val = el("span", "svk-num");
            hd.append(t, val);
            const r = el("input", "svk-range"); r.type = "range"; r.min = lo; r.max = hi; r.step = step;
            const paint = v => { r.value = v; r.style.setProperty("--p", ((v - lo) / (hi - lo || 1)) * 100 + "%"); val.textContent = (opts.fmt ? opts.fmt(v) : Number(v).toFixed(dec)); };
            r.addEventListener("input", () => { const v = isInt ? Math.round(Number(r.value)) : Number(r.value); paint(v); set(v); });
            t.addEventListener("dblclick", () => { if (o.default !== undefined) { set(o.default); paint(o.default); } });
            typeIn(val, v => { if (isInt) v = clamp(Math.round((v - lo) / step) * step + lo, lo, hi); set(v); sync(); }, lo, hi);
            row.append(hd, r);
            sync = () => paint(Number(get() ?? lo));
        }
    }
    sync();
    return { row, sync };
}

// «⋯ ще»: згорнутий список рідкісних налаштувань; items = [[імʼя, підпис, opts?], ...]
export function morePanel(node, specs, items) {
    const det = el("details", "svk-more");
    const body = el("div", "body body-col");
    const ctl = [];
    for (const [name, label, opts] of items) {
        if (!specs[name] || !W(node, name)) continue;
        const c = control(node, name, specs[name], label, opts || {});
        body.append(c.row); ctl.push(c);
    }
    det.append(moreSummary(ctl.length), body);
    // розгорнуте «ще» зберігається у workflow; відкриваємо в sync, бо properties приходять після створення ноди
    det.addEventListener("toggle", () => { node.properties ??= {}; node.properties.svkMoreOpen = det.open; });
    det.sync = () => { if (node.properties?.svkMoreOpen && !det.open) det.open = true; ctl.forEach(c => c.sync()); };
    return det;
}

// ---- картка під розмір ноди
// Nodes 2.0 тягне DOM-віджет під вміст, тож стелю картки ставимо з висоти самої ноди: тягнеш ноду - картка росте, решта гортається.
// В класичному режимі висоту віджета й так дає нода - там це ні на що не впливає.
const NODE_CHROME = 44;          // назва ноди + відступи (класичний режим; у Nodes 2.0 міряється наживо)
const VUE_EXTRA = 6;             // Nodes 2.0: назва внизу (36) мінус висота назви поза node.size (30)
export function fitToNode(node, card) {
    let seen = null;
    const apply = () => {
        const wrap = card.closest("[data-node-id]");
        let chrome = NODE_CHROME;
        if (wrap) {             // Nodes 2.0: над карткою - рядки входів/виходів; міряємо наживо, бо в кожної ноди їх різна кількість
            const k = wrap.getBoundingClientRect().height / (wrap.offsetHeight || 1) || 1;
            chrome = (card.getBoundingClientRect().top - wrap.getBoundingClientRect().top) / k + VUE_EXTRA;
            if (seen !== wrap) { seen = wrap; new ResizeObserver(apply).observe(wrap); }
        }
        const h = Math.max(240, Math.round((node.size?.[1] || 0) - chrome));
        card.style.setProperty("--svk-card-max-h", h + "px");
    };
    requestAnimationFrame(apply);   // обгортка Vue зʼявляється після першого кадру
    const onResize = node.onResize;
    node.onResize = function () { const r = onResize?.apply(this, arguments); apply(); return r; };
    apply();
    return apply;
}

// «нижче ще»: поки картку можна гортати вниз - знизу пігулка, клік гортає на екран нижче
export function scrollHint(card) {
    const pill = el("button", "svk-below", ICON.down + "<span>" + T("more below") + "</span>");
    pill.type = "button"; pill.title = T("scroll the card — more settings below");
    pill.onclick = () => card.scrollBy({ top: card.clientHeight * 0.8, behavior: "smooth" });
    card.append(pill);
    // data-атрибут, не клас: Nodes 2.0 (Vue) перезаписує className віджета при перемальовці
    const upd = () => { card.dataset.below = card.scrollHeight - card.scrollTop - card.clientHeight > 24 ? "1" : ""; };
    card.addEventListener("scroll", upd, { passive: true });
    new ResizeObserver(upd).observe(card);
    card.addEventListener("toggle", () => setTimeout(upd, 0), true);     // відкрили/закрили «ще»
    setTimeout(upd, 0);
    return upd;
}

// ---- рядок стану виконання ноди: сервер шле "svk.stage" {node, stage, k, n, eta, secs, msg}
const STAGE_TEXT = {
    queued: () => T("waiting in queue"),
    read: s => T("reading {n} frame(s)").replace("{n}", s.n ?? "?"),
    cache: () => T("saving preview frames"),
    start: () => T("starting the engine"),
    camera: () => T("measuring camera motion"),
    frame: s => T("frame {k} / {n}").replace("{k}", s.k).replace("{n}", s.n),
    mix: () => T("mixing with the original"),
    strip: () => T("saving the frame strip"),
    done: s => T("done in {s} s").replace("{s}", (s.secs ?? 0).toFixed(1)),
    error: s => T("error") + (s.msg ? ": " + s.msg : ""),
    cancel: () => T("cancelled"),
    cached: () => T("unchanged — taken from cache"),
};
const STAGES = [];                // {box, set, node}; id ноди читаємо наживо - при створенні він ще -1 у всіх
export function stageLine(node) {
    const box = el("div", "svk-status");
    box.innerHTML = '<span class="dot"></span><span class="st"></span><span class="tm"></span><span class="bar"></span>';
    const st = box.querySelector(".st"), tm = box.querySelector(".tm");
    box.style.display = "none";
    const set = s => {
        box.style.display = "";
        const run = !["done", "error", "cancel", "cached"].includes(s.stage);
        box.classList.toggle("is-run", run); box.classList.toggle("is-done", s.stage === "done" || s.stage === "cached"); box.classList.toggle("is-err", s.stage === "error");
        st.textContent = (STAGE_TEXT[s.stage] || (() => s.stage))(s);
        st.title = s.msg || st.textContent;
        tm.textContent = s.stage === "frame" && s.eta > 0.5 ? "~" + fmtSec(s.eta) : "";
        const p = s.stage === "frame" && s.n ? (s.k / s.n) * 100 : (s.stage === "done" ? 100 : (run ? 0 : null));
        if (p != null) box.style.setProperty("--p", p + "%");
    };
    STAGES.push({ box, set, node });
    box.__svkSet = set;
    hookStages();
    return box;
}
function fmtSec(s) { s = Math.round(s); return s < 60 ? s + " " + T("s") : Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); }
function stageFor(id) {
    for (let i = STAGES.length - 1; i >= 0; i--) {             // свіжіша картка ноди - остання
        const v = STAGES[i];
        if (!v.box.isConnected && !v.node.graph) { STAGES.splice(i, 1); continue; }     // нода видалена
        if (String(v.node.id) === String(id)) return v;
    }
}
let stagesHooked = false;
function hookStages() {
    if (stagesHooked) return;
    stagesHooked = true;
    api.addEventListener("svk.stage", ev => stageFor(ev.detail?.node)?.set(ev.detail));
    // вкладку перезавантажили або звʼязок рвався: довгий прогін іде далі, стан беремо з сервера
    const replay = () => api.fetchApi("/svk/stages").then(r => r.ok ? r.json() : {})
        .then(m => { for (const [id, s] of Object.entries(m)) stageFor(id)?.set(s); }).catch(() => {});
    api.addEventListener("reconnected", replay);
    setTimeout(replay, 2000);
    api.addEventListener("execution_start", () => {             // новий прогін: наші ноди в черзі - «чекає»
        for (const v of STAGES) { v.set({ stage: "queued" }); v.box.dataset.queued = "1"; }
    });
    api.addEventListener("svk.stage", ev => { const e = stageFor(ev.detail?.node); if (e) delete e.box.dataset.queued; });
    api.addEventListener("execution_success", () => {           // ноди, яких у цьому прогоні не було, не висять у «чекає»
        for (const v of STAGES) if (v.box.dataset.queued) { delete v.box.dataset.queued; v.box.style.display = "none"; }
    });
    api.addEventListener("execution_cached", ev => {            // нода взята з кешу - прогону не буде
        for (const id of ev.detail?.nodes || []) { const e = stageFor(id); if (e) { delete e.box.dataset.queued; e.set({ stage: "cached" }); } }
    });
    api.addEventListener("execution_error", ev => stageFor(ev.detail?.node_id)?.set({ stage: "error", msg: ev.detail?.exception_message }));
    api.addEventListener("execution_interrupted", () => { for (const v of STAGES) if (v.box.classList.contains("is-run")) v.set({ stage: "cancel" }); });
}
