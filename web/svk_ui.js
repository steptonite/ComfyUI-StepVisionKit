// Спільні дрібниці UI нод StepVisionKit: іконки, підписи луків, приховування нативних віджетів,
// контроли «нативний віджет -> DOM» (повзунок, вибір, перемикач) і панель «⋯ ще».
// Правило паку (CONVENTIONS_UI.md): на самій ноді - жодних абзаців; пояснення живуть у title (підказка на наведенні).
import { app } from "../../scripts/app.js";

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
};

// людські назви луків замість імен файлів; новий пресет без запису тут - імʼя файлу без префіксів
const PRESET_LABELS = { "look_v1": "Лук v1", "test_grade": "Тестовий", "00_оригінал": "Оригінал" };
export function presetLabel(name) {
    return PRESET_LABELS[name] || String(name).replace(/^\d+_/, "").replace(/^look_/, "").replace(/_/g, " ");
}

// ---- контроли, привʼязані до нативного віджета (його значення йде у workflow і в Run)
// spec - рядок INPUT_TYPES з object_info: [тип | список, {min,max,step,tooltip,default}]
export function control(node, name, spec, label, opts = {}) {
    const [type, o = {}] = spec || [];
    const tip = (opts.tip || o.tooltip || "") + (o.default !== undefined && !Array.isArray(type) ? "\nподвійний клік по назві — типове значення" : "");
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
            row = el("div", "svk-sl"); row.title = tip;
            const hd = el("div", "hd"); const t = el("span", "t", label); const val = el("span", "svk-num");
            hd.append(t, val);
            const r = el("input", "svk-range"); r.type = "range"; r.min = lo; r.max = hi; r.step = step;
            const paint = v => { r.value = v; r.style.setProperty("--p", ((v - lo) / (hi - lo || 1)) * 100 + "%"); val.textContent = (opts.fmt ? opts.fmt(v) : Number(v).toFixed(dec)); };
            r.addEventListener("input", () => { const v = isInt ? Math.round(Number(r.value)) : Number(r.value); paint(v); set(v); });
            t.addEventListener("dblclick", () => { if (o.default !== undefined) { set(o.default); paint(o.default); } });
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
    det.append(el("summary", null, ICON.more + "<span>ще</span>"));
    const body = el("div", "body body-col");
    det.append(body);
    const ctl = [];
    for (const [name, label, opts] of items) {
        if (!specs[name] || !W(node, name)) continue;
        const c = control(node, name, specs[name], label, opts || {});
        body.append(c.row); ctl.push(c);
    }
    det.sync = () => ctl.forEach(c => c.sync());
    return det;
}
