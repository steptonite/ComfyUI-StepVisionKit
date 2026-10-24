// Depth Look (look v1): тюнер луку всередині ноди. DOM-віджет з iframe на /depth_look/ui?embed=1&node=<id>
// (той самий tune.html, що й автономний тюнер). Параметри iframe -> віджет params_json (зберігається у workflow).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ICON, control, morePanel, keepWheel, stageLine } from "./svk_ui.js";
import { T } from "./svk_i18n.js";

const NODE = "DepthLook";
const START_W = 1000, START_H = 900;
// на ноді лише тюнер: усі нативні віджети сховані (значення лишаються у workflow і йдуть у Run).
// Пресет обирається в тюнері; режим - сегментом у рядку під тюнером; рідкісне - під «ще».
const MORE = [["output_size", T("Output size")], ["max_side", T("Working resolution")], ["depth_levels", T("Depth levels"), { names: { clip: T("shared across clip"), frame: T("per frame") } }],
              ["clean_depth", T("Clean depth")], ["depth_light_is_near", T("Light = near")]];
function hideAll(node) { for (const w of node.widgets || []) if (w.name !== "look_ui") hideWidget(w); node.__dlBar?.sync(); }
const PANELS = new Set();          // {node, frame}

// --- міграція старих workflow: fps/start_time прибрано зі статичної ноди, порядок віджетів змінився.
// Старий порядок widgets_values (до 11 значень); у новому fps/start_time нема, тож значення зсуваються.
const OLD_ORDER = ["preset", "clean_depth", "depth_light_is_near", "overrides", "palette", "fps", "start_time",
                   "params_json", "max_side", "depth_levels", "mode"];
const NEW_DEFAULTS = { output_size: "як оригінал", max_side: 0, depth_levels: "clip",
                       mode: "налаштування (швидко, 720)", overrides: "{}", palette: "", params_json: "" };
function migrateOld(node, info) {
    const v = info?.widgets_values;
    // старий формат: на місці fps (індекс 5) і start_time (6) числа; у новому на 5-му depth_levels (рядок)
    if (!Array.isArray(v) || v.length < 7 || v.length > OLD_ORDER.length) return;
    if (typeof v[5] !== "number" || typeof v[6] !== "number") return;
    const set = (name, val) => { const w = node.widgets?.find(x => x.name === name); if (w) w.value = val; };
    for (const [name, val] of Object.entries(NEW_DEFAULTS)) set(name, val);   // що не було в старому - дефолт
    OLD_ORDER.forEach((name, i) => { if (i < v.length && name !== "fps" && name !== "start_time") set(name, v[i]); });
}

function gradeConnected(node) {
    const i = node.inputs?.find(x => x.name === "grade");
    return !!(i && i.link != null);
}
// початкові параметри тюнера: params_json, а якщо порожньо й підключено grade - параметри грейду (з останнього запуску)
function initParams(node) {
    const p = readParams(node);
    if (Object.keys(p).length) return p;
    if (node.__dlGrade && gradeConnected(node)) return node.__dlGrade;
    return {};
}

function paramsWidget(node) { return node.widgets?.find(w => w.name === "params_json"); }
function presetWidget(node) { return node.widgets?.find(w => w.name === "preset"); }

function hideWidget(w) {
    if (!w) return;
    w.hidden = true;
    w.options = Object.assign(w.options || {}, { hidden: true });
    w.computeSize = () => [0, -4];
    w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    if (w.element?.style) w.element.style.display = "none";
}

function post(panel, msg) {
    try { panel.frame.contentWindow?.postMessage(msg, "*"); } catch (_) { /* iframe ще не готовий */ }
}

function markChanged() {
    try { app.graph?.setDirtyCanvas(true, true); } catch (_) {}
    try { app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState(); } catch (_) {}
}

function readParams(node) {
    const w = paramsWidget(node);
    try { const v = JSON.parse(w?.value || ""); return v && typeof v === "object" ? v : {}; } catch (_) { return {}; }
}

function onMessage(ev) {
    const m = ev.data;
    if (!m || typeof m !== "object" || typeof m.type !== "string" || !m.type.startsWith("depth_look_")) return;
    let panel = null;
    for (const p of PANELS) if (p.frame.contentWindow === ev.source) { panel = p; break; }
    if (!panel) return;
    const node = panel.node;
    if (m.type === "depth_look_ready") {
        post(panel, { type: "depth_look_init", params: initParams(node), preset: presetWidget(node)?.value });
    } else if (m.type === "depth_look_params") {
        const w = paramsWidget(node);
        if (w && m.params) { w.value = JSON.stringify(m.params); markChanged(); }
    } else if (m.type === "depth_look_presets") {   // збережено пресет у тюнері -> зʼявляється в списку ноди
        const pw = presetWidget(node);
        if (pw && m.preset) {
            const vals = pw.options?.values;
            if (Array.isArray(vals) && !vals.includes(m.preset)) { vals.push(m.preset); vals.sort(); }
            pw.value = m.preset;     // напряму, без callback (він скидає params_json)
            markChanged();
        }
    }
}

function tunerURL(node) { return api.apiURL("/depth_look/ui?embed=1&node=" + encodeURIComponent(node.id)); }

// повний екран: окремий оверлей зі своїм iframe (переносити наявний не можна - він перезавантажиться)
function openFull(node) {
    if (node.__dlFull || node.id == null || node.id < 0) return;
    const ov = document.createElement("div");
    ov.style.cssText = "position:fixed;inset:0;z-index:100000;background:#0e0e10;display:flex;flex-direction:column";
    const bar = document.createElement("div");
    bar.style.cssText = "display:flex;align-items:center;gap:12px;padding:6px 12px;background:#18181b;color:#ccc;font:13px system-ui";
    const title = document.createElement("span");
    title.textContent = T("Depth Look · node #") + node.id;
    const back = document.createElement("button");
    back.textContent = T("✕ back to graph (Esc)");
    back.style.cssText = "margin-left:auto;padding:5px 14px;border:1px solid #444;border-radius:6px;background:#26262b;color:#eee;cursor:pointer";
    bar.append(title, back);
    const frame = document.createElement("iframe");
    frame.style.cssText = "flex:1;width:100%;border:0;display:block;background:#0e0e10";
    frame.setAttribute("title", "Depth Look");
    ov.append(bar, frame);
    document.body.appendChild(ov);
    const panel = { node, frame, started: true, full: true };
    PANELS.add(panel);
    const savedSize = node.size ? [node.size[0], node.size[1]] : null;     // у повному екрані нода не повинна змінювати розмір
    const esc = e => { if (e.key === "Escape") close(); };
    function close() {
        PANELS.delete(panel);
        window.removeEventListener("keydown", esc, true);
        ov.remove();
        node.__dlFull = null;
        const inl = node.__dlPanel;
        if (inl) {               // вбудований тюнер повертається й бере поточні params_json
            inl.frame.style.visibility = "";
            post(inl, { type: "depth_look_init", params: initParams(node), preset: presetWidget(node)?.value });
            post(inl, { type: "depth_look_refresh" });
        }
        if (savedSize && node.size && (node.size[0] !== savedSize[0] || node.size[1] !== savedSize[1])) node.setSize(savedSize);
        const nudge = () => { try { inl?.frame.contentWindow?.dispatchEvent(new Event("resize")); } catch (_) {} };
        requestAnimationFrame(nudge); setTimeout(nudge, 250);       // тюнер перемірює кадр і сітку під справжній розмір
        markChanged();
    }
    back.onclick = close;
    window.addEventListener("keydown", esc, true);
    node.__dlFull = { close };
    // вбудований тюнер ховаємо БЕЗ зміни розміру (display:none обнуляв iframe: тюнер перелічував сітку й кадр у 0×0, нода шрінкалась) - лише visibility
    if (node.__dlPanel) node.__dlPanel.frame.style.visibility = "hidden";
    frame.src = tunerURL(node);
}

function createBar(node, specs) {
    const bar = document.createElement("div");
    bar.className = "svk-card";
    bar.style.cssText = "height:auto;flex:none;padding:6px 2px 0;gap:6px";
    const row = document.createElement("div");
    row.className = "svk-bar";
    const full = document.createElement("button");
    full.className = "svk-ibtn"; full.type = "button"; full.innerHTML = ICON.full; full.title = T("tuner full screen (Esc — back)");
    full.onclick = () => openFull(node);
    row.append(full);
    const ctl = [];
    if (specs.mode) {
        const vals = specs.mode[0] || [];
        const c = control(node, "mode", specs.mode, "", { seg: { [vals[0]]: T("preview"), [vals[1]]: T("final") }, tip: specs.mode[1]?.tooltip });
        row.append(c.row); ctl.push(c);
    }
    // рядок стану (етап / кадр k з N) - у тому ж рядку, що й керування, займає вільне місце
    const st = stageLine(node); st.style.flex = "1"; row.append(st);
    const grow = document.createElement("span"); grow.className = "grow"; row.append(grow);
    const more = morePanel(node, specs, MORE);                       // «ще» - як на інших нодах: останній рядок на всю ширину
    bar.append(row, more);
    keepWheel(bar);
    bar.sync = () => { ctl.forEach(c => c.sync()); more.sync(); };
    return bar;
}

function createPanel(node, specs) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "width:100%;height:100%;box-sizing:border-box;overflow:hidden;background:transparent;display:flex;flex-direction:column";
    const frame = document.createElement("iframe");
    frame.style.cssText = "width:100%;flex:1 1 auto;min-height:0;border:0;display:block;background:#0e0e10;border-radius:6px";
    frame.setAttribute("title", "Depth Look");
    wrap.appendChild(frame);
    node.__dlBar = createBar(node, specs);
    wrap.appendChild(node.__dlBar);
    node.addDOMWidget("look_ui", "depth_look_ui", wrap, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 720,
    });
    const panel = { node, frame, started: false };
    PANELS.add(panel);
    node.__dlPanel = panel;
    return panel;
}

function startPanel(node) {            // id ноди відомий лише після додавання в граф
    const panel = node.__dlPanel;
    if (!panel || panel.started || node.id == null || node.id < 0) return;
    panel.started = true;
    panel.frame.src = tunerURL(node);
}

app.registerExtension({
    name: "StepVisionKit.DepthLook",
    setup() {
        window.addEventListener("message", onMessage);
        // нода виконалась -> тюнер береться за новий кадр із кешу (executed приходить, бо run() віддає ui)
        api.addEventListener("executed", ev => {
            const id = ev.detail?.node;
            const warns = ev.detail?.output?.depth_look_warn || [];     // запобіжники ноди -> плашка в тюнері
            const gr = ev.detail?.output?.depth_look_grade?.[0];        // підключено grade -> база для тюнера
            for (const p of PANELS) if (String(p.node.id) === String(id)) {   // і оверлей теж
                if (!p.full && p.node.__dlFull) continue;       // вбудований під оверлеєм не оновлюємо: без подвійних рендерів
                if (gr?.params) {
                    const key = JSON.stringify(gr.params);
                    if (p.node.__dlGradeKey !== key) {                  // грейд змінився - тюнер бере його, якщо власних правок ще нема
                        p.node.__dlGrade = gr.params;
                        p.node.__dlGradeKey = key;
                        if (!Object.keys(readParams(p.node)).length)
                            post(p, { type: "depth_look_init", params: gr.params, preset: presetWidget(p.node)?.value });
                    }
                }
                post(p, { type: "depth_look_refresh" });
                post(p, { type: "depth_look_warn", warns });
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // українські підписи віджетів з {"label": ...} у INPUT_TYPES: фронтенд сам їх не бере -> ставимо руками (усі три ноди)
        if (/^DepthLook/.test(nodeData?.name || "")) {
            const spec = { ...(nodeData.input?.required || {}), ...(nodeData.input?.optional || {}) };
            const onC = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onC?.apply(this, arguments);
                for (const w of this.widgets || []) { const lb = spec[w.name]?.[1]?.label; if (lb) w.label = lb; }
                return r;
            };
        }
        if (nodeData?.name !== NODE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            hideAll(this);
            const pw = presetWidget(this);
            if (pw) {                  // змінив пресет у ноді -> скинути params_json і показати пресет у тюнері
                const orig = pw.callback;
                const node = this;
                pw.callback = function (v) {
                    const res = orig?.apply(this, arguments);
                    const w = paramsWidget(node);
                    if (w) w.value = "";
                    if (node.__dlPanel) post(node.__dlPanel, { type: "depth_look_init", params: {}, preset: v });
                    markChanged();
                    return res;
                };
            }
            createPanel(this, { ...(nodeData.input?.required || {}), ...(nodeData.input?.optional || {}) });
            hideAll(this);
            this.setSize([Math.max(this.size?.[0] || 0, START_W), Math.max(this.size?.[1] || 0, START_H)]);
            return r;
        };

        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const r = onAdded?.apply(this, arguments);
            startPanel(this);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            migrateOld(this, info);
            hideAll(this);
            startPanel(this);
            return r;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__dlFull?.close();
            if (this.__dlPanel) { PANELS.delete(this.__dlPanel); this.__dlPanel = null; }
            return onRemoved?.apply(this, arguments);
        };
    },
});
