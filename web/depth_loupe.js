import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { nodeById, VIEW, VIDEO } from "./depth_grade_ui.js";

// Depth Grade: лупа 1:1 для View-нод. Payload svk_loupe приходить після Run: кадри повної
// роздільності (PNG у temp). Режими Image / Steps (червона маска сходинок) / Relief.

const VIEW_NODES = [VIEW, VIDEO];
const MODES = ["Image", "Steps", "Relief"];
const ZOOMS = { "1:1": 1, "2:1": 2, "4:1": 4 };
const MIN_S = 0.25, MAX_S = 8, WHEEL_STEP = 1.25;
const HINT = "Steps — red: tone stands in steps (8-bit source). Relief — fine tone ×12: soft bands show as stripes; clean float stays flat grey.";
const BTN = "background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:4px;padding:3px 10px;font:inherit;cursor:pointer";
const BTN_ON = "background:#3b5b8c;color:#fff;border:1px solid #5a7fb8;border-radius:4px;padding:3px 10px;font:inherit;cursor:pointer";

const mk = (tag, css = "", text = "") => {
    const e = document.createElement(tag);
    if (css) e.style.cssText = css;
    if (text) e.textContent = text;
    return e;
};

const fileUrl = (file) => api.apiURL(
    "/view?filename=" + encodeURIComponent(file.filename) + "&type=" + file.type
    + "&subfolder=" + encodeURIComponent(file.subfolder));  // ім'я унікальне на кожен Run — кеш браузера безпечний

function toast(summary) {
    const tm = app.extensionManager?.toast;
    if (tm) tm.add({ severity: "info", summary, life: 2500 });
    else alert(summary);
}

function openLoupe(payload) {
    const frames = payload.frames;
    const W = payload.width, H = payload.height;
    let mode = "Image", frameIdx = 0, s = 1, tx = 0, ty = 0, fitted = true, drag = null;
    let shown = "", stepsOf = -1;

    const overlay = mk("div", "position:fixed;inset:0;z-index:10000;display:flex;flex-direction:column;background:rgba(17,17,17,0.96);color:#ddd;font-family:system-ui,-apple-system,sans-serif;font-size:13px");
    const bar = mk("div", "display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:8px 12px");
    const modeBtns = MODES.map((m) => {
        const b = mk("button", BTN, m);
        b.onclick = () => { mode = m; paint(); };
        bar.append(b);
        return [m, b];
    });
    const zoomBtns = ["Fit", ...Object.keys(ZOOMS)].map((z) => {
        const b = mk("button", BTN, z);
        b.onclick = () => (z === "Fit" ? fit() : zoomTo(ZOOMS[z], ...center()));
        bar.append(b);
        return [z, b];
    });
    if (frames.length > 1) {
        const sel = mk("select", BTN);
        frames.forEach((f, i) => sel.append(new Option(`frame ${f.frame}`, String(i))));
        sel.onchange = () => { frameIdx = +sel.value; paint(); };
        bar.append(sel);
    }
    const info = mk("span", "color:#999;white-space:nowrap");
    const closeBtn = mk("button", BTN + ";margin-left:auto", "✕");
    closeBtn.onclick = closeLoupe;
    bar.append(info, closeBtn);

    const hint = mk("div", "padding:0 12px 6px;font-size:12px;color:#888", HINT);
    const stage = mk("div", "position:relative;flex:1;min-height:0;overflow:hidden;cursor:grab;touch-action:none;user-select:none");
    const wrap = mk("div", "position:absolute;left:0;top:0;transform-origin:0 0");
    const main = mk("img", "position:absolute;left:0;top:0;display:block;pointer-events:none");
    const steps = mk("img", "position:absolute;left:0;top:0;display:none;pointer-events:none");
    for (const img of [main, steps]) { img.draggable = false; img.width = W; img.height = H; }
    wrap.append(main, steps);
    stage.append(wrap);
    overlay.append(bar, hint, stage);
    document.body.append(overlay);

    const center = () => [stage.clientWidth / 2, stage.clientHeight / 2];

    function paint() {
        const f = frames[frameIdx];
        const kind = mode === "Relief" ? "relief" : "image";
        if (shown !== `${frameIdx}:${kind}`) { shown = `${frameIdx}:${kind}`; main.src = fileUrl(f[kind]); }
        if (stepsOf !== frameIdx) { stepsOf = frameIdx; steps.src = fileUrl(f.steps); }
        steps.style.display = mode === "Steps" ? "block" : "none";
        wrap.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`;
        const px = s > 1 ? "pixelated" : "auto";
        main.style.imageRendering = px;
        steps.style.imageRendering = px;
        info.textContent = `steps: ${f.steps_pct}% px · ${W}×${H}`;
        const zoomNow = fitted ? "Fit" : Object.keys(ZOOMS).find((z) => Math.abs(ZOOMS[z] - s) < 1e-3);
        for (const [m, b] of modeBtns) b.style.cssText = m === mode ? BTN_ON : BTN;
        for (const [z, b] of zoomBtns) b.style.cssText = z === zoomNow ? BTN_ON : BTN;
    }

    function fit() {
        const sw = stage.clientWidth, sh = stage.clientHeight;
        s = Math.min(sw / W, sh / H);
        tx = (sw - W * s) / 2;
        ty = (sh - H * s) / 2;
        fitted = true;
        paint();
    }

    function zoomTo(z, ax, ay) {
        z = Math.min(MAX_S, Math.max(MIN_S, z));
        tx = ax - (ax - tx) * (z / s);
        ty = ay - (ay - ty) * (z / s);
        s = z;
        fitted = false;
        paint();
    }

    stage.addEventListener("wheel", (e) => {
        e.preventDefault();
        if (!e.deltaY) return;
        const r = stage.getBoundingClientRect();
        zoomTo(s * (e.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });

    stage.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        stage.setPointerCapture(e.pointerId);
        drag = { x: e.clientX, y: e.clientY };
        stage.style.cursor = "grabbing";
    });
    stage.addEventListener("pointermove", (e) => {
        if (!drag) return;
        tx += e.clientX - drag.x;
        ty += e.clientY - drag.y;
        drag = { x: e.clientX, y: e.clientY };
        fitted = false;
        paint();
    });
    const endDrag = () => { drag = null; stage.style.cursor = "grab"; };
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);

    stage.addEventListener("dblclick", (e) => {
        if (!fitted) return fit();
        const r = stage.getBoundingClientRect();
        zoomTo(1, e.clientX - r.left, e.clientY - r.top);
    });

    const onKey = (e) => {
        if (e.key !== "Escape") return;
        e.stopPropagation();
        closeLoupe();
    };
    const onResize = () => { if (fitted) fit(); };
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("resize", onResize);

    function closeLoupe() {
        window.removeEventListener("keydown", onKey, true);
        window.removeEventListener("resize", onResize);
        overlay.remove();
    }

    fit();
}

function registerView(nodeType) {
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onCreated?.apply(this, arguments);
        const b = this.addWidget("button", "🔍 Loupe 1:1", null, () => {
            if (this._svkLoupe) openLoupe(this._svkLoupe);
            else toast("Run first");
        });
        b.serialize = false;
        return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        const p = message?.svk_loupe?.[0];
        if (p) this._svkLoupe = p;
    };
}

app.registerExtension({
    name: "StepVisionKit.DepthLoupe",
    setup() {
        // Nodes 2.0 не викликає onExecuted — беремо виводи з події виконання
        api.addEventListener("executed", (ev) => {
            const node = nodeById(ev.detail?.node);
            const p = ev.detail?.output?.svk_loupe?.[0];
            if (node && p && VIEW_NODES.includes(node.type)) node._svkLoupe = p;
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (VIEW_NODES.includes(nodeData?.name)) registerView(nodeType);
    },
});
