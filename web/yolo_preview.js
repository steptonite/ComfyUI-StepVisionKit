// Превʼю YOLO у картці: кадри з останнього Run і всі бокси детектора. Поріг, класи, сортування, вибір і кількість
// перераховуються тут за віджетами ноди (та сама логіка, що sort_frame / select_frame у vk/boxes.py), без нового прогону.
// Клік по боксу в YOLO Detect обирає саме цей обʼєкт: track_id, якщо є id, інакше index.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { el, W, markChanged } from "./svk_ui.js";
import { T } from "./svk_i18n.js";

// ключ ноди -> імена віджетів
const FIELDS = {
    LexYOLODetect: { conf: "conf", classes: "class_filter", sort: "sort_by", max: "max_boxes", select: "select", pick: "pick", px: "point_x", py: "point_y" },
    VKDetect: { conf: "confidence", classes: "classes", sort: "sort_by", max: "max_objects" },
};
export const PREVIEW_H = 210;
const ACCENT = "#e0a458", DIM = "rgba(232,230,227,.4)", INK = "#1a1208";

const area = b => (b.x2 - b.x1) * (b.y2 - b.y1);
const NO_ID = 1 << 30;
const SORT = {
    area: (a, b) => area(b) - area(a),
    confidence: (a, b) => b.score - a.score,
    left_to_right: (a, b) => a.x1 - b.x1,
    top_to_bottom: (a, b) => a.y1 - b.y1,
    track_id: (a, b) => (a.id >= 0 ? a.id : NO_ID) - (b.id >= 0 ? b.id : NO_ID),
};

// перший з найкращих, як max()/min() у Python
const best = (f, key, better) => [f.reduce((a, b) => (better(key(b), key(a)) ? b : a))];
function select(f, mode, pick, pt) {
    if (!f.length) return [];
    if (mode === "index") return [f[Math.min(Math.max(pick, 0), f.length - 1)]];
    if (mode === "track_id") return f.filter(b => b.id === pick).slice(0, 1);
    if (mode === "largest") return best(f, area, (x, y) => x > y);
    if (mode === "smallest") return best(f, area, (x, y) => x < y);
    if (mode === "highest_confidence") return best(f, b => b.score, (x, y) => x > y);
    if (mode === "nearest_to_point") return best(f, b => ((b.x1 + b.x2) / 2 - pt[0]) ** 2 + ((b.y1 + b.y2) / 2 - pt[1]) ** 2, (x, y) => x < y);
    return f;
}

function live(node, F, data, raw) {
    const v = n => W(node, F[n])?.value;
    const conf = Number(v("conf") ?? 0);
    const wanted = String(v("classes") ?? "").split(",").map(s => s.trim().toLowerCase()).filter(Boolean);
    let ranked = raw.filter(b => b.score >= conf && (!wanted.length || wanted.includes(b.label.toLowerCase())));
    if (v("sort") !== "detector") ranked = [...ranked].sort(SORT[v("sort")] ?? SORT.area);
    let kept = ranked;
    if (F.select) {
        const m = v("select");
        const pt = m === "center" ? [data.w / 2, data.h / 2] : [Number(v("px")), Number(v("py"))];
        kept = select(ranked, m === "center" ? "nearest_to_point" : m, Number(v("pick")), pt);
    }
    return { ranked, kept: new Set(kept.slice(0, Number(v("max")))), low: conf < data.conf - 1e-6 };
}

function setWidget(node, name, value) {
    const w = W(node, name);
    if (!w || w.value === value) return;
    w.value = value;
    w.callback?.(value, app.canvas, node);
}

export function yoloPreview(node, key) {
    const F = FIELDS[key];
    const root = el("div", "svk-yolo");
    const cv = el("canvas");
    const bar = el("div", "svk-yolo-bar");
    const scrub = el("input");
    scrub.type = "range"; scrub.min = 0; scrub.step = 1; scrub.value = 0;
    const note = el("span");
    bar.append(scrub, note);
    root.append(cv, bar);
    cv.title = F.select ? T("Click a box to pass on exactly this object") : T("Boxes from the last Run; the highlighted ones go to the outputs");
    if (F.select) cv.style.cursor = "pointer";

    let data = null, at = 0, img = null, hits = [];

    function load() {
        const s = data.shots[at];
        img = new Image();
        img.onload = draw;
        img.src = api.apiURL("/view?filename=" + encodeURIComponent(s.filename) + "&type=" + s.type
            + "&subfolder=" + encodeURIComponent(s.subfolder) + "&t=" + data.t);   // імʼя файлу однакове між Run, t скидає кеш браузера
    }

    function draw() {
        const dpr = window.devicePixelRatio || 1;
        const cw = cv.clientWidth || 320, ch = PREVIEW_H;
        cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
        const g = cv.getContext("2d");
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, cw, ch);
        hits = [];
        if (!data || !img?.complete || !img.naturalWidth) {
            g.fillStyle = DIM; g.font = "12px system-ui, sans-serif"; g.textAlign = "center";
            g.fillText(data ? T("loading…") : T("Run the node to see the boxes"), cw / 2, ch / 2);
            note.textContent = "";
            return;
        }
        const s = Math.min(cw / data.w, ch / data.h), ox = (cw - data.w * s) / 2, oy = (ch - data.h * s) / 2;
        g.drawImage(img, ox, oy, data.w * s, data.h * s);
        const shot = data.shots[at];
        const { ranked, kept, low } = live(node, F, data, shot.boxes);
        g.font = "10px ui-monospace, Menlo, monospace"; g.textAlign = "left"; g.textBaseline = "top";
        ranked.forEach((b, j) => {
            const x = ox + b.x1 * s, y = oy + b.y1 * s, w = (b.x2 - b.x1) * s, h = (b.y2 - b.y1) * s, on = kept.has(b);
            hits.push({ x, y, w, h, b, j });
            g.strokeStyle = on ? ACCENT : DIM; g.lineWidth = on ? 2 : 1;
            g.strokeRect(x, y, w, h);
            const t = `#${j} ${b.label}${b.id >= 0 ? " id:" + b.id : ""} ${b.score.toFixed(2)}`;
            const tw = g.measureText(t).width + 6;
            g.fillStyle = on ? ACCENT : "rgba(14,14,16,.7)";
            g.fillRect(x, Math.max(0, y - 13), tw, 13);
            g.fillStyle = on ? INK : "#e8e6e3";
            g.fillText(t, x + 3, Math.max(0, y - 13) + 2);
        });
        note.textContent = `${T("frame")} ${shot.frame + 1} · ${T("found")} ${ranked.length} · ${T("passed on")} ${kept.size}`
            + (low ? " · " + T("threshold below the Run: run again to see more") : "");
    }

    cv.onclick = (e) => {
        if (!F.select || !hits.length) return;
        const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
        const inside = hits.filter(h => px >= h.x && px <= h.x + h.w && py >= h.y && py <= h.y + h.h);
        if (!inside.length) return;
        const { b, j } = inside.reduce((a, c) => (c.w * c.h < a.w * a.h ? c : a));   // вкладені бокси: найменший
        if (b.id >= 0) { setWidget(node, "select", "track_id"); setWidget(node, "pick", b.id); }
        else { setWidget(node, "select", "index"); setWidget(node, "pick", j); }
        markChanged();
        node.__svkCard?.sync();
        draw();
    };
    scrub.oninput = () => { at = +scrub.value; load(); };

    return {
        root,
        draw,
        set(d) {
            data = { ...d, t: Date.now() };
            at = Math.max(0, Math.min(at, d.shots.length - 1));
            scrub.max = d.shots.length - 1; scrub.value = at;
            scrub.style.display = d.shots.length > 1 ? "" : "none";
            if (d.shots.length) load(); else draw();
        },
    };
}
