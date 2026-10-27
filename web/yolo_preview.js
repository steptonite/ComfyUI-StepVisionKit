// Превʼю YOLO у картці: кадри з останнього Run і всі бокси детектора. Поріг, класи, сортування, вибір і кількість
// перераховуються тут за віджетами ноди (та сама логіка, що sort_frame / select_frame у vk/boxes.py), без нового прогону.
// Клік по боксу в YOLO Detect додає його до вибору, ще клік знімає; вибір живе у віджеті picked («id:3,#0») і йде у workflow,
// тож Run бере саме його (pick_frame у vk/boxes.py): id:N ведеться трекером по всьому кліпу, #N - місце в рейтингу кадру.
// Порожній вибір - як раніше, за параметрами «Вибір» і «Макс. рамок».
// Кадр росте з шириною ноди; колесо або кнопки наближають, тягнення посуває, подвійний клік вписує назад.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { el, W, clamp, markChanged, onWheel } from "./svk_ui.js";
import { T } from "./svk_i18n.js";

// ключ ноди -> імена віджетів
const FIELDS = {
    LexYOLODetect: { conf: "conf", classes: "class_filter", sort: "sort_by", max: "max_boxes", select: "select", pick: "pick", px: "point_x", py: "point_y", picked: "picked" },
    VKDetect: { conf: "confidence", classes: "classes", sort: "sort_by", max: "max_objects" },
};
export const PREVIEW_H = 190;   // найменша висота кадру; далі він росте з шириною ноди
const DIM = "rgba(232,230,227,.4)";
// налаштування паку (Settings → StepVisionKit → Cards): колір боксів, підписи, відкинуті бокси
export const YOLO_SETTINGS = { color: "StepVisionKit.YoloBoxColor", labels: "StepVisionKit.YoloLabels", others: "StepVisionKit.YoloShowOthers" };
function setting(id, def) {
    try { return app.extensionManager?.setting?.get(id) ?? app.ui?.settings?.getSettingValue(id) ?? def; } catch (_) { return def; }
}
function look() {
    let hex = String(setting(YOLO_SETTINGS.color, "e0a458")).replace(/^#/, "");
    if (!/^[0-9a-f]{6}$/i.test(hex)) hex = "e0a458";
    const [r, g, b] = [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16));
    return { accent: "#" + hex, tint: `rgba(${r},${g},${b},.16)`, ink: r * .299 + g * .587 + b * .114 > 140 ? "#1a1208" : "#f4f2ef",
             labels: setting(YOLO_SETTINGS.labels, "number"), others: setting(YOLO_SETTINGS.others, true) !== false };
}
const PREVIEW_OUT = 2;          // індекс виходу preview у LexYOLODetect (порожній без make_preview)
const ZOOMS = [1, 2, 4], ZMAX = 8, DRAG = 4;

const area = b => (b.x2 - b.x1) * (b.y2 - b.y1);
const NO_ID = 1 << 30;
const SORT = {
    area: (a, b) => area(b) - area(a),
    confidence: (a, b) => b.score - a.score,
    left_to_right: (a, b) => a.x1 - b.x1,
    top_to_bottom: (a, b) => a.y1 - b.y1,
    track_id: (a, b) => (a.id >= 0 ? a.id : NO_ID) - (b.id >= 0 ? b.id : NO_ID),
};
const keyOf = (b, j) => (b.id >= 0 ? "id:" + b.id : "#" + j);

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

// вибір кліком: «id:3,#0» -> упорядкований список (як parse_picked у vk/boxes.py)
function pickedList(node, F) {
    if (!F.picked) return [];
    const out = [];
    for (const t of String(W(node, F.picked)?.value ?? "").split(",").map(s => s.trim()))
        if (/^(id:|#)\d+$/.test(t) && !out.includes(t)) out.push(t);
    return out;
}

function live(node, F, data, raw) {
    const v = n => W(node, F[n])?.value;
    const conf = Number(v("conf") ?? 0);
    const wanted = String(v("classes") ?? "").split(",").map(s => s.trim().toLowerCase()).filter(Boolean);
    let ranked = raw.filter(b => b.score >= conf && (!wanted.length || wanted.includes(b.label.toLowerCase())));
    if (v("sort") !== "detector") ranked = [...ranked].sort(SORT[v("sort")] ?? SORT.area);
    const picked = pickedList(node, F);
    let kept = ranked;
    if (picked.length) kept = ranked.filter((b, j) => picked.includes(keyOf(b, j)));
    else {
        if (F.select) {
            const m = v("select");
            const pt = m === "center" ? [data.w / 2, data.h / 2] : [Number(v("px")), Number(v("py"))];
            kept = select(ranked, m === "center" ? "nearest_to_point" : m, Number(v("pick")), pt);
        }
        kept = kept.slice(0, Number(v("max")));
    }
    return { ranked, kept: new Set(kept), picked, low: conf < data.conf - 1e-6 };
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
    const zoom = el("div", "svk-seg svk-yolo-zoom");
    zoom.title = T("zoom: wheel over the frame, drag to move, double-click to fit");
    const view = { z: 1, c: null };     // наближення відносно «вписати» і центр погляду в пікселях кадру
    const zbtn = ZOOMS.map(z => {
        const b = el("button", null, z === 1 ? T("fit") : z + "×"); b.type = "button";
        b.onclick = () => { view.z = z; if (z === 1) view.c = null; draw(); };
        zoom.append(b);
        return [z, b];
    });
    bar.append(scrub, zoom, note);
    const outRow = el("div", "svk-yolo-outrow");
    const out = el("div", "svk-yolo-out");
    const clear = el("button", "svk-btn ghost svk-yolo-clear", T("clear pick")); clear.type = "button";
    clear.title = T("forget the clicked boxes: Select and Max boxes decide again");
    outRow.append(out, clear);
    root.append(cv, bar, outRow);
    const hint = F.picked ? T("Click a box to pick it, click again to drop it; you can pick several. Picked boxes go to the outputs and follow their id through the video")
                          : T("Boxes from the last Run; the highlighted ones go to the outputs");
    cv.title = hint;
    if (F.picked) cv.style.cursor = "pointer";

    let data = null, at = 0, img = null, hits = [], drag = null, lastW = 0;
    let geo = null;                     // { s, ox, oy } останнього малювання: кадр -> екран
    const desc = (b, j) => `#${j} ${b.label}${b.id >= 0 ? " id:" + b.id : ""} ${b.score.toFixed(2)}`;

    // вихід preview без make_preview - картинка 1×1; підпис сокета каже про це, імʼя виходу не міняється
    function previewSlot() {
        const o = F.select && node.outputs?.[PREVIEW_OUT];
        if (!o) return;
        const lb = W(node, "make_preview")?.value ? null : T("preview (off)");
        if ((o.label ?? null) !== lb) { o.label = lb ?? undefined; node.setDirtyCanvas?.(true, true); }
    }

    function outLine(ranked, kept, picked) {
        const list = ranked.map((b, j) => [b, j]).filter(([b]) => kept.has(b));
        clear.style.display = picked.length ? "" : "none";
        out.classList.toggle("is-empty", !list.length && !picked.length);
        out.title = "";
        if (picked.length) {         // вибір кліком: усе обране, навіть те, чого не видно на цьому кадрі
            const names = picked.map(k => {
                const hit = list.find(([b, j]) => keyOf(b, j) === k);
                return hit ? `${k} ${hit[0].label}` : `${k} (${T("not in this frame")})`;
            });
            out.textContent = `${T("To the outputs")} (${T("clicked")} ${picked.length}): ${names.join(", ")}`;
            if (list.length < picked.length) out.title = T("a picked id that is missing in a frame keeps its previous box (If no box = previous)");
            return;
        }
        if (!list.length) { out.textContent = T("Nothing goes to the outputs: threshold, classes or selection dropped every box"); return; }
        const all = F.select && W(node, "select")?.value === "all";
        const names = list.slice(0, 3).map(([b, j]) => desc(b, j).replace(/ [\d.]+$/, "")).join(", ") + (list.length > 3 ? " …" : "");
        out.textContent = `${T("To the outputs")}: ${all ? T("all") + " " + list.length + " · " : ""}${names}`;
    }

    function load() {
        const s = data.shots[at];
        img = new Image();
        img.onload = draw;
        img.src = api.apiURL("/view?filename=" + encodeURIComponent(s.filename) + "&type=" + s.type
            + "&subfolder=" + encodeURIComponent(s.subfolder) + "&t=" + data.t);   // імʼя файлу однакове між Run, t скидає кеш браузера
    }

    // висота кадру йде за шириною картки; вертикальний кліп не вищий за квадрат
    function size() {
        const cw = cv.clientWidth || 320;
        const k = data ? Math.min(data.h / data.w, 1) : 9 / 16;
        const ch = Math.max(PREVIEW_H, Math.round(cw * k));
        if (cv.style.height !== ch + "px") cv.style.height = ch + "px";
        return [cw, ch];
    }

    // центр погляду не дає кадру виїхати з рамки
    function place(cw, ch) {
        const s = Math.min(cw / data.w, ch / data.h) * view.z;
        const c = view.c || [data.w / 2, data.h / 2];
        const fit = (v, len, full) => (full * s <= len ? full / 2 : clamp(v, len / 2 / s, full - len / 2 / s));
        view.c = [fit(c[0], cw, data.w), fit(c[1], ch, data.h)];
        return { s, ox: cw / 2 - view.c[0] * s, oy: ch / 2 - view.c[1] * s };
    }

    function draw() {
        const dpr = window.devicePixelRatio || 1;
        const [cw, ch] = size();
        lastW = cw;
        cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
        const g = cv.getContext("2d");
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, cw, ch);
        hits = []; geo = null;
        previewSlot();
        zbtn.forEach(([z, b]) => b.classList.toggle("is-on", Math.abs(view.z - z) < 1e-3));
        if (!data || !img?.complete || !img.naturalWidth) {
            out.textContent = ""; clear.style.display = "none";
            g.fillStyle = DIM; g.font = "12px system-ui, sans-serif"; g.textAlign = "center";
            g.fillText(data ? T("loading…") : T("Run the node to see the boxes"), cw / 2, ch / 2);
            note.textContent = "";
            return;
        }
        const { s, ox, oy } = geo = place(cw, ch);
        g.imageSmoothingQuality = "high";
        g.drawImage(img, ox, oy, data.w * s, data.h * s);
        const shot = data.shots[at];
        const { ranked, kept, picked, low } = live(node, F, data, shot.boxes);
        g.font = "10px ui-monospace, Menlo, monospace"; g.textAlign = "left"; g.textBaseline = "top";
        const tags = [];               // на боксі номер (обраний кліком - з ✓ і id); решта - у підказці під курсором
        const free = r => !tags.some(q => r.x < q.x + q.w && q.x < r.x + r.w && r.y < q.y + q.h && q.y < r.y + r.h);
        const L = look();
        const order = ranked.map((b, j) => [b, j]).sort(([a], [b]) => kept.has(a) - kept.has(b));   // обрані малюються зверху
        for (const [b, j] of order) {
            const x = ox + b.x1 * s, y = oy + b.y1 * s, w = (b.x2 - b.x1) * s, h = (b.y2 - b.y1) * s, on = kept.has(b);
            hits.push({ x, y, w, h, b, j });           // відкинуті без рамки теж клікаються
            if (!on && !L.others) continue;
            if (on && picked.length) { g.fillStyle = L.tint; g.fillRect(x, y, w, h); }
            g.strokeStyle = on ? L.accent : DIM; g.lineWidth = on ? (picked.length ? 3 : 2) : 1;
            g.strokeRect(x, y, w, h);
            if (L.labels === "none" && !(on && picked.length)) continue;
            const t = on && picked.length ? "✓ " + keyOf(b, j) : L.labels === "class" ? `#${j} ${b.label} ${b.score.toFixed(2)}` : "#" + j;
            const tw = g.measureText(t).width + 6;
            const r = [{ x, y: Math.max(0, y - 13), w: tw, h: 13 }, { x, y, w: tw, h: 13 }].find(free);   // над боксом, інакше всередині
            if (!r && !on) continue;
            const q = r || { x, y, w: tw, h: 13 };
            tags.push(q);
            g.fillStyle = on ? L.accent : "rgba(14,14,16,.7)";
            g.fillRect(q.x, q.y, q.w, q.h);
            g.fillStyle = on ? L.ink : "#e8e6e3";
            g.fillText(t, q.x + 3, q.y + 2);
        }
        outLine(ranked, kept, picked);
        note.textContent = `${T("frame")} ${shot.frame + 1} · ${T("found")} ${ranked.length} · ${T("passed on")} ${kept.size}`
            + (low ? " · " + T("threshold below the Run: run again to see more") : "");
    }

    // вкладені бокси: найменший під курсором
    const local = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    function under(e) {
        const [px, py] = local(e);
        const inside = hits.filter(h => px >= h.x && px <= h.x + h.w && py >= h.y && py <= h.y + h.h);
        return inside.length ? inside.reduce((a, c) => (c.w * c.h < a.w * a.h ? c : a)) : null;
    }
    function toggle(h) {
        const list = pickedList(node, F), k = keyOf(h.b, h.j), i = list.indexOf(k);
        if (i >= 0) list.splice(i, 1); else list.push(k);
        setWidget(node, F.picked, list.join(","));
        markChanged();
        draw();
    }
    // наближення довкола точки (mx, my) екрана: точка кадру під нею лишається на місці
    function zoomAt(z, mx, my) {
        if (!geo) return;
        const p = [(mx - geo.ox) / geo.s, (my - geo.oy) / geo.s];
        const s = geo.s / view.z * z;
        view.z = z;
        view.c = [p[0] - (mx - cv.clientWidth / 2) / s, p[1] - (my - cv.clientHeight / 2) / s];
        draw();
    }

    // колесо над кадром - наближення; на «вписати» колесо вниз гортає картку як звичайно
    onWheel(cv, (e) => {
        if (!geo || (view.z <= 1 && e.deltaY > 0)) return;
        e.preventDefault();
        const [mx, my] = local(e);
        zoomAt(clamp(view.z * Math.exp(-e.deltaY * 0.0025), 1, ZMAX), mx, my);
    });
    cv.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.stopPropagation();
        drag = { x: e.clientX, y: e.clientY, c: view.c && [...view.c], moved: false };
        cv.setPointerCapture?.(e.pointerId);
    });
    cv.addEventListener("pointermove", (e) => {
        if (drag) {
            const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
            if (!drag.moved && Math.hypot(dx, dy) < DRAG) return;
            drag.moved = true;
            if (geo && view.z > 1 && drag.c) { view.c = [drag.c[0] - dx / geo.s, drag.c[1] - dy / geo.s]; cv.style.cursor = "grabbing"; draw(); }
            return;
        }
        const h = under(e);
        const act = h && F.picked ? " · " + T(pickedList(node, F).includes(keyOf(h.b, h.j)) ? "click to drop" : "click to pick") : "";
        cv.title = h ? desc(h.b, h.j) + act + "\n" + hint : hint;
    });
    cv.addEventListener("pointerup", (e) => {
        const d = drag; drag = null;
        cv.style.cursor = F.picked ? "pointer" : "";
        if (!d || d.moved || !F.picked || !hits.length) return;
        const h = under(e);
        if (h) toggle(h);
    });
    cv.addEventListener("dblclick", (e) => { e.stopPropagation(); view.z = 1; view.c = null; draw(); });
    clear.onclick = () => { setWidget(node, F.picked, ""); markChanged(); draw(); };
    scrub.oninput = () => { at = +scrub.value; load(); };
    new ResizeObserver(() => { if (cv.clientWidth && cv.clientWidth !== lastW) draw(); }).observe(root);

    return {
        root,
        draw,
        set(d) {
            data = { ...d, t: Date.now() };
            at = Math.max(0, Math.min(at, d.shots.length - 1));
            scrub.max = d.shots.length - 1; scrub.value = at;
            scrub.style.display = d.shots.length > 1 ? "" : "none";
            bar.classList.toggle("one", d.shots.length <= 1);       // один кадр: підпис займає місце повзунка
            if (d.shots.length) load(); else draw();
        },
    };
}
