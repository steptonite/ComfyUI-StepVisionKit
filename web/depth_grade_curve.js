// Криві Depth Grade: гістограма планів, крива LUT, маркери Ч / Б / Г / С і кружок стелі.
// Спільне для фото-View (depth_grade_live.js) і відео-View у режимі «Статика» (depth_grade_video.js).
// Панель не розкладає себе сама: батько ставить this.curve і this.legend (place / display).
// Від батька (host) потрібно:
//   host.widget(name)  — віджет ноди-джерела з полем value;
//   host.set(name, v)  — записати повзунок (з округленням, як у Python-ноді);
//   host.reset(names)  — скинути повзунки до стандартних;
//   host.schedule()    — запланувати перемальовку;
//   host.last          — останній результат M.grade (з полями v, lut, P) або null.
import { C, MONO, h } from "./depth_grade_ui.js";
import * as M from "./depth_grade_math.js";
import { T } from "./svk_i18n.js";

// розкладка кривих — однакова для обох View
export const CURVE_H = 150;     // висота полотна кривих, px
export const LEGEND_COL = 46;   // легенда під кривими (колонка)
export const LEGEND_ROW = 124;  // легенда під кривими (бічна колонка)
export const SIDE_W = 210;      // ширина бічної колонки в портреті
export const ROW_GAP = 8;       // проміжок між превью і бічною колонкою

export class CurvePanel {
    constructor(host) {
        this.host = host;
        this.drag = null;        // що тягнемо зараз: { id, y, mid, pop, tip }
        this.hs = null;          // маркери на осі кривих (після draw): попадання мишкою і підсвітка

        this.curve = h("canvas", `position:absolute;display:none;border-radius:8px;background:${C.bg};
            box-shadow:0 0 0 1px ${C.line};touch-action:none;cursor:default`);
        this.legend = h("div", `position:absolute;display:none;overflow:hidden;color:${C.dim};
            font-size:11px;line-height:1.4`);
        this.legend.innerHTML =
            T("Drag: B / W — black and white points · <b>H</b> — hero (up = stronger) · F — where dimming starts, the dot on the right — how much · empty space up/down — midtones. Double-click resets. Clarity, texture, banding and deflicker apply after Run.")
                .replace("<b>", "<b style='color:" + C.accent + "'>");

        this.curve.addEventListener("pointerdown", (e) => this.onCurveDown(e));
        this.curve.addEventListener("pointermove", (e) => this.onCurveMove(e));
        this.curve.addEventListener("pointerup", (e) => this.onCurveUp(e));
        this.curve.addEventListener("pointercancel", (e) => this.onCurveUp(e));
        this.curve.addEventListener("dblclick", (e) => this.onCurveDbl(e));
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
            { id: "black", label: T("B"), v: P.bp },
            { id: "white", label: T("W"), v: P.wp },
            { id: "hero", label: T("H"), v: M.invert(lv, P.heroP), on: P.heroS > 0, accent: true },
            { id: "from", label: T("F"), v: M.invert(hv, P.k), on: P.c < 1 },
        ];
    }

    // ── малювання: g — результат M.grade (v, lut, P)
    draw(g) {
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
        ctx.fillText(T("far"), X(0) + 3, G.y1 + 11);
        ctx.textAlign = "right"; ctx.fillText(T("near"), X(1) - 3, G.y1 + 11); ctx.textAlign = "left";
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
        if (!this.host.last) return;
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
        pick.mid = this.host.widget("midtones")?.value || 0;
        pick.pop = this.host.widget("hero_pop")?.value || 0;
        this.drag = pick;
        this.curve.setPointerCapture(e.pointerId);
        e.preventDefault();
        this.host.schedule();
    }

    onCurveMove(e) {
        const G = this.geom(), p = this.pos(e);
        if (!this.drag) {
            const near = Math.abs(p.x - (G.x1 + 14)) < 12 ||
                (this.hs || []).some((m) => Math.abs(p.x - (G.x0 + m.v * (G.x1 - G.x0))) < 10 && p.y > G.y0 - 30);
            this.curve.style.cursor = near ? "grab" : "default";
            return;
        }
        const L = this.host.last;
        if (!L) return;
        this.curve.style.cursor = "grabbing";
        const v = Math.min(1, Math.max(0, (p.x - G.x0) / (G.x1 - G.x0)));
        const o = Math.min(1, Math.max(0, (G.y0 - p.y) / (G.y0 - G.y1)));
        const P = L.P;
        const d = this.drag;
        const host = this.host;
        if (d.id === "black") { host.set("blacks", Math.min(v, P.wp - 0.02)); d.tip = T("black from {v}").replace("{v}", v.toFixed(2)); }
        if (d.id === "white") { host.set("whites", 1 - Math.max(v, P.bp + 0.02)); d.tip = T("white from {v}").replace("{v}", v.toFixed(2)); }
        if (d.id === "hero") {
            host.set("hero_plane", M.levels(v, P));
            const pop = Math.min(14, Math.max(0, d.pop + (d.y - p.y) / 6));
            host.set("hero_pop", pop);
            d.tip = T("hero {h} · power {p}").replace("{h}", M.levels(v, P).toFixed(2)).replace("{p}", pop.toFixed(1));
        }
        if (d.id === "from") {
            const k = M.hero(M.levels(v, P), P);
            host.set("ceiling_from", k);
            if ((host.widget("foreground_ceiling")?.value ?? 1) >= 1) host.set("foreground_ceiling", k + 0.3 * (1 - k));
            d.tip = T("dim from {k}").replace("{k}", k.toFixed(2));
        }
        if (d.id === "ceil") { host.set("foreground_ceiling", o); d.tip = T("ceiling {o}").replace("{o}", o.toFixed(2)); }
        if (d.id === "mid") {
            const m = Math.min(2, Math.max(-2, d.mid + (d.y - p.y) / 60));
            host.set("midtones", m);
            d.tip = T("midtones {m}").replace("{m}", (m >= 0 ? "+" : "") + m.toFixed(2));
        }
        host.schedule();
    }

    onCurveDbl(e) {
        const G = this.geom(), p = this.pos(e);
        const X = (v) => G.x0 + v * (G.x1 - G.x0);
        const groups = { black: ["blacks"], white: ["whites"], hero: ["hero_plane", "hero_pop"],
            from: ["ceiling_from", "foreground_ceiling"] };
        if (Math.abs(p.x - (G.x1 + 14)) < 12) return this.host.reset(["foreground_ceiling", "ceiling_from"]);
        for (const m of this.hs || []) if (Math.abs(p.x - X(m.v)) < 12 && p.y > G.y0 - 30) return this.host.reset(groups[m.id]);
        this.host.reset(["midtones"]);
    }

    onCurveUp(e) {
        if (!this.drag) return;
        this.drag = null;
        try { this.curve.releasePointerCapture(e.pointerId); } catch (_) { /* уже відпущено */ }
        this.host.schedule();
    }
}
