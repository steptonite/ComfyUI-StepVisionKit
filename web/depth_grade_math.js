// Тонова частина SVKDepthGrade, перекладена 1:1 з lex/depth_grade.py — для живого прев'ю.
//
// Браузер отримує мініатюру СИРОГО депсу й таблиці квантилів, а далі грейдить сам:
// повзунки й маркери відгукуються миттєво, без Run. Просторові ефекти (clarity,
// texture, smooth_banding) і deflicker тут не рахуються — вони лишаються за Run.
// 🔴 Міняєш формулу в depth_grade.py — міняй і тут, інакше прев'ю збреше.

export const PCT_LO = Array.from({ length: 41 }, (_, i) => i * 0.5);        // 0…20 %
export const PCT_HI = Array.from({ length: 41 }, (_, i) => 80 + i * 0.5);   // 80…100 %

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);

function interp(table, pcts, p) {
    if (p <= pcts[0]) return table[0];
    const last = pcts.length - 1;
    if (p >= pcts[last]) return table[last];
    const i = Math.floor((p - pcts[0]) / 0.5);
    const t = (p - pcts[i]) / 0.5;
    return table[i] * (1 - t) + table[i + 1] * t;
}

function xq(lo, hi, p) {
    return p <= 20 ? interp(lo, PCT_LO, p) : interp(hi, PCT_HI, p);
}

// Межі нормалізації відстані — як _lohi(dist, trim_near, 100 - trim_far) у Python.
// Для «near is white» dist = 1 - x, тож квантиль дзеркалиться: q_dist(p) = 1 - q_x(100 - p).
export function distRange(prev, w) {
    const clip = w.levels_scope === "whole clip" && prev.frames > 1;
    const lo = clip ? prev.clip_lo : prev.frame_lo;
    const hi = clip ? prev.clip_hi : prev.frame_hi;
    const near = w.depth_is === "near is white";
    const q = (p) => (near ? 1 - xq(lo, hi, 100 - p) : xq(lo, hi, p));
    return [q(w.trim_near), q(100 - w.trim_far)];
}

export function params(w) {
    const bp = +w.blacks;
    return {
        look: w.look,
        nearBias: 2.0 * Math.pow(0.01, +w.foreground_pop),
        bp,
        wp: Math.max(1 - +w.whites, bp + 1e-3),
        gamma: Math.pow(2, -w.midtones),
        heroS: +w.hero_pop,
        heroP: +w.hero_plane,
        c: +w.foreground_ceiling,
        k: +w.ceiling_from,
        flip: !!w.flip,
    };
}

// d (0 = найближче) → v: «об'єм V2» або лінійно
export function volume(d, P) {
    if (P.look === "V2 volume") {
        const lo = 1 / (1 + P.nearBias), hi = 1 / P.nearBias;
        return clamp01((1 / (d + P.nearBias) - lo) / Math.max(hi - lo, 1e-6));
    }
    return 1 - d;
}

// v → u: рівні + середні тони
export function levels(v, P) {
    const u = clamp01((v - P.bp) / (P.wp - P.bp));
    return P.gamma !== 1 ? Math.pow(u, P.gamma) : u;
}

// u → h: S-крива навколо героя
export function hero(u, P) {
    if (!(P.heroS > 0)) return u;
    const s = P.heroS, p = P.heroP;
    const g0 = 1 / (1 + Math.exp(s * p));
    const g1 = 1 / (1 + Math.exp(-s * (1 - p)));
    return clamp01((1 / (1 + Math.exp(-s * (u - p))) - g0) / Math.max(g1 - g0, 1e-6));
}

// h → стеля переднього плану
export function ceiling(h, P) {
    const { c, k } = P;
    if (!(c < 1 && k < 1) || h <= k) return h;
    const t = (h - k) / (1 - k);
    let shape;
    if (c > k) {
        const pw = (1 - k) / Math.max(c - k, 1e-3);
        shape = 1 - Math.pow(1 - t, pw);
    } else {
        shape = t * t * (3 - 2 * t);
    }
    return k + (c - k) * shape;
}

export function tone(v, P) {
    const o = ceiling(hero(levels(v, P), P), P);
    return P.flip ? 1 - o : o;
}

// Таблиця v → вихід (кривa для малювання і швидкий прохід по пікселях)
export function buildLUT(P, n = 4096) {
    const lut = new Float32Array(n);
    for (let i = 0; i < n; i++) lut[i] = tone(i / (n - 1), P);
    return lut;
}

// Знайти v, при якому монотонна f(v) досягає target (для маркерів на осі)
export function invert(f, target) {
    let a = 0, b = 1;
    if (f(1) <= target) return 1;
    if (f(0) >= target) return 0;
    for (let i = 0; i < 40; i++) {
        const m = (a + b) / 2;
        if (f(m) < target) a = m; else b = m;
    }
    return (a + b) / 2;
}

// Розкодувати мініатюру з Python: uint16 LE у [x0, x1]
export function decodePreview(payload) {
    const bin = atob(payload.data);
    const n = payload.w * payload.h;
    const x = new Float32Array(n);
    const span = payload.x1 - payload.x0;
    for (let i = 0; i < n; i++) {
        const u = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
        x[i] = payload.x0 + (u / 65535) * span;
    }
    return { ...payload, x };
}

// Повний прохід: сирий x → v (після об'єму) і вихід 0…1
export function grade(prev, w) {
    const P = params(w);
    const [lo, hi] = distRange(prev, w);
    const near = w.depth_is === "near is white";
    const n = prev.x.length;
    const v = new Float32Array(n);
    const out = new Float32Array(n);
    const lut = buildLUT(P);
    const L = lut.length - 1;
    const span = Math.max(hi - lo, 1e-6);
    for (let i = 0; i < n; i++) {
        const dist = near ? 1 - prev.x[i] : prev.x[i];
        const vi = volume(clamp01((dist - lo) / span), P);
        v[i] = vi;
        out[i] = lut[Math.round(vi * L)];
    }
    return { v, out, P, lut };
}
