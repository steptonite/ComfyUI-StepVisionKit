// Мова нод StepVisionKit — окремо від мови самого ComfyUI.
// Базова мова паку англійська (вимога Registry). Українська — словник web/i18n/uk.json:
//   nodes — назви, описи й підказки нод за ключем ноди (підміняємо у відповіді /object_info до реєстрації нод),
//   ui    — підписи наших карток: T("English") → «Українська».
// Чому не locales/ Comfy: та локалізація перемикає весь інтерфейс і підписи в ній пливуть,
// а тут Comfy може лишатись англійським, а ноди паку — українськими.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

export const LANG_SETTING = "StepVisionKit.Language";

function setting(id) {
    try { return app.extensionManager?.setting?.get(id) ?? app.ui?.settings?.getSettingValue(id); } catch (_) { return undefined; }
}
function readLang() {
    const v = setting(LANG_SETTING) || "en";
    if (v !== "auto") return v;
    const loc = setting("Comfy.Locale") || navigator.language || "en";
    return String(loc).toLowerCase().startsWith("uk") ? "uk" : "en";
}

// Мова фіксується на завантаженні сторінки: назви нод уже зареєстровані, тож зміна діє після перезавантаження.
export const LANG = readLang();
let DICT = { nodes: {}, ui: {} };
if (LANG === "uk") {
    try { DICT = await (await fetch(new URL("./i18n/uk.json", import.meta.url))).json(); }
    catch (e) { console.warn("[StepVisionKit] uk.json not loaded, staying in English", e); }
}

export function T(s) { return LANG === "uk" ? (DICT.ui?.[s] ?? s) : s; }

function translateDefs(defs) {
    for (const [key, t] of Object.entries(DICT.nodes || {})) {
        const d = defs[key];
        if (!d) continue;
        if (t.display_name) d.display_name = t.display_name;
        if (t.description) d.description = t.description;
        if (t.output_tooltips) d.output_tooltips = t.output_tooltips;
        for (const [name, it] of Object.entries(t.inputs || {})) {
            const spec = d.input?.required?.[name] ?? d.input?.optional?.[name];
            if (spec && it.tooltip) spec[1] = { ...(spec[1] || {}), tooltip: it.tooltip };
        }
    }
    return defs;
}

app.registerExtension({
    name: "StepVisionKit.I18n",
    settings: [{
        id: LANG_SETTING,
        name: "Node language",
        category: ["StepVisionKit", "Language", "Node language"],
        type: "combo",
        defaultValue: "en",
        options: [
            { text: "English", value: "en" },
            { text: "Українська", value: "uk" },
            { text: "Auto (same as ComfyUI)", value: "auto" },
        ],
        tooltip: "Language of StepVisionKit nodes only; the rest of ComfyUI keeps its own language. Reload the page to apply.",
        onChange: (v) => {
            if (!v || v === LANG || (v === "auto" && readLang() === LANG)) return;
            app.extensionManager?.toast?.add({ severity: "info", summary: "StepVisionKit: reload the page to switch node language", life: 5000 });
        },
    }],
    async init() {
        if (LANG !== "uk") return;
        const orig = api.getNodeDefs.bind(api);
        api.getNodeDefs = async (...a) => translateDefs(await orig(...a));
    },
    nodeCreated(node) {
        const t = LANG === "uk" && DICT.nodes?.[node.comfyClass];
        if (!t?.inputs) return;
        for (const w of node.widgets || []) {
            const lbl = t.inputs[w.name]?.label;
            if (lbl) w.label = lbl;
        }
    },
});
