import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Raw Depth Load: кнопка відкриває системний вибір теки (macOS, osascript на сервері)
// і вписує шлях у поле folder. Поле лишається звичайним текстом — шлях можна й вставити.
// Ноди збереження кнопки не мають: шлях пишеться в імені (a/b/name), на сервері без екрана вибір не працює.

const RAW_NODES = ["SVKRawDepthLoad"];

function toast(severity, summary) {
    const tm = app.extensionManager?.toast;
    if (tm) tm.add({ severity, summary, life: 4000 });
    else alert(summary);
}

async function pickFolder(node) {
    const w = node.widgets.find((x) => x.name === "folder");
    const r = await api.fetchApi("/svk/raw/pick_folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: w.value }),
    }).then((x) => x.json());
    if (r.ok) {
        w.value = r.path;
        w.callback?.(w.value);
        node.setDirtyCanvas(true, true);
    } else if (!r.cancelled) {
        toast("warn", r.message);
    }
}

app.registerExtension({
    name: "StepVisionKit.RawDepthFolder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!RAW_NODES.includes(nodeData.name)) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const b = this.addWidget("button", "📁 Choose folder…", null, () => pickFolder(this));
            b.serialize = false;
            return r;
        };
    },
});
