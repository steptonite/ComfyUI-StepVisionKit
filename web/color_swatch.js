import { app } from "../../scripts/app.js";
import { T } from "./svk_i18n.js";

// Кнопка «обрати колір» біля текстових полів, у яких лежить hex.
//
// Чому кнопка, а не DOM-віджет зі свотчем: addDOMWidget за версіями фронтенда
// поводиться по-різному й уміє не серіалізуватись, а addWidget("button") є в
// litegraph з першого дня. Значення при цьому лишається у звичайному STRING-віджеті,
// тож воркфлоу зберігається/відкривається як завжди, і без цього js нода теж робоча —
// просто hex доведеться вписати руками.

const TARGETS = {
    LexHUDSprite: ["tint"],
    LexHUDBoxes: ["color"],
    VKDrawBoxes: ["color"],
    VKOverlayOnBoxes: ["tint"],
    VKCutByMask: ["color"],
};

function pick(node, widget) {
    const input = document.createElement("input");
    input.type = "color";
    input.value = /^#[0-9a-f]{6}$/i.test(widget.value || "") ? widget.value : "#00ff00";
    input.style.cssText = "position:fixed;left:-9999px;top:-9999px";
    document.body.appendChild(input);
    input.addEventListener("input", () => {
        widget.value = input.value;
        node.onWidgetChanged?.(widget.name, widget.value, widget.value, widget);
        app.graph.setDirtyCanvas(true, true);
    });
    input.addEventListener("change", () => input.remove(), { once: true });
    input.click();
}

app.registerExtension({
    name: "StepVisionKit.ColorSwatch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const names = TARGETS[nodeData?.name];
        if (!names) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            for (const name of names) {
                const w = this.widgets?.find((x) => x.name === name);
                if (!w) continue;
                const btn = this.addWidget("button", `🎨 ${name}`, null, () => pick(this, w));
                btn.serialize = false;
                // кнопка сідає одразу під своє поле, а не в кінець списку
                const i = this.widgets.indexOf(w);
                this.widgets.splice(this.widgets.indexOf(btn), 1);
                this.widgets.splice(i + 1, 0, btn);

                if (name === "tint") {
                    const off = this.addWidget("button", T("✕ no tint"), null, () => {
                        w.value = "";
                        app.graph.setDirtyCanvas(true, true);
                    });
                    off.serialize = false;
                    this.widgets.splice(this.widgets.indexOf(off), 1);
                    this.widgets.splice(i + 2, 0, off);
                }
            }
            return r;
        };
    },
});
