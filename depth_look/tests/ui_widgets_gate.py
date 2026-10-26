"""Гейт видимості віджетів (26.10.2026): нода DepthLook ховає ВСІ нативні віджети (hideAll у web/depth_look.js),
тож новий вхід у INPUT_TYPES без місця в UI зникає непомітно (так загубився auto_run).
Кожен вхід мусить бути показаний (control(...) / MORE) або явно записаний у HIDDEN_BY_DESIGN з причиною.
Статичний розбір тексту, нічого не імпортує й не рендерить. Запуск: python3 depth_look/tests/ui_widgets_gate.py"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKS = {   # клас -> (python-файл, js-файл)
    "DepthLook": (ROOT / "depth_look/__init__.py", Path(__import__("os").environ.get("SVK_GATE_JS", ROOT / "web/depth_look.js"))),
}
HIDDEN_BY_DESIGN = {
    "DepthLook": {"preset": "обирається в тюнері", "overrides": "пише тюнер", "palette": "пише тюнер",
                  "params_json": "пише тюнер", "mode": "ставить кнопка Render final",
                  "image": "вхід-зʼєднання", "depth": "вхід-зʼєднання", "grade": "вхід-зʼєднання"},
}

TUNE = Path(__import__("os").environ.get("SVK_GATE_TUNE", ROOT / "depth_look/tune.html"))
TUNER_WIRED = {"DepthLook": {"auto_run": "depth_look_auto"}}   # вхід -> повідомлення тюнер<->нода

def wired_in_tuner(cls, js):
    """показано в тюнері: tune.html шле повідомлення, а нода його обробляє й пише саме цей віджет"""
    t, j = TUNE.read_text(), js.read_text()
    return {n for n, msg in TUNER_WIRED.get(cls, {}).items()
            if 'type: "%s"' % msg in t and 'm.type === "%s"' % msg in t
            and 'm.type === "%s"' % msg in j and 'x.name === "%s"' % n in j}

def inputs_of(py, cls):
    src = py.read_text()
    body = src[src.index("class %s" % cls):]
    body = body[:body.index("RETURN_TYPES")]
    return set(re.findall(r'^\s*"(\w+)":\s*\(', body, re.M))

def shown_in(js):
    src = js.read_text()
    names = set(re.findall(r'control\(node,\s*"(\w+)"', src))
    for m in re.finditer(r"const (?:MORE|OWN) = \[(.*?)\];", src, re.S):
        names |= set(re.findall(r'\["(\w+)"', m.group(1)))
    return names

bad = []
for cls, (py, js) in CHECKS.items():
    lost = inputs_of(py, cls) - shown_in(js) - wired_in_tuner(cls, js) - set(HIDDEN_BY_DESIGN.get(cls, {}))
    bad += ["%s.%s" % (cls, n) for n in sorted(lost)]
# Camera Realism: входи - SPEC у realism/params.py + кілька службових; показані мають бути control(...) або MORE у web/camera_realism.js
REALISM_HIDDEN = {"image": "вхід-зʼєднання", "depth": "вхід-зʼєднання", "profile": "плитки профілю на картці пишуть цей віджет"}
rp = ROOT / "realism/params.py"
rj = ROOT / "web/camera_realism.js"
if rp.exists() and rj.exists():
    spec_src = rp.read_text()
    spec_src = spec_src[spec_src.index("SPEC = ["):spec_src.index("NAMES =")]
    r_inputs = set(re.findall(r'^\s*\("(\w+)",', spec_src, re.M)) | {"profile", "strength", "seed"}
    r_lost = r_inputs - shown_in(rj) - set(REALISM_HIDDEN)
    bad += ["SVKCameraRealism.%s" % n for n in sorted(r_lost)]
# Спільна картка (web/svk_cards.js): невказані віджети самі йдуть під «ще», тож загубити вхід вона не може.
# Ловимо інше: ключ ноди чи імʼя в MAIN з одруківкою - тоді головне керування мовчки опиняється під «ще».
cj = Path(__import__("os").environ.get("SVK_GATE_CARDS", ROOT / "web/svk_cards.js"))
py_src = {p: p.read_text() for p in ROOT.glob("*/*.py") if "tests" not in p.parts}
for key, body in re.findall(r"^\s*(\w+): (\[.*?\]|null),", re.search(r"const MAIN = \{(.*?)\n\};", cj.read_text(), re.S).group(1), re.M):
    py = next((p for p, s in py_src.items() if re.search(r"^class %s\b" % key, s, re.M)), None)
    if py is None:
        bad.append("svk_cards.js: нода %s не знайдена" % key); continue
    bad += ["svk_cards.js: %s.%s не вхід ноди" % (key, n) for n in re.findall(r'"(\w+)"', body) if n not in inputs_of(py, key)]
if bad:
    print("🔴 віджети без місця в UI (сховані hideAll):", ", ".join(bad))
    print("   постав у control(...)/MORE у web/*.js або додай у HIDDEN_BY_DESIGN з причиною")
    sys.exit(1)
print("✅ усі входи DepthLook і SVKCameraRealism видимі або сховані свідомо; MAIN спільної картки збігається з входами")
