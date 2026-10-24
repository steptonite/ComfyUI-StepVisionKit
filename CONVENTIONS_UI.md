# StepVisionKit — домовленості UI нод

Діє для всіх нод паку. Перші три ноди за цим стандартом — Depth Look (повний, простий, відео), 24.10.2026.
Токени й компоненти лежать у `web/svk_ui.css`, спільний JS — у `web/svk_ui.js`. Кольори й розміри міняються там, а не у файлах нод.

## Правила

1. **На ноді немає абзаців тексту.** Пояснення дається лише через `title`: підказка зʼявляється, коли тримаєш курсор над елементом. Видимими лишаються підпис (1–3 слова), число або стан.
2. **Вигляд один на всі ноди.** Колір ноди — `#2a2a2f`, тло — `#151518`, акцент — бурштин `--svk-accent`. Фіолетового чи неону немає.
3. **Нативні віджети Comfy, що дублюють картку, сховані** через `hideWidget` зі `svk_ui.js`. Їхні значення все одно зберігаються у workflow і йдуть у Run. Картка лише читає й пише ці віджети через `control(node, name, spec, label)`.
4. **На видноті 3–5 головних керувань.** Усе інше ховається під «⋯ ще» (`morePanel`). Порядок такий: кадр або превʼю → вибір (плитки або сегмент) → 3–4 повзунки → «ще».
5. **Висота ноди не росте від вмісту.** Картка має клас `svk-card svk-scroll` і гортається колесом усередині себе (`keepWheel`). Полотно Comfy під нею не масштабується. У Nodes 2.0 обгортка ноди сама перехоплює колесо й віддає полотну, тому `keepWheel`/`onWheel` ловлять його ще на `window` (capture). Висоту картки в Nodes 2.0 обмежує `--svk-card-max-h` (типово 560px), інакше нода росла б разом із «ще». Ctrl + колесо чи щипок і далі масштабують полотно.
6. **Кнопки — іконки в один ряд** (`.svk-ibtn` у `.svk-icons` або `.svk-bar`). Назва й гаряча клавіша вказані в `title`, стан «увімкнено» позначається класом `.is-on`.
7. **Два-три варіанти вибору показуються сегментом** (`.svk-seg`), випадний список для них не потрібен. Короткі підписи — «превʼю / фінал», повна назва — у `title`.
8. **Повзунок** (`.svk-sl`): назва й число стоять в одному рядку, повзунок займає всю ширину з заливкою до значення. Подвійний клік по назві повертає типове значення. Якщо діапазон більший за ~4000 кроків (px, fps), замість повзунка ставиться поле з числом.
9. **Чіпи ефектів переносяться на новий рядок** (`auto-fill, minmax(78px, 1fr)`), підпис не обрізається. Тап вмикає або вимикає ефект. Перетяг чи колесо змінюють силу 0..2, яку видно як заливку чіпа.
10. **Лук має людську назву:** `presetLabel()` перетворює `look_v1` на «Лук v1». Імʼя файлу видно лише в `title`.
11. **Перевірка — у режимі Nodes 2.0.** Там назва ноди стоїть унизу й теж займає висоту. Ширина картки — 340–360 px, мінімальна висота DOM-віджета — 380–420 px.
12. **Ключі нод не перейменовуються.** Збережені workflow шукають ноду за `type`, тому, наприклад, `DepthLook*` лишились із попереднього окремого паку. Змінюються лише категорія меню (`StepVisionKit/<Розділ>`) і відображувана назва.

## Анатомія картки

```
┌ кадр / превʼю (утримати = оригінал)        ┐
│ [плитки луків → гортаються вбік]           │
│ ефекти: чіпи, переносяться на новий рядок  │
│ сила луку ━━━━━━━━○──  80%                 │
│ ⫶ ще                                       │  ← рідкісне: розмір, роздільність, глибина
└ назва ноди (Nodes 2.0 — внизу)             ┘
```

## Таблиця «нода SVK → що змінити»

Таблиця згенерована з `object_info` (24.10.2026). Позначка «✅ зроблено» означає, що нода вже відповідає правилам вище.

| Категорія | Нода | Ключ | Віджетів | Що змінити |
|---|---|---|---|---|
| StepVisionKit/Depth | Depth Curve (DA3 → вигляд як V2) | `LexDepthCurve` | 16 | 16 віджетів: 3–4 головні на видноті, решта під «ще» |
| StepVisionKit/Depth | Depth Grade Live (стара, заміни на Depth Grade + Depth Grade View) | `SVKDepthGradeLive` | 20 | має свій JS (depth_grade_ui.js): звести на svk_ui.css/svk_ui.js, тексти → title |
| StepVisionKit/Depth | Depth Grade View · відео (живе прев'ю) | `SVKDepthGradeVideoView` | 1 | має свій JS (depth_grade_ui.js): звести на svk_ui.css/svk_ui.js, тексти → title |
| StepVisionKit/Depth | Depth Grade View · кадр (калібрування) | `SVKDepthGradeView` | 0 | має свій JS (depth_grade_ui.js): звести на svk_ui.css/svk_ui.js, тексти → title |
| StepVisionKit/Depth | Depth Grade 🎚 (грейд глибини + антимерехтіння) | `SVKDepthGrade` | 20 | має свій JS (depth_grade_ui.js): звести на svk_ui.css/svk_ui.js, тексти → title |
| StepVisionKit/Depth | Depth Look · відео | `DepthLookVideo` | 23 | ✅ зроблено: картка, «ще», підказки |
| StepVisionKit/Depth | Depth Look · повний | `DepthLook` | 10 | ✅ зроблено: картка, «ще», підказки |
| StepVisionKit/Depth | Depth Look · простий | `DepthLookSimple` | 27 | ✅ зроблено: картка, «ще», підказки |
| StepVisionKit/Depth | Depth Receiver (raw DA3 → float depth) | `SVKDepthReceiver` | 1 | лише колір/категорія; тексти tooltip перевірити |
| StepVisionKit/Depth | Raw Depth Load (EXR float32 → DA3 geometry) | `SVKRawDepthLoad` | 1 | має свій JS (depth_raw_ui.js): звести на svk_ui.css/svk_ui.js, тексти → title |
| StepVisionKit/Depth | Raw Depth Save (DA3 geometry → EXR float32) | `SVKRawDepthSave` | 2 | має свій JS (depth_raw_ui.js): звести на svk_ui.css/svk_ui.js, тексти → title |
| StepVisionKit/Depth | Save Depth 16-bit / EXR | `SVKSaveDepth16` | 3 | має свій JS (depth_raw_ui.js): звести на svk_ui.css/svk_ui.js, тексти → title |
| StepVisionKit/Detect | Boxes To BBOX | `VKBoxesToBBOX` | 2 | лише колір/категорія; тексти tooltip перевірити |
| StepVisionKit/Detect | Boxes → BBOX (для SAM2) | `LexBoxesToBBOX` | 5 | лише колір/категорія; тексти tooltip перевірити |
| StepVisionKit/Detect | Detect Objects (YOLO) | `VKDetect` | 9 | 9 віджетів: 3–4 головні на видноті, решта під «ще» |
| StepVisionKit/Detect | Select Box | `VKSelectBox` | 9 | 9 віджетів: 3–4 головні на видноті, решта під «ще» |
| StepVisionKit/Detect | YOLO Detect | `LexYOLODetect` | 16 | 16 віджетів: 3–4 головні на видноті, решта під «ще» |
| StepVisionKit/HUD | Cut By Mask | `VKCutByMask` | 6 | 6 віджетів: 3–4 головні на видноті, решта під «ще» |
| StepVisionKit/HUD | Draw Boxes (HUD) | `VKDrawBoxes` | 15 | 15 віджетів: 3–4 головні на видноті, решта під «ще» |
| StepVisionKit/HUD | HUD Boxes (кути/підписи) | `LexHUDBoxes` | 6 | 6 віджетів: 3–4 головні на видноті, решта під «ще» |
| StepVisionKit/HUD | HUD Sprite (кастомна рамка PNG + дірка) | `LexHUDSprite` | 21 | 21 віджетів: 3–4 головні на видноті, решта під «ще» |
| StepVisionKit/HUD | Overlay On Boxes | `VKOverlayOnBoxes` | 15 | 15 віджетів: 3–4 головні на видноті, решта під «ще» |

