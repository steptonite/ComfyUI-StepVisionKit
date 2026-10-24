"""Воркер рендеру Depth Look: окремий процес, що рахує look для батча кадрів і ВИХОДИТЬ — уся памʼять вертається ОС.
Нащо: рендер усередині Comfy роздуває процес до 8 ГБ і не віддає памʼять назад (8-ГБ Мак свопить).
Це ж CLI для пакетної обробки агентами та хмарних GPU (Vast): жодного torch / Comfy / сервера.

Запуск:   python worker.py <job_dir>
Python:   потрібні лише numpy + opencv (рушій look.py, look_params.py, look_api.py - у цій же теці).

job_dir має містити:
  args.json   аргументи core.apply_numpy, окрім масивів: preset_name, clean_depth, light_is_near, overrides,
              palette, fps, start_time, params_json, max_side, depth_levels, output_size, static, grade, anim,
              frame_step (усі необовʼязкові -> дефолти apply_numpy)
  img.npy     (B,H,W,3) float32 0..1 (читається mmap-ом, не вантажиться цілком)
  depth.npy   (B|1,h,w) або (B|1,h,w,C) float32 (mmap)
  out.npy     (B,Ho,Wo,3) float32 — ПОВИНЕН бути створений заздалегідь батьком (Ho,Wo = фінальний розмір,
              див. core.out_shape): np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(B,Ho,Wo,3))
Вивід:    у stdout рядки "PROGRESS k n" (flush) після кожного кадру; решта stdout — лог рушія.
Код виходу: 0 = ок; !=0 = помилка, трейсбек у stderr.
Приклад (пакет):  python worker.py /шлях/до/job  &&  python -c "import numpy as np; print(np.load('/шлях/до/job/out.npy', mmap_mode='r').shape)"
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)            # `import core` без пакета (воркер запускається як скрипт)

_ARGS = ("preset_name", "clean_depth", "light_is_near", "overrides", "palette", "fps", "start_time",
         "params_json", "max_side", "depth_levels", "output_size", "static", "grade", "anim", "frame_step")


def main(job):
    import numpy as np
    import core
    args = json.load(open(os.path.join(job, "args.json"), encoding="utf-8"))
    kw = {k: args[k] for k in _ARGS if k in args}
    img = np.load(os.path.join(job, "img.npy"), mmap_mode="r")
    dep = np.load(os.path.join(job, "depth.npy"), mmap_mode="r")
    out = np.load(os.path.join(job, "out.npy"), mmap_mode="r+")

    def progress(k, n):
        print("PROGRESS %d %d" % (k, n), flush=True)

    core.apply_numpy(img, dep, out=out, progress=progress, **kw)
    out.flush()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    try:
        os.nice(5)
    except Exception:
        pass
    try:
        main(sys.argv[1])
    except Exception:
        traceback.print_exc()
        sys.exit(1)
