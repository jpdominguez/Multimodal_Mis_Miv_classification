#!/usr/bin/env python3
"""
Vigilancia de una batería exploratoria (x8, x9 + familia; protocolo fijado el 2026-10-20), para poder
retomarla en otra sesión. No lee el test.

  sbatch --export=ALL,ARGS="--tag x8 --train-job 74853 --tasks slurm/tasks_x8.txt --gate 23_gate_x8.py \
      --decision results/tables/x8_decision.json --phase2 none" pipeline/slurm/watch_battery.sbatch

Cada 2 minutos, mientras haya entrenamientos en cola o en marcha:
  * results/<tag>_progress.md: estado actual (trabajos, entrenamientos terminados con su AUC de validación, fallos,
    puerta y siguiente paso);
  * results/<tag>_events.log: una línea por cada paso nuevo (entrenamiento terminado o fallido, relanzamiento,
    puerta, envío de trabajos). Si el vigilante se reinicia, no repite los pasos ya registrados.
Al terminar la fase 1 relanza una vez lo que haya fallado, espera si hace falta a otra batería (--after) y ejecuta
la puerta. Con --phase2 auto, si la puerta pasa, entrena los modelos que falten en las repeticiones 1 y 2 y los
vigila. El test se abre después, una sola vez, como fija el protocolo.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REV = Path(__file__).resolve().parent
ROOT = REV.parent
EXP_DIR = REV / "runs_exploratory"
RES = REV / "results"
CHUNK, POLL, AFTER_TIMEOUT_H = 4, 120, 8
TAG = "x8"
TITLES = {"x8": "Batería x8: ajuste de lr × wd en los dos brazos",
          "x9": "Batería x9 (red base × resolución) y puerta de la familia x8 + x9"}


def now(ts=None):
    return (datetime.fromtimestamp(ts) if ts else datetime.now()).strftime("%Y-%m-%d %H:%M")


def events_file(tag=None):
    return RES / f"{tag or TAG}_events.log"


def event(msg, ts=None):
    with open(events_file(), "a") as f:
        f.write(f"{now(ts)}  {msg}\n")
    print(f"{now(ts)}  {msg}", flush=True)


def read_tasks(path):
    return [tuple(l.split()[:2]) for l in path.read_text().splitlines() if l.strip()]


def queue(jobs):
    """Estado de los trabajos en SLURM: {estado: n}; {} si ya no queda ninguno; None si squeue falla (se reintenta)."""
    r = subprocess.run(["squeue", "-h", "-r", "-o", "%T", "-j", ",".join(map(str, jobs))],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {} if "Invalid job id" in r.stderr else None
    states = {}
    for s in r.stdout.split():
        states[s] = states.get(s, 0) + 1
    return states


def run_info(exp, split, base=EXP_DIR):
    f = base / exp / split / "run.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return {"val_auc": r["best_val_auc"], "epoch": r["best_epoch"], "minutes": r["minutes"], "device": r["device"],
            "ts": f.stat().st_mtime}


def failures(jobs):
    out = []
    for j in jobs:
        for f in sorted((REV / "logs").glob(f"x_{j}_*.out")):
            out += [tuple(l.split()[1:3]) for l in f.read_text(errors="replace").splitlines() if l.startswith("FALLO")]
    return out


def submit(tasks_file, lines):
    tasks_file.write_text("\n".join(f"{e} {s}" for e, s in lines) + "\n")
    n = (len(lines) + CHUNK - 1) // CHUNK
    # sin las variables SLURM_* del propio trabajo de vigilancia, que sbatch heredaría
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    r = subprocess.run(["sbatch", "--parsable", f"--array=0-{n - 1}%8",
                        f"--export=ALL,TASKS={tasks_file.relative_to(ROOT)},CHUNK={CHUNK}",
                        "pipeline/slurm/run_array_x.sbatch"], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        event(f"ERROR al enviar {tasks_file.name}: {r.stderr.strip()}")
        return None
    job = int(r.stdout.strip().split(";")[0])
    event(f"Enviado el job {job}: {len(lines)} entrenamientos ({tasks_file.name})")
    return job


def gate_summary(dec):
    if "best_image" in dec:      # formato de 23_gate_x8.py
        return (f"mejor image lr {dec['best_image'][0]} wd {dec['best_image'][1]}; mejor concat lr "
                f"{dec['best_concat'][0]} wd {dec['best_concat'][1]}; ΔAUC val {dec['val_delta_mean']:+.4f} "
                f"({dec['val_delta_n_positive']}/5), VRUH {dec['val_delta_VRUH_mean']:+.4f} "
                f"({dec['val_delta_VRUH_n_positive']}/5) -> {'PASA' if dec['passes'] else 'NO PASA'}")
    n_ok = sum(c["passes"] for c in dec["candidates"])
    return (f"{n_ok}/{len(dec['candidates'])} candidatos pasan; " +
            (f"candidato final {dec['final_candidate']} (ΔAUC val {dec['val_delta_mean']:+.4f})" if dec["passes"]
             else "ninguno pasa"))


class Watcher:
    def __init__(self, title, gate):
        self.title, self.gate = title, gate
        self.phase, self.jobs, self.tasks = "", [], []
        self.gate_txt, self.next_step = "", ""
        # pasos ya registrados (por si el vigilante se reinicia)
        old = events_file().read_text() if events_file().exists() else ""
        self.seen = set(re.findall(r"  FIN (\S+) (\S+):", old))
        self.seen_fail = set(re.findall(r"  FALLO (\S+) (\S+) ", old))

    def write(self, states=None):
        done = {t: run_info(*t) for t in self.tasks}
        n_done = sum(v is not None for v in done.values())
        fails = [t for t in failures(self.jobs) if t in done and done[t] is None]
        exps = list(dict.fromkeys(e for e, _ in self.tasks))
        splits = sorted({s for _, s in self.tasks})
        q = ("estado desconocido" if states is None
             else ", ".join(f"{v} {k.lower()}" for k, v in states.items()) or "ninguno en cola")
        lines = [
            f"# {self.title}",
            "",
            f"**Actualizado:** {now()} (cada {POLL // 60} min mientras haya trabajos).",
            f"**Protocolo:** fijado antes de entrenar. **Puerta:** `{self.gate}`. "
            f"**Test:** sin abrir. **Pasos:** `results/{TAG}_events.log`.",
            "",
            "## Estado",
            "",
            f"- **Fase:** {self.phase}",
            f"- **Trabajos SLURM:** {', '.join(map(str, self.jobs))} ({q}).",
            f"- **Entrenamientos terminados:** {n_done}/{len(self.tasks)}; fallidos sin repetir: {len(fails)}.",
            "",
            "## AUC en validación interna (por imagen)",
            "",
            "| Experimento | " + " | ".join(splits) + " | Media |",
            "|---|" + "---|" * (len(splits) + 1),
        ]
        # referencia (receta actual, ya entrenada en las 15 particiones) y después la batería
        refs = {(e, s): run_info(e, s, REV / "runs") for e in ("image", "concat_meta") for s in splits}
        for e in ("image", "concat_meta", *exps):
            cells, got = [], []
            for s in splits:
                v = refs.get((e, s)) or done.get((e, s))
                if v:
                    got.append(v["val_auc"])
                cells.append(f"{v['val_auc']:.3f}" if v else "FALLO" if (e, s) in fails
                             else "…" if (e, s) in done else "")
            mean = (f"{sum(got) / len(got):.3f}" + (f" ({len(got)}/{len(splits)})" if len(got) < len(splits) else "")
                    if got else "")
            label = (f"`{e}` (referencia: DenseNet121, 224 px, lr 1e-4, wd 0,01)" if (e, splits[0]) in refs
                     else f"`{e}`")
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {mean} |")
        if fails:
            lines += ["", "## Fallos", ""] + [f"- `{e}` {s}" for e, s in fails]
        if self.gate_txt:
            lines += ["", "## Puerta en validación", "", "```", self.gate_txt.strip(), "```"]
        if self.next_step:
            lines += ["", "## Siguiente paso", "", self.next_step]
        (RES / f"{TAG}_progress.md").write_text("\n".join(lines) + "\n")
        # una línea en el registro por cada paso nuevo
        for t, v in sorted(((t, v) for t, v in done.items() if v and t not in self.seen), key=lambda x: x[1]["ts"]):
            self.seen.add(t)
            event(f"FIN {t[0]} {t[1]}: val_auc={v['val_auc']:.4f}, época {v['epoch']}, "
                  f"{v['minutes']:.1f} min ({v['device']})", ts=v["ts"])
        for t in fails:
            if t not in self.seen_fail:
                self.seen_fail.add(t)
                event(f"FALLO {t[0]} {t[1]} (ver pipeline/logs/x_<job>_*.out)")

    def wait(self, jobs):
        while True:
            states = queue(jobs)
            self.write(states)
            if states == {}:
                return
            time.sleep(POLL)

    def watch(self, name, phase, jobs, tasks):
        self.phase, self.jobs, self.tasks = phase, list(jobs), tasks
        event(f"Vigilando la {name}: {phase} (jobs {', '.join(map(str, jobs))}, {len(tasks)} entrenamientos)")
        self.wait(jobs)
        missing = [t for t in tasks if run_info(*t) is None]
        if missing:
            event(f"{len(missing)} entrenamientos sin terminar; se relanzan una vez")
            job = submit(REV / "slurm" / f"tasks_{TAG}_retry_{len(self.jobs)}.txt", missing)
            if job:
                self.jobs.append(job)
                self.wait([job])
        missing = [t for t in tasks if run_info(*t) is None]
        self.write({})
        event(f"Fin de la {name}: {len(tasks) - len(missing)}/{len(tasks)} entrenamientos terminados")
        return missing


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="x8")
    ap.add_argument("--title", default=None, help="por defecto, según el tag (TITLES)")
    ap.add_argument("--train-job", type=int, required=True)
    ap.add_argument("--tasks", default="slurm/tasks_x8.txt", help="relativo a pipeline/")
    ap.add_argument("--gate", default="23_gate_x8.py")
    ap.add_argument("--decision", default="results/tables/x8_decision.json", help="relativo a pipeline/")
    ap.add_argument("--after", default=None,
                    help="tag de otra batería cuya fase 1 tiene que terminar antes de la puerta (p. ej. x8)")
    ap.add_argument("--phase2", choices=["auto", "none"], default="auto")
    args = ap.parse_args()
    TAG = args.tag
    decision, gate_out = REV / args.decision, RES / "tables" / f"{TAG}_gate_output.txt"
    w = Watcher(args.title or TITLES.get(TAG, f"Batería {TAG}"), args.gate)
    w.next_step = "Esperar a que terminen los entrenamientos de la fase 1."
    tasks = read_tasks(REV / args.tasks)
    missing = w.watch("fase 1", f"Fase 1: repetición 0 ({len(tasks)} entrenamientos)", [args.train_job], tasks)
    if missing:
        w.next_step = (f"Faltan {len(missing)} entrenamientos tras un relanzamiento; revisar los logs antes de "
                       f"ejecutar la puerta (`{args.gate}`).")
        w.write()
        event("Puerta no ejecutada: faltan entrenamientos")
        return

    if args.after:
        w.phase = f"Esperando a que termine la fase 1 de la batería {args.after}"
        w.next_step = f"Ejecutar la puerta (`{args.gate}`) cuando termine la fase 1 de la {args.after}."
        event(f"Esperando a la fase 1 de la batería {args.after}")
        t0 = time.time()
        while "Fin de la fase 1" not in (events_file(args.after).read_text() if events_file(args.after).exists() else ""):
            if time.time() - t0 > AFTER_TIMEOUT_H * 3600:
                w.next_step = (f"La fase 1 de la {args.after} no terminó en {AFTER_TIMEOUT_H} h; ejecutar la puerta "
                               f"a mano (`{args.gate}`).")
                w.write({})
                event(f"Tiempo de espera agotado para la batería {args.after}")
                return
            w.write({})
            time.sleep(POLL)

    event(f"Ejecutando la puerta en validación ({args.gate})")
    r = subprocess.run([sys.executable, str(REV / args.gate)], cwd=ROOT, capture_output=True, text=True)
    w.gate_txt = r.stdout + (f"\n[stderr]\n{r.stderr}" if r.returncode else "")
    gate_out.write_text(w.gate_txt)
    if r.returncode != 0 or not decision.exists():
        w.phase = "Detenida (error en la puerta)"
        w.next_step = (f"La puerta ha fallado al ejecutarse; ver el error arriba y "
                       f"`results/tables/{gate_out.name}`.")
        w.write({})
        event("ERROR en la puerta")
        return
    dec = json.loads(decision.read_text())
    event(f"Puerta: {gate_summary(dec)}")
    if not dec["passes"]:
        w.phase = "Terminada (la puerta no pasa)"
        w.next_step = ("Según el protocolo, termina aquí: registrarla con el número total de "
                       "variantes y candidatos. Cualquier batería nueva necesita su propio protocolo, fijado antes "
                       "de entrenar.")
        w.write({})
        return
    if args.phase2 == "none":
        w.phase = "Terminada (la puerta pasa; el paso a las 15 particiones lo decide la familia)"
        w.next_step = ("Según la aclaración del protocolo, el candidato final y la fase 2 se deciden con la puerta de "
                       "la familia x8 + x9 (`25_gate_family.py`, vigilada en `results/x9_progress.md`).")
        w.write({})
        return

    # fase 2: los modelos del candidato que falten en las repeticiones 1 y 2 (la referencia ya las tiene)
    if "phase2_tasks" in dec:
        lines = [tuple(t) for t in dec["phase2_tasks"]]
        target = dec["final_candidate"]
    else:
        lines = [(e, f"cv_r{r_}_f{k}") for e in (dec["exp_image"], dec["exp_concat"]) if e.startswith("x8_")
                 for r_ in (1, 2) for k in range(5)]
        target = f"{dec['exp_concat']} frente a {dec['exp_image']}"
    todo = [t for t in lines if run_info(*t) is None]
    if todo:
        w.next_step = "Esperar a que terminen las repeticiones 1 y 2 de los modelos del candidato final."
        job = submit(REV / "slurm" / f"tasks_{TAG}_full.txt", todo)
        if job is None:
            w.next_step = f"No se pudo enviar la fase 2; enviarla a mano con `slurm/tasks_{TAG}_full.txt`."
            w.write({})
            return
        missing = w.watch("fase 2", f"Fase 2: candidato {target}, repeticiones 1 y 2", [job], lines)
        if missing:
            w.next_step = f"Faltan {len(missing)} entrenamientos de la fase 2; revisar los logs."
            w.write({})
            return
    w.phase = "Terminada (la puerta pasa; las 15 particiones están entrenadas)"
    w.next_step = (f"Abrir el test UNA vez, según el protocolo, para el candidato final ({target}): AUC por lesión, "
                   "IC bootstrap por paciente, DeLong y Holm sobre todas las fuentes y solo VRUH.")
    w.write({})
    event("Fase 2 terminada: lista para abrir el test según el protocolo")


if __name__ == "__main__":
    main()
