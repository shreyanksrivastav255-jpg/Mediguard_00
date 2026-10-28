"""Mediguard dashboard backend.

Put the `dashboard` folder inside your Medigaurd00-main project, then run from
the project root:   python dashboard\\app.py   and open http://127.0.0.1:5000
"""
import itertools, os, subprocess, sys, threading, time
from flask import Flask, abort, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # project root, where the .py scripts live
app = Flask(__name__, static_folder=None)

S = lambda i, f, g, t, d, a=(), run=True: dict(id=i, file=f, group=g, title=t, desc=d, args=list(a), runnable=run)
SCRIPTS = [
    S("loader", "dataset_loader.py", "Set up", "Check the dataset", "Loads the X-ray data and prints classes and batch counts (expect 163 / 1 / 20)."),
    S("adapter", "model_adapter.py", "Set up", "Model adapter", "Shared wrapper that lets every test run on any model. A library, so view only.", run=False),
    S("baseline", "evaluate_baseline.py", "Test disturbances", "Clean baseline", "Accuracy, sensitivity and specificity on untouched test images."),
    S("noise", "evaluate_noise.py", "Test disturbances", "Noise", "Accuracy as random noise gets stronger."),
    S("blur", "evaluate_blur.py", "Test disturbances", "Blur", "Accuracy as images get blurrier."),
    S("brightness", "evaluate_brightness.py", "Test disturbances", "Brightness", "Accuracy as images get darker or brighter."),
    S("contrast", "evaluate_contrast.py", "Test disturbances", "Contrast", "Accuracy as contrast is reduced or boosted."),
    S("score", "robustness_score.py", "Score and compare", "Robustness score", "Combines all disturbance results into one score."),
    S("rep_cnn", "robustness_report.py", "Score and compare", "Report: SmallCNN", "Full robustness report for the SmallCNN model.", ["smallcnn"]),
    S("rep_res", "robustness_report.py", "Score and compare", "Report: ResNet18", "Full robustness report for the ResNet18 model.", ["resnet18"]),
    S("compare", "compare_models.py", "Score and compare", "Compare models", "SmallCNN and ResNet18 side by side."),
    S("viz", "visualize_results.py", "Score and compare", "Draw graphs", "Saves result graphs to the results folder."),
    S("train_cnn", "train_cnn.py", "Train models", "Train SmallCNN", "Overwrites pneumonia_cnn_best.pt. Slow."),
    S("train_res", "train_resnet.py", "Train models", "Train ResNet18", "Overwrites pneumonia_resnet18_best.pt. Slow."),
]
BY_ID = {s["id"]: s for s in SCRIPTS}
PIPELINE = ["baseline", "noise", "blur", "brightness", "contrast", "score", "rep_cnn", "rep_res", "compare", "viz"]

JOBS, LATEST, LOCK, COUNT = {}, {}, threading.Lock(), itertools.count(1)
GPU = {"cuda": None}


def probe_gpu():
    try:
        out = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=180).stdout.strip()
        GPU["cuda"] = out == "True"
    except Exception:
        GPU["cuda"] = False


def pump(job):
    p = job["proc"]
    for line in p.stdout:
        if len(job["lines"]) < 20000:
            job["lines"].append(line.rstrip("\n"))
    p.wait()
    job["code"] = p.returncode
    job["status"] = "stopped" if job["stopped"] else ("done" if p.returncode == 0 else "failed")


def running_job():
    return next((j for j in JOBS.values() if j["status"] == "running"), None)


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/api/state")
def state():
    scripts = []
    for s in SCRIPTS:
        j = JOBS.get(LATEST.get(s["id"]))
        scripts.append({**s, "exists": os.path.isfile(os.path.join(ROOT, s["file"])),
                        "job": j and {"id": j["id"], "status": j["status"], "code": j["code"]}})
    rj = running_job()
    ok = lambda *p: os.path.exists(os.path.join(ROOT, *p))
    return jsonify(scripts=scripts, running=rj and rj["id"], pipeline=PIPELINE, status=dict(
        dataset=ok("data", "chest_xray", "chest_xray", "test"), cnn=ok("pneumonia_cnn_best.pt"),
        resnet=ok("pneumonia_resnet18_best.pt"), cuda=GPU["cuda"]))


@app.post("/api/run/<sid>")
def run(sid):
    s = BY_ID.get(sid)
    if not s or not s["runnable"]:
        abort(404)
    with LOCK:
        if running_job():
            return jsonify(error="Another script is still running."), 409
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        proc = subprocess.Popen([sys.executable, "-u", s["file"], *s["args"]], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        jid = next(COUNT)
        JOBS[jid] = dict(id=jid, sid=sid, proc=proc, lines=[], status="running", code=None, stopped=False)
        LATEST[sid] = jid
        threading.Thread(target=pump, args=(JOBS[jid],), daemon=True).start()
    return jsonify(job=jid)


@app.get("/api/job/<int:jid>")
def job(jid):
    j = JOBS.get(jid) or abort(404)
    since = request.args.get("since", 0, type=int)
    return jsonify(lines=j["lines"][since:], total=len(j["lines"]), status=j["status"], code=j["code"])


@app.post("/api/stop/<int:jid>")
def stop(jid):
    j = JOBS.get(jid) or abort(404)
    if j["status"] == "running":
        j["stopped"] = True
        j["proc"].terminate()
    return jsonify(ok=True)


@app.get("/api/source/<sid>")
def source(sid):
    s = BY_ID.get(sid) or abort(404)
    with open(os.path.join(ROOT, s["file"]), encoding="utf-8", errors="replace") as f:
        return jsonify(file=s["file"], text=f.read())


@app.get("/api/results")
def results():
    base, out = os.path.join(ROOT, "results"), []
    for d, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                p = os.path.join(d, f)
                out.append(dict(path=os.path.relpath(p, base).replace("\\", "/"), t=os.path.getmtime(p)))
    return jsonify(sorted(out, key=lambda x: x["path"]))


@app.get("/results/<path:p>")
def result_file(p):
    return send_from_directory(os.path.join(ROOT, "results"), p)


if __name__ == "__main__":
    threading.Thread(target=probe_gpu, daemon=True).start()
    print("Mediguard dashboard: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)
