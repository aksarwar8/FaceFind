"""Render sanitized research figures with Matplotlib. No private data is read."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "benchmark-aggregates.json").read_text(encoding="utf-8"))
DEST = ROOT / "figures"
COLORS = ("#226b8e", "#097d69", "#8c5879")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelcolor": "#1a3247", "text.color": "#1a3247",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#a8b8c6", "axes.axisbelow": True,
    "grid.color": "#d7e1e9", "svg.hashsalt": "facefind-docs",
})


def figure(title, subtitle, n=2):
    fig, raw = plt.subplots(1, n, figsize=(10.5, 3.9) if n == 2 else (9, 3.65))
    fig.suptitle(title, x=.07, y=.97, ha="left", fontsize=17, weight="bold")
    fig.text(.07, .87, subtitle, fontsize=9.5)
    fig.subplots_adjust(left=.09, bottom=.18, right=.97, top=.72, wspace=.36)
    return fig, raw


def save(fig, name):
    path = DEST / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def grouped(ax, labels, series, title):
    x = np.arange(len(labels))
    width = .25 if len(series) == 3 else .34
    for i, (name, values) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width,
               label=name, color=COLORS[i])
    ax.set(title=title, xticks=x, xticklabels=labels, ylim=(0, 105), ylabel="Percent")
    ax.grid(axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.16), ncol=len(series),
              frameon=False, fontsize=8, columnspacing=1.0)


def annotated_bars(ax, labels, values, title, unit, limit):
    bars = ax.bar(labels, values, color=COLORS[:len(labels)], width=.59)
    ax.set(title=title, ylabel=unit, ylim=(0, limit))
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.grid(axis="y")


def render():
    DEST.mkdir(exist_ok=True)
    detectors = DATA["widerface_retinaface_full500_seed42"]["detectors"]
    keys = dict(zip(("YuNet", "SCRFD", "RetinaFace"), detectors))
    fig, (left, right) = figure("Detector comparison on a common cohort",
        "500 WIDER FACE images · 5,615 annotated boxes · custom IoU ≥ 0.50")
    grouped(left, ["Precision", "Recall", "F1"], {name: [100 * detectors[key]["custom_metrics"][m]
        for m in ("precision", "recall", "f1")] for name, key in keys.items()}, "Box-level outcomes")
    grouped(right, ["≤20", ">20–40", ">40"], {name: [100 * part["recall"]
        for part in detectors[key]["recall_by_face_height"].values()] for name, key in keys.items()}, "Recall by annotated face height (px)")
    save(fig, "detectors")

    recognition = DATA["recognition_only_lfw"]["models"]
    fig, (left, right) = figure("Recognition quality and CPU cost",
        "Shared LFW probes · 578 evaluated known and 442 evaluated unknown probes per recognizer")
    annotated_bars(left, list(recognition), [x["embedding_latency_ms"]["mean_ms"]
        for x in recognition.values()], "Mean embedding latency", "Milliseconds", 880)
    grouped(right, ["Known accepted", "Unknown falsely\naccepted"], {k: [100*v["correct_known"]/v["known_evaluated"],
        100*v["false_acceptances"]/v["unknown_evaluated"]] for k,v in recognition.items()}, "Conditional recognition rates")
    for i, value in enumerate(recognition.values()):
        right.text(1 + (i - 1) * .25, 7, f'{value["false_acceptances"]}', ha="center", fontsize=8)
    save(fig, "recognition")

    systems = DATA["combined_pipeline_composites"]["systems"]
    fig, (left, right) = figure("Combined pipeline trade-off",
        "Ten synthetic composites · 100 faces · 50-way calibration · separate from the final demo")
    grouped(left, ["Precision", "Recall", "F1"], {label: [100*x["detection_"+k] for k in ("precision","recall","f1")]
        for label,x in zip(("YuNet + SFace","SCRFD + AdaFace"),systems.values())}, "Detection quality")
    annotated_bars(right, ["YuNet + SFace", "SCRFD + AdaFace"],
        [x["group_processing_latency"]["mean_ms"] for x in systems.values()], "Mean group processing time", "Milliseconds", 1900)
    save(fig, "combined-pipelines")

    fig, left = figure("Reference enrollment recovery",
        "Original 17 reference photos · SCRFD score threshold 0.50 in both passes", n=1)
    adaptation = DATA["reference_adaptation"]
    counts = [adaptation[key]["detected"] for key in ("primary_coverage","adaptive_coverage")]
    bars = left.barh(["1,280 px plus empty-pass retry","1,280 px primary only"], counts[::-1],
                    color=COLORS[:2][::-1], height=.45)
    left.bar_label(bars, labels=[f"{x}/17" for x in counts[::-1]], padding=7)
    left.set(xlim=(0,19), xlabel="References with a detection")
    left.grid(axis="x")
    fig.subplots_adjust(left=.30)
    save(fig, "enrollment")

    fig, ax = figure("Evaluation Studio outcome matrix",
        "60 enrolled identities · five synthetic composites · 57 matched boxes · approximate annotations", n=1)
    matrix = DATA["evaluation_studio_demo"]["outcome_matrix"]
    ax.imshow([[v if v is not None else -1 for v in row] for row in matrix], cmap="GnBu", vmin=0, vmax=35, aspect="auto")
    ax.set(xticks=range(5), xticklabels=["Correct identity","Wrong or false\naccepted identity","Rejected as\nUnknown","Missed","Processing\nfailure"],
           yticks=range(2), yticklabels=["Actual known","Actual unknown"])
    ax.tick_params(length=0)
    for y,row in enumerate(matrix):
        for x,v in enumerate(row):
            ax.text(x,y,"N/A" if v is None else v,ha="center",va="center",weight="bold",
                    color="white" if v and v>18 else "#1a3247")
    fig.subplots_adjust(left=.16, bottom=.18)
    save(fig, "studio-matrix")

    fig, ax = figure("Warm FastAPI request timing",
        "One genuine photo without identity annotations · five warm loopback requests per pipeline", n=1)
    runs = DATA["final_app_performance_20261124_01"]["pipelines"]
    parts = [run["mean_timing_breakdown_ms"] for run in runs.values()]
    legend = ["Detection","Alignment","Embedding","Matching","Annotation","Unaccounted backend","API minus backend"]
    left = np.zeros(2)
    palette = ["#226b8e","#7eb3c3","#097d69","#acbd66","#8c5879","#a3a9b8","#d3dde6"]
    for i, key in enumerate(parts[0]):
        step = np.array([part[key] for part in parts])
        ax.barh(["SCRFD + AdaFace","YuNet + SFace"], step, left=left, color=palette[i], height=.5, label=legend[i])
        left += step
    for i,value in enumerate(left):
        ax.text(value+14,i,f"{value:,.0f} ms",va="center",weight="bold")
    ax.set(xlim=(0,1390), xlabel="Mean milliseconds per request")
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(.48,-.33), ncol=4, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.19,bottom=.35)
    save(fig,"application-timing")


if __name__ == "__main__":
    render()
