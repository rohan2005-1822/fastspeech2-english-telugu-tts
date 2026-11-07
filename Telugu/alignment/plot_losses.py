
import os, argparse, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ap = argparse.ArgumentParser()
ap.add_argument("--log_dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--name", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

def load(sub):
    ea = EventAccumulator(os.path.join(a.log_dir, sub), size_guidance={"scalars": 0}); ea.Reload()
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}
tr, va = load("train"), load("val")
print("train tags:", {k: len(v) for k, v in tr.items()})
print("val tags:  ", {k: len(v) for k, v in va.items()})

def plot(d, split, fname):
    tags = sorted([t for t in d if "loss" in t.lower()], key=lambda t: (0 if "total" in t.lower() else 1, t))
    if not tags: print("no loss tags for", split); return
    n = len(tags); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows)); axes = axes.flatten() if n > 1 else [axes]
    for ax, t in zip(axes, tags):
        s, v = zip(*d[t]); ax.plot(s, v, lw=1.2, marker="o" if len(s) < 60 else None, ms=3)
        ax.set_title(t); ax.set_xlabel("step"); ax.grid(alpha=0.3)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle(f"{a.name} {split} loss"); plt.tight_layout(); fig.savefig(os.path.join(a.out, fname), dpi=130); plt.close(fig)
    return tags
plot(tr, "training", "training_loss.png")
vt = plot(va, "validation", "validation_loss.png")

if vt:
    steps = sorted({s for t in vt for s, _ in va[t]})
    with open(os.path.join(a.out, "validation_losses.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step"] + vt)
        for s in steps: w.writerow([s] + [dict(va[t]).get(s, "") for t in vt])
    tot = [t for t in vt if "total" in t.lower()]
    if tot:
        pts = va[tot[0]]; b = min(pts, key=lambda x: x[1])
        print(f"last val total: step {pts[-1][0]} = {pts[-1][1]:.4f} | best: step {b[0]} = {b[1]:.4f}")
