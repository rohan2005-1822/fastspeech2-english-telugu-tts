import sys, os, glob, json, time, gc, random, argparse
import numpy as np, librosa, tgt

ap = argparse.ArgumentParser(description="Recover selected English alignments")
ap.add_argument("--raw", default=os.environ.get(
    "ENGLISH_RAW", "/kaggle/working/FastSpeech2_English/raw_data/English"))
ap.add_argument("--textgrid", default=os.environ.get(
    "ENGLISH_TEXTGRID", "/kaggle/working/FastSpeech2_English/preprocessed_data/English/TextGrid"))
ap.add_argument("--aligner", default=os.environ.get(
    "CTC_ALIGNER_MODEL", "/kaggle/working/ctc_aligner.onnx"))
ap.add_argument("--report", default=None)
ap.add_argument("--current-clip", default="/kaggle/working/current_clip_en.txt")
args = ap.parse_args()

align_script = os.path.join(os.path.dirname(__file__), "align_english.py")
src = open(align_script, encoding="utf-8").read().split("wavs = sorted(")[0]
src = src.replace("> 0.3 or abs(c[-1] - ix[1] / sr) > 0.3",
                  "> 1e9 or abs(c[-1] - ix[1] / sr) > 1e9")
assert "1e9" in src

recovery_report = args.report or os.path.join(
    os.path.dirname(__file__), "recover_report_en.json")
sys.argv = [
    align_script, "0", "5000",
    "--raw", args.raw,
    "--textgrid", args.textgrid,
    "--aligner", args.aligner,
    "--current-clip", args.current_clip,
]
exec(compile(src, align_script, "exec"))  # defines RAW, TG, session, tokenizer, align_one, write_tg

REC = TG.replace("/TextGrid", "/TextGrid_recovered")
HOP, SR = 256, 22050

def metrics(wav, iv):
    y, _ = librosa.load(wav, sr=SR)
    ch = [(s, e) for l, s, e in iv if l.startswith("en_")]
    s0, e1 = ch[0][0], ch[-1][1]
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0] + 1e-6)
    t = librosa.frames_to_time(np.arange(len(db)), sr=SR, hop_length=HOP)
    lvl = np.percentile(db[(t >= s0) & (t <= e1)], 90)
    loud = db > lvl - 12
    return dict(dur=len(y) / SR, rate=len(ch) / (e1 - s0), maxchar=max(e - s for s, e in ch),
                head=float(loud[t < s0 - 0.1].sum() * HOP / SR), tail=float(loud[t > e1 + 0.1].sum() * HOP / SR))

# calibrate limits on already-accepted clips
random.seed(0)
cal = []
for p in random.sample(glob.glob(f"{TG}/*/*.TextGrid"), 300):
    spk, base = p.split("/")[-2], os.path.basename(p)[:-9]
    tier = tgt.io.read_textgrid(p).get_tier_by_name("phones")
    cal.append(metrics(f"{RAW}/{spk}/{base}.wav", [(i.text, i.start_time, i.end_time) for i in tier]))
P = lambda k, q: float(np.percentile([m[k] for m in cal], q))
LIM = dict(head=max(0.15, P("head", 97)), tail=max(0.15, P("tail", 97)),
           rate_lo=P("rate", 1), rate_hi=P("rate", 99), maxchar=max(0.6, P("maxchar", 99)))
print("limits", LIM, flush=True)

todo = [w for w in sorted(glob.glob(f"{RAW}/*/*.wav"))
        if not os.path.exists(f"{TG}/{w.split('/')[-2]}/{os.path.basename(w)[:-4]}.TextGrid")]
print("failed clips to retry:", len(todo), flush=True)
res, t0 = {}, time.time()
for k, w in enumerate(todo):
    spk, base = w.split("/")[-2], os.path.basename(w)[:-4]
    try:
        text = open(w[:-4] + ".lab", encoding="utf-8").read().strip()
        iv, total = align_one(w, text)
        if iv is None:
            res[base] = {"status": total}
        else:
            m = metrics(w, iv)
            good = (m["head"] <= LIM["head"] and m["tail"] <= LIM["tail"] and
                    LIM["rate_lo"] <= m["rate"] <= LIM["rate_hi"] and m["maxchar"] <= LIM["maxchar"])
            m["status"] = "accept" if good else "reject"
            if good: write_tg(iv, total, f"{REC}/{spk}/{base}.TextGrid")
            res[base] = m
    except Exception as ex:
        res[base] = {"status": "error " + str(ex)[:60]}
    if k % 50 == 0:
        gc.collect(); print(k, "/", len(todo), round(time.time() - t0), "s", flush=True)
acc = [v for v in res.values() if v.get("status") == "accept"]
print("ACCEPTED", len(acc), "of", len(todo), "hours", round(sum(v["dur"] for v in acc) / 3600, 2), flush=True)
json.dump({"limits": LIM, "results": res}, open(recovery_report, "w"))
