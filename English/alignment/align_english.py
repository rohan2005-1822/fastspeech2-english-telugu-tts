import torch, os, re, sys, glob, json, time, gc, argparse, numpy as np, librosa, tgt
import onnxruntime as ort
import ctc_forced_aligner as cfa
from collections import Counter
if hasattr(ort, "preload_dlls"): ort.preload_dlls()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description="English CTC forced alignment")
ap.add_argument("start", type=int, help="first sorted WAV index")
ap.add_argument("end", type=int, help="exclusive final sorted WAV index")
ap.add_argument("--raw", default=os.environ.get(
    "ENGLISH_RAW", "/kaggle/working/FastSpeech2_English/raw_data/English"))
ap.add_argument("--textgrid", default=os.environ.get(
    "ENGLISH_TEXTGRID", "/kaggle/working/FastSpeech2_English/preprocessed_data/English/TextGrid"))
ap.add_argument("--aligner", default=os.environ.get(
    "CTC_ALIGNER_MODEL", "/kaggle/working/ctc_aligner.onnx"))
ap.add_argument("--report", default=None)
ap.add_argument("--current-clip", default="/kaggle/working/current_clip_en.txt")
args = ap.parse_args()
START, END = args.start, args.end
RAW, TG = args.raw, args.textgrid
ALIGNER, REPORT, CURRENT_CLIP = args.aligner, args.report, args.current_clip
if REPORT is None:
    REPORT = os.path.join(os.path.dirname(__file__), f"align_report_en_{START}_{END}.json")
MAX_SEC = 12.0
cuda_opts = {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}
session = ort.InferenceSession(ALIGNER,
                               providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"])
tokenizer = cfa.Tokenizer()
print("providers:", session.get_providers(), flush=True)

def align_one(wav_path, text):
    y, sr = librosa.load(wav_path, sr=22050)
    total = len(y) / sr
    if total > MAX_SEC: return None, "too long"
    rom = re.sub(r"[^a-z ]", "", text.lower())
    words = [w for w in rom.split() if w]
    letters = list("".join(words)); n = len(letters)
    if n == 0: return None, "empty text"
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000).astype(np.float32)
    emissions, stride = cfa.generate_emissions(session, y16)
    FR = stride / 1000.0
    need = n + sum(letters[i] == letters[i + 1] for i in range(n - 1))
    if need > emissions.shape[-2] - 2: return None, "toomany_letters"
    segments, scores, blank = cfa.get_alignments(emissions, letters, tokenizer)
    chars = [s for s in segments if s.label != "<blank>"]
    if len(chars) != n: return None, f"count mismatch {len(chars)}/{n}"
    c = np.array([(s.start + s.end + 1) / 2 * FR for s in chars])
    _, ix = librosa.effects.trim(y, top_db=35)
    if abs(c[0] - ix[0] / sr) > 0.3 or abs(c[-1] - ix[1] / sr) > 0.3:
        return None, "edge mismatch"
    word_last = set((np.cumsum([len(w) for w in words])[:-1] - 1).tolist())
    starts, ends, sp = [0.0] * n, [0.0] * n, {}
    starts[0] = max(0.0, c[0] - 0.03)
    for i in range(n - 1):
        mid = (c[i] + c[i + 1]) / 2
        if i in word_last:
            e, s = min(mid, c[i] + 0.05), max(mid, c[i + 1] - 0.05)
            if s - e < 0.02: e, s = mid - 0.01, mid + 0.01
            ends[i], starts[i + 1], sp[i] = e, s, (e, s)
        else:
            ends[i], starts[i + 1] = mid, mid
    ends[n - 1] = min(total, c[n - 1] + 0.07)
    iv = [("sil", 0.0, starts[0])] if starts[0] > 0 else []
    for i in range(n):
        iv.append(("en_" + letters[i], starts[i], ends[i]))
        if i in sp: iv.append(("sp", *sp[i]))
    if total - ends[-1] > 0: iv.append(("sil", ends[-1], total))
    return iv, total

def write_tg(iv, total, path):
    tier = tgt.IntervalTier(start_time=0, end_time=total, name="phones")
    for l, s, e in iv: tier.add_interval(tgt.Interval(s, e, l))
    tg = tgt.TextGrid(); tg.add_tier(tier)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tgt.io.write_to_file(tg, path, format="long")

wavs = sorted(glob.glob(f"{RAW}/*/*.wav"))[START:END]
report, t0 = {}, time.time()
for k, w in enumerate(wavs):
    spk, base = w.split("/")[-2], os.path.basename(w)[:-4]
    out = f"{TG}/{spk}/{base}.TextGrid"
    if os.path.exists(out): report[base] = "ok"; continue
    open(CURRENT_CLIP, "w").write(base)
    text = open(w[:-4] + ".lab", encoding="utf-8").read().strip()
    try:
        iv, info = align_one(w, text)
    except Exception as ex:
        report[base] = "error: " + str(ex)[:80]; continue
    if iv is None: report[base] = info; continue
    write_tg(iv, info, out); report[base] = "ok"
    if k % 50 == 0:
        gc.collect()
        print(k, "/", len(wavs), round(time.time() - t0), "s", flush=True)

print("DONE", Counter(v.split(" ")[0] + (" " + v.split(" ")[1] if v.startswith(("count", "edge")) else "") for v in report.values()), flush=True)

json.dump(report, open(REPORT, "w"))
