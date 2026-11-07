
import os, sys, json, re, argparse, yaml, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
import uroman
from utils.model import get_model, get_vocoder, vocoder_infer
from text import text_to_sequence

ap = argparse.ArgumentParser()
ap.add_argument("--restore_step", type=int, default=20000)
ap.add_argument("--speaker_id", type=int, default=0)
ap.add_argument("--sentences", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

P = yaml.load(open("config/Telugu/preprocess.yaml"), Loader=yaml.FullLoader)
M = yaml.load(open("config/Telugu/model.yaml"), Loader=yaml.FullLoader)
T = yaml.load(open("config/Telugu/train.yaml"), Loader=yaml.FullLoader)
configs = (P, M, T)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(a, configs, dev, train=False).eval()
vocoder = get_vocoder(M, dev)
sr = P["preprocessing"]["audio"]["sampling_rate"]; hop = P["preprocessing"]["stft"]["hop_length"]
stats = json.load(open(os.path.join(P["path"]["preprocessed_path"], "stats.json")))
p_mean, p_std = stats["pitch"][2], stats["pitch"][3]
e_mean, e_std = stats["energy"][2], stats["energy"][3]
ur = uroman.Uroman()

def text_to_tokens(text):
    rom = re.sub(r"[^a-z ]", "", ur.romanize_string(text).lower())
    words = [w for w in rom.split() if w]
    return "{" + " sp ".join(" ".join("tel_" + c for c in w) for w in words) + "}"

for d in ["audio", "arrays", "plots/duration", "plots/pitch", "plots/energy", "plots/mel"]:
    os.makedirs(f"{a.out}/{d}", exist_ok=True)
summary = {}
for s in json.load(open(a.sentences)):
    tok = text_to_tokens(s["spoken"])
    toks = tok.strip("{}").split()
    seq = np.array(text_to_sequence(tok, ["english_cleaners"]))
    assert len(seq) == len(toks), f"{s['id']}: {len(toks)} tokens but {len(seq)} ids (unknown symbol dropped?)"
    texts = torch.from_numpy(seq).long().unsqueeze(0).to(dev)
    src_lens = torch.tensor([len(seq)]).to(dev)
    spk = torch.tensor([a.speaker_id]).to(dev)
    with torch.no_grad():
        o = model(spk, texts, src_lens, len(seq))
        mel_post, p, e, logd, dr, mel_lens = o[1], o[2], o[3], o[4], o[5], o[9]
        wav = vocoder_infer(mel_post.transpose(1, 2), vocoder, M, P, lengths=[int(mel_lens[0]) * hop])[0]
    wavfile.write(f"{a.out}/audio/{s['id']}.wav", sr, wav)
    dur_fr = dr[0].cpu().numpy(); logd = logd[0].cpu().numpy()
    p_raw = p[0].cpu().numpy(); e_raw = e[0].cpu().numpy()
    pitch_hz = np.maximum(p_raw * p_std + p_mean, 0); energy = e_raw * e_std + e_mean
    np.savez(f"{a.out}/arrays/{s['id']}.npz", tokens=np.array(toks), duration_frames=dur_fr, log_duration=logd,
             pitch_norm=p_raw, pitch_hz=pitch_hz, energy_norm=e_raw, energy=energy)
    labels = [t.replace("tel_", "") for t in toks]
    x = np.arange(len(labels))
    for name, vals, yl in [("duration", dur_fr * hop / sr, "seconds"), ("pitch", pitch_hz, "Hz (de-normalized)"),
                           ("energy", energy, "energy (de-normalized)")]:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.22), 3.2))
        ax.bar(x, vals, color="tab:blue" if name == "duration" else ("tab:red" if name == "pitch" else "tab:green"))
        for i, t in enumerate(toks):
            if t == "sp": ax.axvspan(i - .5, i + .5, color="gray", alpha=.25)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_ylabel(yl); ax.set_title(f"{s['id']} predicted {name} per token (gray = word space)")
        plt.tight_layout(); plt.savefig(f"{a.out}/plots/{name}/{s['id']}.png", dpi=110); plt.close()
    ml = int(mel_lens[0]); fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(mel_post[0, :ml].cpu().numpy().T, origin="lower", aspect="auto"); ax.set_title(f"{s['id']} postnet mel")
    plt.tight_layout(); plt.savefig(f"{a.out}/plots/mel/{s['id']}.png", dpi=100); plt.close()
    voiced = pitch_hz[pitch_hz > 1]
    summary[s["id"]] = {"n_tokens": len(toks), "audio_sec": round(len(wav) / sr, 2),
        "total_pred_dur_sec": round(float(dur_fr.sum()) * hop / sr, 2),
        "sp_dur_sec": round(float(sum(d for d, t in zip(dur_fr, toks) if t == "sp")) * hop / sr, 2),
        "pitch_hz_mean_voiced": round(float(voiced.mean()), 1) if len(voiced) else None,
        "pitch_hz_min_max": [round(float(pitch_hz.min()), 1), round(float(pitch_hz.max()), 1)],
        "unvoiced_tokens(<1Hz)": int((pitch_hz <= 1).sum()),
        "energy_min_max": [round(float(energy.min()), 2), round(float(energy.max()), 2)], "tokens": tok}
    print(s["id"], "OK", summary[s["id"]]["audio_sec"], "s")
json.dump(summary, open(f"{a.out}/summary.json", "w"), ensure_ascii=False, indent=1)
