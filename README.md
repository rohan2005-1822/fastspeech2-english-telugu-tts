# FastSpeech 2 Text-to-Speech: English (Svarah) and Telugu (IndicTTS)

Two separate FastSpeech 2 models based on https://github.com/ming024/FastSpeech2.

## Project structure
- English/: English FastSpeech 2 implementation
- Telugu/: Telugu FastSpeech 2 implementation
- Results/: generated audio, variance-adaptor plots and loss plots
- Report/: technical report

## Checkpoints
- English: https://huggingface.co/rohan1822/fastspeech2-tts-checkpoints/resolve/main/English/20000.pth.tar
- Telugu: https://huggingface.co/rohan1822/fastspeech2-tts-checkpoints/resolve/main/Telugu/20000.pth.tar
- Universal HiFi-GAN: https://huggingface.co/rohan1822/fastspeech2-tts-checkpoints/resolve/main/vocoder/generator_universal.pth.tar

## Training
Both models were trained for 20,000 steps with batch size 16.

## Inference
Run from the respective English/ or Telugu/ directory:

```bash
python inference.py --restore_step 20000 --speaker_id ID --sentences test_sentences.json --out out_dir
```

English default speaker ID: 94 (spk028), chosen as the lowest mean WER (0.160; runner-up 0.181) over 20 held-out validation sentences for 12 candidate speakers (Whisper-small). With 20 sentences per speaker this is a noisy proxy, and the top candidates are within its noise. A first run of this selection had a bug and was discarded. Checkpoint 20,000.
Telugu speaker ID: 0.

## Dataset summary
| Language | Train | Validation |
|---|---:|---:|
| English | 2,920 clips, 4.21 h | 570 clips, 0.85 h |
| Telugu | 2,427 clips, 4.13 h | 486 clips, 0.83 h |

Hours are estimated from mel-frame lengths.

## Limitations
- English alignment recovery used a relaxed-threshold heuristic retry.
- No formal human listening evaluation was performed.
- Punctuation is removed by the text tokenizer.
- Telugu T3 source text contained rendering errors; the intended sentence was synthesized.

## Licences
Based on the MIT-licensed FastSpeech2 repository.
The MMS-300M CTC forced-alignment model is CC-BY-NC 4.0.
Check the original dataset licences before redistribution.

## Setup
Install a CUDA build of PyTorch first (developed with torch 2.11, CUDA 12.8, Python 3.13), then `pip install -r requirements.txt`.

## Data preparation (run inside English/ or Telugu/)
1. Download `ai4bharat/Svarah` / `SPRINGLab/IndicTTS_Telugu`; write wav + `.lab` into `raw_data/<Lang>/<speaker>/`. Selected clips and splits are in `data_prep/`.
2. CTC forced alignment (arguments are START END clip indices):
   `python alignment/align_english.py 0 5000` / `python alignment/align_telugu.py 0 5000`
   (the scripts clip the range to the available data; reports are in `data_prep/`). English then uses `alignment/recover_english.py` to retry failed clips.
3. English: `python data_prep/rebuild_split.py` (merges recovered TextGrids, extracts features, rebuilds train/val). Telugu: `python preprocess.py config/Telugu/preprocess.yaml`.
4. `val_size: 0` makes `preprocess.py` write every clip to train.txt, so train/val lists must be rewritten from the provided split files (`preprocessed_data/<Lang>/{train,val}.txt`).

## Train
```bash
python train.py -p config/<Lang>/preprocess.yaml -m config/<Lang>/model.yaml -t config/<Lang>/train.yaml
```
20,000 steps, batch 16, checkpoint every 2,000 steps. `config.yaml` is a combined view; the code reads the three files in `config/<Lang>/`.

## Inference notes
Place the checkpoint at `output/ckpt/<Lang>/20000.pth.tar` and the vocoder at `hifigan/generator_universal.pth.tar`. E4 and T4 are synthesized from their `spoken` field (digits written as words); the original sentence is kept in `text`.

## Results
Final validation total loss at step 20,000: English 2.788, Telugu 2.164. English validation loss plateaus early (best total 2.723 at step 7,000, best mel 0.792 at step 14,000); the 20,000 checkpoint was kept. Plots are in `Results/<Lang>/`.
