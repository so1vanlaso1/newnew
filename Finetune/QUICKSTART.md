# Quickstart — run the fine-tune

You received this `Finetune/` folder. Your job: produce a LoRA adapter and send it
back. It's **3 commands** and runs ~1–2 hours on the GPU.

---

## 0. What you need

- A **Linux machine with an NVIDIA GPU** (12 GB VRAM is enough — e.g. RTX 5070) and the driver installed.
- Confirm the GPU is visible — this must print your card:
  ```bash
  nvidia-smi
  ```
  If it errors, install the NVIDIA driver first (everything else is automatic).
- Internet access and ~15 GB free disk (it downloads PyTorch + the model).

---

## 1. Open a terminal in this folder

```bash
cd Finetune                 # the folder you received
chmod +x setup.sh train.sh
```

## 2. Install everything — one command

```bash
./setup.sh
```

Creates a `.venv`, installs PyTorch + all libraries, and **downloads the model**.
First run takes ~10–20 min. It finishes by printing your GPU and `Environment ready.`

> Bare Ubuntu with no Python tooling? Run this once first, then re-run `./setup.sh`:
> ```bash
> sudo apt-get update && sudo apt-get install -y python3-venv python3-pip build-essential git
> ```

## 3. Train — one command

```bash
./train.sh
```

Runs ~1–2 hours. It logs progress every 10 steps and ends with
`Done. LoRA adapter is in artifacts/translator-lora/.`

## 4. Send the result back

```bash
tar -czf translator-lora.tar.gz -C artifacts translator-lora
```

Send **`translator-lora.tar.gz`** back to whoever gave you this folder. Done.

---

## If something breaks

| Symptom | Fix |
|---|---|
| `./setup.sh: Permission denied` | Run `bash setup.sh` and `bash train.sh` instead. |
| Warning that **flash-attn** failed to build | Ignore it — just train with `ATTN=sdpa ./train.sh` (~10% slower, no build needed). |
| `Unrecognized model type ... qwen3_5` | transformers is too old: `source .venv/bin/activate && pip install -U "git+https://github.com/huggingface/transformers"`, then `./train.sh`. |
| `CUDA out of memory` | Smaller batch: `BATCH=1 GRAD_ACCUM=16 ./train.sh`. |
| Driver too old / CUDA < 12.8 | `CUDA_WHL=cu126 ./setup.sh` (or `cu121`). |
| Download fails or asks to log in | `HF_TOKEN=hf_xxxx ./setup.sh`. |
| Need a different model id | `MODEL_ID=org/name ./setup.sh` then `MODEL_ID=org/name ./train.sh` (keep them the same). |

## Optional: 10-second sanity check before training (CPU only)

```bash
source .venv/bin/activate
pytest tests/test_smoke.py -v
```

If those 4 tests pass, your environment is wired correctly.
