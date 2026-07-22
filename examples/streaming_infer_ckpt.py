"""
Streaming (causal) decoding of input.wav using the fine-tuned causal checkpoint.

uv run examples/streaming_infer_ckpt.py
"""

from pathlib import Path

import torch
import torchaudio

from vocos import VocosStreamer
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone
from vocos.pretrained import Vocos

CKPT_PATH = Path("logs/lightning_logs/version_0/checkpoints/last.ckpt")
INPUT_PATH = Path("input.wav")
OUTPUT_PATH = Path("streamed.wav")
SAMPLE_RATE = 24000
CHUNK_FRAMES = 4  # feed 4 mel frames (~43 ms) at a time

vocos = Vocos(
    feature_extractor=MelSpectrogramFeatures(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100, padding="same"
    ),
    backbone=VocosBackbone(
        input_channels=100, dim=512, intermediate_dim=1536, num_layers=8, causal=True, lookahead_frames=4
    ),
    head=ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="same"),
).eval()

ckpt = torch.load(CKPT_PATH, map_location="cpu")
sd = ckpt["state_dict"]
generator_prefixes = ("feature_extractor.", "backbone.", "head.")
generator_sd = {k: v for k, v in sd.items() if k.startswith(generator_prefixes)}
missing, unexpected = vocos.load_state_dict(generator_sd, strict=False)
print(f"epoch={ckpt['epoch']} step={ckpt['global_step']}")
print(f"missing={missing} unexpected={unexpected}")

y, sr = torchaudio.load(INPUT_PATH)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=SAMPLE_RATE)
mel = vocos.feature_extractor(y)

streamer = VocosStreamer(vocos)
audio_chunks = []
for start in range(0, mel.size(-1), CHUNK_FRAMES):
    audio_chunks.append(streamer.feed(mel[..., start : start + CHUNK_FRAMES]))
audio_chunks.append(streamer.flush())
y_hat = torch.cat(audio_chunks, dim=-1)

torchaudio.save(OUTPUT_PATH, y_hat, SAMPLE_RATE)
print(f"Saved streamed audio to {OUTPUT_PATH} ({y_hat.size(-1) / SAMPLE_RATE:.2f}s)")
