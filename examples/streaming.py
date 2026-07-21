"""
Streaming (causal) decoding example.

Builds a causal Vocos model, warm-started from the pretrained non-causal checkpoint, and decodes a
mel-spectrogram chunk by chunk with VocosStreamer. The streamed output is identical to offline decoding.

Note: the warm-started weights were trained with future context, so quality is degraded until the model
is fine-tuned in causal mode (see configs/vocos-causal.yaml). This example demonstrates the plumbing.

wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O input.wav
uv run examples/streaming.py
"""

import urllib.request
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from vocos import VocosStreamer
from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone
from vocos.pretrained import Vocos

URL = "https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav"
INPUT_PATH = Path("input.wav")
OUTPUT_PATH = Path("streamed.wav")
SAMPLE_RATE = 24000
CHUNK_FRAMES = 4  # feed 4 mel frames (~43 ms) at a time

if not INPUT_PATH.exists():
    urllib.request.urlretrieve(URL, INPUT_PATH)

# Build a causal model. lookahead_frames=4 allows ~43 ms of future context; use 0 for strictly causal.
vocos = Vocos(
    feature_extractor=MelSpectrogramFeatures(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=100, padding="same"
    ),
    backbone=VocosBackbone(
        input_channels=100, dim=512, intermediate_dim=1536, num_layers=8, causal=True, lookahead_frames=4
    ),
    head=ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="same"),
).eval()

# Warm-start from the non-causal checkpoint (replace with your fine-tuned causal checkpoint for full quality)
model_path = hf_hub_download(repo_id="charactr/vocos-mel-24khz", filename="pytorch_model.bin")
vocos.load_state_dict(torch.load(model_path, map_location="cpu"))

y, sr = torchaudio.load(INPUT_PATH)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=SAMPLE_RATE)
mel = vocos.feature_extractor(y)

# Stream: in a real TTS system, feed mel chunks as the acoustic model produces them
streamer = VocosStreamer(vocos)
audio_chunks = []
for start in range(0, mel.size(-1), CHUNK_FRAMES):
    audio_chunks.append(streamer.feed(mel[..., start : start + CHUNK_FRAMES]))
audio_chunks.append(streamer.flush())
y_hat = torch.cat(audio_chunks, dim=-1)

torchaudio.save(OUTPUT_PATH, y_hat, SAMPLE_RATE)
print(f"Saved streamed audio to {OUTPUT_PATH} ({y_hat.size(-1) / SAMPLE_RATE:.2f}s)")
