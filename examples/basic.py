"""
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O input.wav
uv run examples/basic.py
"""

import urllib.request
from pathlib import Path

import torchaudio

from vocos import Vocos

URL = "https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav"
INPUT_PATH = Path("input.wav")
OUTPUT_PATH = Path("reconstructed.wav")
SAMPLE_RATE = 24000

if not INPUT_PATH.exists():
    urllib.request.urlretrieve(URL, INPUT_PATH)

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Normalize the input audio
y, sr = torchaudio.load(INPUT_PATH)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=SAMPLE_RATE)

# Reconstruct the audio using Vocos
mel = vocos.feature_extractor(y)
y_hat = vocos.decode(mel)

# Save the reconstructed audio
torchaudio.save(OUTPUT_PATH, y_hat, SAMPLE_RATE)
print(f"Saved reconstructed audio to {OUTPUT_PATH}")
