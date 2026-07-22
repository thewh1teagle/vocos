# Vocos with causal streaming support

[Audio samples](https://gemelo-ai.github.io/vocos/) |
Paper [[abs]](https://arxiv.org/abs/2306.00814) [[pdf]](https://arxiv.org/pdf/2306.00814.pdf)

[Vocos](https://github.com/gemelo-ai/vocos) with a causal backbone and streaming inference for real-time use: mel frames
go in chunk by chunk, audio comes out with ~85 ms latency, numerically identical to the offline forward pass. Weight-compatible
with the pretrained Vocos checkpoints.

## Installation

```bash
uv pip install git+https://github.com/thewh1teagle/vocos
```

## Usage

See the [examples](examples) folder for basic reconstruction and streaming inference.

## Pre-trained models

| Model Name                                                                          | Dataset       | Training Iterations | Parameters 
|-------------------------------------------------------------------------------------|---------------|-------------------|------------|
| [thewh1teagle/vocos-mel-24khz-causal](https://huggingface.co/thewh1teagle/vocos-mel-24khz-causal) | LibriTTS-R + Hebrew podcasts | causal fine-tune | 13.5M
| [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)         | LibriTTS      | 1M                | 13.5M
| [charactr/vocos-encodec-24khz](https://huggingface.co/charactr/vocos-encodec-24khz) | DNS Challenge | 2M                | 7.9M

## Training

Prepare a filelist of audio files for the training and validation set:

```bash
find $TRAIN_DATASET_DIR -name *.wav > filelist.train
find $VAL_DATASET_DIR -name *.wav > filelist.val
```

Fill a config file, e.g. [vocos.yaml](configs%2Fvocos.yaml), with your filelist paths and start training with:

```bash
uv run train.py -c configs/vocos.yaml
```

Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.
