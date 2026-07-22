from typing import Optional

import torch
from torch import nn

from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone


class _StreamingConv1d:
    """
    Streams a stride-1 Conv1d with fixed (left, right) zero padding, producing outputs identical to the
    offline `conv(pad(x, (left, right)))`. Outputs for a frame are emitted as soon as its full context is
    available, i.e. they lag the input by `right` frames.
    """

    def __init__(self, conv: nn.Conv1d, left: int, right: int):
        self.conv = conv
        self.left = left
        self.right = right
        self.context = conv.kernel_size[0] - 1
        self.cache: Optional[torch.Tensor] = None

    def feed(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache is None:
            self.cache = torch.zeros(x.shape[0], x.shape[1], self.left, device=x.device, dtype=x.dtype)
        buf = torch.cat([self.cache, x], dim=-1)
        if buf.size(-1) <= self.context:
            self.cache = buf
            return buf.new_zeros(buf.shape[0], self.conv.out_channels, 0)
        self.cache = buf[..., -self.context:]
        return self.conv(buf)

    def flush(self) -> torch.Tensor:
        assert self.cache is not None, "flush() called before any feed()"
        zeros = self.cache.new_zeros(self.cache.shape[0], self.cache.shape[1], self.right)
        return self.feed(zeros)


class _StreamingConvNeXtBlock:
    """Streams a causal ConvNeXtBlock. Only the depthwise conv is stateful; the rest is per-frame."""

    def __init__(self, block):
        assert block.pad is not None, "ConvNeXtBlock must be built with an explicit lookahead (causal mode)"
        assert not block.adanorm, "Streaming with AdaLayerNorm conditioning is not supported"
        self.block = block
        self.sconv = _StreamingConv1d(block.dwconv, *block.pad)
        self.residual: Optional[torch.Tensor] = None

    def _emit(self, y: torch.Tensor) -> torch.Tensor:
        m = y.size(-1)
        y = y.transpose(1, 2)
        y = self.block.norm(y)
        y = self.block.pwconv2(self.block.act(self.block.pwconv1(y)))
        if self.block.gamma is not None:
            y = self.block.gamma * y
        y = y.transpose(1, 2)
        residual, self.residual = self.residual[..., :m], self.residual[..., m:]
        return residual + y

    def feed(self, x: torch.Tensor) -> torch.Tensor:
        self.residual = x if self.residual is None else torch.cat([self.residual, x], dim=-1)
        return self._emit(self.sconv.feed(x))

    def flush(self) -> torch.Tensor:
        return self._emit(self.sconv.flush())


class _StreamingISTFT:
    """
    Streaming overlap-add ISTFT with "same" padding, matching the offline `vocos.spectral_ops.ISTFT` output
    exactly. A sample is emitted once no future frame can contribute to it, so emitted audio lags the newest
    frame by (win_length - hop_length) samples.
    """

    def __init__(self, istft):
        assert istft.padding == "same", "Streaming ISTFT requires an ISTFT head with padding='same'"
        self.n_fft = istft.n_fft
        self.hop = istft.hop_length
        self.win = istft.win_length
        self.window = istft.window
        self.pad = (self.win - self.hop) // 2
        self.tail_y: Optional[torch.Tensor] = None
        self.tail_env: Optional[torch.Tensor] = None
        self.start_trim = self.pad

    def _ola(self, frames: torch.Tensor) -> torch.Tensor:
        m = frames.size(-1)
        out_len = (m - 1) * self.hop + self.win
        y = torch.nn.functional.fold(
            frames, output_size=(1, out_len), kernel_size=(1, self.win), stride=(1, self.hop),
        )[:, 0, 0, :]
        window_sq = self.window.square().expand(1, m, -1).transpose(1, 2)
        env = torch.nn.functional.fold(
            window_sq, output_size=(1, out_len), kernel_size=(1, self.win), stride=(1, self.hop),
        ).reshape(-1)
        if self.tail_y is not None:
            overlap = self.win - self.hop
            y[:, :overlap] = y[:, :overlap] + self.tail_y
            env[:overlap] = env[:overlap] + self.tail_env
        emit = m * self.hop
        self.tail_y = y[:, emit:]
        self.tail_env = env[emit:]
        return self._normalize(y[:, :emit], env[:emit])

    def _normalize(self, y: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        trim = min(self.start_trim, y.size(-1))
        self.start_trim -= trim
        y, env = y[:, trim:], env[trim:]
        assert (env > 1e-11).all()
        return y / env

    def feed(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.size(-1) == 0:
            return torch.zeros(spec.shape[0], 0, device=spec.device, dtype=self.window.dtype)
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]
        return self._ola(ifft)

    def flush(self) -> torch.Tensor:
        assert self.tail_y is not None, "flush() called before any feed()"
        keep = self.tail_y.size(-1) - self.pad
        return self._normalize(self.tail_y[:, :keep], self.tail_env[:keep])


class VocosStreamer:
    """
    Stateful streaming decoder for a causal Vocos model with an ISTFT head.

    Feed mel-spectrogram frames in chunks of any size; audio samples are returned incrementally as soon as
    they are final. The concatenated output of all `feed()` calls followed by `flush()` is numerically
    identical to the offline `backbone` + `head` forward pass over the full feature sequence.

    The algorithmic latency is `backbone.lookahead_frames * hop_length` (frame lookahead) plus
    `win_length - hop_length` samples (ISTFT overlap-add).

    Example:
        streamer = VocosStreamer(model)  # model: vocos.pretrained.Vocos with a causal backbone
        for mel_chunk in mel_chunks:     # (B, n_mels, any number of frames)
            play(streamer.feed(mel_chunk))
        play(streamer.flush())
    """

    def __init__(self, model):
        backbone, head = model.backbone, model.head
        assert isinstance(backbone, VocosBackbone), "VocosStreamer requires a VocosBackbone"
        assert backbone.causal, "VocosStreamer requires a backbone built with causal=True"
        assert not backbone.adanorm, "Streaming with AdaLayerNorm conditioning is not supported"
        assert isinstance(head, ISTFTHead), "VocosStreamer requires an ISTFTHead"
        self.backbone = backbone
        self.head = head
        self.reset()

    def reset(self):
        """Reset all internal state to start a new stream."""
        self.embed = _StreamingConv1d(self.backbone.embed, *self.backbone.embed_pad)
        self.blocks = [_StreamingConvNeXtBlock(block) for block in self.backbone.convnext]
        self.istft = _StreamingISTFT(self.head.istft)

    def _post_embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.norm(x.transpose(1, 2)).transpose(1, 2)

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.final_layer_norm(x.transpose(1, 2))
        x = self.head.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        spec = mag * (torch.cos(p) + 1j * torch.sin(p))
        return self.istft.feed(spec)

    @torch.inference_mode()
    def feed(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): Feature chunk of shape (B, C, L) with any number of frames L >= 0.

        Returns:
            Tensor: Audio samples of shape (B, T) finalized by this chunk (possibly T=0 early in the stream).
        """
        x = self._post_embed(self.embed.feed(features))
        for block in self.blocks:
            x = block.feed(x)
        return self._head(x)

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        """Signal end of stream and return the remaining audio samples of shape (B, T)."""
        x = self._post_embed(self.embed.flush())
        for block in self.blocks:
            fed = block.feed(x)
            x = torch.cat([fed, block.flush()], dim=-1)
        audio = self._head(x)
        return torch.cat([audio, self.istft.flush()], dim=-1)
