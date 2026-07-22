from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

from vocos.modules import ConvNeXtBlock, ResBlock1, AdaLayerNorm


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
        causal (bool, optional): If True, convolutions use left-only (or left-heavy) padding so the model is
            suitable for streaming inference. Weight shapes are identical to the non-causal model, so
            pretrained non-causal checkpoints can be loaded for fine-tuning. Defaults to False.
        lookahead_frames (int, optional): Only used when `causal=True`. Total number of future input frames
            the model may attend to, distributed over the earliest conv layers (up to 3 per conv).
            0 means strictly causal; small values (e.g. 4 frames = 4 * hop_length samples of latency)
            recover most of the quality lost to strict causality. Defaults to 0.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        causal: bool = False,
        lookahead_frames: int = 0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.causal = causal
        if causal:
            max_lookahead = 3 * (num_layers + 1)
            if not 0 <= lookahead_frames <= max_lookahead:
                raise ValueError(f"lookahead_frames must be in [0, {max_lookahead}] for num_layers={num_layers}")
            # Distribute the lookahead budget over the earliest convs first, up to 3 frames each,
            # so future context propagates through as many subsequent layers as possible.
            budget = lookahead_frames
            embed_lookahead = min(3, budget)
            budget -= embed_lookahead
            block_lookaheads = []
            for _ in range(num_layers):
                r = min(3, budget)
                block_lookaheads.append(r)
                budget -= r
            self.lookahead_frames = lookahead_frames
            self.embed_pad = (6 - embed_lookahead, embed_lookahead)
            self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=0)
        else:
            self.lookahead_frames = 3 * (num_layers + 1)
            block_lookaheads = [None] * num_layers
            self.embed_pad = None
            self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                    lookahead=block_lookahead,
                )
                for block_lookahead in block_lookaheads
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        if self.embed_pad is not None:
            x = torch.nn.functional.pad(x, self.embed_pad)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self, input_channels, dim, num_blocks, layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(nn.Conv1d(input_channels, dim, kernel_size=3, padding=1))
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x
