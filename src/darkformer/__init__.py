"""DARKformer for PyTorch."""

from darkformer.attention import (
    AttentionMode,
    CrossAttention,
    DarkformerAttention,
    DarkformerKernelAttention,
    SelfAttention,
)
from darkformer.backends import AttentionBackend
from darkformer.model import (
    Darkformer,
    DarkformerBlock,
    DarkformerEncDec,
    DarkformerLM,
)
from darkformer.random_features import DataAwareRandomFeatures

__version__ = "0.1.0"

__all__ = [
    "CrossAttention",
    "AttentionBackend",
    "AttentionMode",
    "Darkformer",
    "DarkformerAttention",
    "DarkformerBlock",
    "DarkformerEncDec",
    "DarkformerKernelAttention",
    "DarkformerLM",
    "DataAwareRandomFeatures",
    "SelfAttention",
    "__version__",
]
