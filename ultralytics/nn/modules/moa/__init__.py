# 🐧 YOLO-Master MoA (Mixture of Attention) Module
# Copyright (C) 2026 Tencent. All rights reserved.
"""Mixture-of-Attention (MoA) modules for YOLO-Master.

Provides three progressive variants:
  - MoABlock          : base MoA building block (local + global head routing)
  - C2fMoA            : C2f-style feature flow with MoA blocks (drop-in for C3k2)
  - NeckMoAFusion     : cross-scale MoA fusion for FPN/PAN neck
"""

from .block import MoABlock
from .router import anneal_moa_temperature
from .wrappers import C2fMoA, NeckMoAFusion, collect_moa_aux_loss

__all__ = ("MoABlock", "C2fMoA", "NeckMoAFusion", "anneal_moa_temperature", "collect_moa_aux_loss")
