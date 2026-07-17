"""Backward-compatible import path for Mixture-of-Attention modules."""

from ultralytics.nn.modules._numeric import all_reduce_mean  # noqa: F401

from . import heads as _heads
from .block import MoABlock
from .heads import (  # noqa: F401
    _GlobalAttnHead,
    _LocalAttnHead,
    _RegionalAttnHead,
    _flash_attn,
    _window_flash_attn,
)
from .router import _MoARouter, _moa_router_aux_loss, anneal_moa_temperature  # noqa: F401
from .wrappers import C2fMoA, NeckMoAFusion, collect_moa_aux_loss

# Historical monkeypatch target retained for plugins and tests.
F = _heads.F

__all__ = ("MoABlock", "C2fMoA", "NeckMoAFusion", "anneal_moa_temperature", "collect_moa_aux_loss")
