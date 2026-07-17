"""Backward-compatible import path for Mixture-of-Transformer modules."""

from ultralytics.nn.modules._numeric import all_reduce_mean  # noqa: F401

from . import experts as _experts
from .block import MoTBlock
from .experts import (  # noqa: F401
    _DeformableTransformerExpert,
    _LocalConvTransformerExpert,
    _WindowTransformerExpert,
    _roll_via_cat,
)
from .router import (  # noqa: F401
    _MoTRouter,
    _mot_router_aux_loss,
    anneal_mot_temperature,
    differentiable_balance_loss,
)
from .wrappers import C2fMoT, collect_mot_aux_loss

# Historical monkeypatch targets retained for plugins and tests.
F = _experts.F
_SDPA_EXPLICIT_MAX_TOKENS = _experts._SDPA_EXPLICIT_MAX_TOKENS
_SDPA_FALLBACK_CHUNK = _experts._SDPA_FALLBACK_CHUNK


def _sdpa(*args, **kwargs):
    _experts._SDPA_EXPLICIT_MAX_TOKENS = _SDPA_EXPLICIT_MAX_TOKENS
    _experts._SDPA_FALLBACK_CHUNK = _SDPA_FALLBACK_CHUNK
    return _experts._sdpa(*args, **kwargs)


__all__ = ("MoTBlock", "C2fMoT", "collect_mot_aux_loss", "anneal_mot_temperature")
