"""Attention backend helpers built on PyTorch SDPA.

KeySync used to call ``xformers.ops.memory_efficient_attention`` for its
attention blocks.  xformers ships no official Windows wheels, so every
attention path now goes through PyTorch's native
``torch.nn.functional.scaled_dot_product_attention`` (SDPA), which is part of
the regular Windows CUDA builds of PyTorch and picks a memory-efficient kernel
on its own.
"""

import logging
import sys
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from packaging import version

logpy = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"
SDPA_IS_AVAILABLE = version.parse(torch.__version__) >= version.parse("2.0.0")

_NEW_SDPA_API = False
SDPBackend = None

if SDPA_IS_AVAILABLE:
    try:  # torch >= 2.3, the non-deprecated entry point
        from torch.nn.attention import SDPBackend, sdpa_kernel

        _NEW_SDPA_API = True
    except ImportError:  # torch 2.0 - 2.2
        from torch.backends.cuda import SDPBackend, sdp_kernel
else:
    logpy.warning(
        f"No SDPA backend available, likely because you are running PyTorch "
        f"{torch.__version__} (< 2.0). Please upgrade PyTorch."
    )


def _legacy_backend_map(backends):
    return {
        "enable_math": SDPBackend.MATH in backends,
        "enable_flash": SDPBackend.FLASH_ATTENTION in backends,
        "enable_mem_efficient": SDPBackend.EFFICIENT_ATTENTION in backends,
    }


def available_backends():
    """Backends SDPA may pick from, ordered by preference.

    The Windows builds of PyTorch are compiled without the FlashAttention
    kernels, so requesting them there only produces warnings and fallbacks.
    """
    if not SDPA_IS_AVAILABLE:
        return []
    if IS_WINDOWS:
        return [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    return [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]


@contextmanager
def sdpa_backend_context(backend=None):
    """Restrict SDPA to ``backend`` (or to whatever this platform supports)."""
    if not SDPA_IS_AVAILABLE:
        yield
        return

    backends = available_backends() if backend is None else [backend]
    if IS_WINDOWS:
        backends = [b for b in backends if b != SDPBackend.FLASH_ATTENTION]
        if not backends:
            backends = [SDPBackend.MATH]

    if _NEW_SDPA_API:
        with sdpa_kernel(backends):
            yield
    else:
        with sdp_kernel(**_legacy_backend_map(backends)):
            yield


def memory_efficient_attention(q, k, v, attn_bias=None, op=None, backend=None):
    """Drop-in replacement for ``xformers.ops.memory_efficient_attention``.

    Accepts the xformers layouts ``(B, M, K)`` and ``(B, M, H, K)`` and
    dispatches to SDPA, which expects ``(B, H, M, K)``.  ``op`` is accepted and
    ignored so call sites that used to pin an xformers operator keep working.
    """
    del op  # SDPA selects its own kernel

    single_head = q.ndim == 3
    if single_head:
        q, k, v = (t.unsqueeze(2) for t in (q, k, v))

    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    with sdpa_backend_context(backend):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    out = out.transpose(1, 2)

    if single_head:
        out = out.squeeze(2)
    return out
