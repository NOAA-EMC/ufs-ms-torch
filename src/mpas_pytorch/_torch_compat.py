# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The private PyTorch names the model needs, for the one PyTorch version it supports.

Code uses public PyTorch APIs wherever they exist. The numerics of the dycore and the physics reach the names below,
which have no public equivalent, through this module only. Compiler configuration and introspection (``torch._dynamo``
and ``torch._inductor``) stay with the compile code that uses them; the Inductor lowering patches of
``ufs_physics_common.compile_policy`` check the version through this module. Private names can change their meaning in
any release, so importing this module raises ``RuntimeError`` unless PyTorch is ``SUPPORTED_TORCH`` and every name
below exists.
"""

from __future__ import annotations

import torch
import torch.autograd.forward_ad as _forward_ad

SUPPORTED_TORCH = "2.8.0"


def require_supported_torch() -> None:
    """Raise ``RuntimeError`` unless the installed PyTorch release is ``SUPPORTED_TORCH`` (a local build suffix such
    as ``+cu128`` is allowed)."""
    release = str(torch.__version__).split("+", 1)[0]
    if release != SUPPORTED_TORCH:
        raise RuntimeError(
            f"this model supports torch=={SUPPORTED_TORCH} only (it relies on PyTorch internals that change between "
            f"releases); installed: torch {torch.__version__}"
        )


def _private(owner, name: str, where: str):
    value = getattr(owner, name, None)
    if value is None:
        raise RuntimeError(f"torch {torch.__version__} lacks {where}.{name}, which this model needs")
    return value


require_supported_torch()

_functorch_c = _private(torch._C, "_functorch", "torch._C")
_maybe_current_level = _private(_functorch_c, "maybe_current_level", "torch._C._functorch")
#: Whether a tensor is a functorch (``torch.func``) wrapper.
is_functorch_wrapped = _private(_functorch_c, "is_functorch_wrapped_tensor", "torch._C._functorch")
#: The tensor a functorch wrapper holds, one level down.
functorch_unwrap = _private(_functorch_c, "get_unwrapped", "torch._C._functorch")
_private(_forward_ad, "_current_level", "torch.autograd.forward_ad")

from torch._functorch import pyfunctorch as _pyfunctorch  # noqa: E402
from torch._subclasses import fake_tensor as _fake_tensor  # noqa: E402

#: A context in which no functorch transform is active (its interpreter stack is cleared, then restored).
temporarily_clear_interpreter_stack = _private(_pyfunctorch, "temporarily_clear_interpreter_stack",
                                               "torch._functorch.pyfunctorch")
#: Whether a tensor is a fake tensor of a compile trace.
is_fake = _private(_fake_tensor, "is_fake", "torch._subclasses.fake_tensor")
#: The device-side assertion: raises ``RuntimeError(message)`` where the predicate is false, without a host read.
assert_async = _private(torch, "_assert_async", "torch")
#: Copy each source into its destination in one fused launch per device and dtype.
foreach_copy_ = _private(torch, "_foreach_copy_", "torch")
#: The thread-local dispatch key sets and the guard that forces them.
DispatchKey = _private(torch._C, "DispatchKey", "torch._C")
dispatch_tls_local_include_set = _private(torch._C, "_dispatch_tls_local_include_set", "torch._C")
dispatch_tls_local_exclude_set = _private(torch._C, "_dispatch_tls_local_exclude_set", "torch._C")
ForceDispatchKeyGuard = _private(torch._C, "_ForceDispatchKeyGuard", "torch._C")


def forward_ad_level() -> int:
    """The innermost active forward-mode AD level (``torch.autograd.forward_ad.dual_level``), -1 outside any."""
    return _forward_ad._current_level


def forward_ad_active() -> bool:
    """Whether a forward-mode AD level is active, so dual tensors may reach the code."""
    return _forward_ad._current_level >= 0


def functorch_level() -> int | None:
    """The innermost active functorch transform level (``torch.func``), None outside any."""
    return _maybe_current_level()
