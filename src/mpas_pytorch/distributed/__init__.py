# Copyright (c) 2026 IC Weather LLC. Licensed under LICENSE and PERMISSIONS.md.
# Portions are modified from the sources listed in NOTICE, under their terms.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Experimental spatial decomposition for the existing PyTorch dycore.

The experimental CPU/Gloo policy executes local timesteps and owns halo AD;
release support remains single-process. Full derivative-matrix completion,
checkpoint replay and CUDA/NCCL commissioning are still required.
"""
