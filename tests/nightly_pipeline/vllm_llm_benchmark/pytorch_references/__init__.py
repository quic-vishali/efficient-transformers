# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Category-specific PyTorch reference generators for benchmark validation."""

from .registry import ReferenceSpec, reference_for

__all__ = ["ReferenceSpec", "reference_for"]
