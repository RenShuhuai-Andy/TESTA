# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .vit import apply_patch as vit
from .timesformer import apply_patch as timesformer

__all__ = ["vit", "timesformer"]
