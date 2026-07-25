# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V0 import surface for the shared Torch allocator extension."""

from gpu_memory_service.core.client.torch.extensions import _allocator_ext

__all__ = ["_allocator_ext"]
