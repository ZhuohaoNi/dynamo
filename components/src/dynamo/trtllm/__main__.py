# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

if __name__ == "__main__":
    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    # In restore mode (DYN_SNAPSHOT_RESTORE_STANDBY=1), before importing TRT-LLM,
    # write selected restore-time env vars to snapshot-control/restore-context.json
    # and exec `sleep infinity` without initializing CUDA or backend/runtime state.
    maybe_run_restore_standby_mode()

    # Opt-in (DYN_ENABLE_MEDIA_DECODERS, off by default): install this backend's
    # media-decoder package(s) before the first lazy cv2 import at request time.
    # After the standby check so a snapshot pod never triggers an install.
    from dynamo.common.utils.media_decoders import maybe_install_media_decoders

    maybe_install_media_decoders("trtllm")

    from dynamo.trtllm.main import main

    main()
