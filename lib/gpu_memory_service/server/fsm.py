# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V0 connection transport state over shared GMS sessions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.core.server.sessions import (
    ServerSession,
    ServerState,
    StateEvent,
)


class InvalidTransition(Exception):
    pass


@dataclass(eq=False)
class Connection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    mode: GrantedLockType
    session_id: str
    recv_buffer: bytearray = field(default_factory=bytearray)
    core_session: ServerSession | None = None

    def __hash__(self) -> int:
        return hash(self.session_id)

    async def close(self) -> None:
        self.writer.close()
        try:
            await self.writer.wait_closed()
        except Exception:
            pass


__all__ = [
    "Connection",
    "InvalidTransition",
    "ServerState",
    "StateEvent",
]
