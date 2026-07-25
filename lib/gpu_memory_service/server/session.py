# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V0 operation policy and transport adapter for shared socket sessions."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Optional

from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.protocol.messages import (
    AllocateRequest,
    CommitRequest,
    ExportAllocationRequest,
    FreeAllocationRequest,
    GetAllocationRequest,
    GetAllocationStateRequest,
    GetLockStateRequest,
    GetStateHashRequest,
    ListAllocationsRequest,
    MetadataDeleteRequest,
    MetadataGetRequest,
    MetadataListRequest,
    MetadataPutRequest,
)
from gpu_memory_service.core.server.sessions import EpochClearReason
from gpu_memory_service.core.server.sessions import (
    GMSSessionManager as CoreSessionManager,
)
from gpu_memory_service.core.server.sessions import (
    ServerSession,
    ServerState,
    SessionSnapshot,
    StateEvent,
)

from .fsm import Connection


class OperationNotAllowed(Exception):
    pass


RW_REQUIRED: frozenset[type] = frozenset(
    {
        AllocateRequest,
        FreeAllocationRequest,
        MetadataPutRequest,
        MetadataDeleteRequest,
        CommitRequest,
    }
)

RO_ALLOWED: frozenset[type] = frozenset(
    {
        ExportAllocationRequest,
        GetAllocationRequest,
        ListAllocationsRequest,
        MetadataGetRequest,
        MetadataListRequest,
        GetLockStateRequest,
        GetAllocationStateRequest,
        GetStateHashRequest,
    }
)

RW_ALLOWED: frozenset[type] = RW_REQUIRED | RO_ALLOWED


class GMSSessionManager:
    """Adapt the V0 async transport and operation policy to shared sessions."""

    def __init__(
        self,
        clear_epoch: Callable[[EpochClearReason, bool], None],
    ):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._owner_thread: int | None = None
        self._core = CoreSessionManager(clear_epoch)
        self._pending: dict[str, ServerSession] = {}
        self._connections: dict[ServerSession, Connection] = {}
        self._next_session_id = 0

    @property
    def state(self) -> ServerState:
        return self._core.state

    @property
    def rw_conn(self) -> Connection | None:
        session = self._core.rw_session
        return self._connections.get(session)

    def next_session_id(self) -> str:
        self._next_session_id += 1
        return f"session_{self._next_session_id}"

    def snapshot(self) -> SessionSnapshot:
        return self._core.snapshot()

    async def acquire_lock(
        self,
        mode: RequestedLockType,
        timeout_ms: Optional[int],
        session_id: str,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Optional[GrantedLockType]:
        loop = asyncio.get_running_loop()
        owner_thread = threading.get_ident()
        if self._loop is None:
            self._loop = loop
            self._owner_thread = owner_thread
        elif self._loop is not loop or self._owner_thread != owner_thread:
            raise RuntimeError("V0 GMS sessions must use one event loop")

        timeout = timeout_ms / 1000 if timeout_ms is not None else None
        cancelled = threading.Event()
        acquire = asyncio.create_task(
            asyncio.to_thread(
                self._core.acquire,
                mode,
                timeout,
                lambda: (
                    cancelled.is_set() or (is_cancelled is not None and is_cancelled())
                ),
            )
        )
        try:
            session = await asyncio.shield(acquire)
        except asyncio.CancelledError:
            cancelled.set()
            self._core.wake_waiters()
            session = await acquire
            if session is not None:
                self._core.close(session)
            raise
        if session is None:
            return None
        self._pending[session_id] = session
        return session.mode

    async def cancel_connect(
        self,
        session_id: str,
        mode: Optional[GrantedLockType],
    ) -> None:
        session = self._pending.pop(session_id, None)
        if session is not None:
            self._core.close(session)

    def on_connect(self, conn: Connection) -> None:
        session = self._pending[conn.session_id]
        if session.mode != conn.mode:
            raise AssertionError(
                f"session mode changed before connect: {conn.session_id}"
            )
        del self._pending[conn.session_id]
        conn.core_session = session
        self._connections[session] = conn

    def on_commit(self, conn: Connection) -> None:
        if conn.core_session is None:
            raise AssertionError("connection has no core session")
        self._core.commit(conn.core_session)
        conn.mode = conn.core_session.mode

    def check_operation(self, msg_type: type, conn: Connection) -> None:
        if conn.mode == GrantedLockType.RW and msg_type not in RW_ALLOWED:
            raise OperationNotAllowed(
                f"{msg_type.__name__} not allowed for RW session in state {self.state.name}"
            )
        if conn.mode == GrantedLockType.RO and msg_type not in RO_ALLOWED:
            raise OperationNotAllowed(
                f"{msg_type.__name__} not allowed for RO session in state {self.state.name}"
            )
        if msg_type in RW_REQUIRED and conn.mode != GrantedLockType.RW:
            raise OperationNotAllowed(
                f"{msg_type.__name__} requires RW session, got {conn.mode.value}"
            )

    def begin_cleanup(self, conn: Optional[Connection]) -> StateEvent | None:
        if conn is None or conn.core_session is None:
            return None
        self._connections.pop(conn.core_session, None)
        event = self._core.close(conn.core_session)
        conn.core_session = None
        return event

    async def finish_cleanup(self, conn: Optional[Connection]) -> None:
        if conn is not None:
            await conn.close()
