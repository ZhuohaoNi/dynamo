# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""V0 operation policy and transport adapter for shared socket sessions."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import replace
from time import monotonic
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

_CANCELLATION_POLL_SECONDS = 0.01


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
        self._state_waiters: set[asyncio.Future[None]] = set()
        self._waiting_writers = 0
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
        return replace(
            self._core.snapshot(),
            waiting_writers=self._waiting_writers,
        )

    def _wake_waiters(self) -> None:
        for waiter in tuple(self._state_waiters):
            if not waiter.done():
                waiter.set_result(None)

    async def _wait_for_state_change(self, timeout: float | None) -> None:
        waiter = asyncio.get_running_loop().create_future()
        self._state_waiters.add(waiter)
        try:
            await asyncio.wait_for(waiter, timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            self._state_waiters.discard(waiter)

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
        deadline = monotonic() + timeout if timeout is not None else None
        is_writer = mode == RequestedLockType.RW
        if is_writer:
            self._waiting_writers += 1
        try:
            # Keep lock waits on the event loop; the default executor also carries FDs.
            while True:
                if is_cancelled is not None and is_cancelled():
                    return None

                session = None
                if is_writer or self._waiting_writers == 0:
                    session = self._core.acquire(mode, 0, is_cancelled)
                if session is not None:
                    self._pending[session_id] = session
                    return session.mode

                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return None

                wait_timeout = remaining
                if is_cancelled is not None:
                    wait_timeout = (
                        _CANCELLATION_POLL_SECONDS
                        if wait_timeout is None
                        else min(wait_timeout, _CANCELLATION_POLL_SECONDS)
                    )
                await self._wait_for_state_change(wait_timeout)
        finally:
            if is_writer:
                self._waiting_writers -= 1
                self._wake_waiters()

    async def cancel_connect(
        self,
        session_id: str,
        mode: Optional[GrantedLockType],
    ) -> None:
        session = self._pending.pop(session_id, None)
        if session is not None:
            self._core.close(session)
            self._wake_waiters()

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
        self._wake_waiters()

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
        self._wake_waiters()
        return event

    async def finish_cleanup(self, conn: Optional[Connection]) -> None:
        if conn is not None:
            await conn.close()
