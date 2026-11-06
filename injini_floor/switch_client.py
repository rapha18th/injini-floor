"""A small Switch agent-protocol client, written from docs/official/internals/agent-protocol.md.

SSE down, HTTP up. Any Python process holding an agent API key can be a room member:
this file is all it takes to put a non-LLM model into a Switch room.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

log = logging.getLogger("switch")

Handler = Callable[[str, dict[str, Any]], Awaitable[None]]

PROTOCOL = 4  # approval_outcome frames arrive from revision 4


class SwitchError(RuntimeError):
    pass


class Switch:
    """HTTP side: operations, messages, media and approval cards."""

    def __init__(self, base_url: str, agent_id: str, api_key: str):
        self.base = base_url.rstrip("/")
        self.agent_id = agent_id
        self.http = httpx.AsyncClient(
            base_url=self.base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def _json(self, r: httpx.Response) -> Any:
        if r.status_code >= 400:
            raise SwitchError(f"{r.request.method} {r.request.url.path} → {r.status_code}: {r.text[:400]}")
        return r.json() if r.content else None

    # operations ------------------------------------------------------------

    async def op(self, name: str, args: dict[str, Any] | None = None, connection_id: str | None = None) -> Any:
        headers = {"X-Switch-Connection-Id": connection_id} if connection_id else {}
        r = await self.http.post(f"/agents/{self.agent_id}/ops/{name}", json=args or {}, headers=headers)
        body = await self._json(r)
        return body.get("result") if isinstance(body, dict) else body

    async def say(self, room_id: str, content: str) -> None:
        """Post into any room the agent belongs to. Needs no connection."""
        await self._json(await self.http.post(f"/agents/{self.agent_id}/message",
                                              json={"room_id": room_id, "content": content}))

    # media -----------------------------------------------------------------

    async def download(self, room_id: str, mxc: str) -> bytes:
        r = await self.http.get(f"/agents/{self.agent_id}/rooms/{room_id}/media", params={"mxc": mxc})
        if r.status_code >= 400:
            raise SwitchError(f"media download {r.status_code}: {r.text[:200]}")
        return r.content

    async def upload(self, room_id: str, filename: str, data: bytes, mime: str, caption: str | None = None,
                     thread_id: str | None = None) -> Any:
        form = {k: v for k, v in {"caption": caption, "thread_id": thread_id}.items() if v}
        r = await self.http.post(f"/agents/{self.agent_id}/rooms/{room_id}/media",
                                 files={"file": (filename, data, mime)}, data=form)
        return await self._json(r)

    # approval cards --------------------------------------------------------

    async def open_approval(self, session_id: str, request_id: str, room_id: str, title: str, detail: str,
                            options: list[dict[str, str]], ttl_minutes: int = 120) -> Any:
        """Put a card in the room. Whoever may address this agent may answer it."""
        expires = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
        body = {
            "request_id": request_id,
            "turn_id": request_id,
            "kind": "approval",
            "title": title,
            "detail": detail,
            "options": options,
            "questions": [],
            "room_id": room_id,
            "thread_id": None,
            "expires_at": expires.isoformat(),
        }
        return await self._json(await self.http.post(f"/agent-sessions/{session_id}/approvals", json=body))

    async def outcomes(self) -> list[dict[str, Any]]:
        return await self._json(await self.http.get("/agent-sessions/approvals/outcomes")) or []

    async def delivered(self, session_id: str, request_id: str) -> None:
        await self._json(await self.http.post(f"/agent-sessions/{session_id}/approvals/{request_id}/delivered"))


class Connection:
    """One event stream plus its heartbeat. The id is kept across reconnects."""

    def __init__(self, sw: Switch, scope: str = "all", filter: str = "all", rooms: list[str] | None = None,
                 client: str = "injini"):
        self.sw = sw
        self.id = str(uuid.uuid4())
        self.scope = scope
        self.filter = filter
        self.rooms = rooms or []
        self.client = client
        self.cursor: int | None = None
        self.attached = asyncio.Event()
        self.state: dict[str, Any] = {}

    async def run(self, on_event: Handler) -> None:
        beat = asyncio.create_task(self._beat())
        try:
            while True:
                try:
                    await self._stream(on_event)
                except (httpx.HTTPError, SwitchError) as e:
                    log.warning("stream dropped (%s); reopening", e)
                self.attached.clear()
                await asyncio.sleep(1.5)
        finally:
            beat.cancel()

    async def _stream(self, on_event: Handler) -> None:
        params = {
            "connection_id": self.id,
            "scope": self.scope,
            "filter": self.filter,
            "start_from": "head",
            "protocol": PROTOCOL,
            "protocol_accepts": 1,
            "client": self.client,
            "client_version": "0.1.0",
        }
        if self.rooms:
            params["rooms"] = ",".join(self.rooms)
        headers = {"Accept": "text/event-stream"}
        if self.cursor is not None:
            headers["Last-Event-ID"] = str(self.cursor)
        timeout = httpx.Timeout(None, connect=10.0)
        async with self.sw.http.stream("GET", f"/agents/{self.sw.agent_id}/events", params=params,
                                       headers=headers, timeout=timeout) as r:
            if r.status_code >= 400:
                raise SwitchError(f"open stream {r.status_code}: {(await r.aread())[:300]!r}")
            kind, data, sid = None, [], None
            async for line in r.aiter_lines():
                if line == "":
                    if data:
                        await self._dispatch(kind or "message", "\n".join(data), sid, on_event)
                    kind, data, sid = None, [], None
                elif line.startswith(":"):
                    continue
                elif line.startswith("event:"):
                    kind = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif line.startswith("id:"):
                    sid = line[3:].strip()

    async def _dispatch(self, kind: str, raw: str, sid: str | None, on_event: Handler) -> None:
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            return
        if sid and sid.isdigit():
            self.cursor = int(sid)
        if kind == "connection_state":
            self.state = frame
            self.cursor = frame.get("cursor", self.cursor)
            self.attached.set()
            log.info("%s connected: protocol %s, rooms %s", self.client, frame.get("protocol"), frame.get("rooms"))
        elif kind == "evicted":
            log.warning("evicted: %s", frame.get("reason"))
            raise SwitchError("evicted")
        try:
            await on_event(kind, frame)
        except Exception:  # one bad event must not drop the stream
            log.exception("handler failed on %s", kind)

    async def _beat(self) -> None:
        while True:
            await asyncio.sleep(2)
            if not self.attached.is_set():
                continue
            try:
                r = await self.sw.http.post(f"/agents/{self.sw.agent_id}/connection/beat",
                                            json={"connection_id": self.id, "cursor": self.cursor or 0})
                if r.status_code in (404, 409):
                    self.attached.clear()
            except httpx.HTTPError as e:
                log.debug("beat failed: %s", e)


async def register(base_url: str, registration_token: str, name: str, description: str,
                   display_name: str, overwrite: bool = False) -> dict[str, str]:
    """Register a custom, always-on agent. Returns {id, api_key}; the key is shown once."""
    profile = {
        "connection_model": "always_on",
        "message_exchange": True,
        "pre_invocation_mediation": [],
        "post_invocation_mediation": [],
        "event_reporting": [],
    }
    body = {"name": name, "description": description, "display_name": display_name,
            "connector_type": "custom", "integration_profile": profile, "overwrite": overwrite,
            "metadata": {"runtime": "injini-floor"}}
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30) as http:
        r = await http.post("/agents", json=body, headers={"Authorization": f"Bearer {registration_token}"})
        if r.status_code >= 400:
            raise SwitchError(f"register {r.status_code}: {r.text[:400]}")
        return r.json()
