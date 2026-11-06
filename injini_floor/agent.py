"""Injini, the agent that listens to machines.

It sits in the plant room as a full member. A voice note of a machine comes in; Injini
scores it against that machine's healthy baseline and posts what it heard. When a machine
is out of baseline it opens a case room, briefs the Foreman, and puts the stop decision in
front of whoever may address it. A recording after the repair closes the case.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import acoustics as A
from . import render as R
from .switch_client import Connection, Switch, SwitchError

log = logging.getLogger("injini")

AUDIO_EXT = (".ogg", ".oga", ".opus", ".wav", ".flac", ".mp3", ".m4a", ".aac", ".webm")
MACHINE_WORDS = "fan|pump|motor|genset|generator|engine|compressor|mill|grinder|blower|conveyor|press|machine|borehole"
ASSET_RE = re.compile(rf"\b({MACHINE_WORDS})[\s_-]?(\d+[a-z]?)\b", re.IGNORECASE)
TAG_RE = re.compile(r"\b([a-z][a-z0-9]*-\d+[a-z0-9]*)\b", re.IGNORECASE)

STOP, RUN, FALSE_ALARM = "stop", "run", "false-alarm"


def asset_from(text: str) -> str | None:
    m = ASSET_RE.search(text or "")
    if m:
        return f"{m.group(1).lower()}-{m.group(2).lower().zfill(2)}"
    m = TAG_RE.search(text or "")
    return m.group(1).lower() if m else None


@dataclass
class Case:
    room_id: str
    asset: str
    number: int
    plant_room_id: str
    opened: float
    sigma: float
    status: str = "awaiting decision"  # awaiting decision, stopped, running, closed
    request_id: str = ""
    session_id: str = ""
    doc_id: str = ""
    log: list[str] = field(default_factory=list)


class Injini:
    def __init__(self, sw: Switch, data_dir: str, foreman: str = "foreman", stop_webhook: str | None = None):
        self.sw = sw
        self.data_dir = data_dir
        self.foreman = foreman
        self.stop_webhook = stop_webhook
        os.makedirs(os.path.join(data_dir, "baselines"), exist_ok=True)
        self.state_path = os.path.join(data_dir, "state.json")
        self.baselines: dict[str, A.Baseline] = {}
        self.baseline_docs: dict[str, str] = {}  # asset -> room document id
        self.cases: dict[str, Case] = {}  # case room id -> case
        self.last_asset: dict[str, str] = {}  # room id -> asset
        self.last_anomaly: dict[str, np.ndarray] = {}  # case room id -> embeddings of the flagged clip
        self.listener = Connection(sw, scope="all", filter="all", client="injini-listener")
        self.actor = Connection(sw, scope="single", filter="addressed", client="injini-actor")
        self.room_lock = asyncio.Lock()
        self.work = asyncio.Lock()
        self._load()

    # persistence ---------------------------------------------------------

    def _load(self) -> None:
        for name in os.listdir(os.path.join(self.data_dir, "baselines")):
            if name.endswith(".json"):
                with open(os.path.join(self.data_dir, "baselines", name), encoding="utf-8") as f:
                    b = A.Baseline.from_json(f.read())
                self.baselines[b.asset] = b
        if os.path.exists(self.state_path):
            with open(self.state_path, encoding="utf-8") as f:
                s = json.load(f)
            self.baseline_docs = s.get("baseline_docs", {})
            self.cases = {k: Case(**v) for k, v in s.get("cases", {}).items()}
            self.last_asset = s.get("last_asset", {})
        log.info("loaded %d baselines, %d cases", len(self.baselines), len(self.cases))

    def _save(self) -> None:
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({"baseline_docs": self.baseline_docs, "cases": {k: asdict(v) for k, v in self.cases.items()},
                       "last_asset": self.last_asset}, f, indent=1)

    async def _store_baseline(self, base: A.Baseline, room_id: str) -> None:
        text = base.to_json()
        with open(os.path.join(self.data_dir, "baselines", f"{base.asset}.json"), "w", encoding="utf-8") as f:
            f.write(text)
        # The machine's memory also lives in the room, where every member can read it.
        try:
            async with self.room_lock:
                await self._enter(room_id)
                doc = self.baseline_docs.get(base.asset)
                meta = dict(
                    description=f"Healthy acoustic baseline of {base.asset}: {base.n_clips} clips, {base.n_windows} windows.",
                    instructions="Injini's model state for this machine. Frozen mn10_as embeddings of healthy windows "
                                 "(float16, base64) plus band and spectrum means. Edited only by Injini.",
                    content=text,
                )
                if doc:
                    await self.sw.op("update_room_document", {"document_id": doc, **meta}, self.actor.id)
                else:
                    r = await self.sw.op("create_room_document", {"name": f"baseline-{base.asset}", **meta},
                                         self.actor.id)
                    self.baseline_docs[base.asset] = r["document_id"]
        except SwitchError as e:
            log.warning("baseline document not stored in the room: %s", e)
        self._save()

    async def _enter(self, room_id: str) -> None:
        await self.actor.attached.wait()
        await self.sw.op("connect_to_room", {"room_id": room_id, "include_general_instructions": False},
                         self.actor.id)

    # main loop -----------------------------------------------------------

    async def run(self) -> None:
        await asyncio.gather(self.listener.run(self.on_event), self.actor.run(self._ignore), self._poll_outcomes())

    async def _ignore(self, kind: str, frame: dict[str, Any]) -> None:
        return None

    async def on_event(self, kind: str, frame: dict[str, Any]) -> None:
        if kind == "approval_outcome":
            await self._outcome(frame)
        elif kind == "message":
            asyncio.create_task(self._message(frame))

    async def _message(self, frame: dict[str, Any]) -> None:
        p = frame.get("payload", {})
        room_id = frame["room_id"]
        if p.get("sender_kind") == "platform":
            return
        body = p.get("body") or ""
        audio = [a for a in p.get("attachments") or [] if _is_audio(a)]
        if not audio:
            if p.get("addressed"):
                await self._text_command(room_id, body)
            return
        async with self.work:  # one clip at a time keeps the laptop responsive
            for att in audio:
                try:
                    await self._clip(room_id, frame.get("bridge_id"), p, body, att)
                except Exception as e:
                    log.exception("clip failed")
                    await self.sw.say(room_id, f"Injini could not process that recording: {e}")

    async def _text_command(self, room_id: str, body: str) -> None:
        b = body.lower()
        if "machines" in b or "status" in b:
            if not self.baselines:
                await self.sw.say(room_id, "No machines enrolled yet. Send a 10-second recording with `enrol pump-3`.")
                return
            lines = [f"• {a}: {b_.n_clips} healthy clips, {b_.n_windows} windows" for a, b_ in sorted(self.baselines.items())]
            open_cases = [c for c in self.cases.values() if c.status != "closed"]
            lines += [f"• case-{c.asset}-{c.number:03d}: {c.status}" for c in open_cases]
            await self.sw.say(room_id, "Machines Injini knows:\n" + "\n".join(lines))
        else:
            await self.sw.say(room_id, HELP)

    async def _clip(self, room_id: str, bridge_id: str | None, p: dict, body: str, att: dict) -> None:
        data = await self.sw.download(room_id, att["mxc"])
        clip = await asyncio.to_thread(A.analyse, data)
        if not clip.quality.ok:
            await self.sw.say(room_id, f"🎙️ {clip.quality.reason}")
            return
        case = self.cases.get(room_id)
        asset = asset_from(body) or (case.asset if case else None) or self.last_asset.get(room_id)
        if not asset:
            await self.sw.say(room_id, "Which machine is this? Add its name to the message, for example `pump-3`.")
            return
        self.last_asset[room_id] = asset
        if re.search(r"\benrol", body, re.IGNORECASE):
            await self._enrol(room_id, asset, clip)
            return
        base = self.baselines.get(asset)
        if base is None:
            await self.sw.say(room_id, f"{asset} has no baseline yet. Send 2 or 3 recordings of it running well, "
                                       f"each with `enrol {asset}`.")
            return
        v = await asyncio.to_thread(A.score, base, clip)
        if case and case.status != "closed":
            await self._recheck(case, v, clip, base)
            return
        open_case = next((c for c in self.cases.values() if c.asset == asset and c.status != "closed"), None)
        png = await asyncio.to_thread(R.verdict_png, v, clip, base,
                                      f"case-{asset}-{open_case.number:03d}" if open_case else "plant floor")
        await self.sw.upload(room_id, f"{asset}-{int(time.time())}.png", png, "image/png", caption=_caption(v))
        if v.anomalous and open_case is None:
            await self._open_case(room_id, bridge_id, asset, v, clip, png, p)
        elif v.anomalous and open_case is not None:
            await self.sw.say(open_case.room_id, f"New recording from the floor: {v.line()}")

    async def _enrol(self, room_id: str, asset: str, clip: A.Clip) -> None:
        prev = self.baselines.get(asset)
        clips_E = (prev.clip_embeddings if prev else []) + [clip.E]
        if prev is None:
            base = await asyncio.to_thread(A.enrol, asset, [clip])
        else:
            base = A.Baseline(asset, clips_E, prev.band_mean, _running_mean(prev.spec_mean, clip.spec, prev.n_clips),
                              ref_lm=prev.ref_lm)
            await asyncio.to_thread(base.fit)
        self.baselines[asset] = base
        await self._store_baseline(base, room_id)
        more = "" if base.n_clips >= 3 else f" Send {3 - base.n_clips} more for a steadier baseline."
        await self.sw.say(room_id, f"✅ Enrolled {asset}: {base.n_clips} healthy clip(s), {base.n_windows} windows. "
                                   f"Alarm at {A.THRESHOLD_SIGMA:.1f}σ.{more}")

    # cases ---------------------------------------------------------------

    async def _open_case(self, plant: str, bridge_id: str | None, asset: str, v: A.Verdict, clip: A.Clip,
                         png: bytes, p: dict) -> None:
        number = 1 + sum(1 for c in self.cases.values() if c.asset == asset)
        name = f"case-{asset}-{number:03d}"
        args: dict[str, Any] = {
            "name": name,
            "description": f"{asset} out of baseline at {v.sigma:.1f}σ, flagged by Injini from {p.get('sender_name', 'the floor')}.",
            "agent_names": [self._me(), self.foreman],
            "instructions": CASE_INSTRUCTIONS.format(asset=asset, foreman=self.foreman),
            "linked_rooms": [{"target_room_id": plant, "label": "parent: the plant floor"}],
            "roles": [{"name": "artisan", "exclusive": True,
                       "instructions": f"You are on call for {asset}. Inspect it, fix it, then send a fresh "
                                       f"10-second recording into this room so Injini can verify the repair."}],
        }
        if bridge_id:
            args.update(bridge_id=bridge_id, channel_type="channel_public")
        room = await self.sw.op("create_room", args)
        case_room = room["id"]
        case = Case(room_id=case_room, asset=asset, number=number, plant_room_id=plant, opened=time.time(),
                    sigma=v.sigma, session_id=f"case-{case_room}", request_id=f"stop-{name}")
        self.cases[case_room] = case
        self.last_anomaly[case_room] = clip.E
        self._save()
        try:
            await self.sw.op("link_rooms", {"source_room_id": plant, "target_room_id": case_room,
                                            "label": f"open case: {asset}"})
        except SwitchError as e:
            log.warning("link back failed: %s", e)
        await self.sw.say(plant, f"🛠️ Opened **{name}**. The supervisor decides there whether {asset} stops.")
        await self.sw.upload(case_room, f"{name}.png", png, "image/png", caption=_caption(v))
        case.doc_id = await self._case_file(case, v, p, create=True)
        await self._brief_foreman(case, v)
        await self.sw.open_approval(
            case.session_id, case.request_id, case_room,
            title=f"Stop {asset.upper()} for inspection?",
            detail=f"{v.line()}\nOnly people allowed to address Injini can answer this card.",
            options=[
                {"id": STOP, "label": f"Stop {asset} now", "decision": "accept"},
                {"id": RUN, "label": "Run to end of shift", "decision": "decline"},
                {"id": FALSE_ALARM, "label": "False alarm: learn this sound", "decision": "decline"},
            ],
        )
        case.log.append(f"{_now()} opened at {v.sigma:.1f}σ; stop card raised")
        self._save()

    async def _brief_foreman(self, case: Case, v: A.Verdict) -> None:
        lo, hi = v.top_range_hz
        text = (f"@{self.foreman} {case.asset} is {v.sigma:.1f}σ out of its healthy baseline. "
                f"The change sits at {lo:,.0f}–{hi:,.0f} Hz"
                + (f" with a new peak at {v.peak_hz:,.0f} Hz (+{v.peak_db:.0f} dB)" if v.peak_db > 3 else "")
                + f". {v.hint} Read the case file document in this room and write a work order: likely cause, "
                  f"isolation steps, parts and tools, time estimate. Then hand it to @artisan. "
                  f"Do not tell anyone to stop the machine: the supervisor decides that on the card.")
        try:
            async with self.room_lock:
                await self._enter(case.room_id)
                await self.sw.op("send_targeted_message",
                                 {"body": text.replace(f"@{self.foreman} ", ""), "target_names": [self.foreman]},
                                 self.actor.id)
        except SwitchError as e:
            log.warning("targeted brief failed (%s); posting plainly", e)
            await self.sw.say(case.room_id, text)

    async def _case_file(self, case: Case, v: A.Verdict | None, p: dict | None, create: bool = False) -> str:
        lines = [f"# case-{case.asset}-{case.number:03d}", "", f"Machine: {case.asset}",
                 f"Status: {case.status}", f"Opened: {time.strftime('%Y-%m-%d %H:%M', time.localtime(case.opened))}",
                 f"Score at opening: {case.sigma:.1f}σ (alarm at {A.THRESHOLD_SIGMA:.1f}σ)"]
        if v is not None:
            lo, hi = v.top_range_hz
            lines += [f"Strongest change: {lo:,.0f}–{hi:,.0f} Hz", f"Peak: {v.peak_hz:,.0f} Hz, +{v.peak_db:.0f} dB",
                      f"Hint: {v.hint}",
                      "Sigma explained per band: " + ", ".join(
                          f"{a:,.0f} Hz {c:+.1f}" for (a, _), c in zip(A.group_ranges_hz(), v.contributions))]
        if p is not None:
            lines.append(f"Reported by: {p.get('sender_name', 'unknown')}")
        lines += ["", "## Log"] + [f"- {entry}" for entry in case.log]
        meta = dict(description=f"Case file for {case.asset}: evidence, decision and verification.",
                    instructions="Written by Injini. Read it before writing the work order.", content="\n".join(lines))
        try:
            async with self.room_lock:
                await self._enter(case.room_id)
                if create or not case.doc_id:
                    r = await self.sw.op("create_room_document", {"name": "case-file", **meta}, self.actor.id)
                    return r["document_id"]
                await self.sw.op("update_room_document", {"document_id": case.doc_id, **meta}, self.actor.id)
        except SwitchError as e:
            log.warning("case file not written: %s", e)
        return case.doc_id

    async def _recheck(self, case: Case, v: A.Verdict, clip: A.Clip, base: A.Baseline) -> None:
        png = await asyncio.to_thread(R.verdict_png, v, clip, base, f"case-{case.asset}-{case.number:03d} · re-check")
        await self.sw.upload(case.room_id, f"recheck-{int(time.time())}.png", png, "image/png", caption=_caption(v))
        if v.anomalous:
            case.log.append(f"{_now()} re-check {v.sigma:.1f}σ, still out of baseline")
            await self.sw.say(case.room_id, f"Still out of baseline at {v.sigma:.1f}σ. {v.hint}")
        else:
            case.status = "closed"
            case.log.append(f"{_now()} re-check {v.sigma:.1f}σ, back within baseline; case closed")
            await self.sw.say(case.room_id, f"✅ {case.asset} sounds like its healthy self again ({v.sigma:+.1f}σ). "
                                            f"Repair verified by sound. Case closed.")
            await self.sw.say(case.plant_room_id, f"✅ case-{case.asset}-{case.number:03d} closed: {case.asset} "
                                                  f"verified healthy by sound at {v.sigma:+.1f}σ.")
        self._save()
        await self._case_file(case, None, None)

    # decisions -----------------------------------------------------------

    async def _poll_outcomes(self) -> None:
        """The stream pushes outcomes; this catches any wake that was lost."""
        while True:
            await asyncio.sleep(5)
            try:
                for o in await self.sw.outcomes():
                    await self._outcome(_camel_to_snake(o))
            except Exception as e:  # noqa: BLE001
                log.debug("outcome poll: %s", e)

    async def _outcome(self, o: dict[str, Any]) -> None:
        o = _camel_to_snake(o)
        case = next((c for c in self.cases.values() if c.request_id == o.get("request_id")), None)
        if case is None:
            return
        try:
            await self.sw.delivered(o["session_id"], o["request_id"])
        except SwitchError:
            pass
        if case.status != "awaiting decision":
            return
        who = _who(o.get("answered_by"))
        answer = o.get("answer")
        if o.get("state") == "expired":
            case.status = "running"
            case.log.append(f"{_now()} stop card expired unanswered")
            await self.sw.say(case.room_id, f"The stop card for {case.asset} expired unanswered. "
                                            f"{case.asset} keeps running; Injini will flag any new recording.")
        elif answer == STOP:
            case.status = "stopped"
            case.log.append(f"{_now()} STOP approved by {who}")
            await self.sw.say(case.room_id, f"■ Stop approved by {who}. {case.asset.upper()} is down for inspection. "
                                            f"@artisan, once it is fixed, send a 10-second recording here.")
            await self.sw.say(case.plant_room_id, f"■ {case.asset.upper()} stopped for inspection ({who}).")
            await self._fire_webhook(case)
        elif answer == RUN:
            case.status = "running"
            case.log.append(f"{_now()} run to end of shift, approved by {who}")
            await self.sw.say(case.room_id, f"{case.asset} runs to the end of the shift ({who}). "
                                            f"Injini will keep scoring every recording.")
        elif answer == FALSE_ALARM:
            case.status = "closed"
            base = self.baselines[case.asset]
            E = self.last_anomaly.get(case.room_id)
            if E is not None:
                base.clip_embeddings.append(E)
                await asyncio.to_thread(base.fit)
                await self._store_baseline(base, case.plant_room_id)
            case.log.append(f"{_now()} marked false alarm by {who}; sound added to the baseline")
            await self.sw.say(case.room_id, f"Understood, {who}. That sound is now part of {case.asset}'s healthy "
                                            f"baseline ({base.n_clips} clips). Case closed.")
        self._save()
        await self._case_file(case, None, None)

    async def _fire_webhook(self, case: Case) -> None:
        if not self.stop_webhook:
            return
        try:
            r = await self.sw.http.post(self.stop_webhook, json={"asset": case.asset, "action": "stop"},
                                        headers={"Authorization": ""})
            await self.sw.say(case.room_id, f"Stop signal sent to {case.asset}'s controller ({r.status_code}).")
        except Exception as e:  # noqa: BLE001
            await self.sw.say(case.room_id, f"Stop signal failed: {e}. Stop {case.asset} by hand.")

    def _me(self) -> str:
        return os.environ.get("INJINI_NAME", "injini")


HELP = (
    "I'm Injini. I listen to machines.\n"
    "• Enrol: record a machine running well for 10 seconds and send it with `enrol pump-3`. Two or three clips is best.\n"
    "• Check: send any recording with the machine's name. I score it against that machine's baseline.\n"
    "• Out of baseline? I open a case room, brief the Foreman, and ask the supervisor whether it stops.\n"
    "• After the repair, send a recording into the case room. If it sounds healthy again, the case closes.\n"
    "• `@injini machines` lists what I know."
)

CASE_INSTRUCTIONS = """This room is one maintenance case for {asset}, opened by Injini, the acoustic agent.
Injini posts evidence (spectrograms, scores in sigma) and keeps the case-file document current.
@{foreman} writes the work order as a room document named work-order and hands it to @artisan.
The decision to stop {asset} belongs to the supervisor, on Injini's card. No agent stops a machine.
After the repair, the artisan sends a fresh 10-second recording here; Injini re-scores it and closes the case
when the machine sounds healthy again."""


def _is_audio(a: dict) -> bool:
    mime = (a.get("mimetype") or "").lower()
    name = (a.get("filename") or "").lower()
    return mime.startswith("audio/") or name.endswith(AUDIO_EXT) or a.get("msgtype") == "m.audio"


def _caption(v: A.Verdict) -> str:
    mark = "🔴" if v.anomalous else "🟢"
    return f"{mark} {v.line()}"


def _running_mean(mean: np.ndarray, new: np.ndarray, n: int) -> np.ndarray:
    return (mean * n + new) / (n + 1)


def _who(answered_by: str | None) -> str:
    if not answered_by:
        return "the supervisor"
    m = re.match(r"@?([^:]+)", answered_by)
    return m.group(1) if m else answered_by


def _camel_to_snake(d: dict[str, Any]) -> dict[str, Any]:
    return {re.sub(r"(?<!^)(?=[A-Z])", "_", k).lower(): v for k, v in d.items()}


def _now() -> str:
    return time.strftime("%H:%M")
