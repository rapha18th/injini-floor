"""Injini's ear as a command-line tool, for the Claude Code agent that hosts it in Switch.

    injini enrol pump-00 a.ogg [b.ogg ...]   learn a machine from healthy recordings
    injini score pump-00 clip.ogg            score a recording, write the spectrogram PNG
    injini machines                          list enrolled machines and open cases
    injini stop pump-00 --case case-pump-00-001 --by "<who asked>"
                                             the gated action: record and signal a stop
    injini learn pump-00 clip.ogg            add a recording the supervisor called a false alarm

Every command prints one JSON object. Model state lives in data/baselines/<asset>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

from . import acoustics as A
from . import render as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("INJINI_DATA", os.path.join(ROOT, "data"))
BASE_DIR = os.path.join(DATA, "baselines")
OUT_DIR = os.path.join(DATA, "out")
EVENTS = os.path.join(DATA, "events.jsonl")


def _path(asset: str) -> str:
    return os.path.join(BASE_DIR, f"{asset}.json")


def _load(asset: str) -> A.Baseline | None:
    p = _path(asset)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return A.Baseline.from_json(f.read())


def _save(base: A.Baseline) -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(_path(base.asset), "w", encoding="utf-8") as f:
        f.write(base.to_json())


def _event(kind: str, **fields) -> dict:
    os.makedirs(DATA, exist_ok=True)
    e = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "event": kind, **fields}
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(e) + "\n")
    return e


def _clip(path: str) -> A.Clip:
    with open(path, "rb") as f:
        return A.analyse(f.read())


def cmd_enrol(a) -> dict:
    clips = [_clip(p) for p in a.files]
    bad = [p for p, c in zip(a.files, clips) if not c.quality.ok]
    good = [c for c in clips if c.quality.ok]
    if not good:
        return {"ok": False, "asset": a.asset, "reason": clips[0].quality.reason}
    prev = _load(a.asset)
    if prev is None:
        base = A.enrol(a.asset, good)
    else:
        n = prev.n_clips
        spec = prev.spec_mean
        for c in good:
            spec = (spec * n + c.spec) / (n + 1)
            n += 1
        base = A.Baseline(a.asset, prev.clip_embeddings + [c.E for c in good], prev.band_mean, spec,
                          ref_lm=prev.ref_lm).fit()
    _save(base)
    _event("enrol", asset=a.asset, clips=base.n_clips)
    return {"ok": True, "asset": a.asset, "clips": base.n_clips, "windows": base.n_windows,
            "threshold_sigma": A.THRESHOLD_SIGMA, "skipped": bad,
            "message": f"Enrolled {a.asset}: {base.n_clips} healthy clip(s), {base.n_windows} windows. "
                       f"Alarm at {A.THRESHOLD_SIGMA:.1f}σ."
                       + ("" if base.n_clips >= 3 else f" Send {3 - base.n_clips} more for a steadier baseline.")}


def cmd_score(a) -> dict:
    base = _load(a.asset)
    if base is None:
        return {"ok": False, "asset": a.asset, "reason": f"{a.asset} has no baseline. Enrol it first."}
    clip = _clip(a.file)
    if not clip.quality.ok:
        return {"ok": False, "asset": a.asset, "reason": clip.quality.reason}
    v = A.score(base, clip)
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"{a.asset}-{time.strftime('%Y%m%d-%H%M%S')}.png")
    with open(png, "wb") as f:
        f.write(R.verdict_png(v, clip, base, a.label or ""))
    lo, hi = v.top_range_hz
    _event("score", asset=a.asset, sigma=round(v.sigma, 2), anomalous=v.anomalous, file=os.path.basename(a.file))
    return {"ok": True, "asset": a.asset, "sigma": round(v.sigma, 2), "anomalous": v.anomalous,
            "threshold_sigma": A.THRESHOLD_SIGMA, "band_hz": [round(lo), round(hi)],
            "peak_hz": round(v.peak_hz), "peak_db": round(v.peak_db, 1), "hint": v.hint,
            "contributions": {f"{round(x)}-{round(y)} Hz": round(c, 2)
                              for (x, y), c in zip(A.group_ranges_hz(), v.contributions)},
            "summary": ("🔴 " if v.anomalous else "🟢 ") + v.line(), "png": png}


def cmd_machines(a) -> dict:
    out = []
    if os.path.isdir(BASE_DIR):
        for name in sorted(os.listdir(BASE_DIR)):
            if name.endswith(".json"):
                b = _load(name[:-5])
                out.append({"asset": b.asset, "clips": b.n_clips, "windows": b.n_windows})
    return {"ok": True, "machines": out}


def cmd_stop(a) -> dict:
    """Reaching this line means a person with authority approved it on the Switch card."""
    signal = "no controller configured; stop the machine by hand"
    hook = os.environ.get("STOP_WEBHOOK_URL")
    if hook:
        try:
            req = urllib.request.Request(hook, data=json.dumps({"asset": a.asset, "action": "stop"}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as r:
                signal = f"controller answered {r.status}"
        except Exception as e:  # noqa: BLE001
            signal = f"controller failed: {e}"
    e = _event("stop", asset=a.asset, case=a.case, requested_by=a.by, signal=signal)
    return {"ok": True, "asset": a.asset, "case": a.case, "stopped_at": e["at"], "signal": signal,
            "message": f"■ STOP recorded for {a.asset.upper()} ({a.case}) at {e['at']}. {signal}."}


def cmd_learn(a) -> dict:
    base = _load(a.asset)
    if base is None:
        return {"ok": False, "reason": f"{a.asset} has no baseline."}
    c = _clip(a.file)
    base = A.Baseline(base.asset, base.clip_embeddings + [c.E], base.band_mean, base.spec_mean,
                      ref_lm=base.ref_lm).fit()
    _save(base)
    _event("learn", asset=a.asset, file=os.path.basename(a.file))
    return {"ok": True, "asset": a.asset, "clips": base.n_clips,
            "message": f"That sound is now part of {a.asset}'s healthy baseline ({base.n_clips} clips)."}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="injini")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enrol")
    e.add_argument("asset")
    e.add_argument("files", nargs="+")
    s = sub.add_parser("score")
    s.add_argument("asset")
    s.add_argument("file")
    s.add_argument("--label", default="")
    sub.add_parser("machines")
    st = sub.add_parser("stop")
    st.add_argument("asset")
    st.add_argument("--case", default="")
    st.add_argument("--by", default="")
    le = sub.add_parser("learn")
    le.add_argument("asset")
    le.add_argument("file")
    a = p.parse_args(argv)
    a.asset = getattr(a, "asset", "").lower() if getattr(a, "asset", None) else None
    try:
        out = {"enrol": cmd_enrol, "score": cmd_score, "machines": cmd_machines, "stop": cmd_stop,
               "learn": cmd_learn}[a.cmd](a)
    except FileNotFoundError as err:
        out = {"ok": False, "reason": f"file not found: {err.filename}"}
    except Exception as err:  # noqa: BLE001
        out = {"ok": False, "reason": f"{type(err).__name__}: {err}"}
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
