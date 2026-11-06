"""Send recordings into a Switch room the way a fitter's phone would, for anyone without a machine.

Registers (once) a small agent called floor-mic, then uploads each clip into the room with its
message text. Injini hears them exactly as it hears a phone voice note.

    python -m injini_floor.replay --room <plant room id> "enrol fan-03" samples/fan-03-healthy-1.ogg ...
"""
from __future__ import annotations

import argparse
import asyncio
import os

from .__main__ import load_env, need, write_env
from .switch_client import Switch, register


async def main() -> None:
    load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--room", default=os.environ.get("PLANT_ROOM_ID"), help="Switch room id of the plant floor")
    p.add_argument("--gap", type=float, default=6.0, help="seconds between clips")
    p.add_argument("items", nargs="+", help='alternating "message text" and audio path')
    a = p.parse_args()
    url = need("SWITCH_URL")
    if not os.environ.get("MIC_AGENT_ID"):
        r = await register(url, need("SWITCH_REGISTRATION_TOKEN"), "floor-mic",
                           "Replays machine recordings into a room, like a fitter's phone.", "Floor mic")
        write_env({"MIC_AGENT_ID": r["id"], "MIC_API_KEY": r["api_key"]})
        os.environ.update(MIC_AGENT_ID=r["id"], MIC_API_KEY=r["api_key"])
        print("Registered floor-mic. Add it to the plant room in Switch Console, then run this again.")
        return
    sw = Switch(url, os.environ["MIC_AGENT_ID"], os.environ["MIC_API_KEY"])
    it = iter(a.items)
    for text, path in zip(it, it):
        with open(path, "rb") as f:
            await sw.upload(a.room, os.path.basename(path), f.read(), "audio/ogg", caption=text)
        print(f"sent {path}  ({text})")
        await asyncio.sleep(a.gap)
    await sw.close()


if __name__ == "__main__":
    asyncio.run(main())
