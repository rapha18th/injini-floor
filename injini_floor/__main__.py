"""python -m injini_floor register | run"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .switch_client import Switch, register

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(ROOT, ".env")


def load_env() -> None:
    if not os.path.exists(ENV):
        return
    for line in open(ENV, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def write_env(updates: dict[str, str]) -> None:
    lines = open(ENV, encoding="utf-8").read().splitlines() if os.path.exists(ENV) else []
    keys = set(updates)
    out = [ln for ln in lines if ln.split("=", 1)[0].strip() not in keys]
    out += [f"{k}={v}" for k, v in updates.items()]
    with open(ENV, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def need(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"{name} is not set. Put it in {ENV} (see .env.example).")
    return v


async def cmd_register(args: argparse.Namespace) -> None:
    url = need("SWITCH_URL")
    token = args.token or need("SWITCH_REGISTRATION_TOKEN")
    name = os.environ.get("INJINI_NAME", "injini")
    r = await register(url, token, name,
                       "Listens to machines. Scores voice notes of machines against their healthy baseline, "
                       "opens maintenance cases and asks the supervisor before anything stops.",
                       "Injini", overwrite=args.overwrite)
    write_env({"INJINI_AGENT_ID": r["id"], "INJINI_API_KEY": r["api_key"]})
    print(f"Registered {name} ({r['id']}). Its key is saved in .env.")


async def cmd_run(args: argparse.Namespace) -> None:
    from .agent import Injini

    sw = Switch(need("SWITCH_URL"), need("INJINI_AGENT_ID"), need("INJINI_API_KEY"))
    r = await sw.http.get("/health")
    print(f"Switch at {sw.base}: health {r.status_code}")
    agent = Injini(sw, os.environ.get("INJINI_DATA", os.path.join(ROOT, "data")),
                   foreman=os.environ.get("FOREMAN_NAME", "foreman"),
                   stop_webhook=os.environ.get("STOP_WEBHOOK_URL") or None)
    print("Injini is listening.")
    await agent.run()


def main() -> None:
    load_env()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(name)s %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(prog="injini_floor")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="register Injini with a Switch server")
    r.add_argument("--token", help="a registration key from the Switch Gateway")
    r.add_argument("--overwrite", action="store_true")
    sub.add_parser("run", help="run Injini")
    a = p.parse_args()
    asyncio.run({"register": cmd_register, "run": cmd_run}[a.cmd](a))


if __name__ == "__main__":
    main()
