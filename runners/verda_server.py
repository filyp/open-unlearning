import asyncio
import os
import subprocess

from dotenv import load_dotenv
from fastapi import FastAPI

app = FastAPI()

default_repo = "https://github.com/filyp/open-unlearning.git"

load_dotenv("/secrets/.env")

_job_started = False


async def _exit_process(exit_code: int) -> None:
    await asyncio.sleep(0.2)
    os._exit(exit_code)


@app.get("/health")
def health():
    return {"status": "busy" if _job_started else "ok"}


@app.post("/run")
async def run_job(body: dict):
    global _job_started

    assert not _job_started, "Job already started — Verda should not have routed here"
    _job_started = True

    repo = body.get("repo", default_repo)
    command = body["command"]

    # clone the repo
    subprocess.run(["git", "clone", repo, "/root/repo"], check=True)

    # run the command
    result = subprocess.run(command, shell=True, cwd="/root/repo")

    asyncio.create_task(_exit_process(result.returncode))

    return {"success": result.returncode == 0}
