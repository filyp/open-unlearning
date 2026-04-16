import asyncio
import os
import subprocess
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

default_repo = "https://github.com/filyp/open-unlearning.git"

load_dotenv("/secrets/.env")

_job_started = False
_start_time = time.monotonic()
_NO_JOB_TIMEOUT = 180  # seconds


async def _exit_process(exit_code: int) -> None:
    await asyncio.sleep(0.2)
    os._exit(exit_code)


@app.get("/health")
def health():
    if _job_started:
        return JSONResponse(status_code=200, content={"status": "busy"})
    if time.monotonic() - _start_time > _NO_JOB_TIMEOUT:
        print("No job received within timeout, exiting.")
        os._exit(1)
    return JSONResponse(status_code=200, content={"status": "healthy"})


@app.post("/run")
async def run_job(body: dict):
    global _job_started

    assert not _job_started, "Job already started — Verda should not have routed here"
    _job_started = True

    repo = body.get("repo", default_repo)
    command = body["command"]

    loop = asyncio.get_event_loop()

    # clone the repo
    await loop.run_in_executor(
        None, lambda: subprocess.run(["git", "clone", repo, "/root/repo"], check=True)
    )

    # run the command
    result = await loop.run_in_executor(
        None, lambda: subprocess.run(command, shell=True, cwd="/root/repo")
    )

    asyncio.create_task(_exit_process(result.returncode))

    return {"success": result.returncode == 0}
