"""Dashboard backend. Serves the static front end and runs episodes on request.

    uvicorn dashboard.api.main:app --reload        # then open http://127.0.0.1:8000

Endpoints
    GET  /api/scenarios                    named scenarios and their parameters
    GET  /api/methods                      scripted agents and trained checkpoints found under results/
    GET  /api/results                      the aggregated results the front end charts (web/data.json)
    POST /api/episodes                     run one episode, return the full record
    WS   /api/episodes/stream              same request as JSON; events are sent one step at a time
It loads checkpoints on demand and never trains anything.
"""
import asyncio
import json
import pathlib

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agrinegotiate import scenarios
from agrinegotiate.eval import replay

ROOT = pathlib.Path(__file__).resolve().parents[2]
WEB = ROOT / "dashboard" / "web"
app = FastAPI(title="AgriNegotiate dashboard")


class EpisodeRequest(BaseModel):
    scenario: str = "PERISH"
    farmer: str = "Linear"
    buyer: str = "HardballBuyer"
    seed: int = 50000
    mechanism: dict | None = None
    overrides: dict | None = None          # e.g. {"duty": 0.6, "rho": 0.97}


def _bilateral(name: str) -> bool:
    sc = scenarios.get(name)
    return sc.n_farmers == 1 and sc.n_buyers == 1


@app.get("/api/scenarios")
def get_scenarios():
    return {k: dict(v.to_dict(), bilateral=_bilateral(k)) for k, v in scenarios.SCENARIOS.items()}


@app.get("/api/methods")
def get_methods():
    runs = sorted(str(p.parent.relative_to(ROOT / "results")) for p in (ROOT / "results").glob("*/*/seed_*/checkpoint.pt"))
    return {"scripted_farmer": replay.SCRIPTED, "scripted_buyer": replay.SCRIPTED + ["HardballBuyer"],
            "trained": ["run:" + r for r in runs]}


@app.get("/api/results")
def get_results():
    f = WEB / "data.json"
    if not f.exists():
        raise HTTPException(404, "run `python dashboard/export_data.py` first")
    return JSONResponse(json.loads(f.read_text()))


def _run(req: EpisodeRequest) -> dict:
    if req.scenario not in scenarios.SCENARIOS:
        raise HTTPException(404, f"unknown scenario {req.scenario}")
    try:
        return replay.record(req.scenario, req.farmer, req.buyer, req.seed, req.mechanism, req.overrides)
    except (KeyError, FileNotFoundError, TypeError) as e:
        raise HTTPException(400, f"cannot run that episode: {e}")


@app.post("/api/episodes")
def post_episode(req: EpisodeRequest):
    return _run(req)


@app.websocket("/api/episodes/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        req = EpisodeRequest(**json.loads(await ws.receive_text()))
        ep = _run(req)
    except Exception as e:                                   # report the problem to the client and stop
        await ws.send_json({"type": "error", "detail": str(e)})
        await ws.close()
        return
    events = ep.pop("events")
    await ws.send_json({"type": "meta", "episode": ep})
    for t in range(ep["T"] + 1):
        step = [e for e in events if e["t"] == t]
        if step:
            await ws.send_json({"type": "step", "t": t, "events": step})
            await asyncio.sleep(0.15)
    await ws.send_json({"type": "done"})
    await ws.close()


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
