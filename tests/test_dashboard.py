"""Backend endpoints and the episode recorder."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient      # noqa: E402

from dashboard.api.main import app             # noqa: E402

client = TestClient(app)


def test_lists():
    sc = client.get("/api/scenarios").json()
    assert sc["SYM"]["bilateral"] and not sc["RURAL"]["bilateral"]
    me = client.get("/api/methods").json()
    assert "HardballBuyer" in me["scripted_buyer"] and "HardballBuyer" not in me["scripted_farmer"]


def test_episode_record_has_reference_lines_and_events():
    r = client.post("/api/episodes", json={"scenario": "PERISH", "farmer": "Conceder", "buyer": "HardballBuyer",
                                           "seed": 50001, "mechanism": {"kind": "fee", "tau": 0.4, "lam": 1.0}})
    assert r.status_code == 200
    ep = r.json()
    assert {"nash", "target", "farmer_fallback", "buyer_walkaway"} <= set(ep["lines"])
    assert ep["events"][0]["kind"] == "reset" and len(ep["lines"]["nash"]) == ep["T"] + 1
    assert ep["lines"]["target"][0] < ep["lines"]["nash"][0]            # tau 0.4 sits below the 0.5 split
    again = client.post("/api/episodes", json={"scenario": "PERISH", "farmer": "Conceder", "buyer": "HardballBuyer",
                                               "seed": 50001, "mechanism": {"kind": "fee", "tau": 0.4, "lam": 1.0}}).json()
    assert again["deals"] == ep["deals"]                                 # same seed, same episode


def test_bad_requests_are_reported():
    assert client.post("/api/episodes", json={"scenario": "NOPE"}).status_code == 404
    assert client.post("/api/episodes", json={"scenario": "SYM", "farmer": "NoSuchAgent"}).status_code == 400


def test_stream_sends_steps_then_done():
    with client.websocket_connect("/api/episodes/stream") as ws:
        ws.send_text('{"scenario": "SYM", "farmer": "Conceder", "buyer": "Conceder", "seed": 50002}')
        kinds = []
        while True:
            m = ws.receive_json()
            kinds.append(m["type"])
            if m["type"] in ("done", "error"):
                break
    assert kinds[0] == "meta" and kinds[-1] == "done" and "step" in kinds
