import pytest


@pytest.mark.asyncio
async def test_create_and_list_eval_run(client):
    payload = {
        "name": "api test run",
        "scorers": ["exact_match"],
        "items": [
            {"id": "1", "input": "q", "actual_output": "Paris", "expected_output": "Paris"}
        ],
    }
    create_resp = await client.post("/eval-runs", json=payload)
    assert create_resp.status_code == 200
    body = create_resp.json()
    assert body["status"] == "completed"
    assert body["results"][0]["score"] == 1.0

    list_resp = await client.get("/eval-runs")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    run_id = body["id"]
    get_resp = await client.get(f"/eval-runs/{run_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == run_id


@pytest.mark.asyncio
async def test_get_nonexistent_run_returns_404(client):
    resp = await client.get("/eval-runs/does-not-exist")
    assert resp.status_code == 404