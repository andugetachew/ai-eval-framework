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

@pytest.mark.asyncio
async def test_variant_comparison(client):
    payload = {
        "name": "variant test",
        "scorers": ["exact_match", "semantic_similarity"],
        "items": [
            {
                "id": "1", "input": "capital of France?",
                "actual_output": "Paris", "expected_output": "Paris",
                "variant": "prompt_v1",
            },
            {
                "id": "2", "input": "capital of France?",
                "actual_output": "The capital city is Paris, France.",
                "expected_output": "Paris", "variant": "prompt_v2",
            },
        ],
    }
    create_resp = await client.post("/eval-runs", json=payload)
    assert create_resp.status_code == 200
    body = create_resp.json()
    variants = {r["variant"] for r in body["results"]}
    assert variants == {"prompt_v1", "prompt_v2"}

    run_id = body["id"]
    compare_resp = await client.get(f"/eval-runs/{run_id}/compare")
    assert compare_resp.status_code == 200
    summary = compare_resp.json()["summary"]

    v1_exact = next(
        s for s in summary if s["variant"] == "prompt_v1" and s["scorer_name"] == "exact_match"
    )
    v2_exact = next(
        s for s in summary if s["variant"] == "prompt_v2" and s["scorer_name"] == "exact_match"
    )
    assert v1_exact["avg_score"] == 1.0
    assert v2_exact["avg_score"] == 0.0


@pytest.mark.asyncio
async def test_compare_nonexistent_run_returns_404(client):
    resp = await client.get("/eval-runs/does-not-exist/compare")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_item_without_variant_defaults_correctly(client):
    payload = {
        "name": "no variant test",
        "scorers": ["exact_match"],
        "items": [
            {"id": "1", "input": "q", "actual_output": "Paris", "expected_output": "Paris"}
        ],
    }
    create_resp = await client.post("/eval-runs", json=payload)
    assert create_resp.json()["results"][0]["variant"] is None

    run_id = create_resp.json()["id"]
    compare_resp = await client.get(f"/eval-runs/{run_id}/compare")
    summary = compare_resp.json()["summary"]
    assert summary[0]["variant"] == "default"
