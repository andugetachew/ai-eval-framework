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
    list_body = list_resp.json()
    assert list_body["total"] == 1
    assert len(list_body["items"]) == 1

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
@pytest.mark.asyncio
async def test_upload_csv(client):
    csv_content = (
        "id,input,actual_output,expected_output,variant\n"
        "1,capital of France?,Paris,Paris,prompt_v1\n"
        "2,capital of France?,London,Paris,prompt_v2\n"
    )
    files = {"file": ("test.csv", csv_content, "text/csv")}
    data = {"name": "csv test", "scorers": "exact_match"}

    resp = await client.post("/eval-runs/upload", data=data, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    scores = {r["item_id"]: r["score"] for r in body["results"]}
    assert scores["1"] == 1.0
    assert scores["2"] == 0.0


@pytest.mark.asyncio
async def test_upload_jsonl(client):
    jsonl_content = (
        '{"id": "1", "input": "q", "actual_output": "Paris", "expected_output": "Paris"}\n'
    )
    files = {"file": ("test.jsonl", jsonl_content, "application/jsonl")}
    data = {"name": "jsonl test", "scorers": "exact_match"}

    resp = await client.post("/eval-runs/upload", data=data, files=files)
    assert resp.status_code == 200
    assert resp.json()["results"][0]["score"] == 1.0


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_file_type(client):
    files = {"file": ("test.txt", "not a valid format", "text/plain")}
    data = {"name": "bad file test", "scorers": "exact_match"}

    resp = await client.post("/eval-runs/upload", data=data, files=files)
    assert resp.status_code == 400

@pytest.mark.asyncio
async def test_create_run_rejects_too_many_items(client):
    items = [
        {"id": str(i), "input": "q", "actual_output": "a", "expected_output": "a"}
        for i in range(201)
    ]
    resp = await client.post(
        "/eval-runs", json={"name": "too many", "scorers": ["exact_match"], "items": items}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_run_rejects_too_many_items_for_llm_scorer(client):
    items = [
        {"id": str(i), "input": "q", "actual_output": "a", "expected_output": "a"}
        for i in range(26)
    ]
    resp = await client.post(
        "/eval-runs", json={"name": "too many for llm", "scorers": ["llm_judge"], "items": items}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_pagination_limit_and_offset(client):
    for i in range(3):
        await client.post(
            "/eval-runs",
            json={
                "name": f"run {i}",
                "scorers": ["exact_match"],
                "items": [{"id": "1", "input": "q", "actual_output": "a", "expected_output": "a"}],
            },
        )

    resp = await client.get("/eval-runs?limit=2&offset=0")
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2

    resp2 = await client.get("/eval-runs?limit=2&offset=2")
    assert len(resp2.json()["items"]) == 1


@pytest.mark.asyncio
async def test_upload_missing_required_field_returns_400(client):
    csv_content = "id,input\n1,capital of France?\n"  # missing actual_output
    files = {"file": ("bad.csv", csv_content, "text/csv")}
    data = {"name": "bad csv", "scorers": "exact_match"}

    resp = await client.post("/eval-runs/upload", data=data, files=files)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_scorer_failure_surfaces_as_result_not_dropped(client):
    # semantic_similarity requires expected_output; omitting it should
    # surface as a failed ScoreResult rather than vanish silently
    payload = {
        "name": "failing scorer test",
        "scorers": ["semantic_similarity"],
        "items": [{"id": "1", "input": "q", "actual_output": "Paris"}],
    }
    resp = await client.post("/eval-runs", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["raw"] if "raw" in results[0] else True  # raw not in schema, check reasoning instead
    assert "error" in results[0]["reasoning"].lower() or "requires" in results[0]["reasoning"].lower()