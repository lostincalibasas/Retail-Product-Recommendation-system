"""User-facing flow and HTTP error handling, without loading a model in the UI."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

from retail_recs import demo_client
from retail_recs.demo_client import DemoError, fetch_recommendations


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def payload():
    # Synthetic strings are test fixtures only; production labels come from the catalog.
    return {"household_id": 1, "snapshot_cutoff": 701, "model_version": "lambdarank-v1",
            "recommendations": [{"product_id": i, "score": -i / 10,
                                  "description": f"Fixture description {i}", "category": "Fixture category",
                                  "department": "Fixture department"} for i in range(1, 6)]}


def test_http_client_preserves_order_scores_and_missing_metadata(monkeypatch, payload):
    payload["recommendations"][0].pop("description")
    calls = []
    def get(url, timeout):
        calls.append((url, timeout))
        return BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(demo_client, "urlopen", get)
    result = fetch_recommendations("http://localhost:8000/", 1)
    assert calls == [("http://localhost:8000/recommendations/1", 15)]
    assert result["recommendations"][0]["description"] is None
    assert [p["score"] for p in result["recommendations"]] == [-.1, -.2, -.3, -.4, -.5]


@pytest.mark.parametrize("error,message", [
    (HTTPError("url", 404, "missing", {}, None), "not available"),
    (HTTPError("url", 500, "failure", {}, None), "could not complete"),
    (URLError("offline"), "Cannot reach"), (TimeoutError(), "Cannot reach"),
])
def test_http_failures_are_actionable(monkeypatch, error, message):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(demo_client, "urlopen", fail)
    with pytest.raises(DemoError, match=message):
        fetch_recommendations("http://localhost:8000", 1)


@pytest.mark.parametrize("corruption", ["json", "household", "length", "duplicates", "score"])
def test_invalid_service_payload_is_handled(monkeypatch, payload, corruption):
    if corruption == "household": payload["household_id"] = 2
    elif corruption == "length": payload["recommendations"].pop()
    elif corruption == "duplicates": payload["recommendations"][1]["product_id"] = 1
    elif corruption == "score": payload["recommendations"][0]["score"] = float("nan")
    data = b"not json" if corruption == "json" else json.dumps(payload).encode()
    monkeypatch.setattr(demo_client, "urlopen", lambda *args, **kwargs: BytesIO(data))
    with pytest.raises(DemoError):
        fetch_recommendations("http://localhost:8000", 1)


def test_streamlit_form_cards_details_and_error_clear_previous_results(monkeypatch, payload):
    calls = []
    def fetch(url, household):
        calls.append((url, household))
        if household == 99999:
            raise DemoError("This household is not available in the historical snapshot.")
        return deepcopy(payload)
    monkeypatch.setattr(demo_client, "fetch_recommendations", fetch)
    monkeypatch.setenv("RETAIL_API_URL", "http://127.0.0.1:8000")
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception and not calls
    app.button[0].click().run()
    assert not app.exception and calls == [("http://127.0.0.1:8000", 1)]
    cards = [m.value for m in app.markdown if '<div class="product-card">' in m.value]
    assert len(cards) == 5
    assert all(f"Fixture description {i}" in cards[i-1] for i in range(1, 6))
    assert app.expander[0].label == "Technical details · model & ranking"
    assert any("not purchase probabilities" in c.value for c in app.caption)
    app.number_input[0].set_value(99999)
    app.button[0].click().run()
    assert not app.exception and "not available" in app.error[0].value
    assert not any('<div class="product-card">' in m.value for m in app.markdown)


def test_streamlit_fallbacks_escape_catalog_text(monkeypatch, payload):
    payload["recommendations"][0].update(description='<script>alert("x")</script>', category="A & B")
    payload["recommendations"][1].update(description=None, category=None, department=None)
    monkeypatch.setattr(demo_client, "fetch_recommendations", lambda *args: payload)
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()
    assert not app.exception
    cards = [m.value for m in app.markdown if '<div class="product-card">' in m.value]
    assert '&lt;script&gt;' in cards[0] and '<script>' not in cards[0]
    assert 'A &amp; B' in cards[0]
    assert 'Description unavailable' in cards[1] and 'Category unavailable' in cards[1]
