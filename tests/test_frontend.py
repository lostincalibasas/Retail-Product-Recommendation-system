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
FEATURED = [
    ("Demo Household — Cedar", 1),
    ("Demo Household — Maple", 2),
    ("Demo Household — Willow", 3),
    ("Demo Household — Oak", 4),
    ("Demo Household — Birch", 5),
]


def cards(app):
    return [m.value for m in app.markdown if '<div class="product-card">' in m.value]


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
    (HTTPError("url", 422, "invalid", {}, None), "could not complete"),
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
    assert app.radio[0].value == "Featured Households"
    assert list(app.selectbox[0].options) == [name for name, _ in FEATURED]
    assert not app.number_input
    app.button[0].click().run()
    assert not app.exception and calls == [("http://127.0.0.1:8000", 1)]
    rendered = cards(app)
    assert len(rendered) == 5
    assert all(f"Fixture description {i}" in rendered[i-1] for i in range(1, 6))
    assert app.expander[0].label == "Technical details · model & ranking"
    assert any("not purchase probabilities" in c.value for c in app.caption)
    app.radio[0].set_value("All Households").run()
    assert not cards(app)
    app.number_input[0].set_value(99999).run()
    app.button[0].click().run()
    assert not app.exception and "not available" in app.error[0].value
    assert not any('<div class="product-card">' in m.value for m in app.markdown)


@pytest.mark.parametrize("alias,household_id", FEATURED)
def test_each_featured_alias_requests_its_unchanged_id(monkeypatch, payload, alias, household_id):
    calls = []
    def fetch(url, household):
        calls.append(household)
        result = deepcopy(payload)
        result["household_id"] = household
        return result
    monkeypatch.setattr(demo_client, "fetch_recommendations", fetch)
    app = AppTest.from_file(str(APP)).run()
    app.selectbox[0].set_value(alias).run()
    assert not calls
    app.button[0].click().run()
    assert not app.exception and calls == [household_id]
    assert app.subheader[0].value == f"Recommended for {alias}"
    assert len(cards(app)) == 5
    assert app.session_state["recommendation_result"]["household_id"] == household_id
    assert any("fictional demo aliases" in c.value for c in app.caption)


@pytest.mark.parametrize("household_id", [6, 4_200_017])
def test_all_households_requests_ids_without_a_population_size_cap(monkeypatch, payload, household_id):
    # The fake service accepts a sparse ID to verify the UI does not assume 1..2500.
    calls = []
    def fetch(url, household):
        calls.append(household)
        result = deepcopy(payload)
        result["household_id"] = household
        return result
    monkeypatch.setattr(demo_client, "fetch_recommendations", fetch)
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value("All Households").run()
    assert not app.selectbox
    app.number_input[0].set_value(household_id).run()
    app.button[0].click().run()
    assert not app.exception and calls == [household_id]
    assert app.subheader[0].value == f"Recommended for Household {household_id}"


@pytest.mark.parametrize("household_id,message", [(99999, "not available"), (0, "could not complete"), (-1, "could not complete")])
def test_all_households_passes_unknown_and_invalid_ids_to_service(monkeypatch, payload, household_id, message):
    requested = []
    def get(url, timeout):
        household = int(url.rsplit("/", 1)[1])
        requested.append(household)
        if household == household_id:
            raise HTTPError(url, 404 if household > 0 else 422, "unavailable", {}, None)
        result = deepcopy(payload)
        result["household_id"] = household
        return BytesIO(json.dumps(result).encode())
    monkeypatch.setattr(demo_client, "urlopen", get)
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value("All Households").run()
    app.number_input[0].set_value(6).run()
    app.button[0].click().run()
    assert len(cards(app)) == 5
    app.number_input[0].set_value(household_id).run()
    assert not cards(app)
    app.button[0].click().run()
    assert not app.exception and message in app.error[0].value
    assert requested == [6, household_id]
    assert not cards(app) and not app.expander


def test_household_and_mode_changes_clear_results_before_next_request(monkeypatch, payload):
    calls = []
    def fetch(url, household):
        calls.append(household)
        result = deepcopy(payload)
        result["household_id"] = household
        return result
    monkeypatch.setattr(demo_client, "fetch_recommendations", fetch)
    app = AppTest.from_file(str(APP)).run()
    app.button[0].click().run()
    assert len(cards(app)) == 5 and calls == [1]
    app.selectbox[0].set_value(FEATURED[1][0]).run()
    assert not cards(app) and not app.expander and calls == [1]
    app.button[0].click().run()
    assert len(cards(app)) == 5 and calls == [1, 2]
    app.radio[0].set_value("All Households").run()
    assert not cards(app) and calls == [1, 2]
    app.button[0].click().run()
    assert len(cards(app)) == 5 and calls == [1, 2, 1]
    app.radio[0].set_value("Featured Households").run()
    # Same numeric ID, different mode: results must still be cleared.
    assert app.selectbox[0].value == FEATURED[0][0]
    assert not cards(app) and not app.expander and calls == [1, 2, 1]


@pytest.mark.parametrize("mode", ["Featured Households", "All Households"])
def test_both_modes_preserve_backend_order_scores_cards_and_details(monkeypatch, payload, mode):
    payload["recommendations"] = [payload["recommendations"][i] for i in [2, 0, 4, 1, 3]]
    monkeypatch.setattr(demo_client, "fetch_recommendations", lambda *args: deepcopy(payload))
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value(mode).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["recommendation_result"] == payload
    rendered = cards(app)
    for card, product in zip(rendered, payload["recommendations"]):
        assert product["description"] in card
        assert product["category"] in card and product["department"] in card
    details = json.loads(app.json[0].value)
    assert details["household_id"] == 1 and details["snapshot_cutoff"] == 701
    assert details["model_version"] == payload["model_version"]
    assert details["ranking"] == [
        {"rank": rank, "product_id": p["product_id"], "raw_ranking_score": p["score"]}
        for rank, p in enumerate(payload["recommendations"], 1)]


def test_old_session_results_without_selection_context_are_cleared(monkeypatch, payload):
    monkeypatch.setattr(demo_client, "fetch_recommendations", lambda *args: deepcopy(payload))
    app = AppTest.from_file(str(APP))
    app.session_state["recommendation_result"] = deepcopy(payload)
    app.run()
    assert not app.exception and not cards(app)


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
