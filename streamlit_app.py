"""Small retail demo: all recommendation computation stays in FastAPI."""
from html import escape
import json
import os
from pathlib import Path
import sys
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retail_recs.demo_client import DemoError, fetch_recommendations


st.set_page_config(page_title="Basket | Retail recommendations", page_icon="🛒", layout="centered")
st.markdown("""
<style>
.stApp { background: #f7f8f4; }
.block-container { max-width: 880px; padding-top: 3rem; }
h1, h2, h3 { color: #183b32; }
.eyebrow { color: #47705c; font-size: .78rem; font-weight: 700; letter-spacing: .15em; }
.product-card { display: flex; align-items: center; gap: 20px; background: white;
  border: 1px solid #e1e7dd; border-radius: 14px; padding: 20px 24px; margin: 10px 0; }
.rank { color: #276749; background: #edf4e9; border-radius: 10px; min-width: 42px;
  height: 42px; display: flex; align-items: center; justify-content: center; font-weight: 700; }
.product-title { color: #19392d; font-size: 1.05rem; font-weight: 650; margin: 4px 0; overflow-wrap: anywhere; }
.product-category { color: #617067; font-size: .82rem; overflow-wrap: anywhere; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">PERSONALIZED RETAIL · PORTFOLIO DEMO</div>', unsafe_allow_html=True)
st.title("A little help with your next basket")
st.write("Five products ranked from household purchase patterns.")
st.caption("Recommendations use historical purchase patterns at snapshot cutoff day 701. "
           "This is a historical offline demo, not live personalized shopping data.")
mapping_path = Path(__file__).resolve().parent / "configs" / "demo_households.json"
mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

household_options = {
    item["alias"]: item["household_id"]
    for item in mapping["households"]
}


def clear_recommendations():
    st.session_state.pop("recommendation_result", None)
    st.session_state.pop("recommendation_selection", None)


# Selection widgets stay outside a form so changes immediately clear old results.
selection_mode = st.radio(
    "Household selection",
    ["Featured Households", "All Households"],
    horizontal=True,
    key="household_selection_mode",
    on_change=clear_recommendations,
)
if selection_mode == "Featured Households":
    selected_alias = st.selectbox(
        "Choose a household",
        options=list(household_options),
        key="featured_household_alias",
        on_change=clear_recommendations,
    )
    household_id = household_options[selected_alias]
    st.caption("Household names are fictional demo aliases.")
else:
    household_id = st.number_input(
        "Household ID", value=1, step=1, key="all_household_id",
        help="Enter a positive integer ID. The API checks whether it is available in the historical snapshot.",
        on_change=clear_recommendations,
    )

selection = (selection_mode, int(household_id))
if st.session_state.get("recommendation_selection") != selection:
    clear_recommendations()
submitted = st.button("Get Recommendations", type="primary", width="stretch")

if submitted:
    clear_recommendations()
    try:
        with st.spinner("Finding your recommendations…"):
            result = fetch_recommendations(os.environ.get("RETAIL_API_URL", "http://127.0.0.1:8000"), int(household_id))
        st.session_state.recommendation_result = result
        st.session_state.recommendation_selection = selection
    except DemoError as exc:
        st.error(str(exc))

result = st.session_state.get("recommendation_result")
if result:
    result_label = f"Household {result['household_id']}"
    if selection_mode == "Featured Households":
        result_label = next(alias for alias, key in household_options.items() if key == result["household_id"])
    st.subheader(f"Recommended for {result_label}")
    st.caption("Shown in recommendation order. Labels come from the dataset's product metadata; "
               "commercial product names and prices are not available.")
    for rank, product in enumerate(result["recommendations"], start=1):
        title = escape(product.get("description") or "Description unavailable")
        category = escape(product.get("category") or "Category unavailable")
        department = product.get("department")
        context = f"{escape(department)} · {category}" if department else category
        st.markdown(f'<div class="product-card"><div class="rank">{rank:02d}</div>'
                    f'<div><div class="product-category">{context}</div>'
                    f'<div class="product-title">{title}</div></div></div>', unsafe_allow_html=True)
    with st.expander("Technical details · model & ranking"):
        st.write(f"Model version: {result['model_version']}")
        st.write(f"Snapshot cutoff day: {result['snapshot_cutoff']}")
        st.caption("Raw ranking scores determine order. Scores are not purchase probabilities.")
        st.json({"household_id": result["household_id"], "model_version": result["model_version"],
                 "snapshot_cutoff": result["snapshot_cutoff"],
                 "ranking": [{"rank": rank, "product_id": p["product_id"], "raw_ranking_score": p["score"]}
                             for rank, p in enumerate(result["recommendations"], start=1)]})
else:
    st.caption("Choose a household and get five recommendations to explore.")
