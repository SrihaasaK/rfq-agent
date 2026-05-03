"""Streamlit UI for the RFQ-to-Quote Agent."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from agent.agent import process_rfq
from agent.models import Route
from evals.eval import run_eval, print_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

SAMPLES_DIR = Path(__file__).parent / "samples"
CUSTOMERS_PATH = Path(__file__).parent / "customers" / "customer_history.json"

st.set_page_config(page_title="RFQ Agent — Paragon Demo", layout="wide")
st.title("RFQ-to-Quote Agent")
st.caption("Hypothesis-driven disambiguation for industrial brass fittings")


@st.cache_data
def load_customers() -> list[dict]:
    with open(CUSTOMERS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_sample_emails() -> dict[str, str]:
    emails = {}
    for p in sorted(SAMPLES_DIR.glob("*.txt")):
        emails[p.name] = p.read_text()
    return emails


customers = load_customers()
sample_emails = load_sample_emails()

# --- Input pane ---
col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("Input")

    input_mode = st.radio("Email source", ["Sample email", "Paste your own"], horizontal=True)

    if input_mode == "Sample email":
        selected_file = st.selectbox("Select sample", list(sample_emails.keys()))
        email_text = sample_emails[selected_file]
        st.text_area("Email content", email_text, height=200, disabled=True)
    else:
        email_text = st.text_area("Paste RFQ email", height=200, placeholder="Paste an RFQ email here...")

    customer_names = ["(none — unknown customer)"] + [f"{c['name']} — {c['company']}" for c in customers]
    customer_selection = st.selectbox("Customer", customer_names)

    if customer_selection.startswith("(none"):
        customer_name = None
    else:
        customer_name = customer_selection.split(" — ")[0]

    run_button = st.button("Process RFQ", type="primary", use_container_width=True)

# --- Output pane ---
with col_output:
    st.subheader("Result")

    if run_button and email_text.strip():
        with st.spinner("Processing RFQ (2 LLM calls)..."):
            try:
                result = process_rfq(email_text, customer_name)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # Route badge
        route_colors = {Route.QUOTE: "green", Route.CLARIFY: "orange", Route.HUMAN: "red"}
        route_labels = {Route.QUOTE: "QUOTE", Route.CLARIFY: "CLARIFY", Route.HUMAN: "HUMAN ESCALATION"}
        st.markdown(f"### Route: :{route_colors[result.route]}[{route_labels[result.route]}]")
        st.markdown(f"*{result.route_reasoning}*")

        # Extracted intent
        with st.expander("Extracted Intent", expanded=True):
            intent = result.extracted_intent
            cols = st.columns(3)
            cols[0].metric("Product Type", intent.product_type or "Unknown")
            cols[1].metric("Quantity", intent.quantity or "Unknown")
            cols[2].metric("Urgency", intent.urgency or "Standard")

            if intent.specs_mentioned:
                st.markdown("**Specs mentioned:**")
                st.json(intent.specs_mentioned)
            if intent.specs_missing:
                st.markdown(f"**Specs missing:** {', '.join(intent.specs_missing)}")
            if intent.customer_signals:
                st.markdown(f"**Customer signals:** {', '.join(intent.customer_signals)}")
            if intent.referenced_past_orders:
                st.markdown("**References past orders:** Yes")

        # Hypotheses
        with st.expander("Ranked Hypotheses", expanded=True):
            for h in result.hypotheses:
                conf_color = "green" if h.confidence > 0.85 else ("orange" if h.confidence >= 0.5 else "red")
                st.markdown(
                    f"**#{h.rank}** `{h.sku}` — {h.description}  \n"
                    f"Confidence: :{conf_color}[{h.confidence:.0%}]  \n"
                    f"_{h.reasoning}_"
                )
                st.divider()

        # Route-specific output
        if result.route == Route.QUOTE and result.quote:
            st.subheader("Quote Draft")
            q = result.quote
            st.markdown(f"**SKU:** `{q.sku}` — {q.description}")
            st.markdown(f"**Quantity:** {q.quantity}")
            st.markdown(f"**Unit Price:** ${q.unit_price_usd:.2f}")
            st.markdown(f"**Total:** ${q.total_price_usd:.2f}")

        elif result.route == Route.CLARIFY and result.clarification:
            st.subheader("Clarifying Question")
            st.info(result.clarification.question)
            st.markdown(f"**Disambiguating specs:** {', '.join(result.clarification.disambiguating_specs)}")

        elif result.route == Route.HUMAN and result.escalation:
            st.subheader("Human Escalation")
            st.warning(result.escalation.reasoning)
            st.markdown(f"**Issues:** {', '.join(result.escalation.issues)}")

    elif run_button:
        st.warning("Please enter or select an email to process.")

# --- Eval Suite ---
st.divider()
st.subheader("Eval Suite")

if st.button("Run Full Eval (20 test cases)", use_container_width=True):
    progress_bar = st.progress(0, text="Starting eval...")

    def update_progress(i: int, total: int, tc_id: str) -> None:
        progress_bar.progress((i + 1) / total, text=f"Running {tc_id} ({i + 1}/{total})")

    with st.spinner("Running eval suite..."):
        report = run_eval(progress_callback=update_progress)

    progress_bar.empty()

    s = report["summary"]
    cols = st.columns(4)
    cols[0].metric("Route Accuracy", f"{s['route_accuracy']:.0%}")
    cols[1].metric("SKU Match Rate", f"{s['sku_match_rate']:.0%}")
    cols[2].metric("Question Quality", f"{s['question_quality_avg']:.0%}")
    cols[3].metric("Human Escalation", f"{s['human_escalation_rate']:.0%}")

    # Show failures
    failures = [r for r in report["results"] if r.get("status") == "error" or not r.get("route_correct", True)]
    if failures:
        st.subheader(f"Failures ({len(failures)})")
        for f in failures:
            if f.get("status") == "error":
                st.error(f"**{f['id']}**: {f['error']}")
            else:
                fa = f.get("failure_analysis", {})
                st.warning(
                    f"**{f['id']}**: expected `{fa.get('expected')}`, "
                    f"got `{fa.get('got')}`, "
                    f"top confidence = {fa.get('top_confidence', 'N/A')}"
                )
    else:
        st.success("All test cases passed!")

    with st.expander("Full Report JSON"):
        st.json(report)
