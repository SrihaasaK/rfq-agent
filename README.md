# RFQ-to-Quote Agent

A hypothesis-driven disambiguation agent for industrial brass fittings RFQs. Built as a technical demonstration for Paragon.

**[Live Demo](https://srihaasak-rfq-agent-app-ezjiba.streamlit.app/)**

## The Problem

> "A contractor sends a request for a brass fitting with several important details left out. How do you know what SKU they mean, based on their history with you, their end market, etc.?"

Industrial distributors receive RFQ emails that are ambiguous by nature. A request for "200 half-inch brass elbows, same as last time" could map to 4 different SKUs. Getting it wrong means a returned shipment. The agent must either resolve the ambiguity confidently or ask the *right* clarifying question.

## Approach: Two-LLM Hypothesis Pipeline

Think of it like a distributor's front desk. The **intake clerk** (Llama 3.1 8B, fast/free) parses the email into structured data. The **senior rep** (Llama 4 Scout, stronger reasoning) looks at that data alongside customer history and the catalog, then generates 2-3 ranked SKU hypotheses with explicit confidence scores.

The routing decision flows from confidence:
- **>= 0.80** — Generate a quote draft
- **0.65 - 0.80** — Ask a targeted clarifying question that disambiguates the top 2 hypotheses
- **< 0.65** — Escalate to human with structured reasoning

Three rule-based guards override LLM confidence when the email text contains unambiguous signals: explicit uncertainty language ("not sure", "probably") forces clarification, out-of-catalog indicators (metric sizes, stainless) force human escalation, and close-confidence hypotheses with missing critical specs force clarification rather than guessing.

## Eval Results

Measured on 20 test cases (15 sample emails + 5 edge cases):

| Metric | v1 (baseline) | v2 (calibrated) | What changed |
|--------|---------------|------------------|-------------|
| Route accuracy | 70% (14/20) | **95% (19/20)** | Few-shot calibration + rule-based guards |
| SKU match rate | 83% (10/12) | **83% (10/12)** | Unchanged — extraction was already solid |
| Question quality | 0% (no clarify routes) | **75%** | Ambiguous cases now correctly route to clarify |
| Human escalation | 75% (3/4) | **100% (4/4)** | Out-of-catalog guard catches metric/custom |

## What I Learned About Llama 4 Scout

The first eval run revealed **bimodal confidence distribution**: the model outputs either 0.90 or 0.60, with nothing in between. This collapsed the clarify band — ambiguous cases jumped straight to quote because customer history inflated confidence to 0.90 even when critical specs were missing.

Three fixes, in order of leverage:
1. **Few-shot calibration examples** in the hypothesis prompt showing what 0.72, 0.82, and 0.88 confidence look like. The model calibrated to these — ambiguous-with-history cases dropped from 0.90 to 0.82.
2. **Rule-based ambiguity guard**: if the email contains "not sure", "or" between sizes, "probably", or "can you advise", force a clarify route regardless of model confidence. This caught 2 cases the model still missed.
3. **Close-hypotheses guard**: when the top 2 SKU candidates are within 0.07 of each other and a critical spec (size, thread standard, material grade) is missing, the agent can't pick one — ask instead of guess.

The remaining failure (tc_08) is debatable: Sarah Chen's marine history strongly implies C46400 naval brass for a saltwater application. The model quotes confidently; the test expects clarification. In production, this is the kind of judgment call that benefits from a configurable risk threshold.

## Running It

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your Groq API key (free at console.groq.com)

streamlit run app.py   # Interactive demo
python -m evals.eval   # Run eval suite
```

## What I'd Build Next

- **Multi-turn clarification** — receive the answer, re-run hypothesis generation with the new constraint
- **Confidence calibration** — track predicted vs. actual correctness, apply Platt scaling over time
- **Catalog ingestion pipeline** — parse real distributor catalogs (CSV/PDF) into structured format
- **Customer history sync** — pull from ERP/CRM instead of static JSON
- **Learning loops** — when a rep overrides the agent's pick, feed that correction back

---

Built as a Paragon FDE application project. The code prioritizes eval rigor and architectural clarity over feature breadth.
