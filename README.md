# RFQ-to-Quote Agent

A hypothesis-driven disambiguation agent that turns ambiguous brass fittings RFQ emails into confident quotes, targeted clarifying questions, or structured human escalations.

**[Live Demo](https://srihaasak-rfq-agent-app-ezjiba.streamlit.app/)** | Run locally: `streamlit run app.py`

## Headline Result

| Metric | v1 | v2 | Delta |
|---|---|---|---|
| Route accuracy | 70% (14/20) | **95% (19/20)** | +25pp |
| Clarify cases routed correctly | 0/4 | **4/4** | fixed |
| Question quality | 0% (no questions generated) | **75%** | +75pp |
| Human escalation | 75% (3/4) | **100% (4/4)** | +25pp |
| SKU match (when quoting) | 83% (10/12) | **83% (10/12)** | unchanged |

## What v1 Revealed (The Diagnosis)

- **Bimodal confidence collapse.** Llama 4 Scout outputs either 0.90 or 0.60 with nothing in between. The clarify band (0.65-0.80) never fires, so all 4 ambiguous cases were misrouted to quote.
- **Customer history inflates confidence.** When a customer has order history, the model assigns 0.90 confidence even when a critical spec (thread standard, material grade) is genuinely missing and history contains multiple values for that spec.
- **No structural guard for obvious ambiguity.** When a customer literally writes "not sure if 1/2 or 3/4" (tc_12), the system quoted anyway because model confidence was 0.90. Pure prompt-based reasoning failed; a rule layer was needed.

## What v2 Did (The Fix)

- **Rule-based guards before model routing.** Regex on ambiguity language ("not sure", "probably", "can you advise", size alternatives) forces clarify. Out-of-catalog indicators (metric sizes, stainless, custom) forces human escalation. Close-confidence hypotheses with missing critical specs forces clarify instead of guessing.
- **Few-shot calibration examples in the hypothesis prompt.** Four examples anchoring specific scenarios to specific confidence scores (0.72, 0.82, 0.88, 0.92). Broke the bimodal distribution — ambiguous-with-history cases dropped from 0.90 to 0.82.
- **Prompt-level guardrails.** "Customer history is probabilistic, not conclusive" prevents history from inflating confidence when multiple spec values exist. "Explicit specs override missing customer context" prevents penalizing clear RFQs that have no history.

## The Problem

> "A contractor sends a request for a brass fitting with several important details left out. How do you know what SKU they mean, based on their history with you, their end market, etc.? How do you make judgment calls better than a 30-year industry veteran? A mistake means a returned product."

Industrial distributors receive RFQ emails that are ambiguous by nature. A request for "200 half-inch brass elbows, same as last time" could map to 4 different SKUs that differ by thread standard or material grade. Getting it wrong means a returned shipment. The agent must either resolve the ambiguity confidently or ask the *right* clarifying question — not a generic "can you provide more details?"

## Approach

Think of it like a distributor's front desk. The **intake clerk** (Llama 3.1 8B via Groq, fast/free) parses the noisy email into structured data: product type, specs mentioned, specs missing, customer signals. The **senior rep** (Llama 4 Scout via Groq, stronger reasoning) looks at that structured data alongside customer order history and the catalog, then generates 2-3 ranked SKU hypotheses with explicit confidence scores and reasoning.

This is structurally the same pattern used in ARC-AGI reasoning: generate multiple hypotheses, score them against available evidence, and act on (or narrow) the highest-confidence candidate. The agent doesn't guess — it shows its work. Two focused LLM calls beat one monolithic prompt: each is simpler, cheaper, and independently evaluable.

## What Still Fails

- **tc_08 (Sarah Chen's marine nipple).** Expected clarify for material_grade, but the model quotes BF-4002 (C46400) at 0.82 confidence. Sarah Chen's history is exclusively marine/naval, and the email says "saltwater exposure." The model's reasoning is arguably correct — this is the kind of judgment call that benefits from a configurable risk threshold rather than a blanket rule.

## What I'd Build Next

- **Multi-turn clarification** — receive the customer's answer, re-run hypothesis generation with the new constraint
- **Confidence calibration** — track predicted vs. actual correctness over time, apply Platt scaling
- **Catalog ingestion pipeline** — parse real distributor catalogs (CSV/PDF) into structured format
- **Customer history sync** — pull from ERP/CRM instead of static JSON
- **Learning loops** — when a sales rep overrides the agent's pick, feed that correction back

---

## Running It

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your Groq API key (free at console.groq.com)

streamlit run app.py   # Interactive demo
python -m evals.eval   # Run eval suite
```

Built as a Paragon FDE application project.
