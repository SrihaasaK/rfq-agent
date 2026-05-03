# RFQ-to-Quote Agent

A hypothesis-driven disambiguation agent for industrial brass fittings RFQs. Built as a technical demonstration for Paragon.

---

## The Problem

> "A contractor sends a request for a brass fitting with several important details left out. How do you know what SKU they mean, based on their history with you, their end market, etc.? How do you make judgment calls better than a 30-year industry veteran? A mistake means a returned product. How do you never make a mistake?"

Industrial distributors receive RFQ emails that are ambiguous by nature. A request for "200 half-inch brass elbows, same as last time" could map to 4 different SKUs that differ by thread standard or material grade. Getting it wrong means a returned shipment. The agent must either resolve the ambiguity confidently or ask the *right* clarifying question — not a generic "can you provide more details?"

## Approach: Hypothesis-Driven Disambiguation

This agent doesn't do match-or-fail lookup. It generates **ranked hypotheses** with explicit confidence scores, then routes based on the top score:

1. **Extract** structured intent from raw email (product type, specs mentioned/missing, customer signals)
2. **Hypothesize** 2-3 candidate SKUs with confidence scores and reasoning, using customer order history and end-market signals to disambiguate
3. **Route** based on top confidence:
   - **>= 0.80** → Generate a quote draft
   - **0.65 – 0.80** → Ask a targeted clarifying question that disambiguates the top 2 hypotheses
   - **< 0.65** → Escalate to human with structured reasoning

This is structurally the same pattern used in ARC-AGI reasoning: generate multiple hypotheses, score them against available evidence, and act on (or narrow) the highest-confidence candidate. The agent doesn't guess — it *shows its work*.

### Why Two LLM Calls, Not One

The extraction step (Llama 3.1 8B via Groq — fast, free) parses noisy email text into structured data. The hypothesis step (Llama 4 Scout via Groq — stronger reasoning) works on clean structured input + customer history + catalog. Separating these makes each prompt simpler, more evaluable, and cheaper. Two focused calls beat one monolithic prompt.

## Eval Methodology

The eval harness (`evals/eval.py`) runs 20 test cases and measures:

| Metric | What It Measures |
|--------|-----------------|
| **Route accuracy** | Did the agent choose the correct route (quote/clarify/human)? |
| **SKU match rate** | For quote cases: did it pick the right SKU? |
| **Question quality** | For clarify cases: does the question address the *actual* disambiguating spec? |
| **Human escalation rate** | For garbage inputs: did it correctly refuse to act? |

**Question quality scoring** is the key differentiator. A generic "can you provide more details?" scores 0. A question that asks about the specific spec difference between the top 2 hypotheses scores 1.0. This measures whether the agent is doing the disambiguation work rather than pushing it back to the customer.

## Architecture

```
                    RFQ Email
                        |
                        v
            +-----------+-----------+
            |   CALL 1: Extraction  |
            |   (Llama 8B / Groq)   |
            |   Raw text -> Intent  |
            +-----------+-----------+
                        |
            ExtractedIntent (Pydantic)
                        |
           +------------+------------+
           |                         |
    Customer History            Catalog (30 SKUs)
           |                         |
           +------------+------------+
                        |
                        v
            +-----------+-----------+
            |  CALL 2: Hypotheses   |
            |  (Llama 4 Scout/Groq) |
            |  Intent + context ->  |
            |  ranked SKU guesses   |
            +-----------+-----------+
                        |
                        v
              +------- conf -------+
              |         |          |
           >0.85    0.5-0.85    <0.5
              |         |          |
           QUOTE    CLARIFY     HUMAN
```

## Running It

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env with your Groq API key (free at console.groq.com)

# Run the UI
streamlit run app.py

# Run evals from CLI
python -m evals.eval
```

## Eval Results (Actual)

Measured on 20 test cases (15 sample emails + 5 edge cases):

| Metric | Score | Notes |
|--------|-------|-------|
| Route accuracy | **70%** (14/20) | Main gap: model over-confident on ambiguous cases |
| SKU match rate | **83.3%** (10/12) | When it quotes, it usually picks the right SKU |
| Human escalation | **75%** (3/4) | 1 garbage case got 0.7 confidence instead of <0.65 |
| Question quality | **0%** | Ambiguous cases routed to quote (too confident), so no questions were generated to score |

Full results in `evals/latest_report.json`.

## Known Failure Modes

1. **Over-confident on ambiguous cases with customer history.** Llama 4 Scout gives 0.9 confidence when customer history exists, even when a critical spec (thread standard, material grade) is genuinely missing. The model treats history as conclusive rather than probabilistic. This collapses the clarify band — ambiguous cases jump straight to quote. A production system would need confidence calibration (e.g., Platt scaling on a held-out set) or explicit guardrails that force a clarify route when critical specs are absent regardless of confidence.

2. **Multi-product RFQs.** Email 12 asks for couplings in "1/2 or 3/4 — the engineer is checking." The agent treats this as a single-SKU problem. Real RFQs often have line items for multiple products.

3. **Fuzzy customer matching.** "Tom B" needs to resolve to "Tom Briggs at Briggs Mechanical." The current name matcher is substring-based. Production would need fuzzy matching against a CRM.

4. **Clarifying question phrasing.** The question is constructed from hypothesis metadata, not generated by the LLM. This makes it predictable and evaluable, but sometimes stilted compared to how a veteran sales rep would phrase it.

5. **No feedback loop.** The agent can't learn from corrections. If a customer consistently orders sweat fittings but their history shows NPT (because of a data entry error), the agent will keep recommending NPT.

## What I'd Build Next (Given Another Week)

- **Multi-turn clarification.** Let the agent ask a question, receive the answer, and re-run hypothesis generation with the new constraint.
- **Catalog ingestion pipeline.** Parse real distributor catalogs (CSV/PDF) into the structured format.
- **Customer history sync.** Pull order history from ERP/CRM instead of static JSON.
- **Confidence calibration.** Track predicted confidence vs. actual correctness over time, apply Platt scaling.
- **Learning loops.** When a sales rep overrides the agent's pick, feed that correction back to improve future predictions for that customer.
- **Email integration.** Ingest from a shared inbox, return quotes via email, track the thread.

---

Built as a Paragon FDE application project. The code prioritizes eval rigor and architectural clarity over feature breadth.
