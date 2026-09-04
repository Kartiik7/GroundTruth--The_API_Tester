# GroundTruth — The API Tester

**GroundTruth tests your API against its own OpenAPI contract — using an LLM to propose test cases, but never to decide what's correct.**

Give it an OpenAPI spec and a running API. It generates realistic test scenarios (happy paths, edge cases, invalid input), runs them against the live endpoint, and checks every response against what the spec actually documents — not against what the LLM guessed would happen.

---

## Why

LLMs are good at *imagining* test scenarios a human might not think of. They are not a reliable source of truth for whether a response is *correct*. GroundTruth keeps those two jobs separate:

- **The LLM proposes** — it generates diverse, structured test cases (inputs + a category + a guessed expected status).
- **The spec disposes** — every actual response is validated against the OpenAPI document itself. The LLM's guess is recorded for informational accuracy tracking only. It never decides pass or fail.

---

## Architecture

Five modules, each with a narrow, testable contract:

| Module | Responsibility |
|---|---|
| **1. Spec Parser** | Parses an OpenAPI 3.x spec, resolves all `$ref` pointers recursively so every downstream schema is fully self-contained, and extracts per-endpoint parameters and response schemas. |
| **2. Test Generator** | Calls an LLM (Groq) to produce structured JSON test cases — inputs, a category (`happy_path` / `invalid_input` / `not_found` / `edge_case`), and a non-authoritative guessed status. Enforces JSON schema on the output with one bounded retry. |
| **3. HTTP Executor** | Sends real HTTP requests for each test case against the target API. Read-only by default (`--allow-mutations` required for non-GET methods), with a 10s timeout per request. |
| **4. Schema Validator** | The ground-truth engine. Compares each actual response against the OpenAPI spec and assigns one of three outcomes (see below) — independent of the LLM's guess. |
| **5. Report Generator** | Aggregates results into a console summary (outcome breakdown, category × outcome cross-table, LLM prediction accuracy, flagged violations) and an optional full JSON export. |

---

## The three-outcome model

Most test tools give you pass/fail. That's not enough here, because a response can be *wrong* in two structurally different ways:

- **`SPEC_MATCH`** — the actual status code is documented for this endpoint, and (if a schema is defined) the response body satisfies it.
- **`SPEC_VIOLATION`** — the actual status code *is* documented, but the response body breaks the documented schema. This is a real contract break.
- **`UNDOCUMENTED_BEHAVIOR`** — the actual status code isn't documented for this endpoint at all. Not automatically a failure — it's a gap between what the spec says can happen and what actually happened, and it's reported as its own category rather than silently folded into pass or fail.

---

## Quickstart

```bash
pip install -r requirements.txt
# create .env with GROQ_API_KEY=gsk_...

python cli.py \
    --spec path/to/openapi.yaml \
    --base-url https://your-api.example.com \
    --endpoint "/your/{resource_id}" \
    --method get \
    --output report.json
```

---

## Demo: three real scenarios

### 1. Clean API — proving it passes correct responses

Run against a local FastAPI app with two well-behaved endpoints (`/items`, `/items/{item_id}`):

```
OUTCOME BREAKDOWN
SPEC_MATCH             [####################]    6  (100.0%)
SPEC_VIOLATION         [....................]    0  (  0.0%)
UNDOCUMENTED_BEHAVIOR  [....................]    0  (  0.0%)
```

Six generated test cases — happy path, not-found, invalid input, and two edge cases — all correctly validated as matching the documented contract.

### 2. Broken API — catching a real contract violation

The same demo app has a third endpoint, `/orders/{order_id}`, with an intentional bug: the spec documents `total_price` as a `number`, but the handler returns it as a formatted string (`"19.98"` instead of `19.98`) — a realistic "forgot to cast a display value back to a number" mistake.

```
tc_005  [FAIL] SPEC_VIOLATION  (LLM guess matched)
    Status 200 is documented but response body violates the 200 schema (1 error(s) found).
    Schema errors:
      - [total_price] '19.98' is not of type 'number'

SPEC VIOLATIONS  (contract breaks: documented code, wrong body)
!! 1 SPEC VIOLATION(S) FOUND !!
```

Caught precisely, with the exact field and the exact violation — while LLM prediction accuracy on this same run was only 60%, underscoring that the *validator*, not the LLM, is what's making the correctness call.

### 3. Live third-party API — handling real-world messiness

Run against the public Swagger Petstore demo (`petstore3.swagger.io`), which — like a lot of real APIs — returns undocumented `500` errors under normal-looking inputs:

```
OUTCOME BREAKDOWN
SPEC_MATCH             [########............]    2  ( 40.0%)
SPEC_VIOLATION         [....................]    0  (  0.0%)
UNDOCUMENTED_BEHAVIOR  [############........]    3  ( 60.0%)
```

No crashes, no false failures — every unexpected `500` is reported as `UNDOCUMENTED_BEHAVIOR`, a distinct, honest category rather than a false positive or a silent pass.

---

## Design notes

- **`$ref` resolution happens once, at parse time.** Early on, extracted response schemas retained unresolved OpenAPI `$ref` pointers (e.g. a `Pet` schema referencing `Category`/`Tag`), which crashed schema validation with `PointerToNowhere` — the reference was valid only relative to the full spec document, not the isolated fragment. Fixed by recursively dereferencing every `$ref` in the spec parser, so every module downstream works with fully self-contained schemas and never needs to understand OpenAPI's reference system.
- **The LLM's guess is never authoritative.** `TestCase.expected_status` is explicitly documented in code as non-authoritative — it's tracked purely for reporting prediction accuracy, and it plays no role in the pass/fail decision.
- **Read-only by default.** The HTTP executor requires an explicit `--allow-mutations` flag before it will send non-GET requests, so pointing it at a real API by mistake doesn't risk writing data.

---

## Limitations

- **GroundTruth catches contract violations, not business-logic bugs the spec doesn't encode.** For example, an API that accepts a negative ID and returns a valid `200` response isn't a spec violation if the spec never said IDs must be positive — that's a gap in the spec itself, not something a contract tester can catch.
- **Correctness is only as good as the spec.** An outdated or incomplete OpenAPI document will produce misleading `UNDOCUMENTED_BEHAVIOR` results for behavior that's actually intentional.
- Scoped to a single endpoint per run by design — no automatic looping across an entire spec (a deliberate scope decision, not a missing feature).

---

## Run via Docker (no clone required)

You can run GroundTruth via Docker, which defaults to the Streamlit UI on port 8501.

```bash
docker pull ghcr.io/kartiik7/groundtruth:latest

docker run -p 8501:8501 --rm ghcr.io/kartiik7/groundtruth:latest \
    -e GROQ_API_KEY=your_key_here
```

Open `http://localhost:8501` to use the UI.

To run the CLI instead of the UI:

```bash
docker run --rm ghcr.io/kartiik7/groundtruth:latest \
    -e GROQ_API_KEY=your_key_here \
    --entrypoint python \
    cli.py \
    --spec /path/to/spec.yaml \
    --base-url https://your-api.com \
    --endpoint "/your/{resource}" \
    --method get
```

*Note: If testing a locally-run API from inside the container, use `--base-url http://host.docker.internal:PORT` instead of `localhost`.*