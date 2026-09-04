# GroundTruth Demo API

A tiny FastAPI application used as a test target for GroundTruth.

## Run

```bash
# from D:\Projects 2026\GroundTruth\
uvicorn demo_api.main:app --reload --port 8001
```

OpenAPI spec is live at: http://localhost:8001/openapi.json

## Endpoints

| Endpoint | Status | Purpose |
|---|---|---|
| `GET /items/{item_id}` | Correct | Produces `SPEC_MATCH` in GroundTruth |
| `GET /items` | Correct | Produces `SPEC_MATCH` in GroundTruth |
| `GET /orders/{order_id}` | **INTENTIONALLY BROKEN** | Produces `SPEC_VIOLATION` — `total_price` is returned as a string instead of a float, simulating a real currency-formatting bug |

## GroundTruth demo commands

```bash
# 1. Save the live spec
curl http://localhost:8001/openapi.json -o demo_api_spec.json

# 2. Test the clean endpoints (expect SPEC_MATCH)
python cli.py --spec demo_api_spec.json --base-url http://localhost:8001 --endpoint "/items/{item_id}" --method get

python cli.py --spec demo_api_spec.json --base-url http://localhost:8001 --endpoint "/items" --method get

# 3. Test the broken endpoint (expect SPEC_VIOLATION with total_price type error)
python cli.py --spec demo_api_spec.json --base-url http://localhost:8001 --endpoint "/orders/{order_id}" --method get
```
