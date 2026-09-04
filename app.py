import os
import sys
import json
import yaml
import requests
import streamlit as st
import pandas as pd
from typing import Dict, Any

from groundtruth.spec_parser import parse_endpoint, SpecParseError
from groundtruth.test_generator import generate_test_cases, GeneratorError
from groundtruth.executor import execute
from groundtruth.validator import validate
from groundtruth.report import build_summary
import dataclasses
from datetime import datetime, timezone
from groundtruth import __version__
from dotenv import load_dotenv

load_dotenv()
# ── Global Config & State ─────────────────────────────────────────────────────

st.set_page_config(page_title="GroundTruth", page_icon="🔍", layout="wide")

# Simple rate limit: N runs per hour (in-memory, resets on app restart)
MAX_RUNS = int(os.environ.get("GROUNDTRUTH_MAX_RUNS", "20"))
if "run_count" not in st.session_state:
    st.session_state.run_count = 0

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Helper Functions ──────────────────────────────────────────────────────────

def fetch_spec(source: str) -> Dict[str, Any]:
    """Fetch spec from URL or parse uploaded file content."""
    if source.startswith("http"):
        resp = requests.get(source, timeout=10)
        resp.raise_for_status()
        content = resp.text
    else:
        content = source

    try:
        return yaml.safe_load(content)
    except yaml.YAMLError:
        return json.loads(content)


def run_pipeline(spec_content: str, base_url: str, endpoint_path: str, method: str, allow_mutations: bool):
    """Runs the 5 GroundTruth modules."""
    st.session_state.run_count += 1

    try:
        # Module 1: Spec Parser
        with st.spinner("Module 1: Parsing OpenAPI spec..."):
            spec = fetch_spec(spec_content)
            endpoint = parse_endpoint(spec, endpoint_path, method.lower())

        # Module 2: Test Generator
        with st.spinner(f"Module 2: Generating test cases (calling LLM)..."):
            test_cases = generate_test_cases(endpoint, GROQ_API_KEY, model="openai/gpt-oss-120b")

        # Module 3: HTTP Executor
        results = []
        progress_text = "Module 3: Executing HTTP requests..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, tc in enumerate(test_cases):
            res = execute(tc, endpoint, base_url, allow_mutations)
            results.append(res)
            my_bar.progress((i + 1) / len(test_cases), text=progress_text)
        my_bar.empty()

        # Module 4: Schema Validator
        with st.spinner("Module 4: Validating responses against spec..."):
            validations = []
            for result in results:
                tc = next(t for t in test_cases if t.id == result.test_case_id)
                vr = validate(endpoint, tc, result)
                validations.append(vr)

        # Module 5: Report Generator
        with st.spinner("Module 5: Generating report..."):
            summary = build_summary(test_cases, validations)
            
            # Build JSON payload
            json_report = {
                "groundtruth_version": __version__,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "endpoint": {
                    "path": endpoint.path,
                    "method": endpoint.method,
                    "operation_id": endpoint.operation_id,
                    "summary": endpoint.summary,
                },
                "summary": dataclasses.asdict(summary),
                "test_cases": [dataclasses.asdict(tc) for tc in test_cases],
                "execution_results": [dataclasses.asdict(r) for r in results],
                "validation_results": [dataclasses.asdict(vr) for vr in validations],
            }

        return test_cases, results, validations, summary, json_report

    except Exception as e:
        st.error(f"Pipeline Error: {str(e)}")
        return None, None, None, None, None


# ── UI Layout ─────────────────────────────────────────────────────────────────

st.title("🔍 GroundTruth")
st.markdown("### The API Tester")
st.markdown("Give it an OpenAPI spec and a running API. It generates test scenarios, runs them against the live endpoint, and validates the responses against the spec.")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable is not set. The LLM cannot generate test cases.")
    st.stop()

if st.session_state.run_count >= MAX_RUNS:
    st.warning(f"Rate limit reached: {MAX_RUNS} runs completed. Please restart the app to run more.")
    st.stop()

# ── Sidebar Form ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Test Configuration")
    
    spec_source_type = st.radio("OpenAPI Spec Source", ["URL", "File Upload"])
    spec_content = ""
    
    if spec_source_type == "URL":
        spec_url = st.text_input("Spec URL", value="http://localhost:8001/openapi.json")
        spec_content = spec_url
    else:
        uploaded_file = st.file_uploader("Upload Spec (.yaml, .yml, .json)", type=["yaml", "yml", "json"])
        if uploaded_file is not None:
            spec_content = uploaded_file.getvalue().decode("utf-8")
    
    base_url = st.text_input("API Base URL", value="http://localhost:8001")
    endpoint_path = st.text_input("Endpoint Path", value="/orders/{order_id}")
    method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE", "PATCH"])
    
    allow_mutations = st.checkbox("Allow Mutations", value=False, help="Must be checked to run non-GET methods.")
    
    run_disabled = (method != "GET" and not allow_mutations) or not spec_content
    
    if run_disabled and method != "GET":
        st.warning(f"'{method}' is a mutating method. You must check 'Allow Mutations' to run it.")

    run_clicked = st.button("Run GroundTruth", disabled=run_disabled, type="primary", use_container_width=True)
    st.caption(f"Runs used: {st.session_state.run_count} / {MAX_RUNS}")


# ── Main Content Area ─────────────────────────────────────────────────────────

if run_clicked:
    st.markdown("---")
    test_cases, results, validations, summary, json_report = run_pipeline(
        spec_content, base_url, endpoint_path, method, allow_mutations
    )

    if summary:
        # Display Summary Metrics
        st.header(f"Report: {method} {endpoint_path}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Cases", summary.total_cases)
        col2.metric("Spec Matches", f"{summary.outcome_pct.get('SPEC_MATCH', 0):.1f}%")
        col3.metric("Undocumented", f"{summary.outcome_pct.get('UNDOCUMENTED_BEHAVIOR', 0):.1f}%")
        col4.metric("LLM Accuracy (Info)", f"{summary.llm_accuracy_pct:.1f}%", help="Informational only. Does not affect pass/fail.")

        # Spec Violations Prominent Display
        if summary.spec_violation_ids:
            st.error(f"🚨 {len(summary.spec_violation_ids)} SPEC VIOLATION(S) FOUND (Contract Breaks)")
            for tc_id in summary.spec_violation_ids:
                vr = next(v for v in validations if v.test_case_id == tc_id)
                tc = next(t for t in test_cases if t.id == tc_id)
                with st.expander(f"{tc_id} [{tc.category}] - Status: {vr.actual_status_code}", expanded=True):
                    st.write(f"**LLM Intent:** {tc.description}")
                    st.write(f"**Reason:** {vr.reason}")
                    if vr.schema_errors:
                        st.write("**Schema Errors:**")
                        for err in vr.schema_errors:
                            st.code(err, language="text")
        else:
            st.success("✅ No Spec Violations Detected (No contract breaks).")
            
        st.markdown("---")

        # Outcome Breakdown Chart
        st.subheader("Outcome Breakdown")
        outcome_df = pd.DataFrame({
            "Outcome": summary.outcome_counts.keys(),
            "Count": summary.outcome_counts.values()
        }).set_index("Outcome")
        st.bar_chart(outcome_df)

        # Test Cases Table
        st.subheader("Individual Test Cases")
        table_data = []
        for tc, vr in zip(test_cases, validations):
            table_data.append({
                "ID": tc.id,
                "Category": tc.category,
                "LLM Guess": tc.expected_status,
                "Actual Status": vr.actual_status_code,
                "Outcome": vr.outcome,
                "Reason": vr.reason
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # Download JSON Report
        st.markdown("---")
        st.download_button(
            label="Download JSON Report",
            data=json.dumps(json_report, indent=2),
            file_name=f"groundtruth_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            type="primary"
        )
