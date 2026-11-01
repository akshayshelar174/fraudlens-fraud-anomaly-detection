"""FraudLens analyst dashboard: model comparison, triage, and drift."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from fraudlens.features import FEATURES
from fraudlens.monitoring import drift_report
from fraudlens.scoring import explain_one, load_bundle, score_frame
from fraudlens.training import ARTIFACT_DIR

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transactions.csv"

st.set_page_config(page_title="FraudLens | Transaction Risk", page_icon="◈", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f4f7f8; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e3e9ec; padding: 15px 18px; border-radius: 12px; }
    .hero { padding: 23px 28px; border-radius: 15px; background: linear-gradient(120deg,#087f5b,#12a879); color: white; margin-bottom: 20px; }
    .hero h1 { color: white; margin: 0; font-size: 30px; }
    .hero p { color: #e4fff5; margin: 6px 0 0; }
    .disclaimer { color: #5c6771; font-size: 12px; }
</style>
<div class="hero"><h1>FraudLens</h1><p>Transaction risk intelligence · synthetic demo data</p></div>
""", unsafe_allow_html=True)

try:
    bundle = load_bundle()
    metrics = json.loads((ARTIFACT_DIR / "model_metrics.json").read_text(encoding="utf-8"))
    transactions = pd.read_csv(DATA_PATH)
except (FileNotFoundError, json.JSONDecodeError) as exc:
    st.error(f"Project artifacts are not ready: {exc}")
    st.code("python -m fraudlens.training --rows 12000")
    st.stop()

with st.sidebar:
    st.title("Workspace")
    page = st.radio("Go to", ["Overview", "Transaction review", "Model performance", "Drift monitoring"])
    threshold = st.slider("Review threshold", 1, 99, int(round(bundle["threshold"] * 100)), 1) / 100
    st.caption(f"Selected model: {bundle['model_name']}")
    st.divider()
    st.caption("Demo dataset is synthetic. Do not enter card numbers or real customer data.")

scored = score_frame(transactions, bundle=bundle, threshold=threshold)
fraud_rate = float(transactions["is_fraud"].mean())
review_rate = float((scored["decision"] == "review").mean())

if page == "Overview":
    st.subheader("Analyst overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(scored):,}")
    c2.metric("Synthetic fraud rate", f"{fraud_rate:.2%}")
    c3.metric("Queued for review", f"{review_rate:.1%}")
    c4.metric("Review threshold", f"{threshold:.0%}")

    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("#### Risk score distribution")
        chart = px.histogram(scored, x="risk_score", color="decision", nbins=36,
                             color_discrete_map={"allow": "#4b83d1", "review": "#e58a3a"},
                             labels={"risk_score": "Relative risk score", "decision": "Queue"})
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.markdown("#### Model comparison · test set")
        comparison = pd.DataFrame([
            {"Model": name, "PR-AUC": values["test"]["average_precision"],
             "Precision": values["test"]["precision"], "Recall": values["test"]["recall"]}
            for name, values in metrics["models"].items()
        ]).sort_values("PR-AUC", ascending=False)
        st.dataframe(comparison.style.format({"PR-AUC": "{:.3f}", "Precision": "{:.1%}", "Recall": "{:.1%}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Thresholds were selected on validation data; reported performance uses a separate test split.")

    st.markdown("#### Recent synthetic activity")
    daily = scored.groupby("event_day", as_index=False).agg(
        transactions=("transaction_id", "count"), queued=("decision", lambda values: (values == "review").sum())
    )
    st.plotly_chart(px.line(daily, x="event_day", y=["transactions", "queued"], markers=False,
                            labels={"event_day": "Synthetic day", "value": "Count", "variable": "Series"}),
                    use_container_width=True)

elif page == "Transaction review":
    st.subheader("Transaction review queue")
    queue = scored[scored["decision"] == "review"].sort_values("risk_score", ascending=False)
    st.write(f"{len(queue):,} transactions meet the current review threshold.")
    display_columns = ["transaction_id", "risk_score", *FEATURES, "risk_indicators"]
    display_queue = queue[display_columns].head(500).copy()
    display_queue["risk_indicators"] = display_queue["risk_indicators"].apply(lambda items: "; ".join(items))
    st.dataframe(display_queue, hide_index=True, use_container_width=True,
                 column_config={"risk_score": st.column_config.NumberColumn("Relative risk", format="%.1f / 100")})
    st.download_button("Download review queue (CSV)", queue.drop(columns="risk_indicators").to_csv(index=False),
                       "fraudlens-review-queue.csv", "text/csv")
    if len(queue):
        selected = st.selectbox("Inspect transaction", queue["transaction_id"].head(500).tolist())
        transaction = queue.loc[queue["transaction_id"] == selected].iloc[0]
        st.markdown(f"**{selected} · relative score {transaction['risk_score']:.1f}/100**")
        for reason in transaction["risk_indicators"]:
            st.write(f"• {reason}")
        st.markdown("**One-feature score sensitivity**")
        for item in explain_one(transaction.to_dict(), bundle=bundle):
            st.write(f"• {item['display_name']}: replacing {item['observed_value']:.2f} with a typical value would lower the score by about {item['score_increase_points']:.1f} points.")
        st.caption("Ablations are local sensitivity checks against reference medians, not causal effects; feature interactions are not additive. Risk indicators are separate reference-range checks. Review source evidence before action.")

elif page == "Model performance":
    st.subheader("Model comparison")
    st.markdown("Average precision (PR-AUC) is emphasized because fraud is rare; accuracy alone can hide missed fraud.")
    rows = []
    for name, result in metrics["models"].items():
        test = result["test"]
        rows.append({"Model": name, "Validation PR-AUC": result["validation_average_precision"],
                     "Test PR-AUC": test["average_precision"], "ROC-AUC": test["roc_auc"],
                     "Precision": test["precision"], "Recall": test["recall"], "F1": test["f1"],
                     "Threshold": test["threshold"]})
    table = pd.DataFrame(rows).sort_values("Validation PR-AUC", ascending=False)
    st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns if c != "Model"}),
                 hide_index=True, use_container_width=True)
    st.info(f"Selected using validation PR-AUC: **{metrics['selected_model']}**. The held-out test split was not used for selection.")
    selected = metrics["models"][metrics["selected_model"]]["test"]["confusion"]
    confusion = pd.DataFrame([[selected["true_negative"], selected["false_positive"]],
                              [selected["false_negative"], selected["true_positive"]]],
                             index=["Actual: allow", "Actual: fraud"], columns=["Predicted: allow", "Predicted: review"])
    st.markdown("#### Held-out confusion matrix at the selected model's validation threshold")
    st.dataframe(confusion, use_container_width=True)
    st.caption("Thresholds trade off the cost of missed fraud against investigation capacity. Set an operating point with your actual costs and labeled validation data.")

else:
    st.subheader("Feature drift monitoring")
    reference = transactions[transactions["event_day"] < 84]
    current = transactions[transactions["event_day"] >= 96]
    report = pd.DataFrame(drift_report(reference, current, FEATURES))
    st.write(f"Reference window: {len(reference):,} transactions · recent window: {len(current):,} transactions")
    chart = px.bar(report, x="psi", y="feature", color="level", orientation="h",
                   color_discrete_map={"low": "#2a9d73", "moderate": "#e0a33a", "high": "#d45d5d"},
                   labels={"psi": "Population Stability Index", "feature": "Feature"})
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(report, hide_index=True, use_container_width=True)
    st.caption("Common screening guide: PSI < 0.10 low, 0.10–0.25 moderate, >= 0.25 high. PSI is a monitoring signal, not proof that a model is invalid.")

st.divider()
st.markdown("<p class='disclaimer'>FraudLens is a portfolio demo using generated data. It is not connected to a payment processor, does not authorize transactions, and is not validated for real-world fraud decisions.</p>", unsafe_allow_html=True)
