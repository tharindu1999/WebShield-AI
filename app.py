"""WebShield AI single-page Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from feature_definitions import FEATURE_NAMES
from feature_extractor import URLSafetyError, WebsiteFetchError, fetch_and_extract


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "web_risk_model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata["selected_features"] != FEATURE_NAMES:
        raise RuntimeError("Saved model metadata does not match the application feature schema.")
    if getattr(model, "n_features_in_", None) != len(FEATURE_NAMES):
        raise RuntimeError("Saved model input count does not match the application feature schema.")
    return model, metadata


def risk_level(probability: float) -> str:
    if probability < 0.30:
        return "LOW"
    if probability < 0.60:
        return "MEDIUM"
    if probability < 0.80:
        return "HIGH"
    return "VERY HIGH"


def metric_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def show_model_details(metadata: dict) -> None:
    st.divider()
    with st.expander("Model information"):
        left, right = st.columns(2)
        left.metric("Algorithm", metadata["algorithm"])
        right.metric("Model features", metadata["selected_feature_count"])
        st.caption(metadata["dataset"])

        metrics = metadata["metrics"]
        columns = st.columns(5)
        labels = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision_phishing"),
            ("Recall", "recall_phishing"),
            ("F1 Score", "f1_phishing"),
            ("ROC AUC", "roc_auc"),
        ]
        for column, (label, key) in zip(columns, labels):
            column.metric(label, metric_percent(metrics[key]))

        matrix = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual Legitimate", "Actual Phishing"],
            columns=["Predicted Legitimate", "Predicted Phishing"],
        )
        st.caption(
            f"Training samples: {metadata['training_sample_count']:,} | "
            f"Testing samples: {metadata['testing_sample_count']:,}"
        )
        st.dataframe(matrix, use_container_width=True)

def main() -> None:
    st.set_page_config(page_title="WebShield AI", layout="wide")
    st.title("WebShield AI")
    st.caption("AI Website Risk Classifier")
    try:
        model, metadata = load_artifacts()
    except Exception as exc:
        st.error(f"Model artifacts could not be loaded: {exc}")
        st.stop()

    url = st.text_input("Website URL", placeholder="https://www.example.com")
    analyze = st.button("Analyze Website", type="primary")

    if analyze:
        with st.spinner("Safely retrieving the public page and extracting model inputs..."):
            try:
                result = fetch_and_extract(url)
                input_frame = pd.DataFrame([result.features], columns=FEATURE_NAMES)
                prediction = int(model.predict(input_frame)[0])
                probabilities = model.predict_proba(input_frame)[0]
                phishing_probability = float(probabilities[list(model.classes_).index(1)])
                legitimate_probability = float(probabilities[list(model.classes_).index(0)])
            except (URLSafetyError, WebsiteFetchError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Analysis could not be completed: {exc}")
            else:
                label = "PHISHING" if prediction == 1 else "LEGITIMATE"
                st.header("AI Prediction")
                if prediction == 0:
                    st.success(label)
                else:
                    st.error(label)
                first, second, third = st.columns(3)
                first.metric("Phishing Probability", metric_percent(phishing_probability))
                second.metric("Legitimate Probability", metric_percent(legitimate_probability))
                third.metric("Risk Level", risk_level(phishing_probability))
                st.progress(phishing_probability)

                with st.expander("Model input features"):
                    table = pd.DataFrame({"Feature": FEATURE_NAMES, "Value": [result.features[name] for name in FEATURE_NAMES]})
                    st.dataframe(table, hide_index=True, use_container_width=True)
                    st.caption(f"Extracted {len(result.features)} of {len(FEATURE_NAMES)} features")


    show_model_details(metadata)


if __name__ == "__main__":
    main()

