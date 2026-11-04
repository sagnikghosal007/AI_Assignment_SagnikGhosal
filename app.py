"""
app.py  (Task 6 - Simple Interface with Streamlit)
--------------------------------------------------
Run:  streamlit run app.py
A browser tab opens with a text box, a Predict button and the result.
"""

import sys
from pathlib import Path

import streamlit as st

# Make the src/ folder importable (predict.py, preprocessing.py, auto_reply.py live there)
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from predict import load_model, predict_ticket  # noqa: E402

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫")


@st.cache_resource          # load the model only once, not on every click
def get_model():
    return load_model()


st.title("🎫 Support Ticket Classifier")
st.write("Type a customer support ticket below and the AI model will predict its category.")

ticket_text = st.text_area(
    "Ticket description",
    placeholder="Example: I forgot my password and cannot log in",
    height=120,
)

if st.button("Predict"):
    if not ticket_text.strip():
        st.warning("Please enter a ticket description first.")
    else:
        result = predict_ticket(get_model(), ticket_text)

        st.subheader("Result")
        st.success(f"Predicted Category: **{result['category']}**")
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        st.progress(result["confidence"])

        if result["low_confidence"]:
            st.warning("Low confidence - a human agent should check this ticket.")

        st.subheader("Suggested reply")
        st.info(result["reply"])

        with st.expander("See the score for every category"):
            scores = {k: float(v) for k, v in result["all_scores"].items()}
            st.bar_chart(scores)