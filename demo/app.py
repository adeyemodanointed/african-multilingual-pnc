"""Visual demo for the 8-epoch AfroXLMR-small PnC checkpoint.

Run from the multilingual-pnc repository root:
    $env:PYTHONPATH = "src"
    streamlit run demo/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pnc.inference import load_checkpoint, restore_text  # noqa: E402


CHECKPOINT = PROJECT_ROOT / "outputs/afroxlmr_small_expanded_v1_8epoch/best.pt"
EXAMPLES = {
    "Yoruba": "awon agbe nilo lati lo irugbin didara ati ajile lati mu ikore won pọ si",
    "Kinyarwanda": "abahinzi bakeneye gukoresha imbuto nziza n ifumbire kugira ngo umusaruro wiyongere",
    "Nigerian Pidgin": "farmers need better seed and fertilizer make their harvest increase",
    "Swahili": "wakulima wanahitaji kutumia mbegu bora na mbolea ili kuongeza mavuno",
    "Igbo": "ndi oru ugbo kwesiri iji mkpuru oma na fatịlaịza mee ka owuwe ihe ubi bawanye",
}


@st.cache_resource(show_spinner="Loading the AfroXLMR-small PnC checkpoint…")
def load_pnc_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    return load_checkpoint(CHECKPOINT)


st.set_page_config(page_title="African Languages PnC Demo", page_icon="✍️", layout="centered")
st.markdown("# ✍️ Punctuation & Capitalization Demo")
st.caption("AfroXLMR-small · shared multilingual 8-epoch checkpoint · experimental demonstration")
st.info(
    "This model restores punctuation and word casing from lowercase, punctuation-free text. "
    "It is an experimental checkpoint, not the final selected PnC model."
)

language = st.selectbox("Example language", list(EXAMPLES))
if st.button("Use example"):
    st.session_state["pnc_text"] = EXAMPLES[language]
text = st.text_area(
    "Unpunctuated text",
    key="pnc_text",
    height=150,
    placeholder="paste lowercase text without punctuation…",
)

if st.button("Restore text", type="primary", use_container_width=True):
    try:
        tokenizer, model, config, device = load_pnc_model()
        result = restore_text(text, tokenizer, model, config, device)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        st.error(str(error))
    else:
        st.subheader("Restored text")
        st.success(str(result["restored_text"]))
        st.caption(f"Processed {len(result['input_words'])} words on {device.type.upper()}.")
        with st.expander("Predicted token labels"):
            st.dataframe(
                {
                    "Word": result["input_words"],
                    "Punctuation after word": result["punctuation_labels"],
                    "Casing": result["case_labels"],
                },
                use_container_width=True,
                hide_index=True,
            )

with st.expander("Important note"):
    st.write(
        "The demo predicts labels token by token and does not provide calibrated confidence. "
        "Use it to show the task and the current model behaviour, not as evidence that the model is final or production-ready."
    )
