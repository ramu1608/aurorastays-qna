
import streamlit as st
import requests


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BACKEND_URL = "http://localhost:7860"


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Aurora Stays",
    page_icon="🏨",
    layout="wide"
)


# ---------------------------------------------------------
# Sidebar  -  Model parameters
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Parameters")

    k = st.slider(
        "k (chunks to retrieve)",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of relevant chunks fetched from the vector store."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make the output more creative; lower values make it more focused."
    )

    max_tokens = st.slider(
        "Max tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
        help="Maximum number of tokens in the generated answer."
    )


# ---------------------------------------------------------
# Main area
# ---------------------------------------------------------

st.title("🏨 Aurora Stays")
st.write("Ask a question about Aurora Stays policies.")


# ---------------------------------------------------------
# Response type selection (two options only)
# ---------------------------------------------------------

response_type = st.radio(
    "Response mode",
    ["Answer only", "Answer + Relevant Chunks"],
    horizontal=True
)


# ---------------------------------------------------------
# Query input
# ---------------------------------------------------------

query = st.text_input(
    "Your question",
    placeholder="e.g., What is the waiting period for hospitalization?"
)


# ---------------------------------------------------------
# Submit
# ---------------------------------------------------------

if st.button("Submit", type="primary"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:

        # Pick endpoint
        if response_type == "Answer only":
            endpoint = "/v1/answer"
        else:
            endpoint = "/v1/answer_with_relevant_chunks"

        # Build request payload with parameters
        payload = {
            "query": query,
            "k": k,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            with st.spinner("Thinking…"):
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload
                )

            if response.status_code == 200:

                result = response.json()

                # -- Answer only --------------------------------
                if response_type == "Answer only":

                    st.subheader("Answer")
                    st.write(result["answer"])

                # -- Answer + Relevant Chunks -------------------
                else:

                    st.subheader("Answer")
                    st.write(result["answer"])

                    st.subheader("Relevant Chunks")
                    for i, chunk in enumerate(result["relevant_chunks"], start=1):
                        with st.expander(f"Chunk {i}"):
                            st.write(chunk)

            else:
                st.error(
                    f"Unable to get a response from the RAG API. "
                    f"Status code: {response.status_code}"
                )

        except requests.exceptions.RequestException:
            st.error(
                "Unable to connect to the RAG API. "
                "Please make sure the backend service is running."
            )
