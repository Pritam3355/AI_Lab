import streamlit as st
import os
from backend_bk import ComplianceAssistant
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

st.set_page_config(page_title="Capital Market Compliance Assistant", layout="wide")
st.title("📋 AI Powered Regulatory Compliance Assistant")

# Model Selector
st.sidebar.header("LLM Model")
models = [
    "genailab-maas-gpt-4o",
    "azure/genailab-maas-gpt-4.1",
    "azure_ai/genailab-maas-Llama-3.3-70B-Instruct",
    "gemini-2.5-pro",
    "genailab-maas-gpt-35-turbo",
    "azure/genailab-maas-gpt-4o-mini"
]
llm_model = st.sidebar.selectbox("Select Model", models, index=0)

if "assistant" not in st.session_state or st.session_state.get("current_model") != llm_model:
    st.session_state.assistant = ComplianceAssistant(llm_model=llm_model)
    st.session_state.current_model = llm_model

assistant = st.session_state.assistant

# Ingestion
st.sidebar.header("📤 Ingest Documents")
uploaded_files = st.sidebar.file_uploader("PDF (OCR), TXT, Excel", accept_multiple_files=True, type=["pdf","txt","xlsx","xls"])

if uploaded_files and st.sidebar.button("Ingest Documents"):
    for file in uploaded_files:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        status = assistant.ingest_document(temp_path)
        st.sidebar.success(status)
        os.remove(temp_path)

# Show ingested docs & metrics
st.sidebar.header("📊 Data")
if st.sidebar.button("Show Ingested Documents"):
    docs = assistant.get_ingested_documents()
    st.sidebar.write(docs if docs else "No documents yet")

if st.sidebar.button("Show Pipeline Metrics"):
    df = assistant.get_metrics()
    if not df.empty:
        st.sidebar.dataframe(df.tail(20), use_container_width=True)
    else:
        st.sidebar.info("No metrics recorded yet")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["💬 Q&A", "📝 Summarization", "⚖️ Key Obligations"])

with tab1:
    st.subheader("Ask any compliance question")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about regulations, risks, obligations..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = assistant.qna(prompt)
                st.markdown(result["answer"])
                st.caption(f"**Risk Tag:** {result['risk_tag']} | **Groundedness:** {result['hallucination_score']:.2f}/1.0")

                with st.expander("📚 Sources"):
                    for i, meta in enumerate(result["sources"], 1):
                        st.write(f"**[{i}]** {meta.get('filename', 'Unknown')}")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

with tab2:
    st.subheader("Document Summarization")
    docs = assistant.get_ingested_documents()
    if docs:
        selected = st.selectbox("Select document", docs)
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                result = assistant.summarize(selected)
                st.markdown(result["answer"])
    else:
        st.info("Ingest documents first")

with tab3:
    st.subheader("Extract Key Compliance Obligations")
    docs = assistant.get_ingested_documents()
    if docs:
        selected = st.selectbox("Select document", docs, key="obl")
        if st.button("Extract Obligations"):
            with st.spinner("Analyzing obligations..."):
                result = assistant.extract_obligations(selected)
                st.markdown(result["answer"])
    else:
        st.info("Ingest documents first")
