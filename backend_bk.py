

import os
import hashlib
import time
import json
import datetime
from typing import List, Dict
import pandas as pd

import chromadb
import httpx
import numpy as np
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from pdf2image import convert_from_path
import pytesseract

load_dotenv()


class ComplianceAssistant:
    def __init__(self, llm_model: str = "genailab-maas-gpt-4o"):
        self.client = httpx.Client(verify=False, timeout=120.0)

        # LLM - You can change model from your available list
        self.llm = ChatOpenAI(
            base_url="https://genailab.tcs.in",
            model=llm_model,
            api_key=os.getenv("api_key"),
            http_client=self.client
        )

        # Embedding Model (fixed as per your environment)
        self.embedding_model = OpenAIEmbeddings(
            base_url="https://genailab.tcs.in",
            model="azure/genailab-maas-text-embedding-3-large",
            api_key=os.getenv("api_key"),
            http_client=self.client
        )

        # ChromaDB Vector Store
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="compliance_docs")

        # Semantic Chunker
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )

        # CSV Files for persistence (replacing SQLite)
        self.ingested_file = "ingested_docs.csv"
        self.metrics_file = "pipeline_metrics.csv"
        self.judge_file = "hallucination_judges.csv"

        self._init_csv_files()

    def _init_csv_files(self):
        """Create CSV files with proper headers if they don't exist"""
        if not os.path.exists(self.ingested_file):
            pd.DataFrame(columns=["md5_hash", "filename", "num_chunks", "ingest_timestamp"]).to_csv(self.ingested_file, index=False)

        if not os.path.exists(self.metrics_file):
            pd.DataFrame(columns=["query_or_doc", "stage", "latency_ms", "timestamp"]).to_csv(self.metrics_file, index=False)

        if not os.path.exists(self.judge_file):
            pd.DataFrame(columns=["query", "answer", "context_snippet", "hallucination_score", "judge_reason", "timestamp"]).to_csv(self.judge_file, index=False)

    def _record_metric(self, query_or_doc: str, stage: str, latency_ms: float):
        """Record pipeline latency in CSV"""
        new_row = pd.DataFrame([{
            "query_or_doc": str(query_or_doc)[:200],
            "stage": stage,
            "latency_ms": round(latency_ms, 2),
            "timestamp": datetime.datetime.now().isoformat()
        }])
        if os.path.exists(self.metrics_file):
            df = pd.read_csv(self.metrics_file)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.metrics_file, index=False)

    def _is_already_ingested(self, md5_hash: str) -> bool:
        if not os.path.exists(self.ingested_file):
            return False
        df = pd.read_csv(self.ingested_file)
        return md5_hash in df["md5_hash"].values

    def _record_ingestion(self, md5_hash: str, filename: str, num_chunks: int):
        new_row = pd.DataFrame([{
            "md5_hash": md5_hash,
            "filename": filename,
            "num_chunks": num_chunks,
            "ingest_timestamp": datetime.datetime.now().isoformat()
        }])
        if os.path.exists(self.ingested_file):
            df = pd.read_csv(self.ingested_file)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.ingested_file, index=False)

    def get_ingested_documents(self) -> List[str]:
        if not os.path.exists(self.ingested_file):
            return []
        df = pd.read_csv(self.ingested_file)
        return df["filename"].tolist()

    # ====================== Text Extraction with OCR ======================
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF (with OCR fallback), TXT, and Excel"""
        ext = file_path.lower()
        text = ""

        if ext.endswith('.pdf'):
            # Try normal text extraction first
            try:
                reader = PdfReader(file_path)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            except:
                text = ""

            # If text is too short → likely scanned PDF → use OCR
            if len(text.strip()) < 300:
                try:
                    images = convert_from_path(file_path, dpi=300)
                    for i, img in enumerate(images):
                        text += f"\n--- Page {i+1} (OCR) ---\n"
                        text += pytesseract.image_to_string(img) + "\n"
                except Exception as e:
                    text += f"\n[OCR Error: {str(e)}]"

        elif ext.endswith(('.xlsx', '.xls')):
            df_dict = pd.read_excel(file_path, sheet_name=None)
            for sheet, df in df_dict.items():
                text += f"\n=== Sheet: {sheet} ===\n"
                text += df.to_markdown(index=False) + "\n"
        else:
            # TXT files
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        return text.strip()

    # ====================== Ingestion ======================
    def ingest_document(self, file_path: str) -> str:
        """Ingest document with MD5 deduplication + Semantic Chunking"""
        start_time = time.perf_counter()

        # Generate MD5 hash for deduplication
        with open(file_path, "rb") as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()

        filename = os.path.basename(file_path)

        if self._is_already_ingested(md5_hash):
            return f"✅ Already ingested (duplicate): {filename}"

        text = self.extract_text(file_path)

        # Semantic Chunking
        langchain_docs = self.semantic_chunker.create_documents([text])
        documents = [doc.page_content for doc in langchain_docs]
        metadatas = [{"filename": filename, "chunk_id": i, "chunk_type": "semantic"} for i in range(len(documents))]
        ids = [f"{filename}_{i}" for i in range(len(documents))]

        # Embed and store in ChromaDB
        embeddings = self.embedding_model.embed_documents(documents)
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        latency = (time.perf_counter() - start_time) * 1000
        self._record_metric(filename, "ingestion_full", latency)
        self._record_ingestion(md5_hash, filename, len(documents))

        return f"✅ Ingested: {filename} → {len(documents)} semantic chunks"

    # ====================== Retrieval - FIXED ======================
    def retrieve_rrf(self, query: str, top_k: int = 12) -> List[Dict]:
        """Reciprocal Rank Fusion (Vector + BM25) - Fixed for ChromaDB"""
        start_time = time.perf_counter()

        # Vector Search
        query_emb = self.embedding_model.embed_query(query)
        vec_res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k * 4,
            include=["documents", "metadatas", "distances"]   # "ids" removed - causes error
        )

        docs = vec_res["documents"][0]
        metas = vec_res["metadatas"][0]
        distances = vec_res.get("distances", [[]])[0]
        ids_list = vec_res["ids"][0]   # IDs are returned by default

        # Vector ranks
        vec_rank = {ids_list[i]: i + 1 for i in range(len(ids_list))}

        # BM25 ranks
        tokenized = [d.split() for d in docs]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.split())
        bm25_idx = np.argsort(bm25_scores)[::-1]
        bm25_rank = {ids_list[bm25_idx[i]]: i + 1 for i in range(len(ids_list))}

        # Reciprocal Rank Fusion (RRF)
        k = 60
        rrf_scores = []
        for i, doc_id in enumerate(ids_list):
            r_v = vec_rank.get(doc_id, 9999)
            r_b = bm25_rank.get(doc_id, 9999)
            score = 1.0 / (k + r_v) + 1.0 / (k + r_b)
            rrf_scores.append((score, i))

        rrf_scores.sort(reverse=True)

        # Prepare final candidates
        candidates = []
        for _, idx in rrf_scores[:top_k]:
            candidates.append({
                "doc": docs[idx],
                "meta": metas[idx]
            })

        latency = (time.perf_counter() - start_time) * 1000
        self._record_metric(query[:100], "retrieval_rrf", latency)
        return candidates

    # ====================== Hallucination Judge ======================
    def _judge_hallucination(self, query: str, context: str, answer: str) -> Dict:
        prompt = f"""Context:\n{context[:5000]}\n\nAnswer:\n{answer}

Rate the hallucination score from 0.0 (completely hallucinated) to 1.0 (fully grounded).
Return ONLY valid JSON: {{"score": 0.85, "reason": "brief explanation"}}"""

        try:
            resp = self.llm.invoke([("system", "You are a strict hallucination judge for regulatory content."), ("human", prompt)])
            data = json.loads(resp.content.strip().strip("```json").strip("```"))
            score = float(data.get("score", 0.5))
            reason = data.get("reason", "N/A")
        except:
            score, reason = 0.5, "Judge parsing failed"

        # Save judge result to CSV
        new_row = pd.DataFrame([{
            "query": query[:200],
            "answer": answer[:300],
            "context_snippet": context[:200],
            "hallucination_score": round(score, 3),
            "judge_reason": reason,
            "timestamp": datetime.datetime.now().isoformat()
        }])
        if os.path.exists(self.judge_file):
            df = pd.read_csv(self.judge_file)
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.judge_file, index=False)

        return {"score": score, "reason": reason}

    # ====================== Response Generation ======================
    def generate_response(self, query: str, context_docs: List[Dict], task: str = "qna") -> Dict:
        start_time = time.perf_counter()

        context = "\n\n".join([
            f"Source {i+1} ({d['meta'].get('filename', 'unknown')}):\n{d['doc']}"
            for i, d in enumerate(context_docs)
        ])

        if task == "qna":
            user_input =  """You are a Strict Regulatory Compliance Assistant for Capital Markets regulations ONLY.

You MUST follow these rules in exact order. These rules override everything else.

RULE 1 (Highest Priority):
- You can ONLY use information from the "CONTEXT" section below.
- You have ZERO external knowledge. You do not know anything outside the CONTEXT.

RULE 2:
- Read the user query.
- Check if the CONTEXT contains direct, explicit information to answer the query about capital markets regulations.

RULE 3 (Refusal Rule):
- If the query is NOT strictly about capital markets regulations, OR
- If the CONTEXT does not contain sufficient relevant information,
- Then you MUST output EXACTLY this sentence and nothing else:
"I could not find any reference to this in the ingested regulatory documents."

Do not add explanations, do not be helpful, do not say "based on" or anything similar. Just the exact sentence above.

RULE 4 (Only if you pass Rule 3):
- Answer using ONLY exact information from the CONTEXT.
- Never add, assume, or infer anything not explicitly stated.
- Cite every piece of information as [Source X] where X is the source number in the CONTEXT.

RULE 5:
- At the very end of your response (even for refusals), add exactly this line and nothing after it:
**Risk Tag: Low/Medium/High**

CONTEXT:
{context}

User Query: {query}
"""
            # user_input = f"Question: {query}\n\nContext:\n{context}"
        elif task == "summarize":
            system_prompt = "You are an expert summarizer for capital market regulatory documents. Provide a clear and concise summary."
            user_input = f"Document content:\n{context}"
        else:  # obligations
            system_prompt = "Extract and list ALL key compliance obligations, requirements, deadlines, and penalties in clear bullet points."
            user_input = f"Document content:\n{context}"
        try:
            response = self.llm.invoke([("system", system_prompt), ("human", user_input)])
            answer = response.content

            latency = (time.perf_counter() - start_time) * 1000
            self._record_metric(query if task == "qna" else "document", f"generation_{task}", latency)

            judge = self._judge_hallucination(query, context[:4000], answer) if task == "qna" else {"score": 1.0, "reason": "N/A"}

            return {
                "answer": answer,
                "sources": [d["meta"] for d in context_docs],
                "risk_tag": "High" if "high" in answer.lower() else "Medium" if "medium" in answer.lower() else "Low",
                "hallucination_score": judge["score"],
                "judge_reason": judge["reason"]
            }
        except Exception as default_response:
            return {"answer": "Sytem has no clue about what you're asking !!",
                    "sources": [],
                    "risk_tag": 'Very High',
                    "hallucination_score": 1.0,
                    "judge_reason": 'Undefined'
                    }

    # ====================== Public APIs ======================
    def qna(self, query: str) -> Dict:
        """Main Q&A Pipeline"""
        candidates = self.retrieve_rrf(query)
        return self.generate_response(query, candidates, task="qna")

    def summarize(self, filename: str) -> Dict:
        """Summarize a specific document"""
        res = self.collection.get(where={"filename": filename}, include=["documents", "metadatas"])
        docs = [{"doc": d, "meta": m} for d, m in zip(res["documents"], res["metadatas"])]
        return self.generate_response("Summarize this document", docs, task="summarize")

    def extract_obligations(self, filename: str) -> Dict:
        """Extract key obligations from a document"""
        res = self.collection.get(where={"filename": filename}, include=["documents", "metadatas"])
        docs = [{"doc": d, "meta": m} for d, m in zip(res["documents"], res["metadatas"])]
        return self.generate_response("Extract key compliance obligations", docs, task="obligations")

    def get_metrics(self) -> pd.DataFrame:
        """Return pipeline metrics as DataFrame"""
        if os.path.exists(self.metrics_file):
            return pd.read_csv(self.metrics_file)
        return pd.DataFrame()