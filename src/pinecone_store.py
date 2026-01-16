from __future__ import annotations

import os
from typing import List, Dict, Any, cast

from pinecone import Pinecone, ServerlessSpec, Metric, VectorType

DEFAULT_DIM = 1536  # for text-embedding-3-small

from pinecone import ServerlessSpec

def _serverless_spec_from_env() -> ServerlessSpec:
    cloud = os.getenv("PINECONE_CLOUD", "aws").lower()
    region = os.getenv("PINECONE_REGION", "us-west-2")
    return ServerlessSpec(cloud=cloud, region=region)


class PineconeVectorStore:
    def __init__(self, pc: Pinecone, index_name: str):
        self.pc = pc
        self.index_name = index_name
        self.index = pc.Index(index_name)

    @staticmethod
    def from_env() -> "PineconeVectorStore":
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing PINECONE_API_KEY in environment (.env).")

        index_name = os.getenv("PINECONE_INDEX_NAME", "acmecorp-rag-demo")
        pc = Pinecone(api_key=api_key)

        existing = [i["name"] for i in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=int(os.getenv("PINECONE_DIM", str(DEFAULT_DIM))),
                metric="cosine",
                spec=_serverless_spec_from_env(),
            )


        return PineconeVectorStore(pc, index_name)

    def upsert_chunks(self, chunks: List[Dict[str, Any]], vectors: List[List[float]], namespace: str = "default") -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        payload = []
        for c, v in zip(chunks, vectors):
            payload.append((c["id"], v, c["metadata"] | {"text": c["text"]}))

        # Pinecone expects list of tuples: (id, vector, metadata)
        self.index.upsert(vectors=payload, namespace=namespace)
        return len(payload)


    def query(self, query_vector: List[float], top_k: int = 6, namespace: str = "default") -> List[Dict[str, Any]]:
        res = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )

        # Pinecone SDK versions differ: sometimes res.matches, sometimes res["matches"].
        raw_matches = getattr(res, "matches", None)

        if raw_matches is None and isinstance(res, dict):
            res_dict = cast(Dict[str, Any], res)
            raw_matches = res_dict.get("matches", [])

        if raw_matches is None:
            raw_matches = []

        matches: List[Dict[str, Any]] = []
        for m in raw_matches:
            # m can be an object or dict depending on SDK
            m_id = getattr(m, "id", None) if not isinstance(m, dict) else m.get("id")
            m_score = getattr(m, "score", None) if not isinstance(m, dict) else m.get("score")
            m_meta = getattr(m, "metadata", None) if not isinstance(m, dict) else m.get("metadata", {}) or {}
            if m_meta is None:
                m_meta = {}

            matches.append({
                "id": m_id,
                "score": float(m_score) if m_score is not None else 0.0,
                "text": m_meta.get("text", ""),
                "metadata": {
                    "source": m_meta.get("source"),
                    "page": m_meta.get("page"),
                    "chunk_id": m_meta.get("chunk_id"),
                }
            })

        return matches

