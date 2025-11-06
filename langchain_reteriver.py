"""
Cypher-Only GraphRAG Pipeline (Clean + Robust Serialization + Grounded Answers)
-------------------------------------------------------------------------------
- LLM -> Cypher (via your cypher_query_utils)
- Run Cypher on Neo4j with SAFE, COMPLETE JSON serialization
- Option A: Return raw rows (JSON-serializable, keeps ids/labels/types)
- Option B: Ask LLM to answer using rows with stricter, source-aware prompt

Requirements:
  pip install neo4j langchain langchain-ollama
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Union

from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# === Config =================================================================

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "riedelbau_model")
LLM_MODEL  = "llama3.2-vision:11b"   # choose your local Ollama model

from RoboMonitoring.BIM_processing.utils.cypher_query_utils import cypher_query_utils  # your existing module


# === Types ==================================================================

class QueryResult(TypedDict, total=False):
    query: str
    cypher: str
    row_count: int
    rows: List[Dict[str, Any]]
    answer: str


# === Robust serialization helpers ===========================================

def _serialize_value(v: Any) -> Any:
    """Recursively convert Neo4j types to JSON-serializable Python primitives."""
    # Neo4j graph types
    if isinstance(v, Node):
        return {
            "_type": "node",
            "id": v.id,
            "labels": list(v.labels),
            "props": dict(v),
        }
    if isinstance(v, Relationship):
        return {
            "_type": "relationship",
            "id": v.id,
            "type": v.type,
            "start": v.start_node.id,
            "end": v.end_node.id,
            "props": dict(v),
        }
    if isinstance(v, Path):
        # Flatten a path as alternating nodes/relationships
        nodes = [ _serialize_value(n) for n in v.nodes ]
        rels  = [ _serialize_value(r) for r in v.relationships ]
        return {"_type": "path", "nodes": nodes, "relationships": rels}

    # Containers
    if isinstance(v, dict):
        return {str(k): _serialize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_serialize_value(x) for x in v]

    # Neo4j temporal/point/byte arrays often stringify well
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


def _serialize_record(rec) -> Dict[str, Any]:
    """Serialize a neo4j.Record to a dict of primitives."""
    out: Dict[str, Any] = {}
    for k in rec.keys():
        out[k] = _serialize_value(rec[k])
    return out


# === Core classes ============================================================

@dataclass
class Neo4jClient:
    uri: str
    auth: tuple
    _driver: Any = None

    def __post_init__(self) -> None:
        self._driver = GraphDatabase.driver(self.uri, auth=self.auth)

    def run(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        try:
            with self._driver.session() as sess:
                result = sess.run(cypher, params or {})
                for rec in result:
                    rows.append(_serialize_record(rec))
        except Exception as e:
            print(f"[Neo4j ERROR] {e}")
        return rows

    def close(self) -> None:
        if self._driver:
            self._driver.close()


class CypherOnlyRAG:
    """
    LLM -> Cypher -> Neo4j -> (optional) grounded answer.
    No ingestion and no vector search.
    """
    def __init__(self, neo4j_uri: str, neo4j_auth: tuple, llm_model: str) -> None:
        self.neo = Neo4jClient(neo4j_uri, neo4j_auth)
        self.llm = OllamaLLM(model=llm_model, temperature=0)
        self.cypher_utils = cypher_query_utils(
            llm=self.llm,
            neo4j_uri=neo4j_uri,
            neo4j_auth=neo4j_auth,
            vectorstore=None,   # unused in cypher-only mode
        )

        # A stricter, source-aware prompt. It:
        #  - forbids inventing fields
        #  - requires citing elementId or node id per fact
        #  - asks to say "I don't know" when missing
        self._answer_prompt = PromptTemplate(
            template=(
                "You are given a QUESTION and DATA (JSON rows) from a database.\n"
                "Answer ONLY using the DATA.\n"
                "- Do NOT invent fields or records.\n"
                "- If the answer isn't present, reply: \"I don't know based on the provided data.\" \n"
                "- When listing facts, include a short source tag using elementId or node id like [source: W-123] or [source: node 42].\n"
                "- Prefer tabular lists when appropriate.\n\n"
                "QUESTION:\n{question}\n\n"
                "DATA (JSON rows):\n{data}\n\n"
                "ANSWER:"
            ),
            input_variables=["question", "data"],
        )
        self._answer_chain = LLMChain(llm=self.llm, prompt=self._answer_prompt, verbose=False)

    def _gen_cypher(self, question: str) -> str:
        try:
            cypher = self.cypher_utils.generate_cypher_from_query(question, AutoGen=False)
            return (cypher or "").strip()
        except Exception as e:
            print(f"[CypherGen ERROR] {e}")
            return ""

    def ask(self, question: str, synthesize_answer: bool = True, max_rows_for_llm: int = 200) -> QueryResult:
        out: QueryResult = {"query": question}

        cypher = self._gen_cypher(question)
        out["cypher"] = cypher
        if not cypher:
            out["row_count"] = 0
            out["rows"] = []
            out["answer"] = "I couldn't generate a Cypher query for that request."
            return out

        rows = self.neo.run(cypher)
        out["row_count"] = len(rows)
        out["rows"] = rows  # always return full rows to the caller

        if not synthesize_answer:
            return out

        if not rows:
            out["answer"] = "I don't know based on the provided data."
            return out

        # Limit LLM context if huge (but keep full rows in `out["rows"]`)
        data_for_llm = rows[:max_rows_for_llm]
        data_str = json.dumps(data_for_llm, ensure_ascii=False, default=str)
        try:
            answer = self._answer_chain.run(question=question, data=data_str)
        except Exception as e:
            print(f"[AnswerSynthesis ERROR] {e}")
            answer = "An error occurred while generating the answer."
        out["answer"] = answer.strip()
        return out

    def close(self) -> None:
        self.neo.close()


# === Example CLI ============================================================

def main() -> None:
    rag = CypherOnlyRAG(NEO4J_URI, NEO4J_AUTH, LLM_MODEL)
    try:
        examples = [
            "can you list all windows in the building along with their positions in meters?",
        ]
        for q in examples:
            print("\n" + "=" * 80)
            print("❓ QUESTION:", q)
            result = rag.ask(q, synthesize_answer=True)

            print("\n🧠 Generated Cypher:\n", result.get("cypher", ""))
            print("\n📊 Rows returned:", result.get("row_count", 0))

            preview = result.get("rows", [])[:5]
            if preview:
                print("\n🔎 Row preview (up to 5):")
                for i, r in enumerate(preview, 1):
                    print(f"  {i}.", json.dumps(r, ensure_ascii=False, default=str))

            print("\n📝 Answer:\n", result.get("answer", ""))

    finally:
        rag.close()
        print("\n✅ Done.")


if __name__ == "__main__":
    main()
