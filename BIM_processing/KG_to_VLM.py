from neo4j import GraphDatabase
import json
import ollama
from typing import Any, Dict, List, Optional

from langchain_ollama import OllamaLLM
from utils.cypher_query_utils import cypher_query_utils


class VLMGraphRAGPipeline:
    def __init__(self, neo4j_uri: str, neo4j_auth: tuple, llm_model: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
        self.llm_model = llm_model

        # IMPORTANT: pass a REAL LangChain LLM object (not a string) to cypher_query_utils

        self._cypher_llm = OllamaLLM(model="qwen2.5-coder:7b", temperature=0)
        self.cypher_utils = cypher_query_utils(
            llm=self._cypher_llm,
            neo4j_uri=self.neo4j_uri,
            neo4j_auth=self.neo4j_auth,
            vectorstore=None
        )

    # === Cypher generation from query =========================================
    def gen_cypher(self, question: str) -> str:
        try:
            cypher = self.cypher_utils.generate_cypher_from_query(question, AutoGen=False)
            return (cypher or "").strip()
        except Exception as e:
            print(f"[CypherGen ERROR] {e}")
            return ""

    # === Neo4j execution =======================================================
    def get_graph_context(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run Cypher and return a list of JSON-serializable rows (dicts)."""
        if not cypher_query:
            raise ValueError("Cannot run an empty query (Cypher generation returned empty).")

        with GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth) as driver:
            with driver.session() as session:
                result = session.run(cypher_query, params or {})
                rows: List[Dict[str, Any]] = []
                for record in result:
                    rows.append(record.data())
                return rows

    # === Ollama chat ===========================================================
    def ollama_chat(self, model_name: str, prompt: str, image_path: Optional[str] = None) -> str:
        try:
            common = {
                "model": model_name,
                "options": {
                    "temperature": 0,
                    "num_predict": -1,  # allow full-length answers
                },
            }
            if image_path:
                response = ollama.chat(**{
                    **common,
                    "messages": [{"role": "user", "content": prompt, "images": [image_path]}],
                })
            else:
                response = ollama.chat(**{
                    **common,
                    "messages": [{"role": "user", "content": prompt}],
                })
            return response.get("message", {}).get("content", "")
        except Exception as e:
            print(f"[Ollama ERROR] {e}")
            return ""


def run() -> None:
    llm_model = "llama3.2-vision:11b"
    neo4j_uri = "neo4j://127.0.0.1:7687"
    neo4j_auth = ("neo4j", "riedelbau_model")

    rag = VLMGraphRAGPipeline(neo4j_uri, neo4j_auth, llm_model)

    user_question = (
        "can you list 134 doors in the building and their positions"
        "if you don't know please say I don't know based on the provided data"
        "Please don't provide any code just use the available data you have, in case the number doesn't match the data you have just provide the available data"
    )

    cypher = rag.gen_cypher(user_question)
    if not cypher:
        print("⚠️ Cypher generation failed (empty query). Check cypher_query_utils/model/prompt.")
        return

    print("🔎 Cypher:\n", cypher)

    try:
        context_rows = rag.get_graph_context(cypher)
    except Exception as e:
        print(f"[Neo4j ERROR] {e}")
        context_rows = []

    print("📊 Row count:", len(context_rows))

    # Ground the model strictly in returned rows
    context_json = json.dumps(context_rows, ensure_ascii=False, default=str)

    grounded_prompt = (
        "You are given DATA from a graph database and a QUESTION.\n"
        "Answer ONLY using the DATA. If not present, reply exactly: "
        "\"I don't know based on the provided data.\"\n"
        "Prefer bullet points. Use values exactly as shown.\n\n"
        f"DATA:\n{context_json}\n\n"
        f"QUESTION:\n{user_question}\n\n"
        "ANSWER:"
    )

    answer = rag.ollama_chat(model_name=llm_model, prompt=grounded_prompt)
    print("\n📝 Answer:\n", answer)


# === Main ====================================================================
if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[ERROR] {e}")
