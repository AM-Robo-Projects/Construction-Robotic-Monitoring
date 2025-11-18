from neo4j import GraphDatabase
import json
import ollama
from typing import Any, Dict, List, Optional

from langchain_ollama import OllamaLLM
from BIM_processing.utils.cypher_query_utils import cypher_query_utils


class VLMGraphRAGPipeline:
    def __init__(self, neo4j_uri: str, neo4j_auth: tuple, llm_model: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
        self.llm_model = llm_model

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
                rows: List[Dict[str, Any]] = [record.data() for record in result]
                return rows

    # === Ollama chat ===========================================================
    def ollama_chat(self, model_name: str, prompt: str, image_path: Optional[str] = None):
        """
        Call Ollama chat with optional image input.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            if image_path:
                print(f"[INFO] Vision mode (path):....")
                messages[0]["images"] = [image_path]  

                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    options={
                        "temperature": 1,
                        "num_predict": -1,
                        "num_ctx": 16384
                    },
                )
            
            else:
                print(f"[INFO] Text mode (no image)....")
                messages[0]["content"] = prompt
                
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    options={
                        "temperature": 1,
                        "num_predict": -1,
                        "num_ctx": 16384
                    },
                )

            return response.get("message", {}).get("content", "")
        
            
        
        except Exception as e:
            print(f"[Ollama ERROR] {e}")
            return ""

    def run(self, model, user_question, image_path: Optional[str] = None, robot_x: float = 0.0, robot_y: float = 0.0, threshold: float = 2.0):
        llm_model = model

        cypher = self.gen_cypher(user_question)
        if not cypher:
            print("⚠️ Cypher generation failed (empty query). Check cypher_query_utils/model/prompt.")
            return

        print("🔎 Cypher:\n", cypher)

        # Prepare parameters for parameterized queries
        params = {
            "robot_x": robot_x,
            "robot_y": robot_y,
            "threshold": threshold
        }
        
        try:
            context_rows = self.get_graph_context(cypher, params=params)
        except Exception as e:
            print(f"[Neo4j ERROR] {e}")
            context_rows = []

        print(f"📊 Retrieved {len(context_rows)} rows from database")

        # Ground the model strictly in returned rows for the DOORS part
        context_json = json.dumps(context_rows, ensure_ascii=False, indent=2, default=str)
        
        print(f"[DEBUG] Context preview (first 500 chars):\n{context_json[:500]}...\n[END_DEBUG]")

        grounded_prompt = (
            "You are provided with CONTEXT from a Neo4j graph database and an IMAGE. "
            "Answer the question based STRICTLY on the CONTEXT and what you can see in the IMAGE. "
            "The CONTEXT contains ALL matching results from the database - do not assume there are more. "
            "Do NOT invent, infer, or fill gaps beyond what is provided. "
            "If CONTEXT is insufficient, reply exactly: \"I don't know based on the provided data.\"\n\n"
            f"DATABASE CONTEXT ({len(context_rows)} results):\n{context_json}\n\n"
            f"QUESTION: {user_question}\n\n"
            "Provide a structured answer with:\n"
            "1. Detection Result: YES or NO\n"
            "2. Visual Evidence: What you see in the image\n"
            "3. Database Evidence: List ALL items from the context with their properties (use exact IDs from context)\n"
            "4. Reasoning: How the evidence supports your answer\n"
            "5. Confidence: High/Medium/Low\n\n"
            "Answer:"
        )
        

        answer = self.ollama_chat(model_name=llm_model, prompt=grounded_prompt, image_path=image_path)
        print("\n📝 Answer:\n", answer)
        
        return answer


# === Main ====================================================================
if __name__ == "__main__":
    try:
        llm_model = "llama3.2-vision:11b"
        neo4j_uri = "neo4j://127.0.0.1:7687"
        neo4j_auth = ("neo4j", "CRANE_HALL")

        rag = VLMGraphRAGPipeline(neo4j_uri, neo4j_auth, llm_model)

        image_path = "workers.jpeg"

        user_question = (
            "Please list all doors from the data provided, and describe what you see in the image ? "
        )

        rag.run(llm_model, user_question,image_path)

    except Exception as e:
        print(f"[ERROR] {e}")
