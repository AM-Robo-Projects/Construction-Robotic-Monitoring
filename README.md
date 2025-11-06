# 🤖 Construction Robotic Monitoring

A modular framework for **robotic perception, navigation, and scene understanding** in construction environments.  
This project bridges **Building Information Modeling (BIM)** data and **robotic AI systems**, enabling robots to reason about, navigate, and monitor construction sites using structured knowledge graphs derived from IFC models.

---

## 🚀 Overview

Construction-Robotic-Monitoring provides an end-to-end pipeline that:

1. **Parses IFC files** (using [IfcOpenShell](https://ifcopenshell.org/)) to extract detailed 3D building information.  
2. **Converts BIM data** into a **Knowledge Graph** (nodes = building elements, edges = spatial/semantic relationships).  
3. **Stores** and queries the graph efficiently using **Neo4j**.  
4. **Integrates LLMs and Vision-Language Models (VLMs)** (e.g., via [Ollama](https://ollama.ai/)) for reasoning and question-answering over the building structure.  
5. **Interfaces with ROS 2** for robotic tasks like monitoring, navigation, and environment awareness.

---

## 🧩 Features

- ✅ IFC-to- Knowledge Graph conversion with geometry, materials, and spatial relations  
- ✅ Fast Neo4j querying via via Cypher and LLM  
- ✅ LangChain-based integration with local LLMs (e.g., LLaMA 3, Qwen 2.5, Mistral)  
- ✅ Visual and language-aware graph retrieval (VLM + Neo4j)  - GraphRAG
- ✅ Modular design compatible with ROS 2 nodes  
- ✅ IFC analytics: extract walls, doors, windows, storeys, and spatial connectivity  

---

## 🏗️ System Architecture

      ┌──────────────────────────────┐
      │        IFC Model (.ifc)      │
      └──────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  IFC Parser (IfcOpenShell) │
        └──────────────┬─────────────┘
                       ▼
        ┌────────────────────────────┐
        │   Graph Extractor (Nodes,  │
        │   Relationships, Geometry) │
        └──────────────┬─────────────┘
                       ▼
            ┌──────────────────┐
            │   Neo4j GraphDB  │
            └───────┬──────────┘
                    ▼
    ┌────────────────────────────────────┐
    │  LLM / VLM Reasoning via Ollama    │
    │  (LangChain, GraphRAG Retrieval)   │
    └────────────────────────────────────┘


---

## ⚙️ Installation

### Prerequisites
- Python ≥ 3.10  
- [IfcOpenShell](https://github.com/IfcOpenShell/IfcOpenShell)  
- [Neo4j Desktop](https://neo4j.com/download/) or AuraDB  
- [Ollama](https://ollama.ai) (for local LLMs)  
- ROS 2 (optional, for robot integration)

### Setup

```bash
git clone https://github.com/AM-Robo-Projects/Construction-Robotic-Monitoring.git
cd Construction-Robotic-Monitoring
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


