# Resolve AI — Autonomous Support & Scheduling Assistant

Resolve AI is an end-to-end, self-hosted customer support agent that combines **QLoRA fine-tuning**, **vLLM inference**, **Qdrant semantic search (RAG)**, and **Model Context Protocol (MCP)** tool orchestration.

## Architecture

```text
User Query
   │
   ▼
LangChain Agent (src/agent.py)
   ├──► Fine-Tuned Qwen2.5-3B via vLLM (:8000, OpenAI-compatible API + Hermes tool parser)
   ├──► Qdrant Vector DB (BAAI/bge-small-en-v1.5 embeddings over company_faq.txt)
   └──► FastMCP Scheduler Server (:8001, Streamable HTTP JSON-RPC 2.0)
          ├── check_availability(date_time)
          └── book_customer_call(customer_name, email, date_time, issue_summary)
```

## Project Structure

```text
resolve-ai/
├── data/
│   ├── company_faq.txt          # Domain knowledge base for RAG retrieval
│   └── support_data.jsonl       # ChatML dataset for QLoRA fine-tuning
├── scripts/
│   └── start_vllm.sh            # vLLM server launch script with API key auth
├── src/
│   ├── __init__.py
│   ├── embeddings.py            # Local BGE sentence-transformers embedding wrapper
│   ├── train_lora.py            # 4-bit QLoRA fine-tuning & FP16 weight merging
│   ├── build_vector_db.py       # Document chunking & local Qdrant collection builder
│   ├── scheduler_mcp.py         # FastMCP scheduling microservice
│   └── agent.py                 # LangChain agent + async-safe MCP HTTP client
├── .env.example                 # Environment configuration template
├── .gitignore                   # Excludes model weights, vector store, logs, and venv
└── requirements.txt             # Python dependencies
```

## Quickstart

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Fine-Tune & Merge Model Weights
Trains a 4-bit QLoRA adapter on `data/support_data.jsonl` and merges it into `./resolve-final-model`:
```bash
python -m src.train_lora
```

### 3. Build Local Qdrant Vector Database
Indexes `data/company_faq.txt` into `./qdrant_local_db` using `BAAI/bge-small-en-v1.5`:
```bash
python -m src.build_vector_db
```

### 4. Start Inference & MCP Services
In Terminal 1, launch the vLLM server:
```bash
bash scripts/start_vllm.sh
```

In Terminal 2, launch the FastMCP scheduling server:
```bash
python -m src.scheduler_mcp
```

### 5. Run the Resolve AI Agent
In Terminal 3:
```bash
python -m src.agent
```