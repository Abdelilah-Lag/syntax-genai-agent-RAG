# GENAI AGENT — RAG SYSTEM

**ENSAM RABAT × SYNTAX · TECHNICAL CASE STUDY 2026**

An Agentic Retrieval-Augmented Generation (RAG) system designed to answer questions using a curated medical knowledge corpus. It combines hybrid retrieval, tool-based reasoning, evidence-grounded generation, and automated answer evaluation.

---

## 1. Project Overview

### Problem Statement

Traditional Large Language Models (LLMs) can generate fluent answers, but they may also produce inaccurate or unsupported information, particularly in specialized domains such as healthcare.

A reliable knowledge-based system must therefore:

- Retrieve relevant information from trusted documents.
- Ground generated answers in retrieved evidence.
- Support different question types, including factual, comparative, and multi-hop questions.
- Identify insufficient evidence and avoid unsupported conclusions.
- Evaluate retrieval quality, answer quality, reliability, and system performance.

### Proposed Solution

This project implements an **Agentic RAG architecture** in which an LLM coordinates a set of specialized tools to answer questions using a medical document corpus.

The system combines:

- **Hybrid Retrieval:** dense vector retrieval and BM25 keyword search.
- **Agentic Orchestration:** an explicit planner selects the tools required for each question.
- **Specialized Tools:** search, extraction, comparison, interaction analysis, glossary, and verification.
- **Evidence Management:** retrieved information is collected and used as the basis for answer generation.
- **Quality Control:** a Critic evaluates generated answers and can trigger a bounded revision.
- **Evaluation Framework:** a curated 40-question benchmark measures retrieval, answer quality, reliability, and latency.

The objective is to make answers more traceable, evidence-based, and reliable than direct LLM generation.

---

## 2. System Architecture

```mermaid
flowchart TD
    A[User Question] --> B[FastAPI]
    B --> C[Agent Orchestrator]
    C --> D[Planner]
    D --> E[Tool Selection]

    E --> F[Search Tool]
    E --> G[Extract Tool]
    E --> H[Compare Tool]
    E --> I[Interaction Tool]
    E --> J[Glossary Tool]
    E --> K[Verify Tool]

    F --> L[Hybrid Retrieval]
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L

    L --> M[Medical Knowledge Corpus]
    M --> N[Evidence Store]

    F --> N
    G --> N
    H --> N
    I --> N
    J --> N
    K --> N

    N --> O[Answer Generator]
    O --> P[Critic]
    P -->|Revision Required| O
    P -->|Accepted or Bounded Result| Q[Final Answer]
    Q --> B
    B --> R[Frontend]
```

### Main Components

| Component | Responsibility |
|---|---|
| FastAPI | Exposes the application API and serves the frontend |
| Planner | Produces a structured execution plan |
| Agent Orchestrator | Coordinates tool execution and answer generation |
| Hybrid Retriever | Retrieves relevant documents using semantic and keyword search |
| Evidence Store | Collects the evidence used by the agent |
| Generator | Produces evidence-grounded answers |
| Critic | Reviews answer quality and grounding |
| Evaluation Module | Computes evaluation metrics and benchmark results |

---

## 3. Specialized Agent Tools

The agent has six specialized tools.

| Tool | Purpose |
|---|---|
| `search` | Retrieves relevant documents and evidence |
| `extract` | Extracts specific information from documents |
| `compare` | Compares information across multiple documents or medications |
| `interaction` | Analyzes potential medication interactions based on available evidence |
| `glossary` | Explains medical terminology |
| `verify` | Checks claims against available evidence |

The planner determines which tools are appropriate for a question. Search is the mandatory retrieval tool.

Tool execution is coordinated by the agent according to the generated execution plan.

---

## 4. Project Structure

The following tree represents the actual project organization.

```text
syntax-genai-agent-RAG/
│
├── app/
│   │
│   ├── main.py
│   ├── run_eval.py
│   ├── public_eval_set.json
│   ├── __init__.py
│   │
│   ├── agent/
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── planner.py
│   │   ├── prompts.py
│   │   ├── state.py
│   │   └── __init__.py
│   │
│   ├── api/
│   │   ├── metrics.py
│   │   ├── routes.py
│   │   └── __init__.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── llm.py
│   │   └── __init__.py
│   │
│   ├── evaluation/
│   │   ├── critic.py
│   │   ├── metrics.py
│   │   └── __init__.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── indexer.py
│   │   ├── loader.py
│   │   └── __init__.py
│   │
│   ├── results/
│   │   └── evaluation_20261113_015533.json
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── __init__.py
│   │
│   ├── schemas/
│   │   ├── response.py
│   │   └── __init__.py
│   │
│   └── tools/
│       ├── base.py
│       ├── compare_tool.py
│       ├── extract_tool.py
│       ├── glossary_tool.py
│       ├── interaction_tool.py
│       ├── registry.py
│       ├── search_tool.py
│       ├── verify_tool.py
│       └── __init__.py
│
├── corpus/
│   ├── dailymed-*.md
│   ├── medlineplus-*.md
│   └── GLOSSARY.md
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── images/
│       ├── ensam.jpg
│       └── syntax.png
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

### Directory Responsibilities

#### `app/agent/` — Agent orchestration

Contains the agent graph, execution nodes, planner, prompts, and shared agent state.

#### `app/api/` — API layer

Contains API routes and metric-related endpoints exposed to the frontend.

#### `app/core/` — Core configuration

Manages application configuration and LLM initialization.

#### `app/evaluation/` — Evaluation and quality control

Contains the Critic and evaluation metrics.

- `critic.py`: reviews generated answers and their grounding.
- `metrics.py`: implements evaluation metrics used to assess system performance.

#### `app/ingestion/` — Document ingestion

Responsible for loading documents, splitting them into chunks, and preparing the retrieval index.

- `loader.py`: loads source documents.
- `chunker.py`: splits documents into retrieval chunks.
- `indexer.py`: prepares the retrieval index.

#### `app/retrieval/` — Information retrieval

Implements embeddings and document retrieval.

- `embeddings.py`: handles document embeddings.
- `retriever.py`: retrieves relevant information from the indexed corpus.

#### `app/tools/` — Specialized tools

Contains the six tools available to the agent, their common interface, and their registry.

#### `app/schemas/` — Data structures

Defines the response schemas used by the application.

#### `app/results/` — Evaluation outputs

Stores generated evaluation reports in JSON format.

Each report preserves the results of a particular benchmark execution, making it possible to compare system performance across different versions.

#### `corpus/` — Knowledge base

Contains the medical documents used as the source of information.

The corpus includes DailyMed and MedlinePlus documents for medications such as warfarin, apixaban, metformin, sertraline, aspirin, and others.

#### `frontend/` — User interface

Contains the web interface, styling, JavaScript logic, and ENSAM/Syntax visual assets.

### Main Evaluation Files

| File | Location | Responsibility |
|---|---|---|
| `public_eval_set.json` | `app/` | Defines the 40 benchmark questions and expected source documents |
| `run_eval.py` | `app/` | Executes the benchmark against the running API |
| `critic.py` | `app/evaluation/` | Reviews generated answers |
| `metrics.py` | `app/evaluation/` | Computes evaluation metrics |
| `evaluation_*.json` | `app/results/` | Stores generated evaluation reports |

---

## 5. Evaluation Framework

The project includes a curated benchmark of **40 questions** specifically selected to evaluate the Agentic RAG system across different question types and evaluation scenarios.

### Benchmark Objectives

The benchmark evaluates several important aspects of the system:

- Retrieval relevance and source selection.
- Answer completeness and required information.
- Evidence grounding and citation quality.
- Correct use of specialized tools.
- Ability to recognize missing or insufficient evidence.
- Planner execution and tool orchestration.
- Response latency and overall execution reliability.

### Question Categories

The benchmark covers different types of questions:

| Category | Evaluation Objective |
|---|---|
| Factual | Retrieve specific information from medical documents |
| Comparative | Compare information across medications and documents |
| Medication interaction | Identify potential interactions based on corpus evidence |
| Multi-document | Combine information from multiple sources |
| Glossary | Explain medical terminology |
| Evidence verification | Check whether claims are supported by retrieved documents |
| Abstention | Recognize missing or insufficient evidence |

The questions are defined in `app/public_eval_set.json`, together with their expected source documents where applicable.

### Evaluation Workflow

1. Load the 40 benchmark questions from `app/public_eval_set.json`.
2. Submit each question to the running agent API.
3. Execute the planner and selected tools.
4. Retrieve relevant evidence from the medical corpus.
5. Generate and review the answer.
6. Calculate evaluation metrics.
7. Save the resulting report in `app/results/`.

This separation distinguishes the benchmark definition, evaluation execution, runtime quality control, and generated reports.

### Running the 40-Question Benchmark

The benchmark evaluates the complete RAG pipeline through the running application API.

**Step 1 — Start the application**

Build the Docker image if it has not already been built:

```bash
docker build -t agent .
```

Start the application:

```bash
docker run --rm -p 8000:8000 --env-file .env agent
```

**Step 2 — Execute the benchmark**

Open another terminal at the project root and run:

```bash
python -m app.run_eval --url http://localhost:8000
```

The evaluation script sends the 40 benchmark questions to the running API, collects the agent's responses, computes the evaluation metrics, and generates a JSON report.

**Step 3 — Check the generated report**

Evaluation reports are stored in:

```text
app/
└── results/
    └── evaluation_YYYYMMDD_HHMMSS.json
```

For example:

```text
app/results/evaluation_20261113_015533.json
```

To list the generated reports using PowerShell:

```powershell
Get-ChildItem app/results/
```

Each report preserves the results of a particular execution, allowing performance comparisons and regression analysis.

**Important:** The benchmark command evaluates the running application. It does not start the application itself. Make sure the API is accessible at `http://localhost:8000` before starting the evaluation.

---

## 6. Evaluation Results

**Baseline evaluation: 13 November 2026**

The following results were obtained from the 40-question benchmark.

### Overall Results

| Metric | Result |
|---|---:|
| Total questions | 40 |
| Successful executions | 40 / 40 |
| Execution success rate | 100% |
| Average source recall | 95.61% |
| Average source precision | 83.38% |
| Average source F1 | 87.12% |
| Average grounding score | 89.92% |
| Required term recall | 95.00% |
| Source policy compliance | 90.00% |
| Correct abstention rate | 100% |
| Unsupported answer rate | 0% |
| False refusal rate | 0% |
| Planner plan validity | 100% |
| Planner fallback rate | 0% |

### Critic and Answer Quality

| Metric | Result |
|---|---:|
| Critic pass rate | 55.88% |
| Critic review availability | 89.47% |
| Answer status accuracy | 52.50% |
| Total revisions | 19 |
| Answers with medical disclaimer | 100% |
| Duplicate citations | 0 |

The Critic pass rate indicates that a significant proportion of answers still require improvement or fail the configured quality checks.

Answer status accuracy also requires attention: the system should better distinguish between supported answers, insufficient evidence, and cases where no documents were retrieved.

### Latency and Performance

| Metric | Result |
|---|---:|
| Average latency | 20.86 seconds |
| Median latency | 19.14 seconds |
| P95 latency | 38.35 seconds |
| P99 latency | 49.13 seconds |
| Maximum latency | 49.13 seconds |
| Requests over 30-second budget | 12 / 40 |
| 30-second latency budget compliance | 70% |

### Latency by Question Complexity

| Question Type | Questions | Average Latency |
|---|---:|---:|
| Simple | 14 | 13.60 s |
| Multi-document | 7 | 19.63 s |
| Complex | 13 | 28.65 s |
| Multi-hop | 6 | 22.36 s |

### Tool Usage

| Tool | Number of Uses |
|---|---:|
| Search | 39 |
| Extract | 30 |
| Compare | 10 |
| Interaction | 7 |
| Glossary | 4 |
| Verify | 1 |

All six tools were used during the benchmark.

### Planner and Execution Metrics

| Metric | Result |
|---|---:|
| Planner plans generated | 39 |
| Planner plan validity | 100% |
| Planner fallback rate | 0% |
| Planned tool execution rate | 100% |
| Questions using at least one tool | 39 / 40 |
| Total revisions | 19 |

### Interpretation

The evaluation demonstrates that the system successfully executed all 40 benchmark questions and achieved strong retrieval performance.

The source recall of 95.61% indicates that the retrieval pipeline generally identifies the expected documents. The grounding score of 89.92% provides an additional signal about how well answers are supported by retrieved evidence.

The planner also demonstrated consistent execution, with a 100% valid plan rate and no fallback during the benchmark.

However, the results identify several areas for improvement:

- Critic pass rate: 55.88%.
- Answer status accuracy: 52.50%.
- Latency budget compliance: 70%.
- Source policy compliance: 90%.

The results are based on a limited benchmark and should not be interpreted as clinical validation or proof of medical correctness.

A 0% unsupported answer rate on this benchmark does not guarantee that the system will never generate unsupported answers on unseen questions. The two unsupported-evidence scenarios in the benchmark are also too few to establish general reliability.

---

## 7. Limitations and Improvement Roadmap

### Current Limitations

**1. Answer status classification**

The answer status accuracy of 52.50% indicates that the system needs better classification of answer outcomes.

**2. Critic reliability**

The Critic pass rate of 55.88% suggests that answer review criteria and revision behavior should be improved.

**3. Retrieval precision**

Source precision of 83.38% leaves room to reduce irrelevant document retrieval.

**4. Latency**

Only 70% of requests met the global 30-second latency budget. Simple and multi-document questions should be optimized first.

**5. Evaluation coverage**

A 40-question benchmark is useful for development but is not sufficient to establish general reliability across all medical questions.

### Proposed Improvements

| Area | Proposed Improvement |
|---|---|
| Retrieval | Improve chunking, metadata filtering, reranking, and top-k selection |
| Planner | Improve tool selection and reduce unnecessary tool calls |
| Critic | Refine evaluation criteria and revision prompts |
| Generation | Improve evidence attribution and structured answer generation |
| Latency | Optimize retrieval, parallelize independent operations, and reduce redundant LLM calls |
| Abstention | Improve detection of missing or insufficient evidence |
| Evaluation | Expand the benchmark and introduce regression tests |
| Observability | Track tool-level latency, failures, and token usage |

---

## 8. Running the Application

### Prerequisites

- Docker installed.
- An environment configuration file.
- The required LLM provider configuration and credentials.

### Environment Configuration

Create your local `.env` file using `.env.example` as a reference.

Configure the required model provider, API credentials, and other application settings according to `app/core/config.py`.

**Never commit `.env` or expose API keys in the repository.**

### Build the Docker Image

From the project root:

```bash
docker build -t agent .
```

### Run the Application

```bash
docker run --rm -p 8000:8000 --env-file .env agent
```

The application should then be accessible at:

- **Web interface:** http://localhost:8000/
- **API documentation:** http://localhost:8000/docs

The Docker commands above launch the application. They are separate from the benchmark execution command described in Section 5.

---

## 9. Methodology

The project follows an iterative development and evaluation methodology.

1. **Knowledge preparation:** collect and organize medical documents from DailyMed and MedlinePlus.
2. **Document ingestion:** load, clean, and split the corpus into retrieval chunks.
3. **Indexing:** generate embeddings and prepare the retrieval index.
4. **Hybrid retrieval:** combine semantic search and BM25 keyword retrieval.
5. **Agent orchestration:** use a planner to select the appropriate tools.
6. **Evidence collection:** collect retrieved information for answer generation.
7. **Grounded generation:** produce an answer based on the available evidence.
8. **Quality control:** review the generated answer using the Critic.
9. **Evaluation:** run the 40-question benchmark and analyze retrieval, quality, and latency metrics.
10. **Iteration:** use evaluation results to identify weaknesses and guide improvements.

---

## 10. Responsible Use

This system is designed for research and informational purposes.

It is not a medical diagnosis system and must not replace professional medical advice.

### Medical Disclaimer

Responses are not medical advice. This system is designed for research and informational purposes only. Always consult qualified healthcare professionals.

---

## 11. Conclusion

This project demonstrates how an LLM can be integrated into an agentic RAG architecture to answer specialized questions using a structured knowledge corpus.

Its main contribution is the combination of hybrid retrieval, explicit planning, specialized tools, evidence-grounded generation, and automated quality evaluation.

The baseline evaluation provides measurable evidence of system behavior and highlights concrete directions for future improvement.

**Knowledge comes from the corpus. Intelligence comes from the LLM and orchestration. Trust comes from evidence, citations, and verification.**

---

**Developed by Abdelilah Lagribi**

**ENSAM Rabat × Syntax — Technical Case Study 2026**
