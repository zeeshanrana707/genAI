# Autonomous Deep Research & Epistemic Intelligence Engine

An autonomous scientific research engine and multi-agent system inspired by Perplexity Pro and frontier cognitive architectures. It deconstructs complex topics, performs real-time web indexing with interactive citations, extracts failure modes across scales, bridges disconnected domains, and formulates falsifiable architectural hypotheses.

---

## ✦ Key Features

- **Perplexity Pro–Style Deep Research UI**:
  - Live **Sources Grid** with real site favicons, domain tags, and interactive citation badges (`[1]`, `[2]`). Hovering over an in-text citation highlights the primary source card.
  - Expandable **4-Stage Cognitive Workflow**:
    1. *Deconstruction* (exact mathematical/algorithmic mechanisms).
    2. *Contradiction & Bottleneck Mapping* ($100\times$ scaling breakages, OOD collapse).
    3. *Cross-Pollinated Synthesis* (thermodynamics, active inference, program synthesis, control theory).
    4. *Hypothesis Generation & Falsification* (concrete architectures & empirical falsification criteria).
  - Streaming real-time markdown synthesis via Server-Sent Events (SSE).
  - Dynamic follow-up questions tailored to deep technical trade-offs.

- **4-Part Frontier Scientific Output Architecture**:
  1. `State-of-the-Art (SOTA) Mechanism Breakdown` (Core Mechanisms & Hidden Assumptions)
  2. `The Frontier Bottleneck (Why Current Approaches Fail)` (Empirical conflicts & explicit Knowledge Gap)
  3. `Novel Research Hypotheses & Architectural Proposals` (Synthesis, Architecture/Information Flow, Theoretical Justification)
  4. `Falsification & Experimental Design` (Baseline vs. Proposed, Benchmark Suite, Exact Falsification Metrics)

- **Executive Brief Dispatcher (Email to Receiver)**:
  - Input any receiver's email address to automatically synthesize and send a short, descriptive executive summary and core theme.
  - Automatic SMTP transmission or 1-click `mailto:` launch into Gmail/Outlook.

- **3-Tier Zero-Downtime Safe-Hand Failover**:
  - **Tier 1**: Google Gemini API (`gemini-3.1-flash-lite`, `gemini-3.5-flash`, `gemini-3.8-flash`).
  - **Tier 2**: Free Serverless AI Provider (Zero-Cost OpenAI/Llama fallback).
  - **Tier 3**: Dynamic Local Context-Aware Synthesis Engine.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd genAI
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Add your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Launch the Application

```bash
# Launch Modern Web Dashboard
python app.py
# Or:
python main.py --ui
```
Navigate to `http://127.0.0.1:8000`.

### 4. CLI Headless Mode

```bash
python main.py --prompt "Why do current test-time search algorithms fail at out-of-distribution reasoning?"
```

---

## 🛠 Project Structure

```text
├── agents/
│   ├── crew_factory.py        # CrewAI agent definitions & sequential pipeline
│   ├── failover.py            # 3-Tier Safe-Hand failover engine
│   ├── hooks.py               # Modular lifecycle extension hooks
│   └── stream_orchestrator.py # Real-time SSE deep research orchestrator
├── config/
│   └── settings.py            # Model priority & environment configuration
├── output/                    # Staged deliverables (.json, .txt, .ics)
├── templates/
│   └── index.html             # Perplexity-style dark frontend canvas
├── tools/
│   └── tools.py               # Live web search & artifact formatting tools
├── app.py                     # FastAPI web server & streaming endpoints
├── main.py                    # Entrypoint CLI & UI runner
├── requirements.txt           # Project dependencies
└── README.md
```

---

## 📄 License
MIT License.
