# Product Requirements Document (PRD)
## genAI: Epistemic Research & Conversational Intelligence Engine

- **Product Name:** genAI Epistemic Engine
- **Document Version:** 2.4.0
- **Status:** Production / Active
- **Target Audience:** Technical Researchers, AI Engineers, Enterprise Intelligence Analysts, Scientific Authors
- **Date:** October 2026

---

## 1. Executive Summary & Vision

Current generative search systems suffer from two systemic failures:
1. **Hallucination & Speculative Claims:** Models regularly invent unverified hardware metrics (e.g., claiming closed-source models require "8x H100 clusters"), assert absolute consistency ("the only architecture capable of..."), state percentages without study denominators, and output phantom citations.
2. **Latency vs. Rigor Trade-off:** Users are forced to choose between shallow, single-turn chatbots (fast but non-grounded) or monolithic multi-minute autonomous pipelines (thorough but frustratingly slow and non-conversational).

**genAI** is a dual-speed, autonomous epistemic research engine. It delivers:
- **Instant Fast Chat (<1s TTFT):** Direct conversational intelligence comparable to ChatGPT and DeepSeek for rapid iterations.
- **5-Agent Deep Search:** Autonomous parallel literature mining across live web and arXiv, adversarial verification by a dedicated peer-review critic, real Python/SciPy mathematical execution, deterministic citation clamping, and conversational synthesis with zero speculative hyperbole.
- **Safe-Hand 0-Downtime Resilience:** Continuous operation across Google Gemini, serverless fallback engines, and offline context synthesis.

---

## 2. Target Personas & Core Use Cases

### Persona 1: Frontier AI & Scientific Researcher (Dr. Elena)
- **Need:** Evaluate complex machine learning papers, verify diffusion formulations, cross-reference benchmark scores, and inspect mathematical phase spaces.
- **Pain Point:** Generic chatbots hallucinate paper citations, state outdated architectures, and cannot execute differential equations.
- **Solution in genAI:** Dr. Elena receives arXiv-grounded research, deterministic citations [1], [2], SciPy ODE dynamical trajectory plots, and bibtex exports.

### Persona 2: Enterprise Strategy & Operations Lead (Marcus)
- **Need:** Understand technological trends (e.g., solid-state batteries, AI video models) without reading 50 raw PDFs.
- **Pain Point:** Needs verifiable numbers with clear denominators (e.g., Phase I-to-approval transition rates, not unconditioned percentages).
- **Solution in genAI:** Receives executive briefing memos with explicit baseline study conditions, downloadable calendar review invites, and verified trade-off tables.

### Persona 3: Academic Student & Literature Reviewer (Aiden)
- **Need:** Upload research PDFs, extract core equations, formulate structured thematic overviews, and export papers to LaTeX with amsmath formatting.
- **Pain Point:** File upload limits, lost citations, and cluttered interfaces.
- **Solution in genAI:** Directly attaches PDFs into the input bar, views multi-turn conversational answers, and exports verified `.tex` and `.bib` artifacts in one click.

---

## 3. Product Features & Epic Breakdown

```
┌────────────────────────────────────────────────────────────────────────┐
│                          genAI PRODUCT EPICS                           │
└────────────────────────────────────────────────────────────────────────┘
  │
  ├── EPIC 1: Dual-Speed Research Pipeline
  │     ├── Mode A: Instant Fast Chat (<1s TTFT, DeepSeek/ChatGPT style)
  │     └── Mode B: 5-Agent Deep Search (Autonomous parallel investigation)
  │
  ├── EPIC 2: Multi-Specialist Agent Collective
  │     ├── Dr. Aris Thorne (Scout / Literature Mining)
  │     ├── Prof. Elena Rostova (Adversarial Critic & Metric Auditor)
  │     ├── Dr. Marcus Vance (Lead Synthesizer & Citation Locker)
  │     ├── Sophia Chen (Executive Comms Strategist)
  │     └── David Sterling (Logistics & Calendar Coordinator)
  │
  ├── EPIC 3: Epistemic Grounding & Fact-Checking Engine
  │     ├── Deterministic Citation Clamping (<1ms zero-hallucination gate)
  │     ├── Denominator & Baseline Guardrails
  │     └── Epistemic Uncertainty Callout Disclosures
  │
  ├── EPIC 4: Scientific Computation Sandbox
  │     ├── NumPy / SciPy ODE Dynamical Solver
  │     ├── Self-Healing Traceback Compiler Loop
  │     └── Matplotlib Phase-Space Visual Artifact Generation
  │
  ├── EPIC 5: Reference Ingestion & Multi-Turn Persistence
  │     ├── PDF Paper Upload & Extraction
  │     ├── SQLite WAL Multi-Turn Conversational Memory
  │     └── Unified Document Chips Carousel
  │
  ├── EPIC 6: Interactive Research Canvas & Simulation Workbench
  │     ├── Inline Section Refinement (OpenAI Canvas style)
  │     └── Dynamic Parameter Regime Tuning & Real-Time ODE Rerun
  │
  └── EPIC 7: Scientific Artifact Deliverables
        ├── LaTeX (.tex with amsmath) Export
        ├── BibTeX (.bib) Citation Export
        ├── Markdown (.md) Knowledge Export
        └── iCalendar (.ics) Meeting Dispatch
```

---

## 4. Benchmark Case Studies (Origin Problem & Validation)

genAI was specifically designed and tested against real failure modes found in frontier LLM research models. Two benchmark cases define the platform's verification criteria:

### Benchmark Case A: AI Video Generation Architectures (2026)
- **The Failure in Generic Search Models:**
  - Claimed *Sora 2* was available despite OpenAI discontinuing the Sora web/API experiences in 2026.
  - Made unsupported hardware claims: asserting that *Sora 2 and Wan 2.5 require >8x H100 GPUs* while *Runway Gen-4 runs on a single A100*.
  - Claimed *Diffusion Transformers (DiTs)* are "the only architecture capable of temporal consistency."
  - Invented arbitrary generation speed benchmarks: *"10–50x slower than real time."*
  - Generated phantom citations `[1, 5]`, `[2, 3]` without matching bibliography entries.
- **How genAI Solves It:**
  - **Temporal & Deprecation Validation:** Live DuckDuckGo / arXiv verification retrieves current availability notices.
  - **Hardware Spec Sanitization:** Critic forces model to state: *"Official hardware requirements for closed-source models have not been publicly disclosed by vendors."*
  - **Empirical Reframing:** Strips absolute terms ("the only architecture"), framing DiTs as empirical state-of-the-art across published benchmarks.
  - **Deterministic Citation Clamping:** Automatically maps inline citations to verified sources and forces a verbatim `### 📚 Sources & References` catalog.

### Benchmark Case B: AI-Driven Drug Discovery & Clinical Transition
- **The Failure in Generic Search Models:**
  - Cited unconditioned statistics: *"Clinical success rate is 15%"* without specifying whether this represents Phase I, Phase II, or Phase I-to-approval transition.
  - Extrapolated computational ADMET prediction improvements into proven clinical failure reductions without clinical trial data.
  - Claimed generalized development acceleration: *"reduces timeline by 1–2 years"* without study cohorts.
- **How genAI Solves It:**
  - **Denominator Enforcement:** Enforces strict population baselines (e.g., *"10–15% transition from Phase I to FDA approval in oncology cohorts"*).
  - **Evidence Stratification:** Clearly separates computational (*in-silico*), *in-vitro*, preclinical animal, and human clinical trial evidence.
  - **Epistemic Uncertainty Callout:** When empirical data is limited, explicitly wraps claims in `> ⚠️ Epistemic Note: [Limitation disclosed]`.

---

## 5. Detailed Feature Specifications

### 4.1 Epic 1: Dual-Speed Research Modes
- **FR-1.1 Fast Chat Mode (Default):**
  - When the "Deep Search" toggle is OFF, the system operates in low-latency single-agent streaming mode.
  - Streaming starts in under 1 second (<1s TTFT) using native Google Gemini streaming.
  - Delivers direct, conversational, markdown-formatted responses without multi-agent handoff latency.
- **FR-1.2 Deep Search Mode:**
  - Activated via the Cosmic Violet / Royal Indigo toggle pill in the search bar.
  - Triggers the 5-agent pipeline: Query Decomposition -> Parallel Snowballing -> Adversarial Audit -> Python Verification -> Lead Revision -> Executive Deliverables.

### 4.2 Epic 2: Autonomous Multi-Specialist Agent Collective
- **FR-2.1 Dr. Aris Thorne (Scout):**
  - Dispatches parallel multi-query search strings across DuckDuckGo and the ArXiv API.
  - Indexes metadata: Title, Domain, Publication Year, URL, and snippet passages.
- **FR-2.2 Prof. Elena Rostova (Adversarial Critic):**
  - Audits preliminary drafts for metric accuracy, mathematical formalisms, and hardware assertions.
  - Rejects speculative GPU cluster numbers (e.g., "8x H100 required") unless documented in official technical reports.
  - Calculates grounding ratios and triggers autonomous DAG reflection if grounding score < 85%.
- **FR-2.3 Dr. Marcus Vance (Lead Research Synthesizer):**
  - Synthesizes high-signal, direct conversational answers.
  - Concludes every research inquiry with a verbatim `### 📚 Sources & References` catalog.
- **FR-2.4 Sophia Chen (Executive Communications):**
  - Formulates strategic executive summaries with synthesized takeaways and operational implications.
- **FR-2.5 David Sterling (Logistics):**
  - Generates RFC-5545 compliant `.ics` calendar files with 3 proposed review slots.

### 4.3 Epic 3: Deterministic Grounding & Verification
- **FR-3.1 Citation Clamping Gate:**
  - Deterministically scans all generated inline citations `[n]` in `<1ms`.
  - Ensures every citation index `n` satisfies `1 <= n <= len(catalog_sources)`. Any out-of-bounds citations are clamped to valid catalog bounds or stripped, guaranteeing zero phantom links.
- **FR-3.2 Denominator & Baseline Rigor:**
  - Forces percentages and success metrics to specify their evaluation cohort and study baseline (e.g., *"10–15% transition from Phase I to FDA approval"*).
- **FR-3.3 Epistemic Uncertainty Callouts:**
  - When technical literature is inconclusive or proprietary, formats claims with blockquote notices:
    `> ⚠️ Epistemic Note: [Specific unverified conditions disclosed]`.

### 5.4 Epic 4: Computational Sandbox & Visual Artifacts
- **FR-4.1 Python ODE Execution:**
  - Identifies differential equations or dynamic trajectories in prompts.
  - Executes isolated NumPy/SciPy simulations with a self-healing loop for runtime syntax errors.
- **FR-4.2 Visual Artifact Rendering:**
  - Produces 300 DPI Matplotlib phase portrait figures encoded as base64 images and disk-persisted `.png` files.

### 5.5 Epic 5: Interactive Canvas & Simulation Workbench
- **FR-5.1 Inline Section Refinement (Canvas Mode):**
  - Users can highlight specific sections or paragraphs in the research report and provide a natural language prompt (e.g., *"fortify with amsmath equations"*, *"simplify explanation"*).
  - Backend executes targeted revision via `POST /api/canvas/refine` and generates `v3 Refined` draft without re-running the entire pipeline.
- **FR-5.2 Live Simulation Parameter Rerun:**
  - Users can adjust parameters (e.g., dissipation coefficient $\gamma$, drive amplitude $F_0$) via `POST /api/simulation/rerun`.
  - Re-executes the ODE numerical integration in real time ($< 1.5\text{s}$) and updates the embedded figure dynamically.

### 5.6 Epic 6: Multi-Format Scientific Deliverables
- **FR-6.1 LaTeX Export:** Exports research reports as publication-grade `.tex` documents with `\usepackage{amsmath,amssymb,hyperref}`.
- **FR-6.2 BibTeX Export:** Generates standardized `.bib` entries for all cited sources with DOIs and URLs.
- **FR-6.3 Calendar Logistics:** Exports RFC-5545 `.ics` meeting invites with proposed debriefing slots.

---

## 6. Non-Functional Requirements (NFRs)

| ID | Category | Requirement Description | Target Metric |
| :--- | :--- | :--- | :--- |
| **NFR-1** | Latency | Fast Chat Time-to-First-Token (TTFT) | < 1,000 ms |
| **NFR-2** | Latency | Fast Chat Complete Turn Completion | < 4.0 seconds |
| **NFR-3** | Throughput | Stream Typewriter Fluidity | 100 FPS (10ms pacing) |
| **NFR-4** | Availability | Zero-Downtime Failover Availability | 99.9% uptime (Safe-Hand) |
| **NFR-5** | Reliability | Out-of-bounds / Hallucinated Citations | 0 broken citations |
| **NFR-6** | Concurrency | SQLite Concurrent Reads/Writes | WAL Mode with 30s timeout |
| **NFR-7** | Portability | Zero-Cost Infrastructure Requirement | Operates on Gemini Free Tier & Pollinations |
| **NFR-8** | Usability | Responsive Breakpoint Support | Mobile (320px) to 4K Ultrawide |

---

## 7. Success Metrics & Key Performance Indicators (KPIs)

1. **Factual Grounding Ratio:** $\ge 95\%$ of quantitative assertions directly traced to indexed catalog sources.
2. **Zero Citation Error Rate:** $100\%$ of rendered inline citation pills `[n]` link to valid, active source domains.
3. **Session Retention:** SQLite WAL storage retains 100% of multi-turn conversational history across browser refreshes and server reboots.
4. **Interactive Abort Latency:** Immediate cessation of network streams and UI animation cleanup in $< 50\text{ ms}$ upon pressing the Stop action button.
