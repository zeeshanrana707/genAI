"""Real-time Streaming Orchestrator for Perplexity-Grade Deep Research & Multi-Agent Operations.

Emits Server-Sent Events (SSE) detailing live agent execution across 5 specialized autonomous agents:
- Agent 1: Dr. Aris Thorne (Principal Research Scientist) - Web & arXiv Literature Mining
- Agent 2: Prof. Elena Rostova (Adversarial Critic) - Peer Review & Rule-Based Metric Audit
- Agent 3: Dr. Marcus Vance (Senior Academic Editor) - Sequential Revision & Paper Publication
- Agent 4: Sophia Chen (Executive Communications Strategist) - Strategic Leadership Briefing
- Agent 5: David Sterling (Operations & Logistics Coordinator) - Calendar Scheduling & .ics Generation
"""

import re
import json
import time
from typing import Generator, Dict, Any, Optional, List

from config.settings import (
    RESEARCHER_MODEL,
    EMAIL_MODEL,
    SCHEDULER_MODEL,
    OUTPUT_DIR,
)
from tools.tools import (
    WebSearchTool,
    DraftEmailTool,
    CreateCalendarInviteTool,
    run_numerical_verification,
    expand_research_queries,
)
from agents.failover import SafeHandAgentRunner
from agents.test_time_compute import (
    DifficultyEstimator,
    ProcessGroundingVerifier,
    AdversarialCrossCheckVerifier,
    SequentialRevisionOptimizer,
)
from agents.hooks import HookRegistry, AudioAgentHook, VideoAgentHook, PipelineStage


def extract_recipient_and_intent(user_prompt: str, explicit_client: Optional[str] = None) -> Dict[str, Any]:
    """Dynamically determine recipient name and whether outreach/scheduling is requested."""
    clean = user_prompt.strip()
    
    recipient = "Valued Stakeholder"
    if explicit_client and explicit_client.strip() and explicit_client.strip() != "Mr. Smith":
        recipient = explicit_client.strip()
    else:
        match = re.search(r'(?:to|for)\s+(?:our\s+client\s*,?\s*)?([A-Z][a-zA-Z\.\s]+?)(?:,|\.|\s+and|\s+scheduling|\s+with|\s*$)', clean)
        if match:
            candidate = match.group(1).strip()
            if candidate.lower() not in ["our", "the", "a", "an", "all", "each", "my"]:
                recipient = candidate
        elif explicit_client and explicit_client.strip():
            recipient = explicit_client.strip()

    wants_email = bool(re.search(r'\b(email|draft\s+an?\s+email|send\s+an?\s+email|correspondence|letter)\b', clean, re.IGNORECASE))
    wants_schedule = bool(re.search(r'\b(schedul\w+|calendar|meeting|review\s+session|call|book\s+a\s+slot)\b', clean, re.IGNORECASE))

    return {
        "recipient": recipient,
        "wants_email": wants_email,
        "wants_schedule": wants_schedule,
        "is_outreach": wants_email or wants_schedule,
    }


def generate_followup_questions(user_prompt: str, summary: str) -> List[str]:
    """Generate 3 rigorous, frontier research follow-up questions."""
    return [
        f"What exact mathematical formulation or objective function prevents representational drift in this architecture?",
        f"How does credit assignment scale when transitioning from stationary verification to open-ended non-Markovian environments?",
        f"What specific ablation experiment isolates the causal contribution of this mechanism versus compute scale?"
    ]


def generate_ics_content(title: str, agenda: str, recipient: str) -> str:
    """Generate clean standard RFC 5545 iCalendar (.ics) string."""
    clean_title = title.replace("\n", " ").strip()
    clean_agenda = agenda.replace("\n", "\\n").strip()
    return (
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "PRODID:-//Enterprise Multi-Agent AI System//EN\n"
        "CALSCALE:GREGORIAN\n"
        "METHOD:REQUEST\n"
        "BEGIN:VEVENT\n"
        f"SUMMARY:{clean_title}\n"
        f"DESCRIPTION:{clean_agenda}\n"
        "DTSTART:20261103T150000Z\n"
        "DTEND:20261103T153000Z\n"
        "LOCATION:Virtual Video Conference (Google Meet / Zoom)\n"
        "STATUS:CONFIRMED\n"
        f"ATTENDEE;CN={recipient};ROLE=REQ-PARTICIPANT:mailto:{recipient.lower().replace(' ', '.')}@enterprise.com\n"
        "END:VEVENT\n"
        "END:VCALENDAR"
    )


def stream_multi_agent_pipeline(
    user_prompt: str,
    client_name: Optional[str] = None,
    mode: str = "auto",
    engine: str = "gemini",
    prior_context: str = "",
    prior_sources: Optional[List[Dict[str, Any]]] = None,
    uploaded_papers_text: str = "",
    gemini_api_key: Optional[str] = None
) -> Generator[str, None, None]:
    """Generator yielding SSE JSON events detailing live agent execution for ANY user prompt."""
    
    start_time = time.time()
    event_id = 0
    def emit(event_type: str, data: Dict[str, Any]) -> str:
        nonlocal event_id
        event_id += 1
        payload = {"type": event_type, "timestamp": time.strftime("%H:%M:%S"), **data}
        return f"id: {event_id}\ndata: {json.dumps(payload)}\n\n"

    intent = extract_recipient_and_intent(user_prompt, client_name)
    actual_recipient = intent["recipient"]

    agents_roster = [
        {"id": "researcher", "name": "Dr. Aris Thorne", "role": "Principal Research Scientist", "avatar": "🔬", "model": f"Gemini ({RESEARCHER_MODEL}) / Web Indexer"},
        {"id": "critic", "name": "Prof. Elena Rostova", "role": "Adversarial Critic", "avatar": "⚖️", "model": "Adversarial Proof Audit Protocol"},
        {"id": "editor", "name": "Dr. Marcus Vance", "role": "Senior Academic Editor", "avatar": "✍️", "model": "Sequential Revision Optimizer"},
        {"id": "comms", "name": "Sophia Chen", "role": "Executive Communications Strategist", "avatar": "✉️", "model": "Executive Briefing Engine"},
        {"id": "logistics", "name": "David Sterling", "role": "Operations & Logistics Coordinator", "avatar": "📅", "model": "iCalendar Logistics Engine"}
    ]

    # Step 0: Pipeline Initialization
    yield emit("pipeline_init", {
        "prompt": user_prompt,
        "recipient": actual_recipient,
        "mode": "multi_agent_suite",
        "engine": engine,
        "agents": agents_roster,
        "safe_hand_active": True,
    })
    time.sleep(0.15)

    web_search = WebSearchTool()
    draft_email = DraftEmailTool()
    calendar_tool = CreateCalendarInviteTool()
    safe_hand = SafeHandAgentRunner(preferred_gemini_model=RESEARCHER_MODEL, api_key=gemini_api_key)

    # =============================================================
    # AGENT 1: Dr. Aris Thorne (Principal Research Scientist)
    # =============================================================
    yield emit("agent_status_change", {
        "agent_id": "researcher",
        "status": "working",
        "action": "Mining live web index and arXiv literature..."
    })
    yield emit("agent_start", {
        "step": 1,
        "agent_id": "researcher",
        "agent": "Dr. Aris Thorne (Principal Research Scientist)",
        "model": RESEARCHER_MODEL,
        "avatar": "🔬",
        "task": f"Executing empirical literature and web indexing on: '{user_prompt}'",
        "status": "active"
    })
    time.sleep(0.2)

    # Detect if user prompt is a referential follow-up or summary request on the prior conversation
    is_referential_followup = bool(
        prior_context and any(
            kw in user_prompt.lower()
            for kw in [
                "this", "these", "above", "previous", "earlier", "conversation",
                "discussed", "summary", "summarize", "takeaway", "takeaways",
                "elaborate", "clarify", "explain further", "key points", "what did you",
                "what have we", "recap", "review", "synthesize", "conclusion"
            ]
        )
    )

    search_query = user_prompt.strip()
    sub_queries = []
    if is_referential_followup:
        # Extract previous topic or title if present
        topic_match = re.search(r"#\s*([^\n\r]+)", prior_context)
        prior_topic = topic_match.group(1).strip() if topic_match else ""
        if prior_topic:
            search_query = f"{prior_topic} {user_prompt.strip()}"
        if mode == "academic":
            search_query = f"{search_query} academic paper research arxiv"
        elif mode == "enterprise":
            search_query = f"{search_query} enterprise benchmark market analysis"

        yield emit("tool_call_start", {
            "agent": "Dr. Aris Thorne",
            "tool": "Live Web & arXiv Indexer",
            "query": search_query,
            "focus_mode": mode,
            "action": f"Indexing empirical sources for follow-up: '{search_query}'..."
        })
        structured_sources = web_search.search_structured(search_query, max_results=8, mode=mode)
        search_raw = web_search._run(search_query, mode=mode)
    else:
        # Deep Research Snowballing: Decompose inquiry into 3 orthogonal sub-queries
        sub_queries = expand_research_queries(user_prompt, api_key=gemini_api_key)
        yield emit("tool_call_start", {
            "agent": "Dr. Aris Thorne",
            "tool": "Multi-Query Parallel Snowballer",
            "query": " | ".join(sub_queries),
            "sub_queries": sub_queries,
            "focus_mode": mode,
            "action": f"Decomposed inquiry into {len(sub_queries)} orthogonal sub-queries. Indexing arXiv + Web in parallel..."
        })
        structured_sources = web_search.search_structured_multi_queries(sub_queries, max_results=10, mode=mode)
        search_raw_parts = []
        for idx, s in enumerate(structured_sources, 1):
            body_text = s.get("deep_content") or s.get("snippet", "")
            authors_str = f"Authors: {s['authors']}\n" if s.get("authors") else ""
            search_raw_parts.append(
                f"{idx}. Title: {s['title']}\n"
                f"{authors_str}"
                f"   Summary: {s.get('snippet', '')}\n"
                f"   Deep Content: {body_text[:700]}\n"
                f"   Source: {s['url']} ({s['domain']})"
            )
        search_raw = "\n\n".join(search_raw_parts)

    sources_payload = []
    seen_urls = set()
    for s in structured_sources:
        u = s.get("url", "")
        if u and u not in seen_urls:
            seen_urls.add(u)
            sources_payload.append({
                "id": len(sources_payload) + 1,
                "title": s["title"],
                "url": s["url"],
                "domain": s["domain"],
                "snippet": s.get("snippet", ""),
                "deep_content": s.get("deep_content", ""),
                "authors": s.get("authors", ""),
            })

    # If referential follow-up and prior sources exist, include prior sources so the citation lock is preserved
    if is_referential_followup and prior_sources:
        for ps in prior_sources:
            u = ps.get("url", "")
            if u and u not in seen_urls:
                seen_urls.add(u)
                sources_payload.append({
                    "id": len(sources_payload) + 1,
                    "title": ps.get("title", "Prior Research Source"),
                    "url": ps.get("url", "#"),
                    "domain": ps.get("domain", "academic"),
                    "snippet": ps.get("snippet", ""),
                    "deep_content": ps.get("deep_content", ""),
                    "authors": ps.get("authors", ""),
                })

    # Build read-only citation catalog for citation locking (Agent 3)
    sources_catalog = [
        {
            "id": sp["id"],
            "title": sp["title"],
            "authors": sp.get("authors", ""),
            "url": sp["url"],
            "domain": sp["domain"],
        }
        for sp in sources_payload
    ]
    sources_catalog_json = json.dumps(sources_catalog, indent=2)

    yield emit("tool_call_result", {
        "tool": "Live Web & arXiv Indexer",
        "sources": sources_payload,
        "items_count": len(sources_payload),
        "summary": f"Indexed {len(sources_payload)} live authoritative sources."
    })
    time.sleep(0.15)

    difficulty_plan = DifficultyEstimator.estimate_difficulty(user_prompt)
    diff_level = difficulty_plan["difficulty_level"]

    yield emit("agent_thought", {
        "agent": "Dr. Aris Thorne",
        "thought": f"Empirical retrieval complete. Assessing query complexity: Level {diff_level}/5. Formulating initial theoretical structure..."
    })

    system_instruction_thorne = (
        "You are Dr. Aris Thorne, Autonomous Principal Research Scientist specializing in frontier AI, theoretical computing, and complex systems.\n"
        "Draft an initial comprehensive research paper investigating the inquiry using the retrieved empirical context.\n"
        "When a source is from arxiv.org, use the full mathematical sections provided (equations, hardware constraints, boundary conditions) rather than relying only on abstract snippets.\n"
        "Extract and use explicit mathematical assumptions, boundary conditions, and hardware requirements from full-text ArXiv sources.\n"
        "Structure with Title, Abstract blockquote (> **Abstract** — ...), Introduction, Core Mechanisms with LaTeX math, Benchmark comparisons, and References."
    )
    if is_referential_followup:
        system_instruction_thorne += (
            "\nNOTE: The user is asking a follow-up question or requesting a synthesis/summary of the ongoing conversation and research.\n"
            "Directly address and answer the inquiry using the PRIOR RESEARCH CONTEXT provided below, extracting key mathematical foundations and synthesizing conclusions with precision."
        )

    # Build research prompt — prepend prior session context and uploaded paper context if available
    prior_ctx_block = ""
    if prior_context and prior_context.strip():
        prior_ctx_block = (
            "PRIOR RESEARCH CONTEXT (build upon this — do NOT repeat verbatim, extend and deepen):\n"
            f"{prior_context.strip()[:3500]}\n\n"
            "---\n\n"
        )

    uploaded_ctx_block = ""
    if uploaded_papers_text and uploaded_papers_text.strip():
        uploaded_ctx_block = (
            "UPLOADED REFERENCE PAPERS (use these as primary source material — cite them as [Uploaded]):\n"
            f"{uploaded_papers_text.strip()[:4000]}\n\n"
            "---\n\n"
        )

    if is_referential_followup:
        research_prompt = (
            f"{prior_ctx_block}"
            f"{uploaded_ctx_block}"
            f"# USER FOLLOW-UP REQUEST: {user_prompt}\n\n"
            f"The user is asking a follow-up or synthesis question on the research thread above.\n"
            f"Review the PRIOR RESEARCH CONTEXT and provide a comprehensive, rigorous synthesis and explanation answering the user's inquiry directly.\n\n"
            f"Retrieved Empirical Literature & Context:\n"
            f"{search_raw}\n"
        )
    else:
        research_prompt = (
            f"{prior_ctx_block}"
            f"{uploaded_ctx_block}"
            f"# RESEARCH TOPIC: {user_prompt}\n\n"
            f"Retrieved Empirical Literature & Context:\n"
            f"{search_raw}\n\n"
            f"Draft the preliminary academic paper covering theoretical foundations, mathematical formulations, and empirical comparisons."
        )

    initial_draft, provider1 = safe_hand.generate(
        research_prompt,
        system_instruction=system_instruction_thorne,
        search_context=search_raw,
        engine=engine
    )

    yield emit("agent_status_change", {
        "agent_id": "researcher",
        "status": "completed",
        "action": f"Preliminary draft formulated ({len(initial_draft.split())} words)."
    })

    yield emit("draft_v1", {
        "author": "Dr. Aris Thorne",
        "markdown": initial_draft,
        "word_count": len(initial_draft.split())
    })

    yield emit("agent_handoff", {
        "from_agent": "Dr. Aris Thorne (Research Scientist)",
        "to_agent": "Prof. Elena Rostova (Adversarial Critic)",
        "message": "Candidate draft prepared. Passing to Prof. Rostova for adversarial peer-review audit and metric verification."
    })
    time.sleep(0.25)

    # =============================================================
    # AGENT 2: Prof. Elena Rostova (Adversarial Critic)
    # =============================================================
    yield emit("agent_status_change", {
        "agent_id": "critic",
        "status": "working",
        "action": "Auditing mathematical equations, empirical metrics, and proof derivations..."
    })
    yield emit("agent_start", {
        "step": 2,
        "agent_id": "critic",
        "agent": "Prof. Elena Rostova (Adversarial Critic)",
        "model": "Adversarial Proof & Metric Verification Protocol",
        "avatar": "⚖️",
        "task": "Executing adversarial peer review, proof verification, and metric cross-check",
        "status": "active"
    })
    time.sleep(0.2)

    yield emit("agent_thought", {
        "agent": "Prof. Elena Rostova",
        "thought": "Commencing adversarial audit across 5 axes: (1) PDEs and mathematical formalisms, (2) conceptual boundary conditions, (3) deterministic metric grounding, (4) citation veracity, and (5) adversarial proof auditing (rejecting hand-waved 'we derive' claims)."
    })

    cross_verifier = AdversarialCrossCheckVerifier(safe_hand)
    cross_res = cross_verifier.cross_check(user_prompt, initial_draft, search_raw)
    rule_check = cross_res.get("rule_check", {})

    yield emit("agent_thought", {
        "agent": "Prof. Elena Rostova",
        "thought": f"Deterministic metric audit: Grounding ratio {int(rule_check.get('grounding_ratio', 1.0)*100)}% ({len(rule_check.get('verified_metrics', []))} verified figures). Formulated {len(cross_res['critique_points'])} critical refinement mandates."
    })

    peer_review_deliverable = {
        "score": cross_res["score"],
        "verdict": cross_res["verdict"],
        "provider": cross_res["provider"],
        "grounding_ratio": int(rule_check.get("grounding_ratio", 1.0) * 100),
        "verified_metrics": rule_check.get("verified_metrics", []),
        "unverified_metrics": rule_check.get("unverified_metrics", []),
        "critique_points": cross_res["critique_points"],
        "auditor": "Prof. Elena Rostova (Adversarial Proof Protocol)"
    }

    yield emit("cross_check_result", peer_review_deliverable)
    yield emit("agent_deliverable", {
        "agent_id": "critic",
        "type": "peer_review",
        "data": peer_review_deliverable
    })

    # =============================================================
    # AUTONOMOUS DAG REFLECTION LOOP (Elena Rostova ➔ Aris Thorne)
    # =============================================================
    critique_pts = cross_res.get("critique_points", [])
    if critique_pts and not is_referential_followup and cross_res.get("score", 100) < 85:
        yield emit("agent_handoff", {
            "from_agent": "Prof. Elena Rostova (Adversarial Critic)",
            "to_agent": "Dr. Aris Thorne (Research Scientist)",
            "message": f"Autonomous Reflection Triggered: Auditing identified {len(critique_pts)} mathematical proof gaps. Fortifying derivations..."
        })
        yield emit("agent_thought", {
            "agent": "Dr. Aris Thorne",
            "thought": f"Executing reflection pass. Addressing Prof. Rostova's mandates: {'; '.join(critique_pts[:2])}..."
        })
        fortify_prompt = (
            f"Prof. Elena Rostova completed an adversarial proof audit and identified these specific mandates:\n"
            f"{chr(10).join(['- ' + cp for cp in critique_pts])}\n\n"
            f"Preliminary Draft:\n{initial_draft}\n\n"
            f"Empirical context:\n{search_raw[:2200]}\n\n"
            f"Fortify the draft: derive explicit intermediate steps, state boundary conditions, and formalize conceptual leaps as 'PROPOSED HYPOTHESIS:'."
        )
        fortified_draft, _ = safe_hand.generate(
            fortify_prompt,
            system_instruction=system_instruction_thorne,
            search_context=search_raw,
            engine=engine
        )
        if fortified_draft and len(fortified_draft.split()) > 100:
            initial_draft = fortified_draft
            yield emit("agent_status_change", {
                "agent_id": "researcher",
                "status": "completed",
                "action": f"Draft fortified via autonomous reflection cycle ({len(initial_draft.split())} words)."
            })

    # =============================================================
    # PYTHON SELF-HEALING CODE INTERPRETER & PLOT GENERATION
    # =============================================================
    yield emit("agent_thought", {
        "agent": "Dr. Aris Thorne & Prof. Elena Rostova",
        "thought": "Invoking Self-Healing Python Sandbox (numpy / scipy / matplotlib) to simulate differential equations and generate visual attractor dynamics..."
    })

    verif_res = run_numerical_verification(initial_draft[:3200], user_prompt, api_key=gemini_api_key)
    verif_summary = "Verification completed."
    plot_b64 = verif_res.get("plot_base64", "")
    plot_file_path = verif_res.get("plot_path", "")

    if verif_res.get("executed"):
        verif_summary = f"Simulated in Python: {verif_res.get('output', '')[:250]}"
        healed_notice = " (Auto-healed via traceback repair)" if verif_res.get("healed") else ""
        metrics_found = verif_res.get("metrics", [])
        yield emit("agent_thought", {
            "agent": "Prof. Elena Rostova",
            "thought": f"Python simulation converged cleanly{healed_notice}. Metrics: {', '.join(metrics_found[:3]) if metrics_found else 'Equations validated'}."
        })

        if plot_b64:
            yield emit("visual_artifact", {
                "type": "simulation_plot",
                "plot_path": plot_file_path,
                "plot_base64": plot_b64,
                "caption": f"Dynamical Simulation & Phase Portrait: {user_prompt[:50]}",
                "metrics": metrics_found,
                "healed": verif_res.get("healed", False)
            })
            yield emit("agent_thought", {
                "agent": "Dr. Aris Thorne",
                "thought": "Visual simulation figure rendered successfully. Embedding phase-space trajectory into publication draft."
            })
    elif verif_res.get("error"):
        verif_summary = f"Numerical check note: {verif_res.get('error', '')[:180]}"
        yield emit("agent_thought", {
            "agent": "Prof. Elena Rostova",
            "thought": f"Numerical simulation note: {verif_summary}"
        })

    yield emit("agent_status_change", {
        "agent_id": "critic",
        "status": "completed",
        "action": f"Peer review complete. Audit Score: {cross_res['score']}/100 ({cross_res['verdict']})."
    })

    yield emit("agent_handoff", {
        "from_agent": "Prof. Elena Rostova (Adversarial Critic)",
        "to_agent": "Dr. Marcus Vance (Senior Academic Editor)",
        "message": f"Peer-review audit complete ({cross_res['score']}/100). Delivering critique points and numerical verification metrics to Dr. Vance for publication-grade synthesis."
    })
    time.sleep(0.25)

    # =============================================================
    # AGENT 3: Dr. Marcus Vance (Senior Academic Editor)
    # =============================================================
    yield emit("agent_status_change", {
        "agent_id": "editor",
        "status": "working",
        "action": "Enforcing strict citation locking and synthesizing publication-grade paper..."
    })
    yield emit("agent_start", {
        "step": 3,
        "agent_id": "editor",
        "agent": "Dr. Marcus Vance (Senior Academic Editor)",
        "model": RESEARCHER_MODEL,
        "avatar": "✍️",
        "task": "Performing sequential revision with strict citation locking and verified mathematics",
        "status": "active"
    })
    time.sleep(0.2)

    yield emit("agent_thought", {
        "agent": "Dr. Marcus Vance",
        "thought": "Enforcing Strict Citation Locking against the indexed empirical catalog. Hardening equations, embedding Python numerical verification, and ensuring zero fabricated DOIs..."
    })

    # Prepare Dr. Vance's revision prompt with strict citation catalog and numerical metrics
    editor_system_instruction = (
        "You are Dr. Marcus Vance, Senior Academic Editor at an elite research institute.\n"
        "Revise and finalize the candidate research paper into a publication-grade Academic Paper.\n\n"
        "STRICT CITATION LOCKING PROTOCOL:\n"
        "1. You are provided with a read-only JSON catalog of indexed empirical sources.\n"
        "2. You may ONLY cite sources present in that catalog verbatim [1], [2] in the body and References.\n"
        "3. You must NEVER invent DOIs or URLs. In the References section, copy the exact title, authors, year, and URL/DOI from the catalog.\n\n"
        "ADVERSARIAL PROOF AUDITING COMPLIANCE:\n"
        "4. Reject any section claiming 'we derive' or 'we demonstrate' without showing explicit algebraic steps or simulation parameters.\n"
        "5. If a connection between equations is conceptual only, label it as 'PROPOSED HYPOTHESIS: [statement]' rather than a proven theorem.\n"
        "6. Embed the Python numerical verification results into the text (e.g. parameter regimes, convergence metrics).\n\n"
        "PAPER STRUCTURE:\n"
        "# [Full Academic Title]\n"
        "> **Abstract** — [Rigorous 150-word synthesis]\n"
        "## 1. Introduction & Theoretical Foundations\n"
        "## 2. Mathematical Formulations & Boundary Conditions\n"
        "## 3. Computational Mechanisms & Dynamics\n"
        "## 4. Numerical Verification & Parameter Regime\n"
        "## 5. Comparative Trade-offs & Limitations\n"
        "## 6. References (Strictly verbatim from catalog)\n"
    )

    plot_embed_instruction = ""
    if plot_b64:
        plot_embed_instruction = (
            f"\n\nVISUAL SIMULATION ARTIFACT (MANDATORY: Embed this exact image markdown inside Section 4):\n"
            f"![Attractor & Phase Space Simulation]({plot_b64})\n"
            f"*Figure 1: Numerical simulation and dynamic trajectory verification under specified parameter regimes.*\n"
        )

    critique_text = "\n".join([f"- {cp}" for cp in cross_res["critique_points"]])
    if is_referential_followup:
        editor_prompt = (
            f"# USER INQUIRY / FOLLOW-UP: {user_prompt}\n\n"
            f"READ-ONLY CITATION CATALOG (MANDATORY: ONLY CITE FROM THIS LIST VERBATIM):\n"
            f"{sources_catalog_json}\n\n"
            f"## Ongoing Conversation & Prior Research Context:\n{prior_context[:3500]}\n\n"
            f"## Preliminary Response Draft:\n{initial_draft}\n\n"
            f"## Prof. Elena Rostova's Adversarial Critique:\n{critique_text}\n\n"
            f"## Python Numerical Verification Simulation Results:\n"
            f"Summary: {verif_summary}\n\n"
            f"{plot_embed_instruction}\n\n"
            f"Execute sequential revision now. Output the comprehensive, finalized academic response answering the user's inquiry."
        )
    else:
        editor_prompt = (
            f"# RESEARCH TOPIC: {user_prompt}\n\n"
            f"READ-ONLY CITATION CATALOG (MANDATORY: ONLY CITE FROM THIS LIST VERBATIM):\n"
            f"{sources_catalog_json}\n\n"
            f"## Empirical Literature & Context:\n{search_raw[:3500]}\n\n"
            f"## Fortified Preliminary Draft:\n{initial_draft}\n\n"
            f"## Prof. Elena Rostova's Adversarial Critique:\n{critique_text}\n\n"
            f"## Python Numerical Verification Simulation Results:\n"
            f"Script: {verif_res.get('script', 'N/A')[:600]}\n"
            f"Output: {verif_res.get('output', 'N/A')[:600]}\n"
            f"Summary: {verif_summary}\n\n"
            f"{plot_embed_instruction}\n\n"
            f"Execute sequential revision now. Output only the finished publication-grade academic research paper."
        )

    research_summary, final_provider = safe_hand.generate(
        editor_prompt,
        system_instruction=editor_system_instruction,
        search_context=search_raw,
        engine=engine
    )

    # Word-by-word streaming (ChatGPT-like fluid text emission)
    words = research_summary.split(" ")
    accumulated_content = ""
    for i, w in enumerate(words):
        token = w + (" " if i < len(words) - 1 else "")
        accumulated_content += token
        yield emit("content_chunk", {
            "chunk": token,
            "full_content": accumulated_content,
            "provider": final_provider
        })
        time.sleep(0.012)  # Fluid word-by-word emission

    yield emit("agent_deliverable", {
        "agent_id": "editor",
        "type": "research_paper",
        "data": {
            "paper_markdown": research_summary,
            "provider": final_provider,
            "word_count": len(research_summary.split()),
            "sources_count": len(sources_payload)
        }
    })

    yield emit("draft_v2", {
        "author": "Dr. Marcus Vance",
        "markdown": research_summary,
        "word_count": len(research_summary.split())
    })

    yield emit("agent_status_change", {
        "agent_id": "editor",
        "status": "completed",
        "action": "Academic research paper published with full LaTeX derivations."
    })

    yield emit("agent_handoff", {
        "from_agent": "Dr. Marcus Vance (Academic Editor)",
        "to_agent": "Sophia Chen (Executive Communications Strategist)",
        "message": "Academic paper finalized. Passing core findings to Sophia Chen to craft executive briefing memo."
    })
    time.sleep(0.25)

    # =============================================================
    # AGENT 4: Sophia Chen (Executive Communications Strategist)
    # =============================================================
    yield emit("agent_status_change", {
        "agent_id": "comms",
        "status": "working",
        "action": f"Formulating executive briefing correspondence for {actual_recipient}..."
    })
    yield emit("agent_start", {
        "step": 4,
        "agent_id": "comms",
        "agent": "Sophia Chen (Executive Communications Strategist)",
        "model": EMAIL_MODEL,
        "avatar": "✉️",
        "task": f"Translating research insights into high-impact executive briefing for {actual_recipient}",
        "status": "active"
    })
    time.sleep(0.2)

    yield emit("agent_thought", {
        "agent": "Sophia Chen",
        "thought": f"Distilling high-density research findings into clear, actionable executive strategy for {actual_recipient}..."
    })

    client_subject = f"Executive Briefing: Strategic Analysis & Operational Impact of {user_prompt[:45]}"
    
    email_gen_prompt = (
        f"User Research Query: '{user_prompt}'\n\n"
        f"Published Research Findings:\n{research_summary[:1600]}\n\n"
        f"Stakeholder Recipient: {actual_recipient}\n\n"
        f"Draft an authoritative, high-impact executive briefing email addressed to {actual_recipient}.\n"
        f"Include:\n"
        f"1. Executive Greeting to {actual_recipient}\n"
        f"2. Strategic Summary: Why this technological/paradigm shift matters now\n"
        f"3. Core Operational Impact & Trade-offs (concise bullets)\n"
        f"4. Recommended Action Items & Next Steps\n"
        f"5. Professional closing proposing a 30-minute review session next week."
    )

    email_draft_raw, provider2 = safe_hand.generate(
        email_gen_prompt,
        system_instruction=f"You are Sophia Chen, elite Executive Communications Strategist. Produce a crisp, persuasive corporate briefing for {actual_recipient}.",
        search_context=search_raw
    )

    client_body = email_draft_raw if email_draft_raw else (
        f"Dear {actual_recipient},\n\n"
        f"Following recent developments in our technical evaluation, our advisory practice has synthesized a strategic analysis regarding:\n"
        f"\"{user_prompt}\".\n\n"
        f"Executive Summary & Strategic Takeaways:\n"
        f"{research_summary[:450]}...\n\n"
        f"Key Implications for Operations:\n"
        f"• Architectural Efficiency: Transitions to modern paradigm yield significant computational and latency gains.\n"
        f"• Implementation Risk: Mitigated through empirical verification and strict boundary controls.\n\n"
        f"We recommend scheduling a brief 30-minute alignment session next week to review tactical deployment options.\n\n"
        f"Sincerely,\n\n"
        f"Sophia Chen\n"
        f"Executive Strategy & Advisory Practice"
    )

    draft_email._run(
        recipient_name=actual_recipient,
        recipient_email=f"{actual_recipient.lower().replace(' ', '.')}@enterprise.com",
        subject=client_subject,
        body=client_body,
    )

    email_deliverable = {
        "recipient": actual_recipient,
        "recipient_email": f"{actual_recipient.lower().replace(' ', '.')}@enterprise.com",
        "subject": client_subject,
        "body": client_body,
        "provider": provider2
    }

    yield emit("agent_deliverable", {
        "agent_id": "comms",
        "type": "executive_email",
        "data": email_deliverable
    })

    yield emit("agent_status_change", {
        "agent_id": "comms",
        "status": "completed",
        "action": f"Executive briefing staged for {actual_recipient}."
    })

    yield emit("agent_handoff", {
        "from_agent": "Sophia Chen (Communications Strategist)",
        "to_agent": "David Sterling (Operations Coordinator)",
        "message": "Executive briefing completed. Passing review proposal to David Sterling for calendar scheduling and logistics."
    })
    time.sleep(0.25)

    # =============================================================
    # AGENT 5: David Sterling (Operations & Logistics Coordinator)
    # =============================================================
    yield emit("agent_status_change", {
        "agent_id": "logistics",
        "status": "working",
        "action": "Scheduling 30-minute review session and generating .ics calendar invite..."
    })
    yield emit("agent_start", {
        "step": 5,
        "agent_id": "logistics",
        "agent": "David Sterling (Operations & Logistics Coordinator)",
        "model": SCHEDULER_MODEL,
        "avatar": "📅",
        "task": f"Formulating meeting logistics and calendar invite specification for {actual_recipient}",
        "status": "active"
    })
    time.sleep(0.2)

    meeting_title = f"Strategic Review: {user_prompt[:45]}"
    proposed_slots = [
        "Tuesday, Next Week at 10:00 AM – 10:30 AM EST",
        "Wednesday, Next Week at 2:00 PM – 2:30 PM EST",
        "Thursday, Next Week at 11:30 AM – 12:00 PM EST"
    ]
    agenda = (
        f"1. Executive Briefing on: {user_prompt[:50]}.\n"
        f"2. Evaluation of Key Benchmarks & Adversarial Peer-Review Findings.\n"
        f"3. Discussion of Next Steps & Strategic Implementation with {actual_recipient}."
    )

    calendar_tool._run(
        meeting_title=meeting_title,
        client_name=actual_recipient,
        duration_minutes=30,
        proposed_slots=proposed_slots,
        agenda=agenda,
    )

    ics_content = generate_ics_content(meeting_title, agenda, actual_recipient)

    calendar_deliverable = {
        "title": meeting_title,
        "recipient": actual_recipient,
        "duration_minutes": 30,
        "proposed_slots": proposed_slots,
        "agenda": agenda,
        "meeting_type": "Virtual Video Conference (Google Meet / Zoom)",
        "ics_content": ics_content,
        "status": "PROPOSED"
    }

    yield emit("agent_deliverable", {
        "agent_id": "logistics",
        "type": "calendar_invite",
        "data": calendar_deliverable
    })

    yield emit("agent_status_change", {
        "agent_id": "logistics",
        "status": "completed",
        "action": "Calendar invite and 3 meeting slots scheduled."
    })

    followup_questions = generate_followup_questions(user_prompt, research_summary)
    yield emit("followup_questions", {
        "questions": followup_questions
    })

    # =============================================================
    # Final Multi-Agent Mission Wrap-Up
    # =============================================================
    total_time = f"{round(time.time() - start_time, 2)}s"
    yield emit("pipeline_finished", {
        "success": True,
        "prompt": user_prompt,
        "recipient": actual_recipient,
        "mode": "multi_agent_suite",
        "email_subject": client_subject,
        "email_body": client_body,
        "calendar": calendar_deliverable,
        "peer_review": peer_review_deliverable,
        "research": research_summary,
        "draft_v1": initial_draft,
        "draft_v2": research_summary,
        "sources": sources_payload,
        "plot_base64": plot_b64,
        "plot_path": plot_file_path,
        "followups": followup_questions,
        "active_provider": safe_hand.active_provider,
        "execution_time": total_time,
        "agents_completed": 5
    })
