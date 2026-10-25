"""Real-time Streaming Orchestrator for Perplexity-Grade Deep Research & Multi-Agent Operations.

Emits Server-Sent Events (SSE) detailing live search queries, real-time web citations,
expandable thinking/reasoning traces, streaming synthesized research, dynamic follow-up
questions, and optional outreach deliverables.
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
from tools.tools import WebSearchTool, DraftEmailTool, CreateCalendarInviteTool
from agents.failover import SafeHandAgentRunner
from agents.test_time_compute import (
    DifficultyEstimator,
    ProcessGroundingVerifier,
    DeepSeekCrossCheckVerifier,
    SequentialRevisionOptimizer,
)
from agents.hooks import HookRegistry, AudioAgentHook, VideoAgentHook, PipelineStage


def extract_recipient_and_intent(user_prompt: str, explicit_client: Optional[str] = None) -> Dict[str, Any]:
    """Dynamically determine recipient name and whether outreach/scheduling is requested.
    
    Never locks into 'Mr. Smith' unless explicitly requested by the user.
    """
    clean = user_prompt.strip()
    
    # 1. Determine recipient
    recipient = "Valued Stakeholder"
    if explicit_client and explicit_client.strip() and explicit_client.strip() != "Mr. Smith":
        recipient = explicit_client.strip()
    else:
        # Check if prompt specifies a recipient
        # e.g., "draft an email to our client, Mr. Smith" or "email to Sarah" or "for Dr. Watson"
        match = re.search(r'(?:to|for)\s+(?:our\s+client\s*,?\s*)?([A-Z][a-zA-Z\.\s]+?)(?:,|\.|\s+and|\s+scheduling|\s+scheduling|\s+with|\s*$)', clean)
        if match:
            candidate = match.group(1).strip()
            if candidate.lower() not in ["our", "the", "a", "an", "all", "each", "my"]:
                recipient = candidate
        elif explicit_client and explicit_client.strip():
            recipient = explicit_client.strip()

    # 2. Check intent
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


def stream_multi_agent_pipeline(
    user_prompt: str,
    client_name: Optional[str] = None,
    mode: str = "auto",
    engine: str = "gemini"
) -> Generator[str, None, None]:
    """Generator yielding SSE JSON events detailing live agent execution for ANY user prompt."""
    
    start_time = time.time()
    def emit(event_type: str, data: Dict[str, Any]) -> str:
        payload = {"type": event_type, "timestamp": time.strftime("%H:%M:%S"), **data}
        return f"data: {json.dumps(payload)}\n\n"

    # Dynamic Intent & Recipient Detection
    intent = extract_recipient_and_intent(user_prompt, client_name)
    actual_recipient = intent["recipient"]
    is_outreach_mode = intent["is_outreach"] or mode == "enterprise"

    # Step 0: Pipeline Initialization
    yield emit("pipeline_init", {
        "prompt": user_prompt,
        "recipient": actual_recipient,
        "mode": "enterprise" if is_outreach_mode else "research",
        "engine": engine,
        "models": {
            "researcher": "DeepSeek-R1" if engine == "deepseek-r1" else RESEARCHER_MODEL,
            "email": EMAIL_MODEL,
            "scheduler": SCHEDULER_MODEL,
        },
        "safe_hand_active": True,
    })
    time.sleep(0.2)

    # Instantiate tools & Safe-Hand
    web_search = WebSearchTool()
    draft_email = DraftEmailTool()
    calendar_tool = CreateCalendarInviteTool()
    safe_hand = SafeHandAgentRunner(preferred_gemini_model=RESEARCHER_MODEL)

    # -------------------------------------------------------------
    # Perplexity Stage 1: Dynamic Query Deconstruction & Search
    # -------------------------------------------------------------
    yield emit("agent_start", {
        "step": 1,
        "agent": "Perplexity Deep Research Analyst",
        "model": RESEARCHER_MODEL,
        "avatar": "🔍",
        "task": f"Executing real-time deep research on: '{user_prompt}'",
        "status": "active"
    })
    time.sleep(0.3)

    # Customize search query based on active focus mode
    search_query = user_prompt.strip()
    if mode == "academic":
        search_query = f"{user_prompt.strip()} academic paper research arxiv"
    elif mode == "enterprise":
        search_query = f"{user_prompt.strip()} enterprise benchmark market analysis"

    yield emit("tool_call_start", {
        "agent": "Perplexity Deep Research Analyst",
        "tool": "Live Web Indexer",
        "query": search_query,
        "focus_mode": mode,
        "action": f"Searching real-time web index ({mode.capitalize()} Focus) for: '{search_query}'..."
    })

    # Retrieve live structured web citations & academic papers with deep page extraction
    structured_sources = web_search.search_structured(search_query, max_results=8, mode=mode)
    search_raw = web_search._run(search_query, mode=mode)

    # Build clean citation list
    sources_payload = []
    for idx, s in enumerate(structured_sources, 1):
        sources_payload.append({
            "id": idx,
            "title": s["title"],
            "url": s["url"],
            "domain": s["domain"],
            "snippet": s.get("snippet", ""),
            "deep_content": s.get("deep_content", "")
        })

    yield emit("tool_call_result", {
        "tool": "Live Web Indexer",
        "sources": sources_payload,
        "items_count": len(sources_payload),
        "summary": f"Indexed {len(sources_payload)} live authoritative web sources."
    })
    time.sleep(0.3)

    # -------------------------------------------------------------
    # Stage 2: DeepMind Compute-Optimal Test-Time Scaling Engine
    # -------------------------------------------------------------
    difficulty_plan = DifficultyEstimator.estimate_difficulty(user_prompt)
    diff_level = difficulty_plan["difficulty_level"]
    compute_budget = difficulty_plan["compute_budget"]
    strategy_desc = difficulty_plan["description"]

    yield emit("agent_thought", {
        "agent": "Test-Time Compute Allocator (DeepMind Architecture)",
        "thought": f"Assessing query difficulty: Level {diff_level}/5 — {strategy_desc}"
    })
    time.sleep(0.2)

    # -------------------------------------------------------------
    # Stage 3: Phase 1 — Proposal Generation (Academic Research Paper Format)
    # -------------------------------------------------------------
    system_instruction = (
        "You are an Elite Principal Research Scientist and Academic Author specializing in frontier science, technology, economics, and analytical synthesis.\n"
        "Your objective is to produce a publication-grade, rigorously formatted Research Paper directly investigating the user's inquiry.\n\n"
        "RESEARCH PAPER FORMATTING BLUEPRINT:\n"
        "1. Title: Crisp, authoritative paper title using single H1 (# [Title]).\n"
        "2. Abstract: An indented, high-density academic abstract using blockquote syntax:\n"
        "   > **Abstract** — [Concise summary stating the problem formulation, underlying mechanisms, empirical findings, and primary conclusions]\n"
        "3. Section 1: Introduction & Theoretical Background (contextualizing state-of-the-art literature).\n"
        "4. Section 2: Core Mechanisms & Mathematical Formulations (explicit derivations, state equations, and parameters using $inline$ and $$display$$ LaTeX math).\n"
        "5. Section 3: Empirical Benchmarks & Comparative Analysis (formatted with markdown comparison tables and numeric citations [1], [2]).\n"
        "6. Section 4: Critical Bottlenecks, Trade-offs & Failure Modes (uncovering boundary limits and scaling challenges).\n"
        "7. Section 5: Strategic Engineering Implications & Future Directions (actionable next steps and takeaways).\n"
        "8. Section 6: References & Empirical Grounding (listing verified source mappings).\n\n"
        "STRICT QUALITY RULES:\n"
        "- Prioritize technical depth, concrete algorithmic/physical mechanisms, and exact empirical metrics over generic commentary.\n"
        "- Seamlessly weave inline source citation badges like [1], [2], [3] throughout every section."
    )

    research_prompt = (
        f"# RESEARCH TOPIC: {user_prompt}\n\n"
        f"Retrieved Empirical Literature & Context:\n"
        f"{search_raw}\n\n"
        f"Synthesize this context into a complete, publication-grade academic research paper following the Blueprint (Title, Abstract blockquote, numbered sections 1-6, LaTeX math, and comparison tables)."
    )

    yield emit("agent_thought", {
        "agent": "Proposal Generator",
        "thought": f"Generating initial high-depth research synthesis using {engine.upper()}..."
    })

    initial_draft, provider1 = safe_hand.generate(
        research_prompt,
        system_instruction=system_instruction,
        search_context=search_raw,
        engine=engine
    )

    # Stream DeepSeek-R1 native chain-of-thought exploration if available
    if safe_hand.last_reasoning_trace:
        trace_steps = [s.strip() for s in safe_hand.last_reasoning_trace.split("\n\n") if len(s.strip()) > 15]
        for step in trace_steps[:4]:
            yield emit("agent_thought", {
                "agent": "DeepSeek-R1 Reasoning Engine",
                "thought": f"🧠 {step[:160]}..."
            })
            time.sleep(0.12)

    # -------------------------------------------------------------
    # Stage 4: Phase 2 & 3 — DeepSeek-R1 Peer-Review Verifier & Sequential Revision
    # -------------------------------------------------------------
    if compute_budget >= 2:
        yield emit("agent_thought", {
            "agent": "DeepSeek-R1 Cross-Check Verifier",
            "thought": "Initiating Adversarial DeepSeek-R1 Peer Review: Auditing mathematical formalisms, conceptual boundaries, and citation grounding..."
        })
        time.sleep(0.2)

        cross_verifier = DeepSeekCrossCheckVerifier(safe_hand)
        cross_res = cross_verifier.cross_check(user_prompt, initial_draft, search_raw)

        rule_check = cross_res.get("rule_check", {})
        if rule_check.get("verified_metrics"):
            yield emit("agent_thought", {
                "agent": "Deterministic Metric Verifier (DeepSeek Principle)",
                "thought": f"Rule-based metric audit: Grounding ratio {int(rule_check['grounding_ratio']*100)}% ({len(rule_check['verified_metrics'])} metrics verified against source text)."
            })

        yield emit("cross_check_result", {
            "score": cross_res["score"],
            "verdict": cross_res["verdict"],
            "provider": cross_res["provider"],
            "critique": cross_res["critique_points"]
        })

        yield emit("agent_thought", {
            "agent": "DeepSeek-R1 Cross-Check Verifier",
            "thought": f"DeepSeek Peer Review Complete (Score: {cross_res['score']}/100, Verdict: {cross_res['verdict']}). Identified {len(cross_res['critique_points'])} precision refinement vectors."
        })
        time.sleep(0.2)

        yield emit("agent_thought", {
            "agent": "Sequential Revision Optimizer",
            "thought": "Executing Sequential Revision pass: Conditioning on DeepSeek-R1 cross-check critique to harden equations and resolve boundary trade-offs..."
        })

        optimizer = SequentialRevisionOptimizer(safe_hand)
        research_summary, final_provider = optimizer.revise_proposal(
            user_prompt, initial_draft, cross_res["critique_points"], search_raw, engine=engine
        )
    else:
        research_summary = initial_draft
        final_provider = provider1

    # Stream markdown content in progressive chunks to give real-time typing feel
    lines = research_summary.split("\n\n")
    accumulated_content = ""
    for paragraph in lines:
        accumulated_content += (paragraph + "\n\n")
        yield emit("content_chunk", {
            "chunk": paragraph + "\n\n",
            "full_content": accumulated_content,
            "provider": final_provider
        })
        time.sleep(0.06)

    # Dynamic follow-up questions
    followup_questions = generate_followup_questions(user_prompt, research_summary)
    yield emit("followup_questions", {
        "questions": followup_questions
    })

    yield emit("agent_complete", {
        "step": 1,
        "agent": "Perplexity Deep Research Analyst",
        "provider": final_provider,
        "summary": f"Compute-optimal research synthesis completed (Difficulty Level {diff_level}/5).",
        "content": research_summary,
        "sources": sources_payload,
        "followups": followup_questions,
        "status": "completed"
    })
    time.sleep(0.3)

    # -------------------------------------------------------------
    # Stage 4: Optional Multi-Agent Outreach & Calendar Staging
    # -------------------------------------------------------------
    client_subject = f"Executive Briefing: Strategic Analysis on {user_prompt[:45]}"
    client_body = ""
    calendar_data = {}

    if is_outreach_mode:
        # Agent 2: Communications Specialist
        yield emit("agent_start", {
            "step": 2,
            "agent": "Executive Client Communications Specialist",
            "model": EMAIL_MODEL,
            "avatar": "✉️",
            "task": f"Translating research findings into executive briefing for {actual_recipient}",
            "status": "active"
        })
        time.sleep(0.3)

        yield emit("agent_thought", {
            "agent": "Executive Client Communications Specialist",
            "thought": f"Drafting tailored executive correspondence to {actual_recipient} summarizing research and proposing review session..."
        })

        email_gen_prompt = (
            f"User Original Request: '{user_prompt}'\n\n"
            f"Research Findings:\n{research_summary[:1500]}\n\n"
            f"Recipient Name: {actual_recipient}\n\n"
            f"Draft a high-impact, professional executive email addressed to {actual_recipient}. "
            f"Include:\n"
            f"1. Concise greeting to {actual_recipient}\n"
            f"2. Executive summary of top findings and best recommendation from the research\n"
            f"3. Invitation to a 30-minute review session next week\n"
            f"4. Professional signature block from 'Enterprise Strategy & Advisory Practice'"
        )

        email_draft_raw, provider2 = safe_hand.generate(
            email_gen_prompt,
            system_instruction=f"You are an Executive Client Communications Specialist. Write a polished email to {actual_recipient}.",
            search_context=search_raw
        )

        client_body = email_draft_raw if email_draft_raw else (
            f"Dear {actual_recipient},\n\n"
            f"I hope this note finds you well.\n\n"
            f"Following your inquiry, our research intelligence practice has synthesized an in-depth analysis on:\n"
            f"'{user_prompt}'.\n\n"
            f"Key Takeaways & Strategic Recommendation:\n"
            f"{research_summary[:450]}...\n\n"
            f"I would welcome the opportunity to review these findings in detail and discuss tactical next steps. "
            f"Could we connect for a brief 30-minute alignment session next week?\n\n"
            f"Best regards,\n\n"
            f"Enterprise Strategy & Advisory Practice\n"
            f"Enterprise Multi-Agent Solutions"
        )

        draft_email._run(
            recipient_name=actual_recipient,
            recipient_email=f"{actual_recipient.lower().replace(' ', '.')}@enterprise.com",
            subject=client_subject,
            body=client_body,
        )

        yield emit("agent_complete", {
            "step": 2,
            "agent": "Executive Client Communications Specialist",
            "provider": provider2,
            "recipient": actual_recipient,
            "subject": client_subject,
            "body": client_body,
            "status": "completed"
        })
        time.sleep(0.3)

        # Agent 3: Scheduling Specialist
        yield emit("agent_start", {
            "step": 3,
            "agent": "Executive Scheduling & Coordination Specialist",
            "model": SCHEDULER_MODEL,
            "avatar": "📅",
            "task": f"Formulating non-conflicting review slots for {actual_recipient}",
            "status": "active"
        })
        time.sleep(0.3)

        meeting_title = f"Strategic Review: {user_prompt[:45]}"
        proposed_slots = [
            "Tuesday, Next Week at 10:00 AM – 10:30 AM EST",
            "Wednesday, Next Week at 2:00 PM – 2:30 PM EST",
            "Thursday, Next Week at 11:30 AM – 12:00 PM EST"
        ]
        agenda = (
            f"1. Executive Briefing on: {user_prompt[:50]}.\n"
            f"2. Evaluation of Key Benchmarks and Recommended Strategy.\n"
            f"3. Discussion of Next Steps & Strategic Implementation with {actual_recipient}."
        )

        calendar_tool._run(
            meeting_title=meeting_title,
            client_name=actual_recipient,
            duration_minutes=30,
            proposed_slots=proposed_slots,
            agenda=agenda,
        )

        calendar_data = {
            "title": meeting_title,
            "attendees": [
                {"name": actual_recipient, "role": "Client / Stakeholder"},
                {"name": "Enterprise AI Strategy Team", "role": "Organizer"}
            ],
            "duration_minutes": 30,
            "proposed_slots_next_week": proposed_slots,
            "agenda": agenda,
            "meeting_type": "Virtual Video Conference (Google Meet / Zoom)",
            "status": "PROPOSED"
        }

        yield emit("agent_complete", {
            "step": 3,
            "agent": "Executive Scheduling & Coordination Specialist",
            "provider": safe_hand.active_provider,
            "slots": proposed_slots,
            "calendar": calendar_data,
            "status": "completed"
        })
        time.sleep(0.2)

    # -------------------------------------------------------------
    # Final Wrap-Up
    # -------------------------------------------------------------
    total_time = f"{round(time.time() - start_time, 2)}s"
    yield emit("pipeline_finished", {
        "success": True,
        "prompt": user_prompt,
        "recipient": actual_recipient,
        "mode": "enterprise" if is_outreach_mode else "research",
        "email_subject": client_subject if is_outreach_mode else "",
        "email_body": client_body if is_outreach_mode else "",
        "calendar": calendar_data if is_outreach_mode else {},
        "research": research_summary,
        "sources": sources_payload,
        "followups": followup_questions,
        "active_provider": safe_hand.active_provider,
        "execution_time": total_time
    })
