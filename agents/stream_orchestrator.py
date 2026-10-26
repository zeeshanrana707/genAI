"""Real-time Streaming Orchestrator for Perplexity-Grade Deep Research & Multi-Agent Operations.

Emits Server-Sent Events (SSE) detailing live agent execution across 5 specialized autonomous agents:
- Agent 1: Dr. Aris Thorne (Principal Research Scientist) - Web & arXiv Literature Mining
- Agent 2: Prof. Elena Rostova (DeepSeek Adversarial Critic) - DeepSeek-R1 Peer Review & Rule-Based Metric Audit
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
    engine: str = "gemini"
) -> Generator[str, None, None]:
    """Generator yielding SSE JSON events detailing live agent execution for ANY user prompt."""
    
    start_time = time.time()
    def emit(event_type: str, data: Dict[str, Any]) -> str:
        payload = {"type": event_type, "timestamp": time.strftime("%H:%M:%S"), **data}
        return f"data: {json.dumps(payload)}\n\n"

    intent = extract_recipient_and_intent(user_prompt, client_name)
    actual_recipient = intent["recipient"]

    agents_roster = [
        {"id": "researcher", "name": "Dr. Aris Thorne", "role": "Principal Research Scientist", "avatar": "🔬", "model": "Gemini 3.1 Flash / Web Indexer"},
        {"id": "critic", "name": "Prof. Elena Rostova", "role": "DeepSeek Adversarial Critic", "avatar": "⚖️", "model": "DeepSeek-R1 Peer Review Protocol"},
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
    safe_hand = SafeHandAgentRunner(preferred_gemini_model=RESEARCHER_MODEL)

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

    search_query = user_prompt.strip()
    if mode == "academic":
        search_query = f"{user_prompt.strip()} academic paper research arxiv"
    elif mode == "enterprise":
        search_query = f"{user_prompt.strip()} enterprise benchmark market analysis"

    yield emit("tool_call_start", {
        "agent": "Dr. Aris Thorne",
        "tool": "Live Web & arXiv Indexer",
        "query": search_query,
        "focus_mode": mode,
        "action": f"Indexing empirical sources for: '{search_query}'..."
    })

    structured_sources = web_search.search_structured(search_query, max_results=8, mode=mode)
    search_raw = web_search._run(search_query, mode=mode)

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
        "Structure with Title, Abstract blockquote (> **Abstract** — ...), Introduction, Core Mechanisms with LaTeX math, Benchmark comparisons, and References."
    )

    research_prompt = (
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

    yield emit("agent_handoff", {
        "from_agent": "Dr. Aris Thorne (Research Scientist)",
        "to_agent": "Prof. Elena Rostova (DeepSeek Adversarial Critic)",
        "message": "Candidate draft prepared. Passing to Prof. Rostova for adversarial peer-review audit and metric verification."
    })
    time.sleep(0.25)

    # =============================================================
    # AGENT 2: Prof. Elena Rostova (DeepSeek Adversarial Critic)
    # =============================================================
    yield emit("agent_status_change", {
        "agent_id": "critic",
        "status": "working",
        "action": "Auditing mathematical equations, empirical metrics, and conceptual boundaries..."
    })
    yield emit("agent_start", {
        "step": 2,
        "agent_id": "critic",
        "agent": "Prof. Elena Rostova (DeepSeek Adversarial Critic)",
        "model": "DeepSeek-R1 Verification Engine",
        "avatar": "⚖️",
        "task": "Executing DeepSeek-R1 adversarial peer review and metric cross-check",
        "status": "active"
    })
    time.sleep(0.2)

    yield emit("agent_thought", {
        "agent": "Prof. Elena Rostova",
        "thought": "Commencing adversarial audit across 4 axes: (1) PDEs and mathematical formalisms, (2) conceptual boundary conditions, (3) deterministic metric grounding, and (4) citation veracity."
    })

    cross_verifier = DeepSeekCrossCheckVerifier(safe_hand)
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
        "auditor": "Prof. Elena Rostova (DeepSeek Protocol)"
    }

    yield emit("cross_check_result", peer_review_deliverable)
    yield emit("agent_deliverable", {
        "agent_id": "critic",
        "type": "peer_review",
        "data": peer_review_deliverable
    })

    yield emit("agent_status_change", {
        "agent_id": "critic",
        "status": "completed",
        "action": f"Peer review complete. Audit Score: {cross_res['score']}/100 ({cross_res['verdict']})."
    })

    yield emit("agent_handoff", {
        "from_agent": "Prof. Elena Rostova (Adversarial Critic)",
        "to_agent": "Dr. Marcus Vance (Senior Academic Editor)",
        "message": f"Peer-review audit complete ({cross_res['score']}/100). Delivering critique points to Dr. Vance for publication-grade synthesis."
    })
    time.sleep(0.25)

    # =============================================================
    # AGENT 3: Dr. Marcus Vance (Senior Academic Editor)
    # =============================================================
    yield emit("agent_status_change", {
        "agent_id": "editor",
        "status": "working",
        "action": "Synthesizing final publication-grade academic paper with LaTeX math..."
    })
    yield emit("agent_start", {
        "step": 3,
        "agent_id": "editor",
        "agent": "Dr. Marcus Vance (Senior Academic Editor)",
        "model": RESEARCHER_MODEL,
        "avatar": "✍️",
        "task": "Performing sequential revision to publish final academic research paper",
        "status": "active"
    })
    time.sleep(0.2)

    yield emit("agent_thought", {
        "agent": "Dr. Marcus Vance",
        "thought": "Conditioning on Prof. Rostova's adversarial critique: hardening equations, clarifying physical boundaries, and formatting IEEE/Nature style abstract..."
    })

    optimizer = SequentialRevisionOptimizer(safe_hand)
    research_summary, final_provider = optimizer.revise_proposal(
        user_prompt, initial_draft, cross_res["critique_points"], search_raw, engine=engine
    )

    paragraphs = research_summary.split("\n\n")
    accumulated_content = ""
    for paragraph in paragraphs:
        accumulated_content += (paragraph + "\n\n")
        yield emit("content_chunk", {
            "chunk": paragraph + "\n\n",
            "full_content": accumulated_content,
            "provider": final_provider
        })
        time.sleep(0.035)

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
        f"2. Evaluation of Key Benchmarks & DeepSeek Peer-Review Findings.\n"
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
        "sources": sources_payload,
        "followups": followup_questions,
        "active_provider": safe_hand.active_provider,
        "execution_time": total_time,
        "agents_completed": 5
    })
