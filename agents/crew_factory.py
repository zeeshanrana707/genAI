"""Crew Factory module defining agents, sequential tasks, and the CrewAI orchestration pipeline."""

from typing import Dict, Any, Tuple, Optional
from crewai import Agent, Task, Crew, Process

from config.settings import (
    RESEARCHER_MODEL,
    EMAIL_MODEL,
    SCHEDULER_MODEL,
    OUTPUT_DIR,
    get_llm,
    is_live_key_configured,
    EXECUTION_MODE,
)
from tools.tools import WebSearchTool, DraftEmailTool, CreateCalendarInviteTool
from agents.hooks import HookRegistry, AudioAgentHook, VideoAgentHook, PipelineStage
from agents.stream_orchestrator import extract_recipient_and_intent


def create_enterprise_crew(user_prompt: str = "") -> Tuple[Crew, Dict[str, Any]]:
    """Build distinct agents and sequential tasks for the enterprise system.
    
    Dynamically adapts to any user prompt and extracted recipient.
    """
    intent = extract_recipient_and_intent(user_prompt)
    recipient = intent["recipient"]

    # 1. Instantiate Custom Tools
    web_search_tool = WebSearchTool()
    draft_email_tool = DraftEmailTool()
    calendar_invite_tool = CreateCalendarInviteTool()

    # 2. Configure LLMs
    researcher_llm = get_llm(RESEARCHER_MODEL)
    email_llm = get_llm(EMAIL_MODEL)
    scheduler_llm = get_llm(SCHEDULER_MODEL)

    # 3. Define the Distinct Agents
    researcher_agent = Agent(
        role="Autonomous Principal Research Scientist",
        goal="Conduct deep epistemic analysis: identify why current paradigms fail, uncover hidden contradictions across sources, bridge disconnected domains, and formulate novel, testable hypotheses.",
        backstory=(
            "You are an Autonomous Principal Research Scientist specializing in frontier AI, cognitive architectures, and foundational science. "
            "You never summarize search results or list basic facts. You extract underlying mechanisms, assumptions, failure modes, "
            "and cross-pollinate with adjacent disciplines (thermodynamics, control theory, program synthesis) to propose testable hypotheses."
        ),
        llm=researcher_llm,
        tools=[web_search_tool],
        verbose=True,
        memory=False,
    )

    email_agent = Agent(
        role="Executive Client Communications Specialist",
        goal=f"Synthesize research insights into an executive-ready client communication tailored for {recipient}.",
        backstory=(
            "You are an elite corporate communications expert. You excel at translating complex technical findings "
            "into clear, persuasive, strategic executive communications that drive client decisions."
        ),
        llm=email_llm,
        tools=[draft_email_tool],
        verbose=True,
        memory=False,
    )

    scheduling_agent = Agent(
        role="Executive Scheduling & Coordination Specialist",
        goal=f"Propose concrete 30-minute review meeting slots for next week for {recipient}.",
        backstory=(
            "You are a meticulous executive coordinator responsible for seamless calendar logistics. You establish clear meeting "
            "agendas, avoid scheduling conflicts, and generate frictionless calendar invite specifications."
        ),
        llm=scheduler_llm,
        tools=[calendar_invite_tool],
        verbose=True,
        memory=False,
    )

    # 4. Define Sequential Tasks
    research_task = Task(
        description=(
            "Conduct deep epistemic analysis on: '{user_prompt}'.\n"
            "Process the data through the 4-stage cognitive workflow: (1) Deconstruction, (2) Contradiction & Bottleneck Mapping, "
            "(3) Cross-Pollinated Synthesis, and (4) Hypothesis Generation.\n"
            "Format the output strictly into:\n"
            "1. State-of-the-Art (SOTA) Mechanism Breakdown (Core Mechanism & Hidden Assumptions)\n"
            "2. The Frontier Bottleneck (Why Current Approaches Fail & Explicit Knowledge Gap)\n"
            "3. Novel Research Hypotheses & Architectural Proposals (Synthesis, Architecture, Theoretical Justification)\n"
            "4. Falsification & Experimental Design (Baseline vs Proposed, Benchmark Environment, Falsification Criteria)."
        ),
        expected_output="An in-depth epistemic research report structured into the 4 required scientific sections without generic definitions.",
        agent=researcher_agent,
        tools=[web_search_tool],
    )

    email_task = Task(
        description=(
            f"Using the research report, compose an executive business email addressed to {recipient}.\n"
            "Highlight top takeaways, provide strategic recommendations, and propose a 30-minute review meeting next week.\n"
            "Use the Draft Email Tool to stage and save the formatted email to the output directory."
        ),
        expected_output=f"A professionally formatted client email addressed to {recipient}, staged in the output directory.",
        agent=email_agent,
        tools=[draft_email_tool],
        context=[research_task],
    )

    schedule_task = Task(
        description=(
            f"Based on the drafted email to {recipient}, generate 3 specific, non-conflicting 30-minute time slots for next week.\n"
            f"Define a structured agenda addressing: '{user_prompt[:50]}'.\n"
            "Use the Create Calendar Invite Tool to generate the structured invite and save it to the output directory."
        ),
        expected_output="A structured calendar invite with 3 proposed 30-minute meeting slots next week, saved in the output directory.",
        agent=scheduling_agent,
        tools=[calendar_invite_tool],
        context=[email_task],
    )

    # 5. Build Crew with Sequential Process
    crew = Crew(
        agents=[researcher_agent, email_agent, scheduling_agent],
        tasks=[research_task, email_task, schedule_task],
        process=Process.sequential,
        verbose=True,
    )

    pipeline_meta = {
        "recipient": recipient,
        "agents": {
            "researcher": researcher_agent,
            "email": email_agent,
            "scheduler": scheduling_agent,
        },
        "tasks": {
            "research": research_task,
            "email": email_task,
            "schedule": schedule_task,
        },
        "tools": {
            "web_search": web_search_tool,
            "draft_email": draft_email_tool,
            "calendar_invite": calendar_invite_tool,
            "create_calendar_invite": calendar_invite_tool,
        },
    }

    return crew, pipeline_meta


def run_pipeline(user_prompt: str) -> Dict[str, Any]:
    """Execute the multi-agent pipeline with hook integration and graceful fallback.
    
    Args:
        user_prompt: Initial instruction prompt for the multi-agent crew.
        
    Returns:
        Dictionary containing pipeline results, artifacts, and hook outputs.
    """
    intent = extract_recipient_and_intent(user_prompt)
    recipient = intent["recipient"]

    print("=" * 80)
    print("[*] INITIALIZING PERPLEXITY RESEARCH & MULTI-AGENT SYSTEM")
    print("=" * 80)
    print(f"User Prompt : '{user_prompt}'")
    print(f"Recipient   : '{recipient}'")
    print(f"Models      : Researcher: {RESEARCHER_MODEL} | Email: {EMAIL_MODEL} | Scheduler: {SCHEDULER_MODEL}")
    print(f"Execution   : {EXECUTION_MODE} | Live Key: {is_live_key_configured()}")
    print("-" * 80)

    # Initialize Extension Hooks
    hooks = HookRegistry()
    hooks.register(AudioAgentHook())
    hooks.register(VideoAgentHook())

    crew, meta = create_enterprise_crew(user_prompt)
    context: Dict[str, Any] = {"user_prompt": user_prompt, "recipient": recipient}

    # Dynamic Safe-Hand Execution
    from agents.failover import SafeHandAgentRunner
    safe_hand = SafeHandAgentRunner(preferred_gemini_model=RESEARCHER_MODEL)

    # Step 1: Research Agent
    print(f"\n[Agent 1/3] Autonomous Principal Research Scientist ({RESEARCHER_MODEL})")
    print(f"  Task: Conducting epistemic analysis & hypothesis generation for: '{user_prompt}'...")
    search_tool = meta["tools"]["web_search"]
    research_raw = search_tool._run(user_prompt)

    system_instruction = (
        "You are an Autonomous Principal Research Scientist specializing in frontier AI, cognitive architectures, and foundational science.\n"
        "Your goal is NOT to summarize search results or list basic facts. Your goal is to conduct deep epistemic analysis: "
        "identify why current paradigms fail, uncover hidden contradictions across sources, bridge disconnected domains, and formulate novel, testable hypotheses.\n\n"
        "STRICT ANTI-SUMMARIZATION RULES:\n"
        "1. NEVER output generic introductory definitions. Assume the user is a domain expert.\n"
        "2. NEVER treat a source's conclusion as settled fact. Extract underlying mechanisms, compute/data assumptions, and failure modes.\n"
        "3. Prioritize primary technical bottlenecks over high-level commentary."
    )

    research_prompt = (
        f"# RESEARCH MISSION: {user_prompt}\n\n"
        f"Retrieved Empirical Literature & Context:\n"
        f"{research_raw}\n\n"
        f"Execute your 4-stage cognitive workflow:\n"
        f"1. Deconstruction (exact mathematical, architectural, or algorithmic mechanisms)\n"
        f"2. Contradiction & Bottleneck Mapping (where sources diverge, 100x scaling breakages, OOD collapse)\n"
        f"3. Cross-Pollinated Synthesis (mechanisms from adjacent fields)\n"
        f"4. Hypothesis Generation (concrete architectural modifications)\n\n"
        f"Structure your response strictly into these 4 sections:\n"
        f"## 1. State-of-the-Art (SOTA) Mechanism Breakdown\n"
        f"- Core Mechanism\n- Hidden Assumptions\n\n"
        f"## 2. The Frontier Bottleneck (Why Current Approaches Fail)\n"
        f"- Conflicting empirical results & explicit Knowledge Gap\n\n"
        f"## 3. Novel Research Hypotheses & Architectural Proposals\n"
        f"- The Synthesis\n- Proposed Architecture / Mechanism\n- Why It Overcomes the Bottleneck\n\n"
        f"## 4. Falsification & Experimental Design\n"
        f"- Baseline vs. Proposed Model\n- Dataset / Environment\n- Falsification Criteria"
    )

    research_summary, provider1 = safe_hand.generate(
        research_prompt,
        system_instruction=system_instruction,
        search_context=research_raw
    )

    context["research_output"] = research_summary
    context["active_provider"] = provider1
    context = hooks.trigger(PipelineStage.POST_RESEARCH, research_summary, context)

    # Step 2: Email Agent
    print(f"\n[Agent 2/3] Executive Client Communications Specialist ({EMAIL_MODEL})")
    print(f"  Task: Drafting client email for {recipient} based on research findings...")
    email_tool = meta["tools"]["draft_email"]

    email_gen_prompt = (
        f"Original User Request: '{user_prompt}'\n\n"
        f"Research Findings:\n{research_summary[:1500]}\n\n"
        f"Recipient Name: {recipient}\n\n"
        f"Draft a high-impact, professional executive email addressed to {recipient}. "
        f"Summarize key findings, state our recommendation, and propose a 30-minute review session next week."
    )
    email_draft_raw, provider2 = safe_hand.generate(
        email_gen_prompt,
        system_instruction=f"You are an Executive Client Communications Specialist. Write a polished email to {recipient}.",
        search_context=research_raw
    )

    client_subject = f"Executive Briefing: Research & Strategy on {user_prompt[:45]}"
    client_body = email_draft_raw if email_draft_raw else (
        f"Dear {recipient},\n\n"
        f"I hope this note finds you well.\n\n"
        f"Following your inquiry, our research intelligence team has synthesized an in-depth analysis regarding:\n"
        f"'{user_prompt}'.\n\n"
        f"Key Takeaways & Findings:\n"
        f"{research_summary[:450]}...\n\n"
        f"I would welcome the opportunity to review these findings in detail and discuss tactical next steps. "
        f"Could we connect for a brief 30-minute alignment session next week?\n\n"
        f"Best regards,\n\n"
        f"Enterprise Strategy & Advisory Practice\n"
        f"Enterprise Multi-Agent Solutions"
    )

    email_tool._run(
        recipient_name=recipient,
        recipient_email=f"{recipient.lower().replace(' ', '.')}@enterprise.com",
        subject=client_subject,
        body=client_body,
    )
    context["email_output"] = client_body
    context["email_subject"] = client_subject
    context = hooks.trigger(PipelineStage.POST_EMAIL, client_body, context)

    # Step 3: Scheduling Agent
    print(f"\n[Agent 3/3] Executive Scheduling & Coordination Specialist ({SCHEDULER_MODEL})")
    print(f"  Task: Formulating non-conflicting 30-minute slots for {recipient}...")
    calendar_tool = meta["tools"]["create_calendar_invite"]

    proposed_slots = [
        "Tuesday, Next Week at 10:00 AM – 10:30 AM EST",
        "Wednesday, Next Week at 2:00 PM – 2:30 PM EST",
        "Thursday, Next Week at 11:30 AM – 12:00 PM EST"
    ]
    agenda = (
        f"1. Executive Briefing on Research Findings: {user_prompt[:50]}.\n"
        f"2. Evaluation of Strategic Options & Implementation Priorities.\n"
        f"3. Alignment with {recipient} on Next Milestones."
    )

    calendar_tool._run(
        meeting_title=f"Review Session: {user_prompt[:45]}",
        client_name=recipient,
        duration_minutes=30,
        proposed_slots=proposed_slots,
        agenda=agenda,
    )
    context["scheduling_output"] = proposed_slots
    context = hooks.trigger(PipelineStage.POST_SCHEDULE, proposed_slots, context)

    print("\n" + "=" * 80)
    print("[SUCCESS] MULTI-AGENT PIPELINE SUCCESSFULLY COMPLETED")
    print(f"1. Research Summary generated ({provider1}).")
    print(f"2. Client email staged for: {recipient} at {OUTPUT_DIR / 'draft_email.txt'}")
    print(f"3. Calendar invite staged at: {OUTPUT_DIR / 'calendar_invite.json'}")
    print("=" * 80)

    return context
