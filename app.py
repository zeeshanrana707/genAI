"""Modern Web Application for Autonomous Multi-Agent Deep Research & Enterprise System.

Provides a Perplexity-style UI for real-time web research on any topic,
interactive citation badges, dynamic agent reasoning, and optional outreach staging.
"""

import sys
import os
import io
import time
import json
import uuid
import smtplib
import base64
import re
import subprocess
import tempfile
import asyncio
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import quote
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
import uvicorn
import pdfplumber

from fastapi.staticfiles import StaticFiles
from config.settings import (
    RESEARCHER_MODEL,
    EMAIL_MODEL,
    SCHEDULER_MODEL,
    OUTPUT_DIR,
    PAPERS_DIR,
    is_live_key_configured,
)
from agents.crew_factory import run_pipeline
from agents.failover import SafeHandAgentRunner, call_free_ai_api
from agents.stream_orchestrator import stream_multi_agent_pipeline
from agents.stream_bus import stream_bus
from tools.secure_sandbox import SecureSandboxRunner
import database

PLOTS_DIR = Path(__file__).resolve().parent / "output" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Perplexity Deep Research & Multi-Agent Enterprise Canvas")
app.mount("/output/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


def resolve_user_id(request: Request, explicit_user_id: Optional[str] = None) -> str:
    """Extract user_id from explicit argument, header, or query param, falling back to guest."""
    if explicit_user_id and explicit_user_id.strip() not in ("", "undefined", "null"):
        return explicit_user_id.strip()
    header_val = request.headers.get("X-User-Id")
    if header_val and header_val.strip() not in ("", "undefined", "null"):
        return header_val.strip()
    query_val = request.query_params.get("user_id")
    if query_val and query_val.strip() not in ("", "undefined", "null"):
        return query_val.strip()
    return "guest"



class CanvasRefineRequest(BaseModel):
    selection: str
    instruction: str
    context: str = ""
    gemini_api_key: Optional[str] = None


@app.post("/api/canvas/refine")
async def canvas_refine(req: CanvasRefineRequest):
    """OpenAI Canvas-style inline section refinement and mathematical fortification."""
    selection = req.selection.strip()
    instruction = req.instruction.strip()
    if not selection:
        return JSONResponse(status_code=400, content={"success": False, "error": "No text selection provided."})

    safe_hand = SafeHandAgentRunner(preferred_gemini_model=RESEARCHER_MODEL, api_key=req.gemini_api_key)
    prompt = (
        f"You are Dr. Marcus Vance, Senior Academic Editor at an elite theoretical research institute.\n"
        f"The user selected the following specific section of an academic research paper to refine:\n\n"
        f"=== SELECTED SECTION ===\n{selection}\n\n"
        f"=== SURROUNDING PAPER CONTEXT ===\n{req.context[:2500]}\n\n"
        f"=== USER REFINEMENT INSTRUCTION ===\n{instruction}\n\n"
        f"Task:\n"
        f"1. Refactor and rewrite the selected section addressing the user instruction with complete mathematical rigor, explicit derivations, and LaTeX equations ($...$ and $$...$$).\n"
        f"2. Preserve any existing citation badges [1], [2] where appropriate.\n"
        f"3. Return ONLY the drop-in replacement text for the selected section. Do not include conversational preambles or repeating text outside the selection."
    )

    refined_text, provider = safe_hand.generate(
        prompt,
        system_instruction="You are an elite academic editor specializing in localized section refactoring and mathematical proof fortification."
    )

    return {
        "success": True,
        "original": selection,
        "refined": refined_text.strip(),
        "provider": provider
    }


@app.get("/api/stream")
async def stream_agents(
    request: Request,
    prompt: str = "",
    client: Optional[str] = None,
    mode: str = "auto",
    engine: str = "gemini",
    session_id: Optional[str] = None,
    stream_id: Optional[str] = None,
    paper_ids: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    user_id: Optional[str] = None
):
    active_user_id = resolve_user_id(request, user_id)
    """Server-Sent Events streaming endpoint with decoupled StreamEventBus, Last-Event-ID replay, and SQLite WAL persistence."""
    # 1. Detect Last-Event-ID for reconnection from either HTTP header or query param
    last_event_id: Optional[int] = None
    header_val = request.headers.get("last-event-id")
    if header_val and header_val.strip().isdigit():
        last_event_id = int(header_val.strip())
    elif request.query_params.get("last_event_id") and request.query_params.get("last_event_id").strip().isdigit():
        last_event_id = int(request.query_params.get("last_event_id").strip())

    # 2. Reconnection check: if stream_id or session_id matches an existing channel
    existing_channel = None
    if stream_id:
        existing_channel = stream_bus.get_channel(stream_id)
    elif session_id and (last_event_id is not None or not prompt.strip()):
        active_sid = stream_bus.get_active_stream_for_session(session_id)
        if active_sid:
            existing_channel = stream_bus.get_channel(active_sid)
            if existing_channel:
                stream_id = active_sid

    # 2a. Handle reconnection without re-running agent computation
    if existing_channel:
        target_stream_id = existing_channel.stream_id

        async def reconnect_stream_generator():
            q = asyncio.Queue()
            existing_channel.subscribe(q)
            latest_seen = last_event_id if last_event_id is not None else 0
            try:
                # Replay any buffered events strictly greater than last_event_id
                missed = existing_channel.get_events_since(latest_seen)
                for ev in missed:
                    if ev.event_id > latest_seen:
                        latest_seen = ev.event_id
                        yield ev.raw

                # If already completed and caught up, terminate gracefully
                if existing_channel.is_completed:
                    return

                # Otherwise stream incoming live events as agents produce them
                while True:
                    event_obj = await q.get()
                    if event_obj is None:
                        break
                    if event_obj.event_id > latest_seen:
                        latest_seen = event_obj.event_id
                        yield event_obj.raw
            finally:
                existing_channel.unsubscribe(q)

        return StreamingResponse(
            reconnect_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Stream-ID": target_stream_id,
            }
        )

    # 3. New Stream Execution: Validate prompt
    if not prompt.strip():
        return JSONResponse(status_code=400, content={"error": "Prompt query parameter is required for a new research run."})

    new_stream_id = stream_id or f"stream_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    channel = stream_bus.create_channel(new_stream_id, session_id=session_id)
    loop = asyncio.get_running_loop()

    # Retrieve prior session research context and sources from persistent SQLite DB
    prior_context = ""
    prior_sources = []
    if session_id:
        history_list = database.get_session_history(session_id, limit=6, user_id=active_user_id)
        if history_list:
            last_t = history_list[-1]
            prior_context = last_t.get("research", "")
            prior_sources = last_t.get("sources", [])

            if len(history_list) > 1:
                history_blocks = []
                for h in history_list[-3:]:
                    history_blocks.append(
                        f"Turn Query: {h.get('prompt', '')}\n"
                        f"Findings Summary:\n{h.get('research', '')[:2000]}"
                    )
                prior_context = (
                    "=== PRIOR CONVERSATION HISTORY ===\n\n" +
                    "\n\n---\n\n".join(history_blocks) +
                    "\n\n=== LATEST RESEARCH REPORT ===\n\n" +
                    last_t.get("research", "")
                )

    # Retrieve text from selected uploaded reference papers via SQLite
    uploaded_text_parts = []
    if paper_ids:
        p_ids = [pid.strip() for pid in paper_ids.split(",") if pid.strip()]
        papers_data = database.get_uploaded_paper_texts(p_ids, user_id=active_user_id)
        for p in papers_data:
            uploaded_text_parts.append(
                f"=== UPLOADED PAPER: {p['filename']} ===\n"
                f"{p['text'][:4000]}"
            )
    uploaded_papers_text = "\n\n".join(uploaded_text_parts)

    # 4. Decoupled Background Agent Worker Thread
    def run_pipeline_worker(target_id: str):
        accumulated_research = ""
        accumulated_sources = []
        plot_b64 = ""
        plot_path = ""
        deliverables_dict = {}

        try:
            for sse_event in stream_multi_agent_pipeline(
                user_prompt=prompt,
                client_name=client,
                mode=mode,
                engine=engine,
                prior_context=prior_context,
                prior_sources=prior_sources,
                uploaded_papers_text=uploaded_papers_text,
                gemini_api_key=gemini_api_key
            ):
                # Cooperative cancellation check
                if not stream_bus.is_channel_running(target_id):
                    print(f"[StreamBus] Worker halted for cancelled stream {target_id}")
                    break

                if "pipeline_finished" in sse_event:
                    try:
                        clean = sse_event.strip().replace("data: ", "", 1)
                        parsed = json.loads(clean)
                        if parsed.get("research"):
                            accumulated_research = parsed["research"]
                        if parsed.get("sources"):
                            accumulated_sources = parsed["sources"]
                        plot_b64 = parsed.get("plot_base64", "")
                        plot_path = parsed.get("plot_path", "")
                        deliverables_dict = {
                            "email_subject": parsed.get("email_subject"),
                            "email_body": parsed.get("email_body"),
                            "calendar": parsed.get("calendar"),
                            "peer_review": parsed.get("peer_review"),
                            "draft_v1": parsed.get("draft_v1", ""),
                            "draft_v2": parsed.get("draft_v2", "")
                        }
                    except Exception as parse_err:
                        print(f"[Pipeline Finished Parse Error]: {parse_err}")
                elif "tool_call_result" in sse_event:
                    try:
                        clean = sse_event.strip().replace("data: ", "", 1)
                        parsed = json.loads(clean)
                        if parsed.get("sources"):
                            accumulated_sources = parsed["sources"]
                    except Exception:
                        pass

                stream_bus.publish(target_id, sse_event, loop=loop)

        except Exception as exc:
            print(f"[Stream Worker Error]: {exc}")
            err_event = f"data: {json.dumps({'type': 'pipeline_error', 'error': str(exc)})}\n\n"
            stream_bus.publish(target_id, err_event, loop=loop)
        finally:
            # Persist state to SQLite WAL
            if session_id and accumulated_research:
                turn_uid = f"turn_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                try:
                    database.save_turn(
                        session_id=session_id,
                        turn_id=turn_uid,
                        prompt=prompt,
                        research=accumulated_research,
                        sources=accumulated_sources,
                        deliverables=deliverables_dict,
                        plot_path=plot_path,
                        user_id=active_user_id
                    )
                    print(f"[SQLite WAL] Persisted turn {turn_uid} for session {session_id}")
                except Exception as db_err:
                    print(f"[Database Save Error]: {db_err}")

            stream_bus.finish_stream(target_id, loop=loop)

    worker_thread = threading.Thread(target=run_pipeline_worker, args=(new_stream_id,), daemon=True)
    worker_thread.start()

    # 5. Async HTTP Streaming Generator
    async def event_stream_wrapper():
        q = asyncio.Queue()
        channel.subscribe(q)
        latest_seen = 0
        try:
            # Yield any events published immediately upon thread launch
            missed = channel.get_events_since(0)
            for ev in missed:
                if ev.event_id > latest_seen:
                    latest_seen = ev.event_id
                    yield ev.raw

            while True:
                event_obj = await q.get()
                if event_obj is None:
                    break
                if event_obj.event_id > latest_seen:
                    latest_seen = event_obj.event_id
                    yield event_obj.raw
        finally:
            channel.unsubscribe(q)

    return StreamingResponse(
        event_stream_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Stream-ID": new_stream_id,
        }
    )


@app.post("/api/stream/cancel")
async def cancel_stream_endpoint(request: Request):
    """Cancel an active agent research stream."""
    try:
        data = await request.json()
        target_sid = data.get("stream_id")
        target_sess = data.get("session_id")
        if not target_sid and target_sess:
            target_sid = stream_bus.get_active_stream_for_session(target_sess)
        if target_sid:
            stream_bus.cancel_stream(target_sid)
            return {"success": True, "message": f"Stream {target_sid} cancelled."}
        return {"success": False, "message": "No active stream found."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/upload-paper")
async def upload_paper(request: Request, file: UploadFile = File(...)):
    """Upload research paper (PDF), extract text with pdfplumber, and index persistently in user's isolated SQLite database."""
    try:
        active_user_id = resolve_user_id(request)
        if not file.filename.lower().endswith(".pdf"):
            return JSONResponse(status_code=400, content={"success": False, "error": "Only PDF files are supported."})

        file_bytes = await file.read()
        paper_id = str(uuid.uuid4())[:8]
        save_path = PAPERS_DIR / f"{paper_id}_{file.filename}"
        save_path.write_bytes(file_bytes)

        # Extract text using pdfplumber
        extracted_text = []
        page_count = 0
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages[:30]:  # Up to 30 pages
                text = page.extract_text()
                if text:
                    extracted_text.append(text)

        full_text = "\n\n".join(extracted_text)
        if not full_text.strip():
            full_text = f"[PDF loaded: {file.filename} ({page_count} pages). No extractable text stream detected.]"

        database.save_uploaded_paper(
            paper_id=paper_id,
            filename=file.filename,
            page_count=page_count,
            text=full_text,
            user_id=active_user_id
        )

        return {
            "success": True,
            "paper": {
                "id": paper_id,
                "filename": file.filename,
                "page_count": page_count,
                "char_count": len(full_text),
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "user_id": active_user_id
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/papers")
async def list_papers(request: Request):
    """List all uploaded reference research papers from user's isolated SQLite database."""
    active_user_id = resolve_user_id(request)
    papers_summary = database.get_all_uploaded_papers(user_id=active_user_id)
    return {"papers": papers_summary, "user_id": active_user_id}


@app.delete("/api/papers/{paper_id}")
async def delete_paper(request: Request, paper_id: str):
    """Delete an uploaded reference paper from user's isolated database."""
    active_user_id = resolve_user_id(request)
    success = database.delete_uploaded_paper(paper_id, user_id=active_user_id)
    if success:
        return {"success": True, "message": "Paper removed."}
    return JSONResponse(status_code=404, content={"success": False, "error": "Paper not found."})


@app.get("/api/sessions")
async def list_sessions(request: Request):
    """Retrieve persistent research sessions from user's isolated SQLite database."""
    active_user_id = resolve_user_id(request)
    sessions_list = database.get_all_sessions(limit=25, user_id=active_user_id)
    return {"sessions": sessions_list, "user_id": active_user_id}


@app.get("/api/session/{session_id}")
async def get_session(request: Request, session_id: str):
    """Retrieve prior research history and all turns for active session from user's isolated SQLite database."""
    active_user_id = resolve_user_id(request)
    history = database.get_session_history(session_id, limit=30, user_id=active_user_id)
    if not history:
        return {"exists": False, "turns": [], "last_research": "", "history_count": 0, "user_id": active_user_id}
    last_t = history[-1]
    return {
        "exists": True,
        "turns": history,
        "last_research": last_t.get("research", ""),
        "history_count": len(history),
        "user_id": active_user_id
    }


@app.delete("/api/session/{session_id}")
async def reset_session(request: Request, session_id: str):
    """Reset research session memory in user's isolated SQLite database."""
    active_user_id = resolve_user_id(request)
    database.reset_session(session_id, user_id=active_user_id)
    return {"success": True, "message": "Session research memory reset.", "user_id": active_user_id}


# =====================================================================
# Identity & Authentication Routes
# =====================================================================

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/signup")
async def auth_signup(req: SignupRequest):
    """Register a new user account with secure SQLite persistence."""
    try:
        user = database.create_user(req.name, req.email, req.password)
        return {"success": True, "user": user}
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"success": False, "error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Registration error: {str(e)}"})


@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    """Authenticate existing credentials against salted sha256 hash in SQLite."""
    try:
        user = database.authenticate_user(req.email, req.password)
        return {"success": True, "user": user}
    except ValueError as ve:
        return JSONResponse(status_code=401, content={"success": False, "error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Authentication error: {str(e)}"})


class PipelineRequest(BaseModel):
    prompt: str
    client_name: Optional[str] = None
    mode: Optional[str] = "auto"
    duration_minutes: Optional[int] = 30
    preferred_model: Optional[str] = None


class SendBriefRequest(BaseModel):
    recipient_email: str
    recipient_name: Optional[str] = "Colleague"
    topic: str
    research_content: str
    custom_note: Optional[str] = ""


@app.post("/api/send-brief")
async def send_brief(req: SendBriefRequest):
    """Synthesize a short, descriptive executive summary and main theme, then dispatch to receiver."""
    try:
        clean_email = req.recipient_email.strip()
        if not clean_email or "@" not in clean_email:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid email address format."})

        recipient_name = req.recipient_name.strip() if req.recipient_name else "Colleague"
        topic = req.topic.strip() or "Research Brief"
        note_text = f"\nNote from Sender: {req.custom_note.strip()}\n" if req.custom_note and req.custom_note.strip() else ""

        # Safe-Hand LLM synthesis for a short, descriptive theme and summary
        safe_hand = SafeHandAgentRunner(preferred_gemini_model=EMAIL_MODEL)
        synth_prompt = (
            f"You are an Executive Briefing Specialist.\n"
            f"Topic: '{topic}'\n"
            f"Research Findings Content:\n{req.research_content[:2500]}\n\n"
            f"Receiver Name: {recipient_name}\n"
            f"Receiver Email: {clean_email}\n"
            f"{note_text}\n"
            f"TASK:\n"
            f"Write a SHORT but DESCRIPTIVE executive email summary capturing the CORE THEME and KEY TAKEAWAYS of this research.\n\n"
            f"FORMAT GUIDELINES:\n"
            f"- Line 1 MUST start with: 'SUBJECT: Executive Brief: [Core Theme in 4-8 words]'\n"
            f"- Greeting: 'Dear {recipient_name},' (or 'Hi {recipient_name},')\n"
            f"- Section 1: Main Theme & Context (2-3 crisp sentences summarizing the big picture and state of play)\n"
            f"- Section 2: Key Takeaways (3 distinct bullet points with data, features, or benchmarks)\n"
            f"- Section 3: Bottom Line / Strategic Next Step (1 punchy closing sentence)\n"
            f"- Sign-off: 'Best regards,\nEnterprise Research Intelligence'"
        )

        generated_raw, provider = safe_hand.generate(
            synth_prompt,
            system_instruction="You are an Executive Briefing Specialist. Create a concise, high-impact research brief email."
        )

        # Parse Subject and Body
        subject = f"Executive Brief: Strategic Insights on {topic[:40]}"
        body = generated_raw.strip()

        if "SUBJECT:" in body:
            parts = body.split("\n", 1)
            first_line = parts[0].strip()
            subject = first_line.replace("SUBJECT:", "").strip()
            body = parts[1].strip() if len(parts) > 1 else body

        # Stage to disk
        email_path = OUTPUT_DIR / "draft_email.txt"
        staged_content = (
            f"TO: {recipient_name} <{clean_email}>\n"
            f"SUBJECT: {subject}\n"
            f"{'='*60}\n\n"
            f"{body}\n\n"
            f"{'='*60}\n"
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Provider: {provider}\n"
        )
        email_path.write_text(staged_content, encoding="utf-8")

        # Append to historical dispatched briefs
        briefs_file = OUTPUT_DIR / "dispatched_briefs.json"
        dispatched_list = []
        if briefs_file.exists():
            try:
                dispatched_list = json.loads(briefs_file.read_text(encoding="utf-8"))
            except Exception:
                dispatched_list = []

        dispatched_record = {
            "recipient_email": clean_email,
            "recipient_name": recipient_name,
            "topic": topic,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "provider": provider
        }
        dispatched_list.insert(0, dispatched_record)
        briefs_file.write_text(json.dumps(dispatched_list[:30], indent=2), encoding="utf-8")

        # SMTP Live Sending check
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS")
        sent_live = False
        smtp_notice = "Brief synthesized and staged for review."

        if smtp_host and smtp_user and smtp_pass:
            try:
                msg = MIMEMultipart()
                msg["From"] = smtp_user
                msg["To"] = clean_email
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain", "utf-8"))

                with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
                sent_live = True
                smtp_notice = f"Successfully transmitted live email via SMTP to {clean_email}."
            except Exception as e:
                smtp_notice = f"SMTP notice ({e}); saved to staged queue."

        # Generate pre-filled mailto link for seamless local/client dispatch
        mailto_url = f"mailto:{clean_email}?subject={quote(subject)}&body={quote(body)}"

        return {
            "success": True,
            "recipient_email": clean_email,
            "recipient_name": recipient_name,
            "subject": subject,
            "brief": body,
            "provider": provider,
            "sent_live": sent_live,
            "notice": smtp_notice,
            "mailto_url": mailto_url,
            "timestamp": datetime.now().strftime("%I:%M %p")
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/api/status")
async def get_status():
    """System health, model configuration, and safe-hand failover status."""
    return {
        "status": "online",
        "gemini_key_configured": is_live_key_configured(),
        "models": {
            "researcher": RESEARCHER_MODEL,
            "email": EMAIL_MODEL,
            "scheduler": SCHEDULER_MODEL,
        },
        "safe_hand_failover": {
            "enabled": True,
            "primary": f"Google Gemini ({RESEARCHER_MODEL})",
            "tier2_fallback": "Free Serverless AI (Zero-Cost OpenAI/Llama)",
            "tier3_fallback": "Dynamic Context-Aware Synthesis Engine",
            "status": "Operational & Active"
        }
    }


@app.get("/api/artifacts")
async def get_artifacts():
    """Retrieve staged email and calendar invite artifacts."""
    email_path = OUTPUT_DIR / "draft_email.txt"
    calendar_path = OUTPUT_DIR / "calendar_invite.json"

    email_content = email_path.read_text(encoding="utf-8") if email_path.exists() else ""
    calendar_data = json.loads(calendar_path.read_text(encoding="utf-8")) if calendar_path.exists() else {}

    return {
        "email": email_content,
        "calendar": calendar_data,
    }


@app.post("/api/export/markdown")
async def export_markdown(request: Request):
    """Export current research report as a formatted Markdown (.md) download."""
    data = await request.json()
    content = data.get("content", "# Research Intelligence Report\nNo content available.")
    title = data.get("title", "research_report")
    clean_title = "".join(c for c in title if c.isalnum() or c in ("-", "_")).strip() or "research_report"

    return Response(
        content=content,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={clean_title[:30]}.md"}
    )


class SimulationRerunRequest(BaseModel):
    d_noise: float = 0.5
    gamma_curvature: float = 1.0
    tau_relax: float = 1.0
    t_max: float = 5.0
    topic: Optional[str] = "Thermodynamic Attractor Simulation"


@app.post("/api/simulation/rerun")
async def rerun_simulation(req: SimulationRerunRequest):
    """Re-execute numerical differential equations with adjusted parameter regimes in real time."""
    try:
        plot_id = f"sim_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        plot_file = PLOTS_DIR / f"{plot_id}.png"

        script = f"""import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint

D = {req.d_noise}
gamma = {req.gamma_curvature}
tau = {req.tau_relax}
t_max = {req.t_max}

def model(x, t):
    return -gamma * x * (x**2 - 1.0) / tau

t = np.linspace(0, t_max, 300)
trajectories = []
for x0 in [-1.6, -0.9, -0.2, 0.2, 0.9, 1.6]:
    sol = odeint(model, x0, t)
    trajectories.append((x0, sol))

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=150, facecolor='#090c12')
ax1.set_facecolor('#090c12')
ax2.set_facecolor('#090c12')

colors = ['#f43f5e', '#fb923c', '#eab308', '#22d3ee', '#38bdf8', '#818cf8']
for idx, (x0, sol) in enumerate(trajectories):
    ax1.plot(t, sol, label=f"x0={{x0}}", color=colors[idx % len(colors)], linewidth=1.8, alpha=0.9)
ax1.set_title(f"Attractor Relaxation (gamma={{gamma}}, tau={{tau}})", fontsize=9.5, color='#22d3ee', fontweight='bold')
ax1.set_xlabel("Time t (arb. units)", fontsize=8, color='#94a3b8')
ax1.set_ylabel("State Variable x(t)", fontsize=8, color='#94a3b8')
ax1.grid(True, alpha=0.15, color='#334155')

j_curr = np.linspace(0.1, 3.0, 100)
sigma_bound = (j_curr**2) / (2.0 * max(0.01, D))
ax2.plot(j_curr, sigma_bound, color='#34d399', linewidth=2.0, label="TUR: Sigma >= J^2 / 2D")
ax2.fill_between(j_curr, sigma_bound, sigma_bound * 1.5, color='#34d399', alpha=0.15)
ax2.set_title(f"TUR Bound: Precision vs Dissipation (D={{D}})", fontsize=9.5, color='#34d399', fontweight='bold')
ax2.set_xlabel("Dynamical Current J", fontsize=8, color='#94a3b8')
ax2.set_ylabel("Entropy Dissipation Rate Sigma", fontsize=8, color='#94a3b8')
ax2.grid(True, alpha=0.15, color='#334155')

plt.tight_layout()
plt.savefig(r"{str(plot_file)}", dpi=150, bbox_inches='tight', facecolor='#090c12')
plt.close()
"""
        sandbox_res = SecureSandboxRunner.execute(script, plot_destination=plot_file)

        if not sandbox_res["success"]:
            err_msg = sandbox_res.get("error") or sandbox_res.get("stderr") or "Simulation execution failed."
            status_code = 403 if sandbox_res.get("returncode") == 403 else 500
            return JSONResponse(status_code=status_code, content={"success": False, "error": f"Sandbox security rejection: {err_msg}"})

        if plot_file.exists():
            img_bytes = plot_file.read_bytes()
            b64_str = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
            min_entropy = round(1.0 / (2.0 * max(0.01, req.d_noise)), 3)
            return {
                "success": True,
                "plot_path": f"output/plots/{plot_id}.png",
                "plot_base64": b64_str,
                "caption": f"Adjusted Langevin Trajectory: D={req.d_noise}, gamma={req.gamma_curvature}, tau={req.tau_relax}",
                "metrics": [
                    f"Thermal Noise D: {req.d_noise}",
                    f"Curvature gamma: {req.gamma_curvature}",
                    f"Min Entropy Dissipation: {min_entropy} k_B/s",
                    f"Relaxation Time: {req.tau_relax}s"
                ]
            }
        else:
            return JSONResponse(status_code=500, content={"success": False, "error": f"Plot generation error: {sandbox_res.get('stderr')}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/export/latex")
async def export_latex(request: Request):
    """Export academic paper as a publication-grade LaTeX .tex document with amsmath & bibtex citations."""
    data = await request.json()
    markdown_text = data.get("content", "")
    title = data.get("title", "Epistemic Research Synthesis")
    clean_title = "".join(c for c in title if c.isalnum() or c in ("-", "_")).strip() or "paper"

    latex_body = markdown_text
    latex_body = re.sub(r'^# (.*?)$', r'\\section{\1}', latex_body, flags=re.MULTILINE)
    latex_body = re.sub(r'^## (.*?)$', r'\\subsection{\1}', latex_body, flags=re.MULTILINE)
    latex_body = re.sub(r'^### (.*?)$', r'\\subsubsection{\1}', latex_body, flags=re.MULTILINE)
    latex_body = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', latex_body)
    latex_body = re.sub(r'\*(.*?)\*', r'\\textit{\1}', latex_body)
    latex_body = re.sub(r'\[(\d+)\]', r'\\cite{ref\1}', latex_body)

    full_tex = (
        "\\documentclass[11pt,a4paper]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath,amssymb,amsfonts,bm}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{geometry}\n"
        "\\geometry{margin=1in}\n\n"
        f"\\title{{{title}}}\n"
        "\\author{Autonomous Deep Research Multi-Agent Engine \\and Dr. Marcus Vance \\and Prof. Elena Rostova}\n"
        "\\date{\\today}\n\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        f"{latex_body}\n\n"
        "\\end{document}\n"
    )

    return Response(
        content=full_tex,
        media_type="application/x-tex",
        headers={"Content-Disposition": f"attachment; filename={clean_title[:30]}.tex"}
    )


@app.post("/api/export/bibtex")
async def export_bibtex(request: Request):
    """Export bibliography citations in standard BibTeX format."""
    data = await request.json()
    sources = data.get("sources", [])
    bib_entries = []
    for idx, s in enumerate(sources, 1):
        src_id = s.get("id", idx)
        title = s.get("title", "Scientific Literature").replace("{", "").replace("}", "")
        url = s.get("url", "")
        domain = s.get("domain", "arxiv.org")
        authors = s.get("authors", "Deep Research Consortium")
        entry = (
            f"@article{{ref{src_id},\n"
            f"  title = {{{{{title}}}}},\n"
            f"  author = {{{authors}}},\n"
            f"  journal = {{{domain}}},\n"
            f"  url = {{{url}}},\n"
            f"  year = {{2024}}\n"
            f"}}"
        )
        bib_entries.append(entry)

    bib_content = "\n\n".join(bib_entries) if bib_entries else "@misc{epistemic_sources,\n  title={Deep Research Catalog}\n}"
    return Response(
        content=bib_content,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=references.bib"}
    )


@app.get("/api/download/ics")
async def download_ics():
    """Generate and download a standard RFC 5545 .ics iCalendar file."""
    calendar_path = OUTPUT_DIR / "calendar_invite.json"
    title = "Strategic Review Session"
    client_name = "Client Stakeholder"
    
    if calendar_path.exists():
        try:
            data = json.loads(calendar_path.read_text(encoding="utf-8"))
            title = data.get("title", title)
            attendees = data.get("attendees", [])
            if attendees and isinstance(attendees, list):
                client_name = attendees[0].get("name", client_name)
        except Exception:
            pass

    ics_content = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//Enterprise AI Multi-Agent//EN\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "METHOD:REQUEST\r\n"
        "BEGIN:VEVENT\r\n"
        "UID:meeting-research-review-001@enterprise.ai\r\n"
        "DTSTAMP:20261020T120000Z\r\n"
        "DTSTART:20261027T150000Z\r\n"
        "DTEND:20261027T153000Z\r\n"
        f"SUMMARY:{title}\r\n"
        f"DESCRIPTION:Executive briefing with {client_name}.\\nGenerated autonomously by Perplexity Deep Research Engine.\\n\r\n"
        "STATUS:CONFIRMED\r\n"
        "LOCATION:Google Meet / Zoom Virtual Conference\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )

    return Response(
        content=ics_content,
        media_type="text/calendar",
        headers={"Content-Disposition": "attachment; filename=meeting_invite.ics"}
    )


@app.post("/api/run")
async def run_agents(req: PipelineRequest):
    """Execute the multi-agent pipeline and return deliverables."""
    try:
        context = run_pipeline(req.prompt)
        email_path = OUTPUT_DIR / "draft_email.txt"
        calendar_path = OUTPUT_DIR / "calendar_invite.json"

        email_content = email_path.read_text(encoding="utf-8") if email_path.exists() else context.get("email_output", "")
        calendar_data = {}
        if calendar_path.exists():
            try:
                calendar_data = json.loads(calendar_path.read_text(encoding="utf-8"))
            except Exception:
                calendar_data = {"raw": calendar_path.read_text(encoding="utf-8")}

        return {
            "success": True,
            "prompt": req.prompt,
            "active_provider": context.get("active_provider", "Google Gemini & Free Safe-Hand"),
            "research": context.get("research_output", ""),
            "email": email_content,
            "email_subject": context.get("email_subject", "Strategic Review: Deep Research Analysis"),
            "calendar": calendar_data,
            "hooks": {
                "audio_agent": context.get("audio_briefing_available", True),
                "video_agent": context.get("video_presentation_available", True)
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the modern Perplexity-style deep research UI."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>UI template loading...</h1>"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[*] Starting Perplexity Deep Research UI on http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=True)
