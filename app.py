"""Modern Web Application for Autonomous Multi-Agent Deep Research & Enterprise System.

Provides a Perplexity-style UI for real-time web research on any topic,
interactive citation badges, dynamic agent reasoning, and optional outreach staging.
"""

import sys
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import quote
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
import uvicorn

from config.settings import (
    RESEARCHER_MODEL,
    EMAIL_MODEL,
    SCHEDULER_MODEL,
    OUTPUT_DIR,
    is_live_key_configured,
)
from agents.crew_factory import run_pipeline
from agents.failover import SafeHandAgentRunner, call_free_ai_api
from agents.stream_orchestrator import stream_multi_agent_pipeline

app = FastAPI(title="Perplexity Deep Research & Multi-Agent Enterprise Canvas")


@app.get("/api/stream")
async def stream_agents(prompt: str, client: Optional[str] = None, mode: str = "auto"):
    """Server-Sent Events streaming endpoint providing Perplexity-grade real-time research."""
    return StreamingResponse(
        stream_multi_agent_pipeline(prompt, client, mode),
        media_type="text/event-stream"
    )


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
    uvicorn.run("app:app", host=host, port=port)
