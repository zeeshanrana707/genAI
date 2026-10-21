"""Main execution script for Autonomous Multi-Agent Enterprise System.

University MVP: Autonomous Research, Emailing, and Scheduling.
Models:
- Researcher Agent: gemini-4-argon (Heavy research tasks)
- Email Agent: gemini-3.1-pro (Orchestration & client communications)
- Scheduling Agent: gemini-3.1-pro (Scheduling & calendar coordination)
"""

import sys
import argparse
from pathlib import Path

# Configure utf-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agents.crew_factory import run_pipeline
from config.settings import OUTPUT_DIR

DEFAULT_PROMPT = (
    "Analyze the top 3 competitors in AI video generation and draft an email to our "
    "client, Mr. Smith, scheduling a 30-minute review next week."
)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Agent Enterprise System (Research -> Email -> Scheduling)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Input user prompt to initiate the multi-agent pipeline",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the modern web UI dashboard on http://127.0.0.1:8000",
    )
    args = parser.parse_args()

    if args.ui:
        import uvicorn
        from app import app
        print("[*] Launching Enterprise Modern UI on http://127.0.0.1:8000 ...")
        uvicorn.run(app, host="127.0.0.1", port=8000)
        return 0

    # Execute pipeline
    results = run_pipeline(args.prompt)

    # Display deliverables
    email_path = OUTPUT_DIR / "draft_email.txt"
    calendar_path = OUTPUT_DIR / "calendar_invite.json"

    print("\n" + "=" * 80)
    print("[DELIVERABLES SUMMARY]")
    print("=" * 80)
    
    if email_path.exists():
        print(f"\n[Artifact 1] Client Email Draft ({email_path.name}):")
        print("-" * 60)
        print(email_path.read_text(encoding="utf-8"))
        print("-" * 60)

    if calendar_path.exists():
        print(f"\n[Artifact 2] Calendar Invite Specification ({calendar_path.name}):")
        print("-" * 60)
        print(calendar_path.read_text(encoding="utf-8"))
        print("-" * 60)

    print("\n[SUCCESS] Multi-Agent Pipeline execution completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
