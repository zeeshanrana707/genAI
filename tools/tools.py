"""Custom tools for the Multi-Agent Research System.

Includes:
- fetch_arxiv_full_text: Full ArXiv paper ingestion (math sections, hardware constraints)
- run_numerical_verification: Python/numpy numerical verification sandbox
- WebSearchTool: Live DuckDuckGo + arXiv + deep page scraping
- NumericalVerificationTool: Wraps the Python verification sandbox for agent use
- DraftEmailTool: Stages executive client correspondence
- CreateCalendarInviteTool: Stages structured calendar invites
"""

import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import urllib3
from pathlib import Path
from urllib.parse import urlparse
from typing import Type, Optional, List, Dict, Any

from pydantic import BaseModel, Field

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool:
        name: str = ""
        description: str = ""
        args_schema: Optional[Type[BaseModel]] = None

        def _run(self, *args, **kwargs):
            raise NotImplementedError

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

from config.settings import OUTPUT_DIR, GEMINI_API_KEY, is_live_key_configured


# -------------------------------------------------------------------------
# ArXiv Full-Text Ingestion
# -------------------------------------------------------------------------

def fetch_arxiv_full_text(arxiv_url: str) -> Dict[str, Any]:
    """Fetch and parse full ArXiv paper: title, authors, abstract, math sections, hardware constraints.

    Args:
        arxiv_url: Any URL containing 'arxiv.org' (abstract, HTML, or PDF link)

    Returns:
        dict with keys: title, authors, abstract, math_sections, hardware_constraints, raw_text
        Returns empty dict on any failure — never crashes.
    """
    if not arxiv_url or "arxiv.org" not in arxiv_url:
        return {}

    try:
        # Extract paper ID from various arxiv URL formats
        # Handles: arxiv.org/abs/2501.12345, arxiv.org/pdf/2501.12345, arxiv.org/html/2501.12345
        paper_id_match = re.search(
            r'arxiv\.org/(?:abs|pdf|html|e-print)/(\d{4}\.\d{4,5}(?:v\d+)?)',
            arxiv_url
        )
        if not paper_id_match:
            # Try alternate format: arxiv.org/abs/cs/0604005
            paper_id_match = re.search(
                r'arxiv\.org/(?:abs|pdf|html)/([a-z-]+/\d+)',
                arxiv_url
            )
        if not paper_id_match:
            return {}

        paper_id = paper_id_match.group(1)
        result = {
            "paper_id": paper_id,
            "title": "",
            "authors": "",
            "abstract": "",
            "math_sections": [],
            "hardware_constraints": [],
            "raw_text": ""
        }

        # Step 1: Fetch structured metadata from ArXiv Atom API
        api_url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
        headers = {"User-Agent": "Mozilla/5.0 (research-agent/1.0)"}
        api_resp = requests.get(api_url, headers=headers, timeout=10, verify=False)

        if api_resp.status_code == 200:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(api_resp.text)
            entry = root.find("atom:entry", ns)
            if entry is not None:
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                authors = entry.findall("atom:author", ns)

                result["title"] = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
                result["abstract"] = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                result["authors"] = ", ".join(
                    a.find("atom:name", ns).text.strip()
                    for a in authors
                    if a.find("atom:name", ns) is not None
                )[:300]

        # Step 2: Fetch HTML version for deep mathematical content
        html_url = f"https://arxiv.org/html/{paper_id}"
        html_resp = requests.get(html_url, headers=headers, timeout=10, verify=False)

        if html_resp.status_code == 200 and BeautifulSoup is not None:
            soup = BeautifulSoup(html_resp.text, "html.parser")

            # Remove navigation, headers, footers
            for tag in soup(["nav", "header", "footer", "script", "style"]):
                tag.decompose()

            # Extract mathematical sections (look for section elements with equations)
            math_sections = []
            for section in soup.find_all(["section", "div"], limit=30):
                section_text = section.get_text(separator=" ", strip=True)
                # Keep sections that contain mathematical notation indicators
                if any(indicator in section_text for indicator in
                       ["=", "∇", "∂", "σ", "γ", "∫", "Σ", "→", "≈", "dx", "dt",
                        "equation", "theorem", "lemma", "proof", "derivat"]):
                    clean = re.sub(r'\s+', ' ', section_text).strip()
                    if 50 < len(clean) < 1500:
                        math_sections.append(clean)

            result["math_sections"] = math_sections[:8]

            # Extract hardware / constraint mentions
            full_text = soup.get_text(separator=" ", strip=True)
            result["raw_text"] = full_text[:5000]

            hw_patterns = [
                r'[A-Z]\d+[A-Z]?\d*\s+GPU', r'H\d00\s+SXM', r'A\d00', r'V\d00',
                r'\d+\s*GB\s+(?:VRAM|HBM|RAM)', r'\d+\s*kW', r'PCIe', r'NVLink',
                r'FP(?:8|16|32|64)', r'INT(?:4|8)', r'memristor', r'crossbar',
                r'CMOS', r'analog', r'neuromorphic', r'\d+\s*nm\s+node',
                r'conductance', r'memristive', r'colloidal'
            ]
            hardware_hits = set()
            for pattern in hw_patterns:
                for match in re.finditer(pattern, full_text, re.IGNORECASE):
                    # Get context around match
                    start = max(0, match.start() - 40)
                    end = min(len(full_text), match.end() + 60)
                    ctx = full_text[start:end].strip().replace("\n", " ")
                    hardware_hits.add(ctx[:120])

            result["hardware_constraints"] = list(hardware_hits)[:6]

        print(f"[ArXiv Ingester] Full text fetched for {paper_id}: "
              f"{len(result['math_sections'])} math sections, "
              f"{len(result['hardware_constraints'])} hardware constraints.")
        return result

    except Exception as e:
        print(f"[ArXiv Ingester] Notice for {arxiv_url}: {e}")
        return {}


# -------------------------------------------------------------------------
# Python Numerical Verification Sandbox
# -------------------------------------------------------------------------

def run_numerical_verification(equations_context: str, topic: str) -> Dict[str, Any]:
    """Generate and execute a Python numerical verification script for the given equations.

    Uses numpy (and scipy if available) to numerically validate differential equations
    and compute concrete metrics such as dissipation rates, stability bounds, and error estimates.

    Args:
        equations_context: String describing the equations to verify
        topic: Research topic for context

    Returns:
        dict: {executed, script, output, metrics, error}
    """
    result = {"executed": False, "script": "", "output": "", "metrics": [], "error": ""}

    if not is_live_key_configured():
        result["error"] = "Gemini API not configured — skipping numerical verification."
        return result

    try:
        from google import genai as _genai
        client = _genai.Client(api_key=GEMINI_API_KEY)

        gen_prompt = (
            f"Topic: {topic}\n\n"
            f"Equations & Context:\n{equations_context[:2000]}\n\n"
            "Write a self-contained Python script (using only numpy and optionally scipy) that:\n"
            "1. Defines concrete numerical parameters (e.g. gamma=1.0, kBT=0.025, tau=0.1)\n"
            "2. Numerically integrates or evaluates the key differential equations for t=[0,1]\n"
            "3. Computes and prints specific metrics: steady-state values, dissipation rates, SNR, or error bounds\n"
            "4. Uses print() statements to output all computed values clearly\n"
            "Rules: No file I/O. No imports except numpy and scipy. No plt.show(). Must complete in <10 seconds.\n"
            "Output ONLY the raw Python code, no markdown fences, no explanations."
        )

        from config.settings import GEMINI_MODELS_PRIORITY, RESEARCHER_MODEL
        gen_res = None
        for m_name in [RESEARCHER_MODEL] + [m for m in GEMINI_MODELS_PRIORITY if m != RESEARCHER_MODEL]:
            try:
                gen_res = client.models.generate_content(
                    model=m_name,
                    contents=gen_prompt
                )
                if gen_res and gen_res.text:
                    break
            except Exception:
                continue

        if not gen_res or not gen_res.text:
            result["error"] = "Could not generate verification script."
            return result

        script_raw = gen_res.text.strip()
        # Strip any accidental markdown fences
        script_raw = re.sub(r'^```python\n?', '', script_raw)
        script_raw = re.sub(r'\n?```$', '', script_raw)
        result["script"] = script_raw

        # Execute the script in an isolated subprocess with strict timeout
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(script_raw)
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=tempfile.gettempdir()
            )
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            result["executed"] = True
            result["output"] = stdout[:2000] if stdout else "(No output produced)"
            if stderr:
                result["error"] = stderr[:500]

            # Extract numeric metrics from stdout
            metric_pattern = re.compile(
                r'(?:steady[- ]state|dissipation|SNR|error|bound|rate|metric|value|result)[^\n:]*:\s*([-\d.eE+]+)',
                re.IGNORECASE
            )
            metrics = metric_pattern.findall(stdout)
            # Also grab any lines with = sign and numbers
            for line in stdout.split('\n'):
                nums = re.findall(r'[-\d.eE+]{3,}', line)
                if nums and '=' in line:
                    metrics.append(line.strip()[:80])
            result["metrics"] = list(dict.fromkeys(metrics))[:8]

        except subprocess.TimeoutExpired:
            result["error"] = "Numerical verification timed out (>15s)."
        except Exception as exec_err:
            result["error"] = f"Execution error: {exec_err}"
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    except Exception as e:
        result["error"] = f"Verification setup failed: {e}"

    print(f"[NumericalVerification] Executed: {result['executed']}, "
          f"Metrics found: {len(result['metrics'])}, "
          f"Error: {result['error'][:60] if result['error'] else 'None'}")
    return result


# -------------------------------------------------------------------------
# Helper: page body scraper
# -------------------------------------------------------------------------

def scrape_page_body(url: str, max_chars: int = 750) -> str:
    """Fetch and extract clean paragraph text from a web article."""
    if not url or not url.startswith("http") or BeautifulSoup is None:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=4, verify=False)
        if resp.status_code == 200 and resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                element.decompose()
            clean_text = " ".join(soup.stripped_strings)
            return (clean_text[:max_chars] + "...") if len(clean_text) > max_chars else clean_text
    except Exception:
        pass
    return ""


def search_arxiv(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """Retrieve peer-reviewed preprints and research abstracts directly from arXiv API."""
    arxiv_items = []
    clean_q = query.replace("arxiv", "").replace("scholarly", "").replace("paper", "").strip()
    encoded = urllib.parse.quote(clean_q)
    url = f"https://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={max_results}"

    try:
        resp = requests.get(url, timeout=6, verify=False)
        if resp.status_code == 200 and resp.text:
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            for entry in entries:
                title_elem = entry.find("atom:title", ns)
                summary_elem = entry.find("atom:summary", ns)
                id_elem = entry.find("atom:id", ns)
                authors = entry.findall("atom:author", ns)

                title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "arXiv Preprint"
                summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                paper_url = id_elem.text.strip() if id_elem is not None else "https://arxiv.org"
                author_names = ", ".join(
                    a.find("atom:name", ns).text.strip()
                    for a in authors[:3]
                    if a.find("atom:name", ns) is not None
                )

                arxiv_items.append({
                    "title": f"[arXiv] {title}",
                    "snippet": summary[:320] + ("..." if len(summary) > 320 else ""),
                    "deep_content": summary[:800],
                    "url": paper_url,
                    "domain": "arxiv.org",
                    "authors": author_names,
                    "source_type": "arxiv"
                })
    except Exception as e:
        print(f"[Tool: ArXiv] Notice: {e}")

    return arxiv_items


# -------------------------------------------------------------------------
# Tool 1: WebSearchTool (with ArXiv full-text ingestion)
# -------------------------------------------------------------------------

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query string to search on the web")


class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Searches the web and arXiv with deep content scraping and full ArXiv paper ingestion."
    args_schema: Type[BaseModel] = WebSearchInput

    def search_structured(self, query: str, max_results: int = 8, mode: str = "all") -> List[Dict[str, str]]:
        """Retrieve structured web results with full ArXiv content where available."""
        items: List[Dict[str, str]] = []

        # 1. Academic mode: prioritize arXiv API papers
        if mode == "academic" or "arxiv" in query.lower() or "paper" in query.lower():
            arxiv_results = search_arxiv(query, max_results=4)
            items.extend(arxiv_results)

        # 2. DuckDuckGo for live web intelligence
        ddg_needed = max_results - len(items)
        if ddg_needed > 0 and DDGS is not None:
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=ddg_needed + 2))
                    for r in raw:
                        href = r.get("href", "")
                        domain = urlparse(href).netloc.replace("www.", "") if href else "web"
                        items.append({
                            "title": r.get("title", "Untitled Source"),
                            "snippet": r.get("body", ""),
                            "url": href,
                            "domain": domain,
                            "source_type": "web"
                        })
            except Exception as e:
                print(f"[Tool: WebSearch] Search notice: {e}")

        # 3. Fallback
        if not items:
            clean_q = query.strip()
            items = [
                {
                    "title": f"Research & Intelligence: {clean_q}",
                    "snippet": f"In-depth analysis addressing: '{clean_q}'.",
                    "url": f"https://duckduckgo.com/?q={clean_q.replace(' ', '+')}",
                    "domain": "duckduckgo.com",
                    "source_type": "web"
                }
            ]

        # 4. Deep content extraction
        for idx, item in enumerate(items[:3]):
            url = item.get("url", "")
            if "deep_content" not in item and url:
                # ArXiv full-text ingestion
                if "arxiv.org" in url:
                    full_data = fetch_arxiv_full_text(url)
                    if full_data:
                        item["full_arxiv_content"] = full_data
                        # Build enriched deep_content from math sections + abstract
                        parts = []
                        if full_data.get("abstract"):
                            parts.append(f"ABSTRACT: {full_data['abstract'][:400]}")
                        if full_data.get("math_sections"):
                            parts.append("MATHEMATICAL SECTIONS:\n" + "\n\n".join(full_data["math_sections"][:3]))
                        if full_data.get("hardware_constraints"):
                            parts.append("HARDWARE/PARAMETER CONSTRAINTS:\n" + "; ".join(full_data["hardware_constraints"][:4]))
                        item["deep_content"] = "\n\n".join(parts)[:1200]
                else:
                    deep_text = scrape_page_body(url, max_chars=700)
                    if deep_text:
                        item["deep_content"] = deep_text

        return items[:max_results]

    def _run(self, query: str, mode: str = "all") -> str:
        """Execute web search and return formatted text string with deep article content."""
        print(f"\n[Tool: WebSearch] Querying ({mode}): '{query}'")
        items = self.search_structured(query, max_results=8, mode=mode)

        formatted = []
        for idx, item in enumerate(items, 1):
            body_text = item.get("deep_content") or item.get("snippet", "")
            authors_line = f"   Authors: {item['authors']}\n" if item.get("authors") else ""
            formatted.append(
                f"{idx}. Title: {item['title']}\n"
                f"{authors_line}"
                f"   Summary: {item['snippet']}\n"
                f"   Deep Excerpt: {body_text[:600]}\n"
                f"   Source: {item['url']} ({item['domain']})"
            )

        print(f"[Tool: WebSearch] Retrieved {len(items)} enriched sources.")
        return "\n\n".join(formatted)


# -------------------------------------------------------------------------
# Tool 2: NumericalVerificationTool
# -------------------------------------------------------------------------

class NumericalVerificationInput(BaseModel):
    equations_context: str = Field(description="Description of equations and parameters to numerically verify")
    topic: str = Field(description="Research topic for context")


class NumericalVerificationTool(BaseTool):
    name: str = "Numerical Verification Tool"
    description: str = "Generates and executes Python (numpy/scipy) code to numerically verify proposed differential equations and compute concrete metrics."
    args_schema: Type[BaseModel] = NumericalVerificationInput

    def _run(self, equations_context: str, topic: str) -> str:
        """Execute numerical verification and return results."""
        result = run_numerical_verification(equations_context, topic)
        if result["executed"]:
            out = f"[Numerical Verification EXECUTED]\n\nScript:\n```python\n{result['script'][:800]}\n```\n\nOutput:\n{result['output']}"
            if result["metrics"]:
                out += f"\n\nExtracted Metrics:\n" + "\n".join(f"  - {m}" for m in result["metrics"])
            return out
        else:
            return f"[Numerical Verification] Not executed. Reason: {result.get('error', 'Unknown')}"


# -------------------------------------------------------------------------
# Tool 3: DraftEmailTool
# -------------------------------------------------------------------------

class DraftEmailInput(BaseModel):
    recipient_name: str = Field(description="Name of the client")
    recipient_email: str = Field(default="client@example.com", description="Email address of the client")
    subject: str = Field(description="Subject line of the email")
    body: str = Field(description="Body content of the email")


class DraftEmailTool(BaseTool):
    name: str = "Draft Email Tool"
    description: str = "Drafts an executive client email and stages it to the output directory."
    args_schema: Type[BaseModel] = DraftEmailInput

    def _run(self, recipient_name: str, subject: str, body: str, recipient_email: str = "client@example.com") -> str:
        """Format and stage draft client email."""
        print(f"\n[Tool: DraftEmail] Drafting email to {recipient_name}...")

        email_content = (
            f"TO: {recipient_name} <{recipient_email}>\n"
            f"SUBJECT: {subject}\n"
            f"{'='*60}\n\n"
            f"{body}\n\n"
            f"{'='*60}\n"
            f"Status: Staged for review. Ready to send.\n"
        )

        output_path = OUTPUT_DIR / "draft_email.txt"
        output_path.write_text(email_content, encoding="utf-8")
        print(f"[Tool: DraftEmail] Email saved to: {output_path}")
        return f"Email successfully drafted. Subject: {subject}\nRecipient: {recipient_name}"


# -------------------------------------------------------------------------
# Tool 4: CreateCalendarInviteTool
# -------------------------------------------------------------------------

class CreateCalendarInviteInput(BaseModel):
    meeting_title: str = Field(description="Title of the review meeting")
    client_name: str = Field(description="Name of the client attending the meeting")
    duration_minutes: int = Field(default=30, description="Duration of meeting in minutes")
    proposed_slots: list[str] = Field(description="List of 3 proposed meeting time slots next week")
    agenda: str = Field(description="Agenda summary for the meeting")


class CreateCalendarInviteTool(BaseTool):
    name: str = "Create Calendar Invite Tool"
    description: str = "Schedules a 30-minute review meeting and generates a structured calendar invite."
    args_schema: Type[BaseModel] = CreateCalendarInviteInput

    def _run(
        self,
        meeting_title: str,
        client_name: str,
        proposed_slots: list[str],
        agenda: str,
        duration_minutes: int = 30
    ) -> str:
        """Create structured calendar invite."""
        print(f"\n[Tool: CreateCalendarInvite] Creating invite '{meeting_title}' for {client_name}...")

        invite_data = {
            "title": meeting_title,
            "attendees": [
                {"name": client_name, "role": "Client"},
                {"name": "Enterprise AI Strategy Team", "role": "Organizer"}
            ],
            "duration_minutes": duration_minutes,
            "proposed_slots_next_week": proposed_slots,
            "agenda": agenda,
            "meeting_type": "Virtual Video Conference (Google Meet / Zoom)",
            "status": "PROPOSED"
        }

        output_path = OUTPUT_DIR / "calendar_invite.json"
        output_path.write_text(json.dumps(invite_data, indent=2), encoding="utf-8")
        print(f"[Tool: CreateCalendarInvite] Invite saved to: {output_path}")
        return f"Calendar invite created with {len(proposed_slots)} proposed slots for {client_name}."
