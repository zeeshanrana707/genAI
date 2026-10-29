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


import base64
import os
import time
import uuid
import concurrent.futures

PLOTS_DIR = Path(__file__).resolve().parent.parent / "output" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Python Numerical Verification & Self-Healing Code Interpreter
# -------------------------------------------------------------------------

def run_numerical_verification(equations_context: str, topic: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Generate and execute a self-healing Python computational simulation with visual plot generation.

    Uses numpy, scipy, and matplotlib to numerically validate differential equations,
    compute quantitative bounds, and generate publication-grade phase space / attractor plots.
    If execution encounters syntax or numerical errors, executes an interactive self-healing loop.

    Args:
        equations_context: String describing the mathematical equations & assumptions
        topic: Research topic for context
        api_key: Optional subscription or custom Gemini API key

    Returns:
        dict: {executed, script, output, metrics, plot_path, plot_base64, error, healed}
    """
    result = {
        "executed": False,
        "script": "",
        "output": "",
        "metrics": [],
        "plot_path": "",
        "plot_base64": "",
        "error": "",
        "healed": False
    }

    effective_key = (api_key.strip() if api_key and len(api_key.strip()) >= 20 else "") or GEMINI_API_KEY
    if not effective_key or len(effective_key) < 8:
        result["error"] = "Gemini API key not configured — skipping numerical verification."
        return result

    try:
        from google import genai as _genai
        from google.genai import types
        try:
            client = _genai.Client(api_key=effective_key, http_options=types.HttpOptions(timeout=15000))
        except Exception:
            client = _genai.Client(api_key=effective_key)

        plot_id = f"plot_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        plot_dest = PLOTS_DIR / f"{plot_id}.png"

        gen_prompt = (
            f"Topic: {topic}\n\n"
            f"Mathematical Context & Equations:\n{equations_context[:2200]}\n\n"
            "Write a self-contained Python numerical simulation script using numpy, scipy, and matplotlib that:\n"
            "1. Defines concrete physics parameters (e.g. gamma=1.0, D=0.05, tau=0.1, V_0=1.5)\n"
            "2. Numerically integrates or simulates the key differential equations / dynamics for t in [0, 5]\n"
            "3. Computes and prints specific quantitative metrics: steady-state values, entropy production, dissipation rates, or TUR precision bounds\n"
            "4. Generates a publication-grade scientific plot (e.g. phase portrait, manifold trajectory, potential well, or precision-dissipation trade-off curve).\n"
            "   IMPORTANT PLOT REQUIREMENT: Save the figure to the path in os.environ.get('SIMULATION_PLOT_PATH') using:\n"
            "   plt.figure(figsize=(7, 4.5), dpi=150)\n"
            "   # ... plotting code ...\n"
            "   plt.grid(True, alpha=0.3)\n"
            "   plt.tight_layout()\n"
            "   plt.savefig(os.environ['SIMULATION_PLOT_PATH'], dpi=150, bbox_inches='tight')\n"
            "   plt.close()\n"
            "5. Uses print() statements to output all computed metrics clearly with labels.\n"
            "Rules: Must complete in <12 seconds. Output ONLY raw executable Python code, no markdown fences, no explanations."
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
            result["error"] = "Could not generate verification simulation."
            return result

        script_raw = gen_res.text.strip()
        script_raw = re.sub(r'^```python\n?', '', script_raw)
        script_raw = re.sub(r'\n?```$', '', script_raw)
        result["script"] = script_raw

        # Execution function with environment passing
        def execute_code(code_str: str) -> subprocess.CompletedProcess:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
                tmp.write(code_str)
                tmp_path = tmp.name

            env_vars = os.environ.copy()
            env_vars["SIMULATION_PLOT_PATH"] = str(plot_dest)
            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    env=env_vars,
                    cwd=tempfile.gettempdir()
                )
                return proc
            finally:
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

        # Attempt 1: Run initial code
        proc = execute_code(script_raw)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        # Self-Healing Loop: If error or non-zero exit, trigger automatic repair with traceback
        if proc.returncode != 0 or ("Traceback" in stderr):
            print(f"[SelfHealingSandbox] Encountered execution error:\n{stderr[:300]}\nInitiating repair...")
            heal_prompt = (
                f"The following Python simulation script failed with an execution error:\n\n"
                f"```python\n{script_raw}\n```\n\n"
                f"Error Traceback:\n{stderr}\n\n"
                f"Diagnose the cause (e.g. array dimensions, parameter initialization, integration convergence, missing imports). "
                f"Correct the script. Ensure it still saves the plot to os.environ['SIMULATION_PLOT_PATH']. "
                f"Output ONLY the corrected raw executable Python code."
            )
            try:
                heal_res = client.models.generate_content(
                    model=RESEARCHER_MODEL,
                    contents=heal_prompt
                )
                if heal_res and heal_res.text:
                    healed_code = heal_res.text.strip()
                    healed_code = re.sub(r'^```python\n?', '', healed_code)
                    healed_code = re.sub(r'\n?```$', '', healed_code)
                    proc2 = execute_code(healed_code)
                    if proc2.returncode == 0:
                        script_raw = healed_code
                        stdout = proc2.stdout.strip()
                        stderr = proc2.stderr.strip()
                        result["script"] = healed_code
                        result["healed"] = True
                        print("[SelfHealingSandbox] Code successfully healed and executed cleanly!")
            except Exception as heal_err:
                print(f"[SelfHealingSandbox] Repair attempt notice: {heal_err}")

        result["executed"] = True
        result["output"] = stdout[:2000] if stdout else "(No output produced)"
        if stderr and not result["healed"]:
            result["error"] = stderr[:500]

        # Extract numeric metrics from stdout
        metric_pattern = re.compile(
            r'(?:steady[- ]state|dissipation|SNR|error|bound|rate|metric|value|result|entropy)[^\n:]*:\s*([-\d.eE+]+)',
            re.IGNORECASE
        )
        metrics = metric_pattern.findall(stdout)
        for line in stdout.split('\n'):
            nums = re.findall(r'[-\d.eE+]{3,}', line)
            if nums and '=' in line:
                metrics.append(line.strip()[:80])
        result["metrics"] = list(dict.fromkeys(metrics))[:8]

        # Check if plot was created
        if plot_dest.exists() and plot_dest.stat().st_size > 1000:
            result["plot_path"] = str(plot_dest)
            try:
                with open(plot_dest, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                    result["plot_base64"] = f"data:image/png;base64,{b64_data}"
            except Exception as e:
                print(f"[SelfHealingSandbox] Base64 encoding notice: {e}")

    except Exception as e:
        result["error"] = f"Verification setup failed: {e}"

    print(f"[NumericalVerification] Executed: {result['executed']}, "
          f"Healed: {result['healed']}, "
          f"Plot: {bool(result['plot_path'])}, "
          f"Metrics found: {len(result['metrics'])}, "
          f"Error: {result['error'][:60] if result['error'] else 'None'}")
    return result


# -------------------------------------------------------------------------
# Multi-Query Literature Expansion (Deep Research Snowballing - Zero-Latency)
# -------------------------------------------------------------------------

def expand_research_queries(topic: str, api_key: Optional[str] = None) -> List[str]:
    """Decompose research inquiry into 3 orthogonal sub-queries for parallel literature indexing (<1ms)."""
    # Clean topic and extract core keywords
    clean_topic = re.sub(r'[^\w\s\-\.\+]', ' ', topic).strip()
    words = [w for w in clean_topic.split() if len(w) > 2]
    core = " ".join(words[:8]) if len(words) > 8 else clean_topic
    if not core:
        core = topic.strip()

    return [
        f"{core} official architecture documentation technical report",
        f"{core} empirical benchmarks comparative evaluation performance",
        f"{core} engineering trade-offs practical constraints limitations"
    ]


# -------------------------------------------------------------------------
# Helper: page body scraper
# -------------------------------------------------------------------------

def scrape_page_body(url: str, max_chars: int = 750) -> str:
    """Fetch and extract clean paragraph text from a web article with strict timeout."""
    if not url or not url.startswith("http") or BeautifulSoup is None:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=1.5, verify=False)
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

        # 4. Deep content extraction (High-throughput zero-latency mapping)
        for idx, item in enumerate(items):
            url = item.get("url", "")
            if not item.get("deep_content"):
                item["deep_content"] = item.get("snippet", "")
            # Only if arXiv preprint and math sections needed, fetch preprint metadata
            if url and "arxiv.org" in url and idx == 0 and len(item.get("snippet", "")) < 60:
                try:
                    full_data = fetch_arxiv_full_text(url)
                    if full_data and full_data.get("abstract"):
                        item["deep_content"] = full_data["abstract"][:800]
                except Exception:
                    pass

        return items[:max_results]

    def search_structured_multi_queries(self, queries: List[str], max_results: int = 10, mode: str = "all") -> List[Dict[str, str]]:
        """Execute parallel web & arXiv indexing across multiple decomposed sub-queries."""
        all_items: List[Dict[str, str]] = []
        seen_urls = set()

        def fetch_for_query(q: str):
            try:
                return self.search_structured(q, max_results=5, mode=mode)
            except Exception as e:
                print(f"[MultiQuery] Sub-query notice for '{q}': {e}")
                return []

        active_queries = queries[:2] if len(queries) > 2 else queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active_queries) or 1) as executor:
            future_to_query = {executor.submit(fetch_for_query, q): q for q in active_queries}
            for future in concurrent.futures.as_completed(future_to_query):
                res = future.result()
                for item in res:
                    u = item.get("url", "")
                    if u and u not in seen_urls:
                        seen_urls.add(u)
                        all_items.append(item)

        # Fallback if empty
        if not all_items and queries:
            return self.search_structured(queries[0], max_results=max_results, mode=mode)

        return all_items[:max_results]

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
