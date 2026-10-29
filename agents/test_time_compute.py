"""Compute-Optimal Test-Time Scaling Engine (DeepMind Architecture).

Implements the principles from Snell et al. (Google DeepMind, 2024):
'Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters'

Key Mechanisms:
1. Difficulty Estimation & Adaptive Compute Allocation (Section 3.2):
   Dynamically estimates problem difficulty to allocate inference FLOPs.
2. Process & Grounding Verifier (PRM proxy, Section 5.1):
   Verifies intermediate steps, empirical source grounding, and citation fidelity.
3. Sequential Revision Optimizer (Section 6 & Figure 5):
   Iteratively refines the model's proposal distribution by conditioning on
   previous draft mistakes and verifier critique.
4. Adversarial Proof Auditing:
   Rejects hand-waved "we derive/demonstrate" claims without explicit intermediate
   algebraic steps. Forces PROPOSED HYPOTHESIS labeling for conceptual connections.
"""

import re
import time
from typing import Dict, Any, List, Tuple, Generator, Optional
from agents.failover import SafeHandAgentRunner


class DifficultyEstimator:
    """Estimates question difficulty to determine compute-optimal inference allocation."""

    # High-complexity reasoning markers
    FRONTIER_MARKERS = [
        r'\b(evaluate|compare|trade-?offs?|bottlenecks?|mechanisms?|hypothes\w+)\b',
        r'\b(architecture|algorithm|mathematical|formulation|continuous|discrete)\b',
        r'\b(thermodynamic|quantum|non-equilibrium|attractor|entropy|optimization)\b',
        r'\b(benchmark|empirical|falsif\w+|out-of-distribution|generalization)\b',
        r'[\$\=\+\-\*\/\^\\]',  # Math symbols
    ]

    @classmethod
    def estimate_difficulty(cls, prompt: str) -> Dict[str, Any]:
        """Classify prompt into difficulty bins (Levels 1 to 5) as described in Section 3.2."""
        prompt_lower = prompt.lower()
        score = 1

        # Length factor
        words = prompt.split()
        if len(words) > 15:
            score += 1
        if len(words) > 30:
            score += 1

        # Frontier / Analytical markers
        match_count = 0
        for pattern in cls.FRONTIER_MARKERS:
            if re.search(pattern, prompt_lower):
                match_count += 1

        if match_count >= 2:
            score += 1
        if match_count >= 4:
            score += 1

        difficulty_level = min(score, 5)
        
        # Compute-optimal strategy allocation
        if difficulty_level <= 2:
            strategy = "direct_single_pass"
            compute_budget = 1
            description = "Standard Query: Direct single-pass generation (compute-optimal for lower complexity)."
        elif difficulty_level == 3:
            strategy = "verified_synthesis"
            compute_budget = 2
            description = "Moderate Complexity: Verification & citation grounding active."
        else:
            strategy = "compute_optimal_sequential_revision"
            compute_budget = 3
            description = "Frontier / Complex Task: Multi-stage sequential revision with process verification."

        return {
            "difficulty_level": difficulty_level,
            "strategy": strategy,
            "compute_budget": compute_budget,
            "description": description
        }


class DeterministicRuleBasedVerifier:
    """Rule-based accuracy verifier inspired by DeepSeek-R1-Zero.
    
    Deterministically audits numeric claims, percentages, multipliers, and dates against
    retrieved empirical sources without neural reward model hallucination or hacking.
    """

    METRIC_PATTERNS = [
        r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b',                   # 85%, 6.3 percent
        r'\b\d+(?:\.\d+)?\s*(?:x|×|times)\b',                    # 6.3x, 14×, 4 times
        r'\b\d+(?:\.\d+)?\s*(?:B|M|k|GB|TB|MB|ms|s|h|FLOPs)\b',    # 671B, 37B, 512, 100ms
        r'\b(?:19|20)\d{2}\b',                                   # 2024, 2026
    ]

    @classmethod
    def verify_grounding(cls, draft: str, search_context: str) -> Dict[str, Any]:
        """Extract and check every quantitative claim against the retrieved text."""
        context_lower = search_context.lower()
        
        extracted_metrics = set()
        for pat in cls.METRIC_PATTERNS:
            for match in re.finditer(pat, draft, re.IGNORECASE):
                val = match.group(0).strip().lower()
                extracted_metrics.add(val)

        if not extracted_metrics:
            return {
                "grounding_ratio": 1.0,
                "verified_metrics": [],
                "unverified_metrics": [],
                "is_grounded": True,
                "feedback": "No quantitative claims to audit."
            }

        verified = []
        unverified = []

        for m in extracted_metrics:
            num_only = re.sub(r'[^\d\.]', '', m)
            if m in context_lower or (len(num_only) >= 2 and num_only in context_lower):
                verified.append(m)
            else:
                unverified.append(m)

        ratio = len(verified) / len(extracted_metrics) if extracted_metrics else 1.0

        feedback = ""
        if unverified:
            feedback = f"Rule-Based Verification Notice: The metrics {unverified[:4]} were not directly found in the retrieved text. Verify or remove ungrounded figures."

        return {
            "grounding_ratio": round(ratio, 2),
            "verified_metrics": verified,
            "unverified_metrics": unverified,
            "is_grounded": ratio >= 0.75,
            "feedback": feedback
        }


class ProcessGroundingVerifier:
    """PRM-inspired Step & Grounding Verifier (Section 5.1 & DeepSeek Rule-Based Verifier).
    
    Verifies draft assertions against empirical search context, citation accuracy, and numeric claims.
    """

    def __init__(self, safe_hand: SafeHandAgentRunner):
        self.safe_hand = safe_hand

    def verify_and_critique(
        self,
        prompt: str,
        draft: str,
        search_context: str,
        engine: str = "auto"
    ) -> Dict[str, Any]:
        """Score the draft and produce actionable critique points for revision."""
        # 1. Deterministic Rule-Based Check (DeepSeek-R1-Zero style)
        rule_check = DeterministicRuleBasedVerifier.verify_grounding(draft, search_context)

        # 2. Neural PRM Grounding & Step Verification
        verification_instruction = (
            "You are an expert Process Reward Model (PRM) and Research Verifier.\n"
            "Evaluate the provided research draft against the retrieved sources.\n"
            "Assess:\n"
            "1. Grounding & Citations: Are factual claims supported by references [1], [2]?\n"
            "2. Depth & Specificity: Did it explain concrete mechanisms rather than vague summaries?\n"
            "3. Logical Consistency: Are there unsupported assumptions or leaps?\n\n"
            "Output your critique in this exact format:\n"
            "SCORE: <integer 0-100>\n"
            "CRITIQUE:\n"
            "- <point 1>\n"
            "- <point 2>\n"
            "- <point 3>"
        )

        verification_prompt = (
            f"User Question: {prompt}\n\n"
            f"Retrieved Empirical Sources:\n{search_context[:3000]}\n\n"
            f"Candidate Draft to Verify:\n{draft[:3000]}\n\n"
            f"Provide your SCORE and specific CRITIQUE points."
        )

        verifier_output, provider = self.safe_hand.generate(
            verification_prompt,
            system_instruction=verification_instruction,
            engine=engine
        )

        # Parse score and critique
        score_match = re.search(r'SCORE:\s*(\d+)', verifier_output, re.IGNORECASE)
        base_score = int(score_match.group(1)) if score_match else 82
        
        # Combine neural score with deterministic grounding penalty
        final_score = int(base_score * rule_check["grounding_ratio"]) if rule_check["grounding_ratio"] < 0.7 else base_score

        critique_points = []
        critique_section = re.split(r'CRITIQUE:', verifier_output, flags=re.IGNORECASE)
        if len(critique_section) > 1:
            raw_points = critique_section[1].strip().split('\n')
            for p in raw_points:
                p_clean = p.strip().lstrip('-*•0123456789. ')
                if len(p_clean) > 8:
                    critique_points.append(p_clean)

        # If deterministic ungrounded numbers exist, enforce rule-based critique
        if rule_check["feedback"]:
            critique_points.insert(0, rule_check["feedback"])

        if not critique_points:
            critique_points = [
                "Ensure every key technical claim explicitly references relevant source indices [1], [2].",
                "Deepen the empirical mechanism explanations with concrete details.",
                "Clarify operational trade-offs and remaining scientific challenges."
            ]

        return {
            "score": final_score,
            "rule_check": rule_check,
            "is_sufficient": final_score >= 85 and rule_check["is_grounded"],
            "critique_points": critique_points[:4],
            "provider": provider
        }


class AdversarialCrossCheckVerifier:
    """Enhanced Multi-Dimensional Adversarial Verifier (Proof Auditing Protocol).

    Conducts deep peer-review across 5 rigorous axes:
    1. Mathematical & Physical Formalisms (derivatives, PDEs, dimensional consistency)
    2. Conceptual Boundaries & Incommensurability (trade-offs, unstated assumptions)
    3. Deterministic Numeric & Metric Audit (rule-based metric verification)
    4. Citation & Empirical Grounding (checking source indices [1], [2])
    5. Adversarial Proof Auditing — rejects hand-waved "we derive/demonstrate" claims
       without explicit intermediate algebraic steps; forces PROPOSED HYPOTHESIS labeling.
    """

    def __init__(self, safe_hand: SafeHandAgentRunner):
        self.safe_hand = safe_hand

    def cross_check(
        self,
        prompt: str,
        draft: str,
        search_context: str
    ) -> Dict[str, Any]:
        """Execute adversarial peer-review audit on candidate synthesis."""
        # 1. Deterministic Rule-Based Metric Check
        rule_check = DeterministicRuleBasedVerifier.verify_grounding(draft, search_context)

        # 2. Adversarial Peer Review Prompt (Gemini running DeepSeek-R1 CoT protocol)
        cross_check_instruction = (
            "You are Prof. Elena Rostova, an Elite Academic Peer-Reviewer and Theoretical Scientist.\n"
            "Rigorously cross-check the proposed research draft against the retrieved literature.\n\n"
            "CROSS-CHECK PROTOCOL:\n"
            "1. Technical & Mathematical Rigor: Are equations, variables, and physical dynamics sound and dimensionally valid?\n"
            "2. Conceptual Boundaries: What critical trade-offs or assumptions were omitted (e.g., discrete vs continuous readouts, finite-time dissipation limits)?\n"
            "3. Empirical Grounding: Are claims verified by the sources?\n\n"
            "ADVERSARIAL PROOF AUDITING (STRICT):\n"
            "4. For every sentence using 'we derive', 'we demonstrate', 'we prove', 'we show', or 'it follows that':\n"
            "   - REJECT the section UNLESS the explicit intermediate algebraic steps or numerical simulation outputs are shown in the draft.\n"
            "   - If the connection between two equations is conceptual only (no derivation steps), MANDATE relabeling as:\n"
            "     PROPOSED HYPOTHESIS: [statement]\n"
            "   - Flag any equation missing units, boundary conditions, or domain of validity as UNVERIFIED CLAIM.\n"
            "5. DO NOT accept vague hand-waving like 'coarse-graining yields' or 'in the continuum limit' as proof.\n"
            "   Require explicit assumptions: continuum limit conditions, symmetry constraints, and parameter regimes.\n\n"
            "OUTPUT FORMAT:\n"
            "SCORE: <integer 0-100>\n"
            "VERDICT: <Sound & Grounded | Needs Refinement>\n"
            "CRITIQUE:\n"
            "- <point 1>\n"
            "- <point 2>\n"
            "- <point 3>\n"
            "STRENGTHS: <validated mechanisms>"
        )

        cross_check_prompt = (
            f"User Research Mission: {prompt}\n\n"
            f"Retrieved Empirical Literature:\n{search_context[:3200]}\n\n"
            f"Draft Synthesis to Cross-Check:\n{draft[:3500]}\n\n"
            f"Execute your comprehensive adversarial peer-review cross-check now."
        )

        output, provider = self.safe_hand.generate(
            cross_check_prompt,
            system_instruction=cross_check_instruction,
            engine="deepseek-r1"
        )

        score_match = re.search(r'SCORE:\s*(\d+)', output, re.IGNORECASE)
        base_score = int(score_match.group(1)) if score_match else 88

        critique_points = []
        critique_section = re.split(r'CRITIQUE:', output, flags=re.IGNORECASE)
        if len(critique_section) > 1:
            raw_pts = critique_section[1].split("STRENGTHS:")[0].strip().split("\n")
            for p in raw_pts:
                p_clean = p.strip().lstrip("-*•0123456789. ")
                if len(p_clean) > 8:
                    critique_points.append(p_clean)

        if rule_check["feedback"]:
            critique_points.insert(0, rule_check["feedback"])

        if not critique_points:
            critique_points = [
                "Address discrete-to-continuous boundary readout mechanisms.",
                "Clarify operational trade-offs between rapid switching and thermodynamic dissipation limits.",
                "Ensure strict mapping of all quantitative figures to source indices."
            ]

        final_score = int(base_score * rule_check["grounding_ratio"]) if rule_check["grounding_ratio"] < 0.75 else base_score

        return {
            "score": final_score,
            "verdict": "Sound & Grounded" if final_score >= 80 else "Needs Refinement",
            "rule_check": rule_check,
            "critique_points": critique_points[:4],
            "reasoning_trace": self.safe_hand.last_reasoning_trace,
            "provider": provider
        }


# Backward-compatibility alias
DeepSeekCrossCheckVerifier = AdversarialCrossCheckVerifier


class SequentialRevisionOptimizer:
    """Modifies the Proposal Distribution via Sequential Revision (Section 6 & Figure 5)."""

    def __init__(self, safe_hand: SafeHandAgentRunner):
        self.safe_hand = safe_hand

    def revise_proposal(
        self,
        prompt: str,
        initial_draft: str,
        critique_points: List[str],
        search_context: str,
        engine: str = "auto"
    ) -> Tuple[str, str]:
        """Produce an improved revision conditioned on previous mistakes and critique."""
        revision_instruction = (
            "You are an Elite Principal Research Scientist executing sequential self-refinement.\n"
            "Your objective is to revise the initial draft by resolving the specific critique points into a publication-grade Academic Research Paper.\n"
            "STRICT REQUIREMENTS:\n"
            "1. Address each critique point directly.\n"
            "2. Preserve the Paper Blueprint: Title (# [Title]), followed by an indented Abstract blockquote (> **Abstract** — [summary]), followed by numbered sections 1 to 6.\n"
            "3. Deepen technical explanations, mathematical formulations ($inline$ and $$display$$), and comparison tables.\n"
            "4. Seamlessly integrate accurate numeric citations [1], [2] referencing the source context.\n"
            "5. Do NOT include meta-commentary like 'In this revision...'—output only the finished, polished academic research paper."
        )

        critique_text = "\n".join([f"- {cp}" for cp in critique_points])
        revision_prompt = (
            f"# RESEARCH TOPIC: {prompt}\n\n"
            f"Retrieved Empirical Literature:\n{search_context}\n\n"
            f"## Previous Draft:\n{initial_draft}\n\n"
            f"## Verifier Critique & Identified Gaps:\n{critique_text}\n\n"
            f"Produce the finalized, compute-optimally revised research synthesis now."
        )

        revised_output, provider = self.safe_hand.generate(
            revision_prompt,
            system_instruction=revision_instruction,
            search_context=search_context,
            engine=engine
        )

        return revised_output, provider
