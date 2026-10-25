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


class ProcessGroundingVerifier:
    """PRM-inspired Step & Grounding Verifier (Section 5.1).
    
    Verifies draft assertions against empirical search context and citation accuracy.
    """

    def __init__(self, safe_hand: SafeHandAgentRunner):
        self.safe_hand = safe_hand

    def verify_and_critique(
        self,
        prompt: str,
        draft: str,
        search_context: str
    ) -> Dict[str, Any]:
        """Score the draft and produce actionable critique points for revision."""
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
            system_instruction=verification_instruction
        )

        # Parse score and critique
        score_match = re.search(r'SCORE:\s*(\d+)', verifier_output, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 82

        critique_points = []
        critique_section = re.split(r'CRITIQUE:', verifier_output, flags=re.IGNORECASE)
        if len(critique_section) > 1:
            raw_points = critique_section[1].strip().split('\n')
            for p in raw_points:
                p_clean = p.strip().lstrip('-*•0123456789. ')
                if len(p_clean) > 8:
                    critique_points.append(p_clean)

        if not critique_points:
            critique_points = [
                "Ensure every key technical claim explicitly references relevant source indices [1], [2].",
                "Deepen the empirical mechanism explanations with concrete details.",
                "Clarify operational trade-offs and remaining scientific challenges."
            ]

        return {
            "score": score,
            "is_sufficient": score >= 85,
            "critique_points": critique_points[:3],
            "provider": provider
        }


class SequentialRevisionOptimizer:
    """Modifies the Proposal Distribution via Sequential Revision (Section 6 & Figure 5)."""

    def __init__(self, safe_hand: SafeHandAgentRunner):
        self.safe_hand = safe_hand

    def revise_proposal(
        self,
        prompt: str,
        initial_draft: str,
        critique_points: List[str],
        search_context: str
    ) -> Tuple[str, str]:
        """Produce an improved revision conditioned on previous mistakes and critique."""
        revision_instruction = (
            "You are an Elite Research Scientist executing sequential self-refinement.\n"
            "Your objective is to revise the initial draft by resolving the specific critique points.\n"
            "STRICT REQUIREMENTS:\n"
            "1. Address each critique point directly.\n"
            "2. Preserve all accurate findings while deepening technical explanations, data, or equations.\n"
            "3. Seamlessly integrate accurate numeric citations [1], [2] referencing the source context.\n"
            "4. Organize with clean, natural thematic Markdown headings tailored to the topic.\n"
            "5. Do NOT include meta-commentary like 'In this revision...'—output only the finished, polished research report."
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
            search_context=search_context
        )

        return revised_output, provider
