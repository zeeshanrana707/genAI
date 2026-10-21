"""Modular Hook System for Extensible Multi-Agent Architectures.

Provides lifecycle hooks and extension points to easily integrate future agents
such as Audio Agents (e.g. voice briefings, podcast generation) and
Video Agents (e.g. AI avatar generation, slide decks).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List
from pathlib import Path


class PipelineStage(str, Enum):
    """Execution stages across the multi-agent pipeline."""
    PRE_PIPELINE = "pre_pipeline"
    POST_RESEARCH = "post_research"
    POST_EMAIL = "post_email"
    POST_SCHEDULE = "post_schedule"
    POST_PIPELINE = "post_pipeline"


class BaseAgentHook(ABC):
    """Abstract base class for pipeline extension hooks."""

    @property
    @abstractmethod
    def hook_name(self) -> str:
        """Name of the hook."""
        pass

    def on_stage_complete(self, stage: PipelineStage, stage_output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Triggered when a pipeline stage finishes.
        
        Args:
            stage: The completed stage.
            stage_output: Output data produced by the stage.
            context: Shared pipeline state dictionary.
            
        Returns:
            Updated context or metadata.
        """
        return context


class AudioAgentHook(BaseAgentHook):
    """Extension hook for future Audio Agent (e.g., ElevenLabs / Gemini Audio).
    
    Converts research summaries or client emails into an executive audio briefing.
    """

    @property
    def hook_name(self) -> str:
        return "AudioAgentHook"

    def on_stage_complete(self, stage: PipelineStage, stage_output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if stage == PipelineStage.POST_EMAIL:
            print("\n[Modular Hook: AudioAgent] Hook triggered after email drafting.")
            print("[Modular Hook: AudioAgent] Audio pipeline ready: synthesizes executive briefing podcast from email/research.")
            # Future integration point:
            # audio_bytes = elevenlabs_client.generate(text=stage_output)
            context["audio_briefing_available"] = True
            context["audio_briefing_spec"] = {
                "format": "mp3",
                "voice": "Executive Briefing Narrator",
                "status": "Ready for Audio Agent integration"
            }
        return context


class VideoAgentHook(BaseAgentHook):
    """Extension hook for future Video Agent (e.g., HeyGen / Runway / Sora API).
    
    Generates video pitch slides or synthetic avatar presentation for the client review.
    """

    @property
    def hook_name(self) -> str:
        return "VideoAgentHook"

    def on_stage_complete(self, stage: PipelineStage, stage_output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if stage == PipelineStage.POST_SCHEDULE:
            print("\n[Modular Hook: VideoAgent] Hook triggered after meeting scheduling.")
            print("[Modular Hook: VideoAgent] Video pipeline ready: prepares AI avatar presentation slides for Mr. Smith's review.")
            # Future integration point:
            # video_job = heygen_client.create_presentation(script=stage_output)
            context["video_presentation_available"] = True
            context["video_presentation_spec"] = {
                "aspect_ratio": "16:9",
                "presenter": "Enterprise AI Avatar",
                "status": "Ready for Video Agent integration"
            }
        return context


class HookRegistry:
    """Central registry managing modular hooks."""

    def __init__(self):
        self._hooks: List[BaseAgentHook] = []

    def register(self, hook: BaseAgentHook) -> None:
        """Register a new extension hook."""
        self._hooks.append(hook)
        print(f"[HookRegistry] Registered extension hook: {hook.hook_name}")

    def trigger(self, stage: PipelineStage, stage_output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all registered hooks for the specified stage."""
        for hook in self._hooks:
            try:
                context = hook.on_stage_complete(stage, stage_output, context)
            except Exception as e:
                print(f"[HookRegistry] Error in hook {hook.hook_name}: {e}")
        return context
