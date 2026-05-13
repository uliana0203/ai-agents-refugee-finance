"""
Public API for the AI agents package.

Three agents form the pipeline: RefugeeAgent generates synthetic client profiles,
ConsultantAgent produces grounded pension advice, EvaluatorAgent scores the output.
"""

from agents.refugee import RefugeeAgent, RefugeeAgentConfig
from agents.consultant import ConsultantAgent, ConsultantAgentConfig
from agents.evaluator import EvaluatorAgent, EvaluatorConfig

__all__ = [
    "RefugeeAgent",
    "RefugeeAgentConfig",
    "ConsultantAgent",
    "ConsultantAgentConfig",
    "EvaluatorAgent",
    "EvaluatorConfig",
]
