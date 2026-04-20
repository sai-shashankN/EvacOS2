"""EvacOS-MA OpenEnv Package.

Re-exports the multi-agent schema types and exposes the OpenEnv
server/client surface for the benchmark.
"""

from __future__ import annotations

VERSION: str = "0.1.0"

# Re-export key types for convenience
from evacos_ma.schemas.multi_agent import (  # noqa: E402, F401
    REWARD_SCHEMA_VERSION,
    PROMPT_TEMPLATE_VERSION,
    TRACE_SCHEMA_VERSION,
    ObservationEnvelopeMA,
    FloorAgentObservationMA,
    OrchestratorObservationMA,
    RoleObservationMA,
    ActionEnvelopeMA,
    ActionBundleMA,
    StepResultMA,
    StepResultInfo,
    AgentRole,
    Tier,
    ActionTypeMA,
)
from evacos_ma.schemas.rewards import (  # noqa: E402, F401
    RewardBreakdown,
    RoleReward,
    RewardsByRole,
)
