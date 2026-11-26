from dataclasses import dataclass, field


@dataclass
class GuardrailResult:
    passed: bool
    category: str
    details: str
    confidence: float
    latency_ms: float


@dataclass
class GuardrailReport:
    input_results: list = field(default_factory=list)
    output_results: list = field(default_factory=list)
    blocked: bool = False
    block_reason: str = ""
    total_latency_ms: float = 0.0
