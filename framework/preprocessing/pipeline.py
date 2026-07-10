from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.progress import progress_iter, suppress_warnings
from src.breast_mri_ai.breast_dce_moe_pinn_foundation.framework.utils.security import PathPolicy


class PreprocessingCallable(Protocol):
    def __call__(self, context: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PreprocessingStep:
    name: str
    operation: PreprocessingCallable
    enabled: bool = True
    skip_if_output_exists: Path | None = None


@dataclass
class StepResult:
    name: str
    status: str
    elapsed_seconds: float
    message: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)


class PreprocessingPipeline:
    """Small resumable pipeline runner for dataset-specific adapters."""

    def __init__(self, steps: list[PreprocessingStep], policy: PathPolicy):
        self.steps = steps
        self.policy = policy

    def run(self, context: dict[str, Any]) -> list[StepResult]:
        suppress_warnings()
        results: list[StepResult] = []
        for step in progress_iter(self.steps, total=len(self.steps), desc="Run preprocessing", unit="step"):
            start = time.perf_counter()
            if not step.enabled:
                results.append(StepResult(step.name, "skipped_disabled", 0.0))
                continue
            if step.skip_if_output_exists is not None:
                output_path = self.policy.assert_write_allowed(step.skip_if_output_exists)
                if output_path.exists():
                    results.append(StepResult(step.name, "skipped_existing", 0.0, str(output_path)))
                    continue
            try:
                update = step.operation(context)
                context.update(update)
            except Exception as exc:
                results.append(
                    StepResult(step.name, "failed", time.perf_counter() - start, message=str(exc))
                )
                raise
            results.append(
                StepResult(
                    step.name,
                    "ok",
                    time.perf_counter() - start,
                    outputs={key: value for key, value in update.items() if key.endswith("_path")},
                )
            )
        return results
