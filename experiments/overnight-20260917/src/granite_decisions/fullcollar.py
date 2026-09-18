"""Provider for the inspected Fullcollar 0.1.0 Python interface."""

from .contracts import ContractError, digest
from .llamacpp import BackendError


class GraniteProvider:
    name = "granite-decisions"

    def __init__(self, engine, shadow_only=False):
        self.engine, self.shadow_only = engine, shadow_only
        self.model = engine.backend.identity.get("runtime", {}).get("model_id", "granite-4.1")
        self.probability_basis = engine.backend.calibration
        self.revision = digest({"backend": engine.backend.identity, "policy": engine.policy.as_dict(), "shadow_only": shadow_only})

    def evaluate(self, state, questions):
        from fullcollar.contracts import ProviderError
        try:
            response = self.engine.evaluate(state, questions)
            if response["status"] == "abstained" and not self.shadow_only:
                raise ProviderError("granite_decisions_abstained")
            return response
        except (ContractError, BackendError) as exc:
            raise ProviderError(str(exc)) from None


def shadow_engine(engine, **kwargs):
    """Uncalibrated experiments must use this factory, which fixes shadow mode."""
    from fullcollar.engine import Engine
    if kwargs.pop("mode", "shadow") != "shadow":
        raise ContractError("this_factory_only_allows_shadow_mode")
    return Engine(provider=GraniteProvider(engine, shadow_only=True), mode="shadow", **kwargs)
