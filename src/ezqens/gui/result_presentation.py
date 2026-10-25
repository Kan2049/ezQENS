"""GUI-only names and colors for typed Result component identities."""

from ezqens.fitting import ComponentIdentity
from ezqens.gui.scientific_canvas import SCIENTIFIC_LORENTZIAN_COLORS
from ezqens.workflow import ManualModelState, UnifiedResult

SERIES_ALIAS_NOTE = "Series labels apply to this Result only"


def lorentzian_slots(
    result: UnifiedResult | None, model: ManualModelState | None = None
) -> dict[ComponentIdentity, int]:
    """Number first appearances in authoritative Result Q/component order."""
    identities: list[ComponentIdentity] = []
    if result is not None:
        for point in result.points:
            fit = point.fit_result
            if fit is not None and fit.fitted_model is not None:
                for component in fit.fitted_model.lorentzians:
                    if component.identity not in identities:
                        identities.append(component.identity)
    if model is not None:
        for working_component in model.lorentzians:
            if working_component.identity not in identities:
                identities.append(working_component.identity)
    return {identity: index for index, identity in enumerate(identities)}


def series_color(slot: int) -> str:
    return SCIENTIFIC_LORENTZIAN_COLORS[slot % len(SCIENTIFIC_LORENTZIAN_COLORS)]
