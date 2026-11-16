"""GUI-only names and colors for typed Result component identities."""

from ezqens.fitting import ComponentIdentity
from ezqens.gui.scientific_canvas import SCIENTIFIC_LORENTZIAN_COLORS
from ezqens.workflow import ManualModelState, UnifiedResult


def lorentzian_slots(
    result: UnifiedResult | None,
    model: ManualModelState | None = None,
    *,
    group_index: int = 0,
) -> dict[ComponentIdentity, int]:
    """Project one Q's working-state numbers to zero-based plotting slots."""
    if model is None and result is not None:
        model = result.point(group_index).working_state
    return (
        {}
        if model is None
        else {identity: number - 1 for identity, number in model.lorentzian_numbers}
    )


def series_color(slot: int) -> str:
    return SCIENTIFIC_LORENTZIAN_COLORS[slot % len(SCIENTIFIC_LORENTZIAN_COLORS)]
