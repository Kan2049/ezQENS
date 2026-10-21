"""Exact, non-refitting re-expression of standard AutoFit Results for Manual use."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .core import _parameter_slots
from .models import (
    BACKGROUND_COMPONENT,
    BackgroundModel,
    CenterGroup,
    ComponentIdentity,
    FitResult,
    ParameterConfiguration,
    ParameterEstimate,
    ParameterFamily,
    ParameterReference,
    SpectralModelDefinition,
)


def relabel_fit_components(
    fit: FitResult,
    identities: dict[ComponentIdentity, ComponentIdentity],
) -> FitResult:
    """Exactly relabel Lorentzian identities without moving numerical evidence."""

    original = {item.identity for item in fit.configuration.lorentzians}
    if set(identities) != original or len(set(identities.values())) != len(identities):
        raise ValueError("component relabel must be a one-to-one complete mapping")
    if (
        fit.fitted_model is None
        or {item.identity for item in fit.fitted_model.lorentzians} != original
    ):
        raise ValueError("fitted component identities differ from configuration")

    def reference(value: ParameterReference) -> ParameterReference:
        replacement = identities.get(value.component)
        return value if replacement is None else replace(value, component=replacement)

    def model(value: SpectralModelDefinition) -> SpectralModelDefinition:
        return replace(
            value,
            lorentzians=tuple(
                replace(item, identity=identities[item.identity])
                for item in value.lorentzians
            ),
            parameter_ties=tuple(
                replace(tie, members=tuple(reference(item) for item in tie.members))
                for tie in value.parameter_ties
            ),
        )

    return replace(
        fit,
        configuration=model(fit.configuration),
        fitted_model=model(fit.fitted_model),
        parameters=tuple(
            replace(item, references=tuple(reference(ref) for ref in item.references))
            for item in fit.parameters
        ),
        evaluation=replace(
            fit.evaluation,
            component_curves=tuple(
                replace(
                    curve,
                    component=identities.get(curve.component, curve.component),
                )
                for curve in fit.evaluation.component_curves
            ),
        ),
    )


def _require_standard_model(model: SpectralModelDefinition) -> None:
    if (
        model.elastic_area is None
        or model.energy_shift is None
        or model.lorentzian_count > 2
        or model.center_groups
        or model.parameter_ties
        or model.elastic_center_group is not None
        or any(
            component.center is not None or component.center_group is not None
            for component in model.lorentzians
        )
    ):
        raise ValueError("Result is not in the standard shared-center AutoFit family")


def _manual_representation(
    model: SpectralModelDefinition,
    center_group_id: str,
) -> SpectralModelDefinition:
    _require_standard_model(model)
    energy_shift = model.energy_shift
    if energy_shift is None:  # validated above
        raise RuntimeError("standard AutoFit model lost its shared center")
    shared = bool(model.lorentzians)
    return SpectralModelDefinition(
        energy_shift=None if shared else energy_shift,
        elastic_area=model.elastic_area,
        lorentzians=tuple(
            replace(component, center_group=center_group_id if shared else None)
            for component in model.lorentzians
        ),
        background=(
            BackgroundModel.LINEAR
            if model.background is BackgroundModel.CONSTANT
            else model.background
        ),
        b0=model.b0,
        b1=(
            ParameterConfiguration(0.0, free=False)
            if model.background is BackgroundModel.CONSTANT
            else model.b1
        ),
        center_groups=((CenterGroup(center_group_id, energy_shift),) if shared else ()),
        elastic_center_group=center_group_id if shared else None,
    )


def convert_standard_auto_fit_result_to_manual(
    fit: FitResult,
    *,
    center_group_id: str,
) -> FitResult:
    """Re-express one successful standard fit with exact Manual slots, without fitting.

    The original AutoFit evidence remains separate. All values, numerical evidence,
    and context are retained; only the parameter representation is changed.
    """

    if not isinstance(fit, FitResult) or not fit.diagnostics.optimizer_success:
        raise ValueError("conversion requires a successful standard AutoFit FitResult")
    if (
        not isinstance(center_group_id, str)
        or not center_group_id.strip()
        or center_group_id != center_group_id.strip()
    ):
        raise ValueError("center_group_id must be a canonical nonempty string")
    original = fit.configuration
    fitted = fit.fitted_model
    if fitted is None:
        raise ValueError("successful AutoFit Result has no fitted model")
    if original.background is not fitted.background or {
        item.identity for item in original.lorentzians
    } != {item.identity for item in fitted.lorentzians}:
        raise ValueError("AutoFit submitted and fitted model topology differs")
    converted_configuration = _manual_representation(original, center_group_id)
    converted_fitted = _manual_representation(fitted, center_group_id)
    old_slots = _parameter_slots(fitted)
    new_slots = _parameter_slots(converted_fitted)
    if tuple(slot.name for slot in old_slots) != tuple(
        parameter.name for parameter in fit.parameters
    ) or any(
        slot.key != frozenset(parameter.references)
        for slot, parameter in zip(old_slots, fit.parameters, strict=True)
    ):
        raise ValueError("AutoFit Result parameter schema differs from its model")
    old_by_key = {slot.key: index for index, slot in enumerate(old_slots)}
    slope_key = frozenset(
        (ParameterReference(BACKGROUND_COMPONENT, ParameterFamily.SLOPE),)
    )
    indices: list[int | None] = []
    parameters: list[ParameterEstimate] = []
    for slot in new_slots:
        index = old_by_key.get(slot.key)
        if index is None and not (
            original.background is BackgroundModel.CONSTANT and slot.key == slope_key
        ):
            raise ValueError("converted Manual parameter slot has no AutoFit source")
        indices.append(index)
        if index is None:
            parameters.append(
                ParameterEstimate(
                    slot.name,
                    0.0,
                    None,
                    slot.configuration.lower_bound,
                    slot.configuration.upper_bound,
                    False,
                    references=slot.references,
                )
            )
        else:
            parameters.append(
                replace(
                    fit.parameters[index], name=slot.name, references=slot.references
                )
            )
    if {index for index in indices if index is not None} != set(range(len(old_slots))):
        raise ValueError("conversion did not preserve every AutoFit parameter slot")

    def remap_matrix(matrix: np.ndarray | None) -> np.ndarray | None:
        if matrix is None:
            return None
        remapped = np.full((len(indices), len(indices)), np.nan, dtype=np.float64)
        for row, source_row in enumerate(indices):
            if source_row is None:
                continue
            for column, source_column in enumerate(indices):
                if source_column is not None:
                    remapped[row, column] = matrix[source_row, source_column]
        return remapped

    names = {
        old.name: new.name
        for old, new in zip(
            (fit.parameters[index] for index in indices if index is not None),
            (
                parameter
                for index, parameter in zip(indices, parameters, strict=True)
                if index is not None
            ),
            strict=True,
        )
    }
    diagnostics = fit.diagnostics
    remapped_starts = tuple(
        replace(
            start,
            start_parameter_values=tuple(
                start.start_parameter_values[index] if index is not None else 0.0
                for index in indices
            ),
            fitted_parameter_values=tuple(
                start.fitted_parameter_values[index] if index is not None else 0.0
                for index in indices
            ),
        )
        for start in diagnostics.alternative_starts
    )
    return replace(
        fit,
        configuration=converted_configuration,
        fitted_model=converted_fitted,
        parameters=tuple(parameters),
        covariance=remap_matrix(fit.covariance),
        correlation=remap_matrix(fit.correlation),
        diagnostics=replace(
            diagnostics,
            active_bounds=tuple(
                f"{names[name]}:{side}"
                for name, side in (
                    item.rsplit(":", 1) for item in diagnostics.active_bounds
                )
            ),
            relative_standard_errors=tuple(
                diagnostics.relative_standard_errors[index]
                if index is not None
                else None
                for index in indices
            ),
            alternative_starts=remapped_starts,
        ),
    )
