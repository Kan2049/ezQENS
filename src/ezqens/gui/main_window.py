"""Native main-window shell for ezQENS."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QRect, QSignalBlocker, QSize, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ezqens.batch import (
    MultiQFitOutcome,
    MultiQFitStatus,
)
from ezqens.domain import (
    DiagnosticSeverity,
    ImportDiagnostic,
    ImportValidationError,
    QBins,
    ReducedDataset,
    SpectrumRole,
)
from ezqens.fitting import (
    BACKGROUND_COMPONENT,
    ELASTIC_COMPONENT,
    BackgroundModel,
    CandidateFitResult,
    ComponentIdentity,
    FitResult,
    FittingError,
    ManualModelPreview,
    ParameterFamily,
    ParameterMetadata,
    ParameterReference,
    manual_parameter_metadata,
)
from ezqens.gui import masking
from ezqens.gui.auto_fit import AutoFitCandidateDialog, ResultNamesDialog
from ezqens.gui.dataset_view import FitDisplayStatus, ReducedDatasetView
from ezqens.gui.dialogs import (
    DialogChoice,
    choose_dialog,
    confirm_dialog,
    show_message_dialog,
)
from ezqens.gui.icons import IconName, apply_disclosure_icon, load_icon
from ezqens.gui.manual_fit import ManualFitEditor, ManualFitLifecycle
from ezqens.gui.manual_fit_all_q import (
    FitAllQTransferDialog,
    FitAllQWorker,
    TransferDefaultsStore,
)
from ezqens.gui.masking import MaskTaskDraft
from ezqens.gui.q_rebin import QRebinDialog, QRebinPreview
from ezqens.gui.theme import (
    DEFAULT_LAYOUT_TOKENS,
    Appearance,
    application_appearance_controller,
    indicator_tokens_for,
    tokens_for,
)
from ezqens.gui.units_editor import SourceUnitsDialog
from ezqens.gui.workspace import (
    DatasetState,
    ProjectState,
    QMethodState,
    WorkspaceSidebar,
)
from ezqens.io import parse_dave_q_bins
from ezqens.io.importers import import_reduced_data
from ezqens.preprocessing import BoundarySide, FittingSelection
from ezqens.workflow import (
    AutoFitCandidateBranchResult,
    FittingWorkspaceState,
    ManualComponentKind,
    ManualFitContext,
    ManualFitDraft,
    ManualFitExecutionOutcome,
    ManualMultiQFitExecutionOutcome,
    ManualParameterEdit,
    MultiQConstraintTransferPolicy,
    PendingManualInteraction,
    ProjectDataset,
    ResultDisplayStatus,
    ResultSnapshot,
    SavedFitResult,
    SingleQAutoFitOutcome,
    WorkflowDiagnostic,
    WorkflowDiagnosticCode,
    WorkflowError,
    WorkflowProject,
    active_manual_draft,
    add_project_dataset,
    adopt_selected_auto_fit_branch,
    adopt_single_q_auto_fit_candidate,
    append_saved_result,
    applied_resolution,
    apply_resolution,
    begin_component_interaction,
    commit_fitting_selection,
    complete_background_interaction,
    complete_elastic_interaction,
    complete_lorentzian_interaction,
    create_parameter_tie,
    create_project,
    draft_edit_items,
    fresh_analysis,
    join_center_group,
    join_parameter_tie,
    load_saved_result,
    manual_workflow_readiness,
    materialize_manual_setup,
    measured_point_correspondence_exactly_unchanged,
    open_manual_fit_draft,
    preflight_apply_resolution,
    preview_manual_fit,
    preview_pending_background_interaction,
    preview_pending_elastic_interaction,
    preview_pending_lorentzian_interaction,
    remove_manual_component,
    remove_project_dataset,
    remove_saved_result,
    rename_saved_result,
    replace_current_q_result,
    replace_project_dataset,
    resolve_manual_fit_context,
    result_q_display_status,
    revert_saved_result,
    run_and_adopt_manual_fit,
    run_single_q_auto_fit,
    save_current_result,
    save_current_result_as,
    set_best_saved_result,
    set_current_result,
    unsaved_result_items,
    untie_center_group_parameter,
    untie_parameter,
    update_manual_parameter,
    update_working_draft,
)

INSPECTOR_PREFERRED_WIDTH = 280
INSPECTOR_BASE_MINIMUM_WIDTH = 220
SUPPORTED_REDUCED_DATA_SUFFIXES = frozenset({".csv", ".dat", ".txt"})


@dataclass(frozen=True)
class ImportBatchFailure:
    """One file skipped during a GUI import operation."""

    path: Path
    detail: str


@dataclass(frozen=True)
class ImportBatchResult:
    """The compact outcome of importing one or more reduced-data files."""

    imported: tuple[DatasetState, ...]
    failures: tuple[ImportBatchFailure, ...]
    q_methods: tuple[QMethodState, ...] = ()

    @property
    def summary(self) -> str:
        """Return a small user-facing result summary."""

        if not self.q_methods:
            return f"{len(self.imported)} imported · {len(self.failures)} skipped"
        return (
            f"{len(self.imported)} data · {len(self.q_methods)} Q Methods · "
            f"{len(self.failures)} skipped"
        )


@dataclass(eq=False, slots=True)
class ManualFitExecutionState:
    """Transient run diagnostics and overlay visibility, never Result truth."""

    session: ManualFitSession
    group_index: int
    running: bool = False
    overlay_hidden: bool = False
    diagnostics: tuple[WorkflowDiagnostic, ...] = ()

    @property
    def fit_result(self) -> FitResult | None:
        current = self.session.result_workspace.current_result
        if current is None or self.overlay_hidden or self.running:
            return None
        point = current.point(self.group_index)
        model = self.session.draft.setup(self.group_index).model
        fitted_state = point.working_state
        # Match visible component sections only, not scientific validity. This
        # also keeps structural-overlay dismissal on a cached entry's reload.
        if model is None or fitted_state is None:
            return None
        if (
            (model.elastic_area is None) != (fitted_state.elastic_area is None)
            or model.background is not fitted_state.background
            or {item.identity for item in model.lorentzians}
            != {item.identity for item in fitted_state.lorentzians}
        ):
            return None
        return point.fit_result

    @property
    def lifecycle(self) -> ManualFitLifecycle:
        if self.running:
            return ManualFitLifecycle.FITTING
        status = self.session.display_status(self.group_index)
        if status is ResultDisplayStatus.FITTED_CURRENT:
            return ManualFitLifecycle.CURRENT
        if status is ResultDisplayStatus.NEEDS_FIT:
            return ManualFitLifecycle.NEEDS_FIT
        return ManualFitLifecycle.READY


class ManualFitSession:
    """Project the active workflow working copy into the Manual editor."""

    def __init__(
        self, owner: tuple[ProjectState, str], draft: ManualFitDraft, group_index: int
    ) -> None:
        self.owner = owner
        self._template = draft
        self.group_index = group_index
        self.result_workspace = FittingWorkspaceState()
        self.execution_by_group: dict[int, ManualFitExecutionState] = {}
        self.context_for: Callable[[int], ManualFitContext | None] = lambda _index: None
        self.draft_available = True
        self.editor_expanded = True
        self.result_expanded = False
        if any(setup.model is not None for setup in draft.setups):
            self.result_workspace = update_working_draft(self.result_workspace, draft)

    @property
    def draft(self) -> ManualFitDraft:
        if self.draft_available:
            self._template = active_manual_draft(self.result_workspace, self._template)
        return self._template

    @draft.setter
    def draft(self, draft: ManualFitDraft) -> None:
        self.result_workspace = update_working_draft(self.result_workspace, draft)
        self._template = draft

    def display_status(self, group_index: int) -> ResultDisplayStatus:
        return result_q_display_status(
            self.result_workspace,
            self.draft,
            group_index,
            self.context_for(group_index),
        )

    def resume(self, template: ManualFitDraft) -> None:
        """Re-open editing after a consent-gated role/topology retirement."""
        self.result_workspace, self._template = fresh_analysis(
            self.result_workspace, template, discard_untitled=True
        )
        self.draft_available = True

    def execution_state(
        self,
        group_index: int | None = None,
    ) -> ManualFitExecutionState:
        """Return the retained execution state for one validated draft Group."""

        target = self.group_index if group_index is None else group_index
        self.draft.setup(target)
        return self.execution_by_group.setdefault(
            target, ManualFitExecutionState(self, target)
        )

    @property
    def lifecycle(self) -> ManualFitLifecycle:
        return self.execution_state().lifecycle

    @lifecycle.setter
    def lifecycle(self, lifecycle: ManualFitLifecycle) -> None:
        self.execution_state().running = lifecycle is ManualFitLifecycle.FITTING

    @property
    def fit_result(self) -> FitResult | None:
        return self.execution_state().fit_result

    @property
    def execution_diagnostics(self) -> tuple[WorkflowDiagnostic, ...]:
        return self.execution_state().diagnostics

    @execution_diagnostics.setter
    def execution_diagnostics(
        self,
        diagnostics: tuple[WorkflowDiagnostic, ...],
    ) -> None:
        self.execution_state().diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class FitAllQRunContext:
    """Immutable application identity captured for one worker generation."""

    generation: int
    session: ManualFitSession
    owner: tuple[ProjectState, str]
    workflow: WorkflowProject
    draft: ManualFitDraft
    anchor_group: int
    anchor_fit: FitResult
    dataset: ReducedDataset
    selection: FittingSelection | None


@dataclass(slots=True)
class SampleViewSession:
    """Presentation-only scientific-view state retained for one Sample."""

    group_index: int = 0
    y_scale: str = "linear"
    heatmap_scale: str = "linear"


def _changed_fitting_selection_groups(
    before: FittingSelection | None,
    after: FittingSelection | None,
    group_count: int,
) -> tuple[int, ...]:
    """Identify Groups whose public effective fitting selection changed."""

    if before is None or after is None:
        return () if before is after else tuple(range(group_count))
    if (
        len(before.dataset.spectra) != group_count
        or len(after.dataset.spectra) != group_count
    ):
        return tuple(range(group_count))
    return tuple(
        group_index
        for group_index in range(group_count)
        if before.ranges[group_index] != after.ranges[group_index]
        or not np.array_equal(
            before.excluded_mask(group_index),
            after.excluded_mask(group_index),
        )
    )


def _active_application() -> QApplication:
    """Return the process QApplication required to construct this window."""
    application = QApplication.instance()
    if not isinstance(application, QApplication):
        message = "ezQENS MainWindow requires an active QApplication."
        raise RuntimeError(message)
    return application


def inspector_outward_expansion_width(
    window_geometry: QRect,
    available_geometry: QRect,
    requested_width: int,
    *,
    is_maximized: bool,
) -> int:
    """Return a bounded rightward Inspector expansion, or zero when unavailable."""
    if is_maximized or requested_width <= 0:
        return 0
    if not available_geometry.contains(window_geometry):
        return 0
    if window_geometry.width() + requested_width > available_geometry.width():
        return 0
    right_space = (
        available_geometry.x()
        + available_geometry.width()
        - (window_geometry.x() + window_geometry.width())
    )
    return requested_width if right_space >= requested_width else 0


class MainWindow(QMainWindow):
    """Responsive application shell with workspace, canvas, and inspector."""

    @property
    def _manual_owner(self) -> tuple[ProjectState, str] | None:
        session = self._active_manual_session
        return None if session is None else session.owner

    @property
    def _manual_draft(self) -> ManualFitDraft | None:
        session = self._active_manual_session
        return None if session is None else session.draft

    @_manual_draft.setter
    def _manual_draft(self, draft: ManualFitDraft) -> None:
        if self._active_manual_session is None:
            raise RuntimeError("Manual draft assignment requires an active session")
        session = self._active_manual_session
        session.draft = draft

    @property
    def _manual_fit_result(self) -> FitResult | None:
        session = self._active_manual_session
        return None if session is None else session.fit_result

    @property
    def _manual_fit_lifecycle(self) -> ManualFitLifecycle:
        session = self._active_manual_session
        return ManualFitLifecycle.READY if session is None else session.lifecycle

    @_manual_fit_lifecycle.setter
    def _manual_fit_lifecycle(self, lifecycle: ManualFitLifecycle) -> None:
        if self._active_manual_session is None:
            if lifecycle is ManualFitLifecycle.READY:
                return
            raise RuntimeError("Manual lifecycle assignment requires an active session")
        self._active_manual_session.lifecycle = lifecycle

    @property
    def _manual_execution_diagnostics(self) -> tuple[WorkflowDiagnostic, ...]:
        session = self._active_manual_session
        return () if session is None else session.execution_diagnostics

    @_manual_execution_diagnostics.setter
    def _manual_execution_diagnostics(
        self,
        diagnostics: tuple[WorkflowDiagnostic, ...],
    ) -> None:
        if self._active_manual_session is None:
            if not diagnostics:
                return
            raise RuntimeError(
                "Manual diagnostics assignment requires an active session"
            )
        self._active_manual_session.execution_diagnostics = diagnostics

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("mainWindow")
        self._configure_native_window_chrome()
        self.resize(1200, 760)
        self.setMinimumSize(760, 520)
        self._inspector_expansion_width = 0
        self._inspector_preopen_width: int | None = None
        self._inspector_preopen_minimum_width: int | None = None
        self._inspector_forced_minimum_width: int | None = None
        self._appearance_controller = application_appearance_controller(
            _active_application(),
        )
        self._appearance_listener_connected = False
        self._last_inspector_width = INSPECTOR_PREFERRED_WIDTH
        self._manual_inspector_minimum_width = INSPECTOR_BASE_MINIMUM_WIDTH

        self.workspace = WorkspaceSidebar()
        self.workspace.set_resolution_icon_color(
            tokens_for(self._appearance_controller.current_scheme).accent,
        )
        self._apply_workspace_state_colors()
        self.dataset_view = ReducedDatasetView()
        self._apply_dataset_view_theme()
        self.scientific_canvas = self.dataset_view.canvas
        self._active_project: ProjectState | None = None
        self._open_project: ProjectState | None = None
        self._open_dataset: DatasetState | None = None
        self._mask_draft: MaskTaskDraft | None = None
        self._mask_draft_owner: DatasetState | None = None
        self._workflow_projects: dict[ProjectState, WorkflowProject] = {}
        self._preserved_q_sources: dict[tuple[ProjectState, str], ReducedDataset] = {}
        self.q_rebin_dialog: QRebinDialog | None = None
        self._manual_sessions: dict[tuple[ProjectState, str], ManualFitSession] = {}
        self._active_manual_session: ManualFitSession | None = None
        self._sample_view_sessions: dict[
            tuple[ProjectState, str], SampleViewSession
        ] = {}
        self._pending_manual_interaction: PendingManualInteraction | None = None
        self._manual_preview: ManualModelPreview | None = None
        self._transfer_defaults = TransferDefaultsStore()
        self._fit_all_q_generation = 0
        self._active_fit_all_q_generation: int | None = None
        self._fit_all_q_worker: FitAllQWorker | None = None
        self._fit_all_q_thread: QThread | None = None
        self._fit_all_q_cancel_event: Event | None = None
        self._fit_all_q_anchor_group: int | None = None
        self._fit_all_q_session: ManualFitSession | None = None
        self._fit_all_q_run_context: FitAllQRunContext | None = None
        self._fit_all_q_live_statuses: list[FitDisplayStatus] | None = None
        self._fit_all_q_close_pending = False
        self._mask_boundary_preview_active = False
        self.workspace.set_scientific_replacement_resolver(
            self._resolve_scientific_dataset_replacement,
        )
        self.workspace.set_dataset_replacement_guard(self._approve_dataset_replacement)
        self.workspace.set_fitting_started_resolver(self._dataset_fitting_started)
        self.workspace.set_saved_results_resolver(self._workspace_saved_results)
        self.workspace.set_active_result_resolver(self._workspace_active_result)
        self.inspector = self._build_inspector()
        self._create_actions()
        self.central_workspace = self._build_central_workspace()
        self._apply_control_icons()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("workspaceSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        self.splitter.addWidget(self.workspace)
        self.splitter.addWidget(self.central_workspace)
        self.splitter.addWidget(self.inspector)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([230, 760, INSPECTOR_PREFERRED_WIDTH])
        self.splitter.splitterMoved.connect(self._remember_inspector_width)
        self.inspector.hide()

        self._create_menus()
        self.workspace.project_created.connect(self._set_active_project)
        self.workspace.project_selected.connect(self._set_active_project)
        self.workspace.project_renamed.connect(self._on_project_renamed)
        self.workspace.import_files_requested.connect(self._prompt_import_files)
        self.workspace.import_folder_requested.connect(self._prompt_import_folder)
        self.workspace.dataset_open_requested.connect(self._open_workspace_dataset)
        self.workspace.dataset_fresh_requested.connect(self._fresh_workspace_dataset)
        self.workspace.saved_result_open_requested.connect(self._open_saved_result)
        self.workspace.saved_result_delete_requested.connect(self._delete_saved_result)
        self.workspace.saved_result_rename_requested.connect(self._rename_saved_result)
        self.workspace.saved_result_best_requested.connect(self._best_saved_result)
        self.workspace.saved_result_revert_requested.connect(self._revert_saved_result)
        self.workspace.dataset_added.connect(self._on_dataset_added)
        self.workspace.dataset_role_changed.connect(self._on_dataset_role_changed)
        self.workspace.dataset_role_change_requested.connect(self.change_dataset_role)
        self.workspace.dataset_renamed.connect(self._on_dataset_renamed)
        self.workspace.dataset_updated.connect(self._on_dataset_updated)
        self.workspace.units_requested.connect(self.show_units_editor)
        self.workspace.q_assignment_requested.connect(self.show_q_editor)
        self.workspace.mask_edit_requested.connect(self.show_mask_editor)
        self.workspace.manual_fit_requested.connect(self.show_manual_fit)
        self.workspace.auto_fit_requested.connect(self.show_auto_fit)
        self.workspace.apply_resolution_requested.connect(
            self.apply_resolution_for_sample
        )
        self.workspace.q_method_dropped.connect(self._apply_q_method)
        self.workspace.project_removal_requested.connect(self.remove_project)
        self.workspace.dataset_removal_requested.connect(self.remove_dataset)
        self.workspace.q_method_removal_requested.connect(self.remove_q_method)
        self.dataset_view.group_changed.connect(self._refresh_inspector_context)
        self.dataset_view.group_changed.connect(self._sync_mask_controls)
        self.dataset_view.group_changed.connect(self._on_sample_group_changed)
        self.dataset_view.y_scale_changed.connect(self._on_y_scale_changed)
        self.dataset_view.heatmap_scale_changed.connect(self._on_heatmap_scale_changed)
        self.dataset_view.q_assignment_requested.connect(self.show_q_editor)
        self.dataset_view.units_requested.connect(self.show_units_editor)
        self.dataset_view.q_method_dropped.connect(self._apply_dragged_q_method)
        self.dataset_view.mask_edit_requested.connect(self.enter_mask_task)
        self.dataset_view.manual_fit_requested.connect(self.show_manual_fit)
        self.dataset_view.manual_component_completed.connect(
            self._complete_manual_component_interaction,
        )
        self.dataset_view.manual_component_preview_requested.connect(
            self._preview_manual_component_interaction,
        )
        self.dataset_view.manual_component_cancelled.connect(
            self._cancel_manual_component_interaction,
        )
        self.dataset_view.group_changed.connect(self._on_manual_group_changed)
        self.dataset_view.mask_rectangle_requested.connect(self._exclude_rectangle)
        self.dataset_view.mask_lasso_requested.connect(self._exclude_lasso)
        self.dataset_view.mask_boundary_previewed.connect(self._preview_auto_boundary)
        self.dataset_view.mask_boundary_requested.connect(self._edit_auto_boundary)
        self.dataset_view.mask_boundary_preview_cancelled.connect(
            self._cancel_mask_boundary_preview,
        )
        self.dataset_view.mask_operation_feedback_changed.connect(
            self._set_mask_operation_feedback,
        )
        self.dataset_view.q_editor.set_apply_handler(self._apply_q_bins)
        self.dataset_view.q_editor.save_q_method_requested.connect(self._save_q_method)
        self.dataset_view.q_editor.q_file_imported.connect(
            self._save_imported_q_method,
        )
        self.dataset_view.q_editor.draft_state_changed.connect(
            self._refresh_manual_result_actions,
        )
        self._appearance_controller.theme_changed.connect(self._on_theme_changed)
        self._appearance_listener_connected = True

        shell = QWidget()
        shell.setObjectName("applicationShell")
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.splitter)
        self.setCentralWidget(shell)
        self._apply_layout_defaults()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt virtual.
        """Cancel active work and close only after its QThread has stopped."""

        thread = self._fit_all_q_thread
        if thread is not None and thread.isRunning():
            self._fit_all_q_close_pending = True
            self._fit_all_q_generation += 1
            self._active_fit_all_q_generation = None
            self._fit_all_q_run_context = None
            self._fit_all_q_live_statuses = None
            self.setEnabled(False)
            if self._fit_all_q_cancel_event is not None:
                self._fit_all_q_cancel_event.set()
            event.ignore()
            return

        self.setEnabled(True)
        if not self._resolve_result_close(
            tuple(self._manual_sessions.values()), quitting=True
        ):
            event.ignore()
            return
        if self._appearance_listener_connected:
            self._appearance_controller.theme_changed.disconnect(self._on_theme_changed)
            self._appearance_listener_connected = False
        super().closeEvent(event)

    def set_scientific_figure_visible(self, visible: bool) -> None:
        """Reveal the white Matplotlib surface only for an actual scientific view."""
        self.scientific_canvas_container.setVisible(visible)

    def set_inspector_visible(self, visible: bool) -> None:
        """Reveal Inspector outward when possible, otherwise resize internally."""
        if visible == (not self.inspector.isHidden()):
            self._sync_inspector_controls(visible)
            return

        if visible:
            self._inspector_preopen_width = self.width()
            self._inspector_preopen_minimum_width = self.minimumWidth()
            canvas_width = self.scientific_canvas.width()
            desired_width = max(
                self.inspector.minimumWidth(),
                self._last_inspector_width,
            )
            self._inspector_expansion_width = self._expand_for_inspector()
            self.inspector.show()
            self._enforce_inspector_minimum_width()
            self.splitter.setSizes(
                [
                    max(self.workspace.width(), 200),
                    max(canvas_width, 320),
                    desired_width,
                ],
            )
        else:
            self._last_inspector_width = max(
                self.inspector.minimumWidth(),
                self.inspector.width(),
            )
            self.inspector.hide()
            restore_width = self._inspector_preopen_width
            restore_minimum = self._inspector_preopen_minimum_width or 760
            current_minimum = self.minimumWidth()
            if current_minimum != self._inspector_forced_minimum_width:
                restore_minimum = max(restore_minimum, current_minimum)
            self.setMinimumWidth(restore_minimum)
            self._inspector_expansion_width = 0
            if restore_width is not None:
                self.resize(max(restore_width, self.minimumWidth()), self.height())
            self._inspector_preopen_width = None
            self._inspector_preopen_minimum_width = None
            self._inspector_forced_minimum_width = None
        self._sync_inspector_controls(visible)

    def _enforce_inspector_minimum_width(self) -> None:
        """Grow the shell before its sidebar can overlap the scientific area."""

        required = (
            self.workspace.minimumWidth()
            + self.central_workspace.minimumWidth()
            + self.inspector.minimumWidth()
            + 2 * self.splitter.handleWidth()
        )
        self._inspector_forced_minimum_width = required
        self.setMinimumWidth(required)

    def _create_actions(self) -> None:
        self.new_project_action = QAction("New Project", self)
        self.new_project_action.setShortcut(QKeySequence.StandardKey.New)
        self.new_project_action.triggered.connect(self.workspace.new_project)

        self.import_data_action = QAction("Import Data…", self)
        self.import_data_action.setShortcut("Ctrl+I")
        self.import_data_action.setEnabled(False)
        self.import_data_action.triggered.connect(self._prompt_import_files)

        self.edit_mask_action = QAction("Edit Mask…", self)
        self.edit_mask_action.setEnabled(False)
        self.edit_mask_action.triggered.connect(
            lambda _checked=False: self.enter_mask_task(),
        )

        self.q_rebin_action = QAction("Q Rebin…", self)
        self.q_rebin_action.setEnabled(False)
        self.q_rebin_action.triggered.connect(
            lambda _checked=False: self.show_q_rebin(),
        )

        self.manual_fit_action = QAction("Fitting Parameters…", self)
        self.manual_fit_action.setEnabled(False)
        self.manual_fit_action.triggered.connect(
            lambda _checked=False: self.show_manual_fit(),
        )

        self.auto_fit_action = QAction("AutoFit…", self)
        self.auto_fit_action.setEnabled(False)
        self.auto_fit_action.triggered.connect(
            lambda _checked=False: self.show_auto_fit(),
        )

        self.toggle_inspector_action = QAction("Inspector", self)
        self.toggle_inspector_action.setCheckable(True)
        self.toggle_inspector_action.setShortcut("Ctrl+Shift+I")
        self.toggle_inspector_action.toggled.connect(self.set_inspector_visible)

        self.appearance_action_group = QActionGroup(self)
        self.appearance_action_group.setExclusive(True)
        self.system_appearance_action = self._appearance_action(
            "System",
            Appearance.SYSTEM,
            self._appearance_controller.appearance,
        )
        self.light_appearance_action = self._appearance_action(
            "Light",
            Appearance.LIGHT,
            self._appearance_controller.appearance,
        )
        self.dark_appearance_action = self._appearance_action(
            "Dark",
            Appearance.DARK,
            self._appearance_controller.appearance,
        )

        self.quit_action = QAction("Quit ezQENS", self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        self.quit_action.triggered.connect(QApplication.closeAllWindows)

        self.about_action = QAction("About ezQENS", self)
        self.about_action.setMenuRole(QAction.MenuRole.AboutRole)
        self.about_action.triggered.connect(self._show_about)

    def _appearance_action(
        self,
        text: str,
        appearance: Appearance,
        current_appearance: Appearance,
    ) -> QAction:
        action = QAction(text, self)
        action.setCheckable(True)
        action.setChecked(appearance is current_appearance)
        action.triggered.connect(
            lambda: self._appearance_controller.set_appearance(appearance),
        )
        self.appearance_action_group.addAction(action)
        return action

    def _create_menus(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        assert file_menu is not None
        file_menu.addAction(self.new_project_action)
        file_menu.addAction(self.import_data_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        analysis_menu = menu_bar.addMenu("&Analysis")
        assert analysis_menu is not None
        analysis_menu.addAction(self.q_rebin_action)
        analysis_menu.addSeparator()
        analysis_menu.addAction(self.edit_mask_action)
        analysis_menu.addAction(self.auto_fit_action)
        analysis_menu.addAction(self.manual_fit_action)

        view_menu = menu_bar.addMenu("&View")
        assert view_menu is not None
        view_menu.addAction(self.toggle_inspector_action)
        appearance_menu = view_menu.addMenu("Appearance")
        assert appearance_menu is not None
        appearance_menu.addAction(self.system_appearance_action)
        appearance_menu.addAction(self.light_appearance_action)
        appearance_menu.addAction(self.dark_appearance_action)

        help_menu = menu_bar.addMenu("&Help")
        assert help_menu is not None
        help_menu.addAction(self.about_action)

    def _build_central_workspace(self) -> QWidget:
        central_workspace = QWidget()
        central_workspace.setObjectName("centralScientificWorkspace")
        central_workspace.setMinimumWidth(
            DEFAULT_LAYOUT_TOKENS.scientific_workspace_min_width,
        )
        central_header = QFrame()
        central_header.setObjectName("centralHeader")
        central_header.setFixedHeight(38)

        self.project_context_label = QLabel()
        self.project_context_label.setObjectName("projectContextLabel")
        self.project_context_label.setVisible(False)

        self.manual_fit_button = QToolButton()
        self.manual_fit_button.setObjectName("centralManualFitButton")
        self.manual_fit_button.setText("Manual Fit")
        self.manual_fit_button.setEnabled(False)
        self.manual_fit_button.setToolTip(
            "Open Fitting Parameters for the current Sample"
        )
        self.manual_fit_button.clicked.connect(
            lambda _checked=False: self.show_manual_fit(),
        )

        self.auto_fit_button = QToolButton()
        self.auto_fit_button.setObjectName("centralAutoFitButton")
        self.auto_fit_button.setText("AutoFit…")
        self.auto_fit_button.setEnabled(False)
        self.auto_fit_button.setToolTip(
            "Evaluate Single-Q AutoFit candidates for the current Group",
        )
        self.auto_fit_button.clicked.connect(
            lambda _checked=False: self.show_auto_fit(),
        )

        self.inspector_button = QToolButton()
        self.inspector_button.setObjectName("inspectorButton")
        self.inspector_button.setCheckable(True)
        self.inspector_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.inspector_button.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.inspector_button.toggled.connect(self.set_inspector_visible)

        header_layout = QHBoxLayout(central_header)
        header_layout.setContentsMargins(12, 5, 8, 5)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.project_context_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.manual_fit_button)
        header_layout.addWidget(self.auto_fit_button)
        header_layout.addWidget(self.inspector_button)
        self._central_header_layout = header_layout

        canvas_container = QWidget()
        canvas_container.setObjectName("scientificCanvasContainer")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(12, 12, 12, 12)
        canvas_layout.setSpacing(0)
        canvas_surface = QWidget()
        canvas_surface.setObjectName("scientificCanvasSurface")
        surface_layout = QVBoxLayout(canvas_surface)
        surface_layout.setContentsMargins(0, 0, 0, 0)
        surface_layout.setSpacing(0)
        surface_layout.addWidget(self.dataset_view)
        self.mask_task_bar = self._build_mask_task_bar()
        self.dataset_view.content_layout.insertWidget(3, self.mask_task_bar)
        canvas_layout.addWidget(canvas_surface, 1)

        layout = QVBoxLayout(central_workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(central_header, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(canvas_container, 1)

        self.central_header = central_header
        self.scientific_canvas_container = canvas_container
        self.scientific_canvas_container.hide()
        self._sync_inspector_controls(False)
        return central_workspace

    def _build_mask_task_bar(self) -> QFrame:
        """Create the compact local controls for the Mask editing task."""

        bar = QFrame()
        bar.setObjectName("maskTaskBar")
        title = QLabel("Mask editing")
        title.setObjectName("maskTaskTitle")
        self.mask_boundary_button = QToolButton()
        self.mask_boundary_button.setText("Boundary")
        self.mask_boundary_button.setToolTip("Drag a visible Mask boundary")
        self.mask_boundary_all_button = QToolButton()
        self.mask_boundary_all_button.setText("Apply to All Groups…")
        self.mask_boundary_all_button.setToolTip(
            "Apply the current energy boundary to every Group",
        )
        self.mask_boundary_all_button.clicked.connect(
            lambda _checked=False: self._apply_mask_boundary_to_all_groups(),
        )
        self.mask_rectangle_button = QToolButton()
        self.mask_rectangle_button.setText("Rectangle")
        self.mask_rectangle_button.setToolTip(
            "Exclude or restore points in a rectangle"
        )
        self.mask_lasso_button = QToolButton()
        self.mask_lasso_button.setText("Lasso")
        self.mask_lasso_button.setToolTip("Exclude or restore points in a lasso")
        for button, tool in (
            (self.mask_boundary_button, "boundary"),
            (self.mask_rectangle_button, "rectangle"),
            (self.mask_lasso_button, "lasso"),
        ):
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, value=tool: self._set_mask_tool(value, checked),
            )
        self.mask_exclude_button = QToolButton()
        self.mask_exclude_button.setText("Exclude")
        self.mask_exclude_button.setToolTip("Exclude selected points")
        self.mask_exclude_button.setCheckable(True)
        self.mask_exclude_button.clicked.connect(
            lambda checked: self._set_mask_operation("exclude", checked),
        )
        self.mask_restore_button = QToolButton()
        self.mask_restore_button.setText("Restore")
        self.mask_restore_button.setToolTip("Restore selected points")
        self.mask_restore_button.setCheckable(True)
        self.mask_restore_button.clicked.connect(
            lambda checked: self._set_mask_operation("restore", checked),
        )
        self.mask_undo_button = QToolButton()
        self.mask_undo_button.setText("Undo")
        self.mask_undo_button.setToolTip("Undo the last Mask edit")
        self.mask_undo_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.mask_undo_button.clicked.connect(self._undo_mask)
        self.mask_redo_button = QToolButton()
        self.mask_redo_button.setText("Redo")
        self.mask_redo_button.setToolTip("Redo the last Mask edit")
        self.mask_redo_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.mask_redo_button.clicked.connect(self._redo_mask)
        self.mask_disable_auto_button = QToolButton()
        self.mask_disable_auto_button.setText("Disable AutoMask")
        self.mask_disable_auto_button.setToolTip(
            "Restore all valid AutoMask points in this draft",
        )
        self.mask_disable_auto_button.clicked.connect(self._disable_auto_mask)
        self.mask_reset_group_button = QToolButton()
        self.mask_reset_group_button.setText("Reset Group")
        self.mask_reset_group_button.clicked.connect(self._reset_mask_group)
        self.mask_reset_all_button = QToolButton()
        self.mask_reset_all_button.setText("Reset All")
        self.mask_reset_all_button.clicked.connect(
            lambda _checked=False: self._reset_mask_all(),
        )
        self.mask_rerun_button = QToolButton()
        self.mask_rerun_button.setText("Re-run AutoMask…")
        self.mask_rerun_button.clicked.connect(
            lambda _checked=False: self.rerun_auto_mask(),
        )
        # Keep task actions on one native control family so their text baselines
        # and compact heights remain consistent across themes.
        self.mask_save_button = QToolButton()
        self.mask_save_button.setText("Save")
        self.mask_save_button.setToolTip("Save Mask changes")
        self.mask_save_button.clicked.connect(self.save_mask_task)
        self.mask_close_button = QToolButton()
        self.mask_close_button.setText("Close")
        self.mask_close_button.setToolTip("Close Mask editing")
        self.mask_close_button.clicked.connect(lambda: self.close_mask_task())

        self.mask_tool_row = QWidget(bar)
        self.mask_tool_row.setObjectName("maskToolRow")
        tool_layout = QHBoxLayout(self.mask_tool_row)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.setSpacing(4)
        self._mask_tool_layout = tool_layout
        tool_layout.addWidget(title)
        tool_layout.addStretch(1)
        tool_layout.addWidget(self.mask_boundary_button)
        tool_layout.addWidget(self.mask_boundary_all_button)
        tool_layout.addWidget(self.mask_rectangle_button)
        tool_layout.addWidget(self.mask_lasso_button)
        tool_layout.addSpacing(6)
        tool_layout.addWidget(self.mask_exclude_button)
        tool_layout.addWidget(self.mask_restore_button)

        self.mask_action_row = QWidget(bar)
        self.mask_action_row.setObjectName("maskActionRow")
        action_layout = QHBoxLayout(self.mask_action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        self._mask_action_layout = action_layout
        action_layout.addStretch(1)
        action_layout.addWidget(self.mask_undo_button)
        action_layout.addWidget(self.mask_redo_button)
        action_layout.addWidget(self.mask_disable_auto_button)
        action_layout.addWidget(self.mask_reset_group_button)
        action_layout.addWidget(self.mask_reset_all_button)
        action_layout.addWidget(self.mask_rerun_button)
        action_layout.addWidget(self.mask_save_button)
        action_layout.addWidget(self.mask_close_button)

        layout = QVBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(4)
        self._mask_task_layout = layout
        layout.addWidget(self.mask_tool_row)
        layout.addWidget(self.mask_action_row)
        bar.hide()
        return bar

    def _build_inspector(self) -> QWidget:
        inspector = QWidget()
        inspector.setObjectName("inspectorPanel")
        inspector.setMinimumWidth(INSPECTOR_BASE_MINIMUM_WIDTH)

        self.inspector_context_label = QLabel("Open a dataset to view its details.")
        self.inspector_context_label.setObjectName("inspectorContextLabel")
        self.inspector_context_label.setProperty("secondary", True)
        self.inspector_context_label.setWordWrap(True)
        self.inspector_dataset_toggle = QToolButton()
        self.inspector_dataset_toggle.setObjectName("inspectorDatasetToggle")
        self.inspector_dataset_toggle.setText("Dataset")
        self.inspector_dataset_toggle.setCheckable(True)
        self.inspector_dataset_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.inspector_dataset_toggle.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft,
        )
        self.inspector_dataset_toggle.toggled.connect(
            self._set_inspector_dataset_expanded,
        )
        self.inspector_dataset_toggle.hide()
        self.inspector_source_title = QLabel("Source")
        self.inspector_source_title.setObjectName("inspectorSourceTitle")
        self.inspector_source_title.hide()
        self.inspector_source_metadata_label = QLabel()
        self.inspector_source_metadata_label.setObjectName("inspectorSourceMetadata")
        self.inspector_source_metadata_label.setProperty("secondary", True)
        self.inspector_source_metadata_label.setWordWrap(True)
        self.inspector_source_metadata_label.hide()
        self.inspector_source_toggle = QToolButton()
        self.inspector_source_toggle.setObjectName("inspectorSourceToggle")
        self.inspector_source_toggle.setText("Source")
        self.inspector_source_toggle.setCheckable(True)
        self.inspector_source_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
        self.inspector_source_toggle.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft,
        )
        self.inspector_source_toggle.toggled.connect(
            self._set_inspector_source_expanded,
        )
        self.inspector_source_toggle.hide()
        self._inspector_source_details_available = False
        self.inspector_resolution_title = QLabel("Resolution")
        self.inspector_resolution_title.setObjectName("inspectorResolutionTitle")
        self.inspector_resolution_title.hide()
        self.inspector_resolution_label = QLabel()
        self.inspector_resolution_label.setObjectName("inspectorResolution")
        self.inspector_resolution_label.setProperty("secondary", True)
        self.inspector_resolution_label.setWordWrap(True)
        self.inspector_resolution_label.hide()
        self.inspector_resolution_button = QToolButton()
        self.inspector_resolution_button.setObjectName("inspectorResolutionButton")
        self.inspector_resolution_button.clicked.connect(
            lambda _checked=False: self.apply_resolution_for_sample(),
        )
        self.inspector_resolution_button.hide()
        self.manual_fit_editor = ManualFitEditor()
        self.manual_fit_editor.hide()
        self.manual_fit_editor.component_interaction_requested.connect(
            self._begin_manual_component_interaction,
        )
        self.manual_fit_editor.parameter_edit_requested.connect(
            self._update_manual_parameter,
        )
        self.manual_fit_editor.parameter_focused.connect(
            self._emphasize_manual_parameter,
        )
        self.manual_fit_editor.chain_requested.connect(self._handle_manual_chain)
        self.manual_fit_editor.new_tie_group_requested.connect(
            self._begin_new_manual_tie_group,
        )
        self.manual_fit_editor.join_tie_requested.connect(self._join_manual_tie)
        self.manual_fit_editor.component_removal_requested.connect(
            self._remove_manual_component,
        )
        self.manual_fit_editor.apply_resolution_requested.connect(
            self.apply_resolution_for_sample,
        )
        self.manual_fit_editor.run_requested.connect(self._run_manual_fit)
        self.manual_fit_editor.fit_all_q_requested.connect(self._show_fit_all_q)
        self.manual_fit_editor.cancel_fit_all_q_requested.connect(
            self._cancel_fit_all_q
        )
        self.manual_fit_editor.save_result_requested.connect(self._save_manual_result)
        self.manual_fit_editor.save_result_as_requested.connect(
            self._save_manual_result_as
        )
        self.manual_fit_editor.clear_model_requested.connect(
            self.clear_manual_model,
        )
        self.manual_fit_editor.expanded_changed.connect(
            self._on_manual_fit_expanded_changed,
        )

        self.inspector_scroll_area = QScrollArea(inspector)
        self.inspector_scroll_area.setObjectName("inspectorScrollArea")
        self.inspector_scroll_area.setWidgetResizable(True)
        self.inspector_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.inspector_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.inspector_scroll_area.viewport().setObjectName(
            "inspectorScrollViewport",
        )
        self.inspector_scroll_body = QWidget()
        self.inspector_scroll_body.setObjectName("inspectorScrollBody")
        self.inspector_scroll_body.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )
        layout = QVBoxLayout(self.inspector_scroll_body)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(5)
        layout.addWidget(self.manual_fit_editor)
        layout.addWidget(self.inspector_resolution_title)
        layout.addWidget(self.inspector_resolution_label)
        layout.addWidget(
            self.inspector_resolution_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(
            self.inspector_dataset_toggle,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(self.inspector_context_label)
        layout.addWidget(
            self.inspector_source_toggle,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(self.inspector_source_title)
        layout.addWidget(self.inspector_source_metadata_label)
        layout.addStretch(1)
        self.inspector_scroll_area.setWidget(self.inspector_scroll_body)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(0)
        inspector_layout.addWidget(self.inspector_scroll_area)
        self._inspector_layout = layout
        return inspector

    def _expand_for_inspector(self) -> int:
        screen = self.screen()
        if screen is None:
            return 0
        expansion_width = inspector_outward_expansion_width(
            self.geometry(),
            screen.availableGeometry(),
            INSPECTOR_PREFERRED_WIDTH,
            is_maximized=self.isMaximized(),
        )
        if expansion_width:
            self.resize(self.width() + expansion_width, self.height())
        return expansion_width

    def _remember_inspector_width(self, _position: int, _index: int) -> None:
        """Retain the user's splitter width across ordinary visibility changes."""

        if not self.inspector.isHidden():
            self._last_inspector_width = max(
                self.inspector.minimumWidth(),
                self.inspector.width(),
            )

    def _set_manual_inspector_mode(self, active: bool) -> None:
        """Prioritize the active Manual task while retaining collapsible context."""

        self.inspector_dataset_toggle.setVisible(active)
        self.inspector_source_toggle.setVisible(active)
        if active:
            self.inspector_dataset_toggle.setChecked(False)
            self.inspector_source_toggle.setChecked(False)
            self.inspector_context_label.hide()
            self.inspector_source_title.hide()
            self.inspector_source_metadata_label.hide()
            self.inspector_source_toggle.setEnabled(
                self._inspector_source_details_available,
            )
            return
        self.inspector_context_label.show()
        self.inspector_source_title.setVisible(
            self._inspector_source_details_available,
        )
        self.inspector_source_metadata_label.setVisible(
            self._inspector_source_details_available,
        )

    def _set_inspector_dataset_expanded(self, expanded: bool) -> None:
        active = self._manual_draft is not None
        apply_disclosure_icon(
            self.inspector_dataset_toggle,
            expanded=expanded,
            color=tokens_for(
                self._appearance_controller.current_scheme,
            ).text_secondary,
        )
        self.inspector_context_label.setVisible(active and expanded)

    def _set_inspector_source_expanded(self, expanded: bool) -> None:
        active = self._manual_draft is not None
        apply_disclosure_icon(
            self.inspector_source_toggle,
            expanded=expanded,
            color=tokens_for(
                self._appearance_controller.current_scheme,
            ).text_secondary,
        )
        self.inspector_source_metadata_label.setVisible(
            active and expanded and self._inspector_source_details_available,
        )

    def _sync_inspector_context_visibility(self) -> None:
        """Respect Manual-task disclosure without changing stored context text."""

        active = self._manual_draft is not None
        self.inspector_dataset_toggle.setVisible(active)
        self.inspector_source_toggle.setVisible(active)
        self.inspector_source_toggle.setEnabled(
            self._inspector_source_details_available,
        )
        if active:
            self.inspector_context_label.setVisible(
                self.inspector_dataset_toggle.isChecked(),
            )
            self.inspector_source_title.hide()
            self.inspector_source_metadata_label.setVisible(
                self.inspector_source_toggle.isChecked()
                and self._inspector_source_details_available,
            )
            return
        self.inspector_context_label.show()
        self.inspector_source_title.setVisible(
            self._inspector_source_details_available,
        )
        self.inspector_source_metadata_label.setVisible(
            self._inspector_source_details_available,
        )

    def _sync_inspector_controls(self, visible: bool) -> None:
        action_blocker = QSignalBlocker(self.toggle_inspector_action)
        self.toggle_inspector_action.setChecked(visible)
        del action_blocker
        button_blocker = QSignalBlocker(self.inspector_button)
        self.inspector_button.setChecked(visible)
        icon_name = IconName.INSPECTOR_HIDE if visible else IconName.INSPECTOR_SHOW
        icon_color = tokens_for(
            self._appearance_controller.current_scheme,
        ).text_secondary
        self.inspector_button.setIcon(load_icon(icon_name, icon_color))
        self.inspector_button.setText("" if visible else "More info")
        self.inspector_button.setFixedWidth(
            DEFAULT_LAYOUT_TOKENS.control_height
            if visible
            else self.inspector_button.sizeHint().width()
        )
        tooltip = "Hide more info" if visible else "Show more info"
        self.inspector_button.setToolTip(tooltip)
        self.inspector_button.setAccessibleName(tooltip)
        del button_blocker

    def _set_active_project(self, project: ProjectState) -> None:
        self._active_project = project
        self.import_data_action.setEnabled(True)

    def _workflow_project_for(self, project: ProjectState) -> WorkflowProject:
        """Synchronize small GUI identity state into the immutable workflow Project."""

        workflow = self._workflow_projects.get(project)
        if workflow is None:
            workflow = create_project(project.name)
        states = {state.workflow_dataset_id: state for state in project.datasets}
        retained_workflow_ids = {
            association.resolution_id
            for association in workflow.resolution_associations
        }
        for reference in tuple(workflow.datasets):
            state = states.pop(reference.dataset_id, None)
            if state is None:
                if reference.dataset_id in retained_workflow_ids:
                    continue
                workflow = remove_project_dataset(workflow, reference)
            elif reference.dataset is not state.dataset:
                workflow, _ = replace_project_dataset(
                    workflow,
                    reference,
                    state.dataset,
                )
        added_states = tuple(states.values())
        for state in added_states:
            workflow, _ = add_project_dataset(
                workflow,
                state.dataset,
                dataset_id=state.workflow_dataset_id,
            )
        # A selection enters workflow state only with a newly added dataset or an
        # explicit mask-baseline update. Dataset replacement owns preservation or
        # invalidation; recommitting GUI-rebound state here would resurrect a
        # selection intentionally invalidated by unit/scientific replacement.
        for state in added_states:
            reference = self._workflow_dataset(workflow, state)
            selection = None if state.auto_mask is None else state.auto_mask.selection
            if (
                reference.dataset.role is SpectrumRole.SAMPLE
                and selection is not None
                and selection.dataset is reference.dataset
            ):
                workflow = commit_fitting_selection(workflow, reference, selection)
        self._workflow_projects[project] = workflow
        return workflow

    def _manual_context_uses_dataset(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> bool:
        """Return whether a dataset participates in the open Manual context."""

        if self._manual_owner is None or self._manual_owner[0] is not project:
            return False
        sample_id = self._manual_owner[1]
        if dataset.workflow_dataset_id == sample_id:
            return True
        workflow = self._workflow_projects.get(project)
        return workflow is not None and any(
            item.sample_id == sample_id
            and item.resolution_id == dataset.workflow_dataset_id
            for item in workflow.resolution_associations
        )

    def _manual_sessions_using_dataset(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> tuple[ManualFitSession, ...]:
        """Return retained Sample sessions scientifically using one dataset."""

        workflow = self._workflow_projects.get(project)
        associated_sample_ids = (
            set()
            if workflow is None
            else {
                item.sample_id
                for item in workflow.resolution_associations
                if item.resolution_id == dataset.workflow_dataset_id
            }
        )
        associated_sample_ids.add(dataset.workflow_dataset_id)
        return tuple(
            session
            for owner, session in self._manual_sessions.items()
            if owner[0] is project and owner[1] in associated_sample_ids
        )

    def _dataset_fitting_started(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> bool:
        """Return whether any Group has a model in the Sample's retained draft."""

        session = self._manual_sessions.get(
            (project, dataset.workflow_dataset_id),
        )
        return (
            session is not None
            and session.draft_available
            and any(setup.model is not None for setup in session.draft.setups)
        )

    def _store_open_sample_view_state(self) -> None:
        """Retain current display-only state under stable Sample identity."""

        if (
            self._open_project is None
            or self._open_dataset is None
            or self._open_dataset.dataset.role is not SpectrumRole.SAMPLE
        ):
            return
        owner = (
            self._open_project,
            self._open_dataset.workflow_dataset_id,
        )
        session = self._sample_view_sessions.setdefault(owner, SampleViewSession())
        session.group_index = self.dataset_view.current_group_index
        session.y_scale = self.dataset_view.y_scale
        session.heatmap_scale = self.dataset_view.heatmap_scale

    def _on_sample_group_changed(self, group_index: int) -> None:
        """Write through Group navigation independently of fitting state."""

        session = self._open_sample_view_session()
        if session is not None:
            session.group_index = group_index

    def _on_y_scale_changed(self, scale: str) -> None:
        """Write through one display-only scale change to the open Sample session."""

        session = self._open_sample_view_session()
        if session is not None:
            session.y_scale = scale

    def _on_heatmap_scale_changed(self, scale: str) -> None:
        """Write through one overview color-scale change to its Sample session."""

        session = self._open_sample_view_session()
        if session is not None:
            session.heatmap_scale = scale

    def _open_sample_view_session(self) -> SampleViewSession | None:
        if (
            self._open_project is None
            or self._open_dataset is None
            or self._open_dataset.dataset.role is not SpectrumRole.SAMPLE
        ):
            return None
        owner = (self._open_project, self._open_dataset.workflow_dataset_id)
        return self._sample_view_sessions.setdefault(owner, SampleViewSession())

    def _commit_explicit_fitting_selection(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> None:
        """Commit only a newly produced AutoMask/Mask baseline."""

        workflow = self._workflow_project_for(project)
        reference = self._workflow_dataset(workflow, dataset)
        selection = None if dataset.auto_mask is None else dataset.auto_mask.selection
        if (
            reference.dataset.role is SpectrumRole.SAMPLE
            and selection is not None
            and selection.dataset is reference.dataset
        ):
            self._workflow_projects[project] = commit_fitting_selection(
                workflow,
                reference,
                selection,
            )

    @staticmethod
    def _workflow_dataset(
        workflow: WorkflowProject,
        state: DatasetState,
    ) -> ProjectDataset:
        """Resolve a stable GUI dataset identity through the public workflow Project."""

        return next(
            item
            for item in workflow.datasets
            if item.dataset_id == state.workflow_dataset_id
        )

    def import_data(
        self,
        path: str | Path,
        *,
        project: ProjectState | None = None,
    ) -> DatasetState:
        """Import ordinary Data through the core without opening it."""

        target_project = project or self.workspace.current_project
        if target_project is None:
            raise RuntimeError("create or select a Project before importing data")
        source_path = Path(path)
        dataset = self._import_reduced_dataset(source_path)
        return self.workspace.add_dataset(
            target_project,
            dataset,
            source_path=source_path,
        )

    def import_paths(
        self,
        paths: tuple[str | Path, ...],
        *,
        project: ProjectState | None = None,
    ) -> ImportBatchResult:
        """Import every selected source, retaining successful files on failures."""

        target_project = project or self.workspace.current_project
        if target_project is None:
            raise RuntimeError("select a Project before importing data")
        imported: list[DatasetState] = []
        q_methods: list[QMethodState] = []
        failures: list[ImportBatchFailure] = []
        for source in paths:
            path = Path(source)
            try:
                imported_dataset, imported_method = self._route_import_path(
                    path,
                    target_project,
                )
                if imported_dataset is not None:
                    imported.append(imported_dataset)
                if imported_method is not None:
                    q_methods.append(imported_method)
            except ImportValidationError as error:
                failures.append(
                    ImportBatchFailure(
                        path,
                        _format_import_diagnostics(error.diagnostics),
                    ),
                )
            except OSError as error:
                failures.append(ImportBatchFailure(path, str(error)))
        return ImportBatchResult(tuple(imported), tuple(failures), tuple(q_methods))

    def _route_import_path(
        self,
        path: Path,
        project: ProjectState,
    ) -> tuple[DatasetState | None, QMethodState | None]:
        """Route a file only after supported parsers establish its content type."""

        reduced_dataset = None
        reduced_error: ImportValidationError | None = None
        try:
            reduced_dataset = self._import_reduced_dataset(path)
        except ImportValidationError as error:
            reduced_error = error
        dave_result = None
        dave_error: ImportValidationError | None = None
        try:
            dave_result = parse_dave_q_bins(path)
        except ImportValidationError as error:
            dave_error = error
        if reduced_dataset is not None and dave_result is not None:
            raise ImportValidationError(
                (
                    ImportDiagnostic(
                        code="gui_import_ambiguous_type",
                        severity=DiagnosticSeverity.ERROR,
                        message=(
                            "File is valid as both reduced data and DAVE Q-bin "
                            "parameters; choose an explicit import path."
                        ),
                    ),
                ),
            )
        if dave_result is not None:
            method = self.workspace.save_q_method(
                project,
                dave_result.q_bins,
                name=path.stem or None,
            )
            return None, method
        if reduced_dataset is not None:
            return (
                self.workspace.add_dataset(
                    project,
                    reduced_dataset,
                    source_path=path,
                ),
                None,
            )
        if reduced_error is not None:
            raise reduced_error
        if dave_error is not None:
            raise dave_error
        raise OSError("no supported importer accepted the selected file")

    @staticmethod
    def _import_reduced_dataset(path: Path) -> ReducedDataset:
        """Use one metadata default policy for direct and batch GUI imports."""

        return import_reduced_data(
            path,
            role=SpectrumRole.SAMPLE,
            intensity_unit="arb. unit",
            uncertainty_unit="arb. unit",
        )

    def open_dataset(
        self,
        project: ProjectState,
        dataset: DatasetState,
        *,
        mask_task_decision: str | None = None,
    ) -> bool:
        """Open an imported Workspace dataset, or preserve an exact open match."""

        if self._fit_all_q_mutation_blocked() and (
            project is not self._open_project or dataset is not self._open_dataset
        ):
            return False
        self.workspace.validate_dataset_membership(project, dataset)
        if project is self._open_project and dataset is self._open_dataset:
            return True
        if (
            project is not self._open_project or dataset is not self._open_dataset
        ) and not self._resolve_mask_task_transition(mask_task_decision):
            return False
        self._store_open_sample_view_state()
        self._deactivate_manual_fit()
        owner = (project, dataset.workflow_dataset_id)
        view_session = (
            self._sample_view_sessions.setdefault(owner, SampleViewSession())
            if dataset.dataset.role is SpectrumRole.SAMPLE
            else None
        )
        self._open_project = project
        self._open_dataset = dataset
        self.dataset_view.open_dataset(
            dataset.dataset,
            overview_source=self._preserved_q_sources.get(owner),
            group_index=0 if view_session is None else view_session.group_index,
            y_scale="linear" if view_session is None else view_session.y_scale,
            heatmap_scale=(
                "linear" if view_session is None else view_session.heatmap_scale
            ),
        )
        self.dataset_view.set_selection(
            None if dataset.auto_mask is None else dataset.auto_mask.selection,
        )
        self.dataset_view.set_mask_edit_available(dataset.mask_editable)
        self.workspace.set_active_dataset(project, dataset)
        self.set_scientific_figure_visible(True)
        self.edit_mask_action.setEnabled(
            dataset.mask_editable and not self._fit_all_q_mutation_blocked()
        )
        self._sync_manual_fit_entry()
        self._refresh_context_label()
        self._refresh_inspector_context()
        session = self._manual_sessions.get(owner)
        if session is not None and session.draft_available:
            session.group_index = self.dataset_view.current_group_index
            self._activate_manual_session(session)
        else:
            self._refresh_result_status_projection()
        return True

    def _open_workspace_dataset(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> None:
        """Restore the last active item, then Best, save order, or raw data."""

        if self.open_dataset(project, dataset):
            session = self._manual_sessions.get((project, dataset.workflow_dataset_id))
            if session is not None and session.draft_available:
                self._default_result_view(session)
                self._refresh_manual_fit()

    def show_q_rebin(
        self,
        project: ProjectState | None = None,
        dataset: DatasetState | None = None,
    ) -> QRebinDialog | None:
        """Open one modal, preview-first Q-rebin task for the selected Sample."""

        if self._fit_all_q_mutation_blocked():
            return None

        if self.q_rebin_dialog is not None:
            self.q_rebin_dialog.raise_()
            self.q_rebin_dialog.activateWindow()
            return self.q_rebin_dialog
        target_project, target_dataset = self._editing_target(project, dataset)
        if (
            target_project is None
            or target_dataset is None
            or target_dataset.dataset.role is not SpectrumRole.SAMPLE
        ):
            return None
        if (
            target_project is not self._open_project
            or target_dataset is not self._open_dataset
        ) and not self.open_dataset(target_project, target_dataset):
            return None
        workflow = self._workflow_project_for(target_project)
        sample = self._workflow_dataset(workflow, target_dataset)
        resolution = applied_resolution(workflow, sample)
        preserved_sample = self._preserved_q_sources.get(
            (target_project, sample.dataset_id),
            sample.dataset,
        )
        preserved_resolution = (
            None
            if resolution is None
            else self._preserved_q_sources.get(
                (target_project, resolution.dataset_id),
                resolution.dataset,
            )
        )
        dialog = QRebinDialog(
            workflow,
            sample,
            sample_name=target_dataset.name,
            preserved_sample=preserved_sample,
            preserved_resolution=preserved_resolution,
            resolution_name=(
                None
                if resolution is None
                else self._dataset_name_for_workflow_id(
                    target_project,
                    resolution.dataset_id,
                )
            ),
            apply_handler=lambda preview: self._apply_q_rebin_preview(
                target_project,
                target_dataset,
                preview,
            ),
            parent=self,
        )
        self.q_rebin_dialog = dialog
        dialog.finished.connect(lambda _result: self._q_rebin_finished(dialog))
        dialog.open()
        return dialog

    def _q_rebin_finished(self, dialog: QRebinDialog) -> None:
        if self.q_rebin_dialog is dialog:
            self.q_rebin_dialog = None

    def _apply_q_rebin_preview(
        self,
        project: ProjectState,
        sample_state: DatasetState,
        preview: QRebinPreview,
    ) -> bool:
        """Adopt one already-validated workflow transaction without replaying it."""

        if self._fit_all_q_mutation_blocked():
            return False

        if (
            project is self._open_project
            and sample_state is self._open_dataset
            and not self._resolve_mask_task_transition()
        ):
            return False
        if project is self._open_project and sample_state is not self._open_dataset:
            return False
        if project is self._open_project and self._open_dataset is not None:
            sample_state = self._open_dataset
        if sample_state not in project.datasets:
            return False
        if self._workflow_projects.get(project) is None:
            return False
        visible_resolution = (
            None
            if preview.resolution is None
            else next(
                (
                    state
                    for state in project.datasets
                    if state.workflow_dataset_id == preview.resolution.dataset_id
                ),
                None,
            )
        )

        def publish_workflow() -> None:
            self._workflow_projects[project] = preview.workflow
            self._preserved_q_sources.setdefault(
                (project, preview.sample.dataset_id), preview.sample_source
            )
            if preview.resolution is not None and preview.resolution_source is not None:
                self._preserved_q_sources.setdefault(
                    (project, preview.resolution.dataset_id), preview.resolution_source
                )

        transaction = self.workspace.apply_q_rebin_transaction(
            project,
            sample_state,
            preview.sample.dataset,
            resolution=visible_resolution,
            rebinned_resolution=(
                None
                if visible_resolution is None or preview.resolution is None
                else preview.resolution.dataset
            ),
            before_publish=publish_workflow,
        )
        if transaction is None:
            return False
        current_sample, _current_resolution = transaction
        view_session = self._sample_view_sessions.setdefault(
            (project, current_sample.workflow_dataset_id),
            SampleViewSession(),
        )
        view_session.group_index = (
            self.dataset_view.current_group_index
            if project is self._open_project and current_sample is self._open_dataset
            else 0
        )
        self._commit_explicit_fitting_selection(project, current_sample)
        self._sync_manual_fit_entry()
        return True

    def show_manual_fit(
        self,
        project: ProjectState | None = None,
        dataset: DatasetState | None = None,
    ) -> None:
        """Open group-local Fitting Parameters without requiring a Resolution."""

        if self._fit_all_q_mutation_blocked():
            return
        target_project, target_dataset = self._editing_target(project, dataset)
        if target_project is None or target_dataset is None:
            return
        if (
            target_project is not self._open_project
            or target_dataset is not self._open_dataset
        ) and not self.open_dataset(target_project, target_dataset):
            return
        owner = (target_project, target_dataset.workflow_dataset_id)
        workflow = self._workflow_project_for(target_project)
        session = self._manual_sessions.get(owner)
        if session is None or not session.draft_available:
            try:
                draft = open_manual_fit_draft(
                    workflow,
                    self._workflow_dataset(workflow, target_dataset),
                )
            except WorkflowError as error:
                self._sync_manual_fit_entry()
                self._show_workflow_error("Fitting Parameters", error)
                return
            if session is None:
                session = ManualFitSession(
                    owner,
                    draft,
                    self.dataset_view.current_group_index,
                )
                self._manual_sessions[owner] = session
            else:
                session.resume(draft)
                session.group_index = self.dataset_view.current_group_index
                session.execution_by_group.clear()
                session.draft_available = True
            self.manual_fit_editor.reset_result_disclosure()
            self._manual_inspector_minimum_width = INSPECTOR_BASE_MINIMUM_WIDTH
        self._activate_manual_session(session, expand=True)

    def show_auto_fit(
        self,
        project: ProjectState | None = None,
        dataset: DatasetState | None = None,
    ) -> None:
        """Evaluate and explicitly adopt one candidate for the current Group."""

        if self._fit_all_q_mutation_blocked():
            return None
        target_project, target_dataset = self._editing_target(project, dataset)
        if target_project is None or target_dataset is None:
            return
        if (
            target_project is not self._open_project
            or target_dataset is not self._open_dataset
        ) and not self.open_dataset(target_project, target_dataset):
            return
        owner = (target_project, target_dataset.workflow_dataset_id)
        workflow = self._workflow_project_for(target_project)
        session = self._manual_sessions.get(owner)
        if session is not None and not session.draft_available:
            try:
                session.resume(
                    open_manual_fit_draft(
                        workflow,
                        self._workflow_dataset(workflow, target_dataset),
                    )
                )
            except WorkflowError as error:
                self._show_workflow_error("AutoFit", error)
                return
            session.group_index = self.dataset_view.current_group_index
            session.execution_by_group.clear()
            session.draft_available = True
        try:
            draft = (
                session.draft
                if session is not None
                else open_manual_fit_draft(
                    workflow,
                    self._workflow_dataset(workflow, target_dataset),
                )
            )
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                outcome = run_single_q_auto_fit(
                    workflow,
                    draft,
                    self.dataset_view.current_group_index,
                )
            finally:
                QApplication.restoreOverrideCursor()
        except WorkflowError as error:
            self._show_workflow_error("AutoFit", error)
            return
        except (FittingError, ValueError) as error:
            show_message_dialog(self, "AutoFit", str(error))
            return

        dialog = self._create_auto_fit_dialog(
            target_project, target_dataset, workflow, draft, outcome
        )
        dialog.exec()

    def _create_auto_fit_dialog(
        self,
        project: ProjectState,
        dataset: DatasetState,
        workflow: WorkflowProject,
        draft: ManualFitDraft,
        outcome: SingleQAutoFitOutcome,
    ) -> AutoFitCandidateDialog:
        owner = (project, dataset.workflow_dataset_id)
        session = self._manual_sessions.get(owner)
        workspace = (
            FittingWorkspaceState() if session is None else session.result_workspace
        )
        dialog = AutoFitCandidateDialog(
            outcome,
            self,
            project=workflow,
            draft=draft,
            has_active_saved_result=workspace.associated_saved_result is not None,
        )
        dialog.finalization_requested.connect(
            lambda action, selections: self._finalize_auto_fit(
                dialog, owner, draft, action, selections
            )
        )
        return dialog

    def _finalize_auto_fit(
        self,
        dialog: AutoFitCandidateDialog,
        owner: tuple[ProjectState, str],
        original_draft: ManualFitDraft,
        action: str,
        selections: tuple[CandidateFitResult | AutoFitCandidateBranchResult, ...],
    ) -> None:
        """Validate all selections before atomically changing application state."""

        if self._fit_all_q_mutation_blocked() or dialog.is_running or not selections:
            return
        session = self._manual_sessions.get(owner)
        workspace = (
            FittingWorkspaceState() if session is None else session.result_workspace
        )
        if action not in {"apply", "save"} or (
            action == "apply"
            and (workspace.associated_saved_result is None or len(selections) != 1)
        ):
            return
        names: tuple[str, ...] = ()
        if action == "save":
            taken = {entry.name for entry in workspace.saved_results}
            defaults: list[str] = []
            for item in selections:
                base = (
                    item.anchor_evidence.candidate.name
                    if isinstance(item, AutoFitCandidateBranchResult)
                    else item.candidate.name
                )
                name = base
                suffix = 1
                while name in taken:
                    name = f"{base} ({suffix})"
                    suffix += 1
                taken.add(name)
                defaults.append(name)
            naming = ResultNamesDialog(
                tuple(defaults),
                dialog,
            )
            if naming.exec() != QDialog.DialogCode.Accepted:
                return
            names = naming.names
            if not all(name.strip() for name in names):
                return
        draft = original_draft if session is None else session.draft
        active_copy = workspace.active_working_copy
        load_first = active_copy is None or not (
            active_copy.has_result or active_copy.has_draft_edits
        )
        workflow = self._workflow_project_for(owner[0])
        adoptions: list[
            ManualFitExecutionOutcome | ManualMultiQFitExecutionOutcome
        ] = []
        adoption: ManualFitExecutionOutcome | ManualMultiQFitExecutionOutcome
        try:
            if (
                self._open_project is not owner[0]
                or self._open_dataset is None
                or self._open_dataset.workflow_dataset_id != owner[1]
            ):
                raise WorkflowError(
                    (
                        WorkflowDiagnostic(
                            WorkflowDiagnosticCode.AUTO_FIT_CONTEXT_CHANGED,
                            DiagnosticSeverity.ERROR,
                            "the active Dataset changed after AutoFit; "
                            "run AutoFit again",
                        ),
                    )
                )
            for selection in selections:
                if action == "apply" and isinstance(
                    selection, AutoFitCandidateBranchResult
                ):
                    selection = selection.anchor_evidence
                if isinstance(selection, AutoFitCandidateBranchResult):
                    adoption = adopt_selected_auto_fit_branch(
                        workflow,
                        draft,
                        dialog.outcome,
                        selection,
                        center_group_id=f"center-{uuid4().hex}",
                    )
                else:
                    single_adoption = adopt_single_q_auto_fit_candidate(
                        workflow,
                        draft,
                        dialog.outcome,
                        selection,
                        group_index=self.dataset_view.current_group_index,
                        current=workspace.current if action == "apply" else None,
                    )
                    if not single_adoption.success:
                        raise WorkflowError(single_adoption.diagnostics)
                    adoption = single_adoption
                adoptions.append(adoption)
            if action == "save":
                first_saved_index = len(workspace.saved_results)
                for name, adoption in zip(names, adoptions, strict=True):
                    assert adoption.snapshot is not None
                    workspace = append_saved_result(workspace, name, adoption.snapshot)
                if load_first:
                    workspace, _projected = load_saved_result(
                        workspace, draft, workspace.saved_results[first_saved_index]
                    )
            else:
                adoption = adoptions[0]
                assert adoption.snapshot is not None
                adopt_current = (
                    set_current_result
                    if isinstance(adoption, ManualMultiQFitExecutionOutcome)
                    else replace_current_q_result
                )
                workspace = adopt_current(
                    workspace,
                    adoption.snapshot,
                    draft=adoption.adopted_draft,
                )
        except WorkflowError as error:
            self._show_workflow_error("AutoFit", error)
            return
        except (FittingError, ValueError) as error:
            show_message_dialog(self, "AutoFit", str(error))
            return
        if session is None:
            session = ManualFitSession(
                owner, original_draft, dialog.outcome.group_index
            )
            self._manual_sessions[owner] = session
        session.result_workspace = workspace
        if action == "save":
            self.workspace.refresh_dataset_analysis(owner[0])
            if load_first:
                session.execution_by_group.clear()
                self._activate_manual_session(session)
            dialog.set_active_saved_result(
                workspace.associated_saved_result is not None
            )
            self._refresh_manual_result_actions()
            return
        adoption = adoptions[0]
        assert adoption.adopted_draft is not None and adoption.snapshot is not None
        self._discard_pending_manual_interaction()
        session.draft = adoption.adopted_draft
        session.group_index = dialog.outcome.group_index
        assert workspace.current is not None
        self._project_current_snapshot(
            session, workspace.current, group_indices=(dialog.outcome.group_index,)
        )
        self._activate_manual_session(session, expand=True)
        dialog.accept()

    def _activate_manual_session(
        self,
        session: ManualFitSession,
        *,
        expand: bool = False,
    ) -> None:
        """Bind one retained Sample session to the current editor and plot view."""

        self._active_manual_session = session
        session.context_for = lambda index: self._manual_context(session, index)
        self.set_inspector_visible(True)
        self.manual_fit_editor.show()
        self.manual_fit_editor.set_expanded(True if expand else session.editor_expanded)
        self.manual_fit_editor.result_toggle_button.setChecked(
            session.result_expanded,
        )
        self._set_manual_inspector_mode(True)
        self._sync_manual_fit_entry()
        self._refresh_manual_fit()

    def _deactivate_manual_fit(self, *, discard_session: bool = False) -> None:
        """Unbind the active editor while optionally deleting its Sample session."""

        session = self._active_manual_session
        if session is None:
            return
        if (
            self._open_project is session.owner[0]
            and self._open_dataset is not None
            and self._open_dataset.workflow_dataset_id == session.owner[1]
        ):
            session.group_index = self.dataset_view.current_group_index
        session.editor_expanded = self.manual_fit_editor.collapse_button.isChecked()
        session.result_expanded = (
            self.manual_fit_editor.result_toggle_button.isChecked()
        )
        self._discard_pending_manual_interaction()
        self._manual_preview = None
        self.dataset_view.set_manual_preview(None)
        self.dataset_view.set_manual_fit_result(None)
        self._active_manual_session = None
        if discard_session:
            self._manual_sessions.pop(session.owner, None)
        if hasattr(self, "manual_fit_editor"):
            self.manual_fit_editor.hide()
            self._manual_inspector_minimum_width = INSPECTOR_BASE_MINIMUM_WIDTH
            self.inspector.setMinimumWidth(INSPECTOR_BASE_MINIMUM_WIDTH)
            if not self.inspector.isHidden():
                self._enforce_inspector_minimum_width()
            self._set_manual_inspector_mode(False)
            self._sync_manual_fit_entry()
            self._refresh_inspector_context()
        self.workspace.refresh_dataset_analysis(session.owner[0])

    def close_manual_fit(self, *, discard_session: bool = False) -> None:
        """Hide the local editor without losing its Sample working state."""

        self._deactivate_manual_fit(discard_session=discard_session)

    def apply_resolution_for_sample(
        self,
        project: ProjectState | None = None,
        dataset: DatasetState | None = None,
        resolution: DatasetState | None = None,
        *,
        replace_confirmed: bool | None = None,
    ) -> bool:
        """Associate one selected Resolution through the transactional workflow seam."""

        if self._fit_all_q_mutation_blocked():
            return False

        target_project, target_dataset = self._editing_target(project, dataset)
        if (
            target_project is None
            or target_dataset is None
            or target_dataset.dataset.role is not SpectrumRole.SAMPLE
        ):
            return False
        candidates = tuple(
            item
            for item in target_project.datasets
            if item.dataset.role is SpectrumRole.RESOLUTION
        )
        candidate = resolution
        if candidate is None:
            if not candidates:
                show_message_dialog(
                    self,
                    "Apply Resolution",
                    "Add or mark a reduced dataset as Resolution first.",
                )
                return False
            names = tuple(item.name for item in candidates)
            choice, accepted = QInputDialog.getItem(
                self,
                "Apply Resolution",
                "Resolution:",
                names,
                editable=False,
            )
            if not accepted:
                return False
            candidate = candidates[names.index(choice)]
        if candidate not in candidates:
            return False
        workflow = self._workflow_project_for(target_project)
        sample_reference = self._workflow_dataset(workflow, target_dataset)
        resolution_reference = self._workflow_dataset(workflow, candidate)
        preflight = preflight_apply_resolution(
            workflow,
            sample_reference,
            resolution_reference,
        )
        confirmed = bool(replace_confirmed)
        if preflight.replacement_confirmation_required and replace_confirmed is None:
            existing = self._dataset_name_for_workflow_id(
                target_project,
                preflight.existing_resolution.dataset_id
                if preflight.existing_resolution is not None
                else "",
            )
            confirmed = confirm_dialog(
                self,
                "Replace Resolution",
                f"{existing} is currently applied. Replace it with {candidate.name}?",
                accept_text="Replace Resolution",
            )
        if preflight.replacement_confirmation_required and not confirmed:
            return False
        try:
            updated_workflow = apply_resolution(
                workflow,
                sample_reference,
                resolution_reference,
                replace_confirmed=confirmed,
            )
        except WorkflowError as error:
            self._show_workflow_error("Apply Resolution", error)
            return False
        replayed = applied_resolution(updated_workflow, sample_reference)
        synchronized_resolution = None
        if replayed is not None and replayed.dataset is not candidate.dataset:
            if replayed.dataset_id == candidate.workflow_dataset_id:
                synchronized_resolution = (
                    self.workspace.synchronize_replayed_resolution(
                        target_project,
                        candidate,
                        replayed.dataset,
                    )
                )
            self._preserved_q_sources.setdefault(
                (target_project, replayed.dataset_id),
                self._preserved_q_sources.get(
                    (target_project, candidate.workflow_dataset_id),
                    candidate.dataset,
                ),
            )
        self._workflow_projects[target_project] = updated_workflow
        if synchronized_resolution is not None:
            self.workspace.dataset_updated.emit(
                target_project,
                candidate,
                synchronized_resolution,
            )
        association_changed = (
            updated_workflow.resolution_associations != workflow.resolution_associations
        )
        target_session = self._manual_sessions.get(
            (
                target_project,
                target_dataset.workflow_dataset_id,
            )
        )
        active_manual_sample = self._manual_owner == (
            target_project,
            target_dataset.workflow_dataset_id,
        )
        if association_changed and target_session is not None:
            if active_manual_sample:
                self._discard_pending_manual_interaction()
            self._invalidate_manual_session(target_session)
        self._sync_manual_fit_entry()
        self._refresh_inspector_context()
        self._refresh_manual_fit()
        return True

    def _invalidate_manual_session(
        self,
        session: ManualFitSession,
        group_indices: Iterable[int] | None = None,
        *,
        retain_fit_result: bool = False,
    ) -> None:
        """Mark the affected Groups in one retained Sample session stale."""

        if not session.draft_available:
            return

        targets = (
            range(len(session.draft.setups)) if group_indices is None else group_indices
        )
        for group_index in targets:
            execution = session.execution_state(group_index)
            if not retain_fit_result:
                execution.overlay_hidden = (
                    session.display_status(group_index)
                    is not ResultDisplayStatus.FITTED_CURRENT
                )
                execution.diagnostics = ()
        # Working-Method edits invalidate the editable per-Group projection only.
        # Current and Saved Results are immutable application lifecycle state.

    def _approve_dataset_replacement(
        self, project: ProjectState, previous: DatasetState, replacement: ReducedDataset
    ) -> bool:
        """Veto a retiring replacement before Workspace/workflow publication."""

        retires = previous.dataset.role is SpectrumRole.SAMPLE and (
            replacement.role is not SpectrumRole.SAMPLE
            or len(previous.dataset.spectra) != len(replacement.spectra)
        )
        if not retires:
            return True
        if self._fit_all_q_mutation_blocked():
            return False
        session = self._manual_sessions.get((project, previous.workflow_dataset_id))
        return session is None or self._resolve_untitled(session)

    def _retire_manual_editing(self, session: ManualFitSession) -> None:
        """Retire a preflight-approved draft; notifications cannot offer a veto."""

        session.result_workspace, _draft = fresh_analysis(
            session.result_workspace, session.draft, discard_untitled=True
        )
        session.draft = _draft
        session.draft_available = False
        session.execution_by_group.clear()
        if session is self._active_manual_session:
            self._deactivate_manual_fit()

    def _invalidate_manual_fit(self, *, retain_fit_result: bool = False) -> None:
        """Mark working edits stale, optionally retaining last fitted evidence."""

        if self._active_manual_session is not None:
            self._invalidate_manual_session(
                self._active_manual_session,
                (self.dataset_view.current_group_index,),
                retain_fit_result=retain_fit_result,
            )

    def _run_manual_fit(self) -> None:
        """Run and atomically adopt one group through the public workflow seam."""

        if (
            self._fit_all_q_mutation_blocked()
            or self._manual_fit_lifecycle is ManualFitLifecycle.FITTING
            or self._manual_draft is None
            or self._open_project is None
        ):
            return
        self._discard_pending_manual_interaction()
        self._manual_fit_lifecycle = ManualFitLifecycle.FITTING
        self.dataset_view.set_manual_fit_result(None)
        self._manual_execution_diagnostics = ()
        self._refresh_manual_fit()
        QApplication.processEvents()
        workflow = self._workflow_project_for(self._open_project)
        outcome = run_and_adopt_manual_fit(
            workflow,
            self._manual_draft,
            self.dataset_view.current_group_index,
        )
        self._manual_execution_diagnostics = outcome.diagnostics
        if outcome.success:
            assert outcome.adopted_draft is not None
            assert outcome.fit_result is not None
            assert outcome.snapshot is not None
            self._manual_draft = outcome.adopted_draft
            session = self._active_manual_session
            assert session is not None
            session.result_workspace = replace_current_q_result(
                session.result_workspace,
                outcome.snapshot,
                draft=outcome.adopted_draft,
            )
            assert session.result_workspace.current is not None
            self._project_current_snapshot(
                session,
                session.result_workspace.current,
                outcome.diagnostics,
                group_indices=(self.dataset_view.current_group_index,),
            )
        self._manual_fit_lifecycle = ManualFitLifecycle.READY
        self.dataset_view.set_manual_fit_result(self._manual_fit_result)
        self._refresh_manual_fit()

    def _fit_all_q_available(self, session: ManualFitSession) -> bool:
        """Require the active authoritative Current Single-Q Result as anchor."""

        if (
            self._active_fit_all_q_generation is not None
            or self._has_unresolved_fitting_context_draft()
        ):
            return False
        current = session.result_workspace.current_result
        if current is None:
            return False
        group_index = self.dataset_view.current_group_index
        current_fit = current.point(group_index).fit_result
        if current_fit is None:
            return False
        return (
            session.display_status(group_index) is ResultDisplayStatus.FITTED_CURRENT
            and current_fit.context_binding is not None
            and current_fit.context_binding.group_index == group_index
        )

    def _has_unresolved_fitting_context_draft(self) -> bool:
        """Keep worker inputs aligned with the scientific state visible in the GUI."""

        return (
            (self._mask_draft is not None and self._mask_draft.is_dirty)
            or self._mask_boundary_preview_active
            or self.dataset_view.q_editor.has_unapplied_changes
        )

    def _fit_all_q_mutation_blocked(self) -> bool:
        """Return whether one captured Manual/scientific context owns execution."""

        return self._active_fit_all_q_generation is not None

    def _set_fit_all_q_context_locked(self, locked: bool) -> None:
        """Disable the existing mutation entrances while a run owns its context."""

        self.manual_fit_editor.set_context_locked(locked)
        self.dataset_view.q_editor.setEnabled(not locked)
        self.mask_task_bar.setEnabled(not locked)
        self.dataset_view.set_mask_interaction_enabled(not locked)
        self.inspector_resolution_button.setEnabled(not locked)
        self._sync_manual_fit_entry()
        self.edit_mask_action.setEnabled(
            not locked
            and self._open_dataset is not None
            and self._open_dataset.mask_editable
        )

    def _fit_all_q_context_is_current(self, context: FitAllQRunContext) -> bool:
        """Verify exact application ownership before adopting worker evidence."""

        if self._active_manual_session is not context.session:
            return False
        if (
            self._open_project is not context.owner[0]
            or self._open_dataset is None
            or self._open_dataset.workflow_dataset_id != context.owner[1]
        ):
            return False
        if self._manual_sessions.get(context.owner) is not context.session:
            return False
        if context.session.draft is not context.draft:
            return False
        current = context.session.result_workspace.current_result
        if (
            current is None
            or current.point(context.anchor_group).fit_result is not context.anchor_fit
        ):
            return False
        if self._workflow_projects.get(context.owner[0]) is not context.workflow:
            return False
        dataset_state = next(
            (
                item
                for item in context.owner[0].datasets
                if item.workflow_dataset_id == context.owner[1]
            ),
            None,
        )
        if dataset_state is None or dataset_state.dataset is not context.dataset:
            return False
        selection = (
            None
            if dataset_state.auto_mask is None
            else dataset_state.auto_mask.selection
        )
        return selection is context.selection

    def _show_fit_all_q(self) -> None:
        """Open active-constraint selection without reconstructing core metadata."""

        session = self._active_manual_session
        if (
            session is None
            or self._open_project is None
            or self._open_dataset is None
            or not self._fit_all_q_available(session)
        ):
            return
        group_index = self.dataset_view.current_group_index
        model = session.draft.setup(group_index).model
        current = session.result_workspace.current_result
        if model is None or current is None:
            return
        current_fit = current.point(group_index).fit_result
        if current_fit is None:
            return
        workflow = self._workflow_project_for(self._open_project)
        try:
            materialized = materialize_manual_setup(
                workflow,
                session.draft,
                group_index,
            )
        except WorkflowError as error:
            self._show_workflow_error("Fit to all Q", error)
            return
        spectrum = self._open_dataset.dataset.spectra[group_index]
        dialog = FitAllQTransferDialog(
            model,
            self,
            metadata=manual_parameter_metadata(
                materialized.fit_model,
                energy_unit=spectrum.energy_unit,
                intensity_unit=spectrum.intensity_unit,
            ),
            defaults=self._transfer_defaults,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._start_fit_all_q(session, current_fit, dialog.transfer_policy())

    def _start_fit_all_q(
        self,
        session: ManualFitSession,
        current_fit: FitResult,
        transfer_policy: MultiQConstraintTransferPolicy,
    ) -> None:
        """Start one generation-guarded Manual Multi-Q execution."""

        if (
            self._open_project is None
            or self._active_fit_all_q_generation is not None
            or self._has_unresolved_fitting_context_draft()
        ):
            return
        anchor = self.dataset_view.current_group_index
        self._fit_all_q_generation += 1
        generation = self._fit_all_q_generation
        cancel_event = Event()
        workflow = self._workflow_project_for(self._open_project)
        owner = session.owner
        dataset_state = next(
            item for item in owner[0].datasets if item.workflow_dataset_id == owner[1]
        )
        selection = (
            None
            if dataset_state.auto_mask is None
            else dataset_state.auto_mask.selection
        )
        worker = FitAllQWorker(
            workflow,
            session.draft,
            anchor,
            current_fit,
            transfer_policy,
            cancel_event,
            generation,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        worker.group_started.connect(self._on_fit_all_q_group_started)
        worker.progress.connect(self._on_fit_all_q_progress)
        worker.completed.connect(self._on_fit_all_q_completed)
        worker.failed.connect(self._on_fit_all_q_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            lambda active_thread=thread: self._release_fit_all_q_thread(active_thread)
        )
        self._active_fit_all_q_generation = generation
        self._fit_all_q_worker = worker
        self._fit_all_q_thread = thread
        self._fit_all_q_cancel_event = cancel_event
        self._fit_all_q_anchor_group = anchor
        self._fit_all_q_session = session
        self._fit_all_q_run_context = FitAllQRunContext(
            generation,
            session,
            owner,
            workflow,
            session.draft,
            anchor,
            current_fit,
            dataset_state.dataset,
            selection,
        )
        self._fit_all_q_live_statuses = list(self._result_fit_statuses(session))
        self._set_fit_all_q_context_locked(True)
        self.dataset_view.set_q_navigation_enabled(False)
        self.dataset_view.set_fit_statuses(
            tuple(self._fit_all_q_live_statuses),
        )
        self._refresh_manual_fit()
        thread.start()

    def _cancel_fit_all_q(self) -> None:
        if self._fit_all_q_cancel_event is not None:
            self._fit_all_q_cancel_event.set()

    def _on_fit_all_q_group_started(self, generation: int, group_index: int) -> None:
        if generation != self._active_fit_all_q_generation:
            return
        statuses = self._fit_all_q_live_statuses
        if statuses is None:
            return
        if self._active_manual_session is self._fit_all_q_session:
            self.dataset_view.set_fit_statuses(
                tuple(statuses),
                processing_group_index=group_index,
            )
            self.dataset_view.set_current_group(group_index)

    def _on_fit_all_q_progress(
        self,
        generation: int,
        outcome: MultiQFitOutcome,
    ) -> None:
        if generation != self._active_fit_all_q_generation:
            return
        statuses = self._fit_all_q_live_statuses
        if statuses is None:
            return
        statuses[outcome.group_index] = outcome.status
        if self._active_manual_session is self._fit_all_q_session:
            self.dataset_view.set_fit_statuses(tuple(statuses))

    def _on_fit_all_q_completed(
        self,
        generation: int,
        outcome: ManualMultiQFitExecutionOutcome,
    ) -> None:
        if generation != self._active_fit_all_q_generation:
            return
        session = self._fit_all_q_session
        context = self._fit_all_q_run_context
        if (
            session is None
            or context is None
            or context.generation != generation
            or not self._fit_all_q_context_is_current(context)
        ):
            self._finish_fit_all_q_run(generation)
            return
        session.draft = outcome.adopted_draft
        workspace = session.result_workspace
        session.result_workspace = set_current_result(
            workspace,
            outcome.snapshot,
            draft=outcome.adopted_draft,
        )
        self._project_current_snapshot(session, outcome.snapshot)
        self._finish_fit_all_q_run(generation)

    def _on_fit_all_q_failed(self, generation: int, error: Exception) -> None:
        if generation != self._active_fit_all_q_generation:
            return
        context = self._fit_all_q_run_context
        current = context is not None and self._fit_all_q_context_is_current(context)
        self._finish_fit_all_q_run(generation)
        if current and not self._fit_all_q_close_pending:
            show_message_dialog(self, "Fit to all Q", str(error))

    def _finish_fit_all_q_run(self, generation: int) -> None:
        """Restore the anchor and discard only this run's presentation state."""

        if generation != self._active_fit_all_q_generation:
            return
        anchor = self._fit_all_q_anchor_group
        run_session = self._fit_all_q_session
        self._active_fit_all_q_generation = None
        self._fit_all_q_cancel_event = None
        self._fit_all_q_anchor_group = None
        self._fit_all_q_session = None
        self._fit_all_q_run_context = None
        self._fit_all_q_live_statuses = None
        self._set_fit_all_q_context_locked(False)
        self.dataset_view.set_q_navigation_enabled(True)
        if (
            anchor is not None
            and self.dataset_view.dataset is not None
            and self._active_manual_session is run_session
        ):
            self.dataset_view.set_current_group(anchor)
        self._refresh_result_status_projection()
        self._refresh_manual_fit()

    def _release_fit_all_q_thread(self, thread: QThread) -> None:
        if self._fit_all_q_thread is thread:
            self._fit_all_q_thread = None
            self._fit_all_q_worker = None
            if self._fit_all_q_close_pending:
                anchor = self._fit_all_q_anchor_group
                self._fit_all_q_cancel_event = None
                self._fit_all_q_anchor_group = None
                self._fit_all_q_session = None
                self._fit_all_q_run_context = None
                self.setEnabled(True)
                self._set_fit_all_q_context_locked(False)
                self.dataset_view.set_q_navigation_enabled(True)
                if anchor is not None:
                    self.dataset_view.set_current_group(anchor)
                self._refresh_manual_fit()
        thread.deleteLater()
        if self._fit_all_q_close_pending:
            self._fit_all_q_close_pending = False
            QTimer.singleShot(0, self.close)

    @staticmethod
    def _project_current_snapshot(
        session: ManualFitSession,
        snapshot: ResultSnapshot,
        diagnostics: tuple[WorkflowDiagnostic, ...] = (),
        *,
        group_indices: tuple[int, ...] | None = None,
    ) -> None:
        """Project execution evidence only when an issued snapshot becomes Current."""

        for point in snapshot.result.points:
            outcome = point.outcome
            if group_indices is not None and outcome.group_index not in group_indices:
                continue
            execution = session.execution_state(outcome.group_index)
            execution.running = False
            execution.overlay_hidden = False
            execution.diagnostics = ()

    def _save_manual_result(self) -> None:
        session = self._active_manual_session
        if session is None or session.result_workspace.current_result is None:
            return
        if session.result_workspace.current_saved_result_index is None:
            return
        session.result_workspace = save_current_result(session.result_workspace)
        self.workspace.refresh_dataset_analysis(session.owner[0])
        self._refresh_manual_fit()

    def _save_manual_result_as(self) -> None:
        session = self._active_manual_session
        if session is None or session.result_workspace.current_result is None:
            return
        name, accepted = QInputDialog.getText(self, "Save Result As", "Result name")
        name = name.strip()
        if not accepted or not name:
            return
        session.result_workspace = save_current_result_as(
            session.result_workspace,
            name,
        )
        self.workspace.refresh_dataset_analysis(session.owner[0])
        self._refresh_manual_fit()

    def _workspace_saved_results(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> tuple[SavedFitResult, ...]:
        """Project only explicitly saved snapshots from their owning session."""

        session = self._manual_sessions.get((project, dataset.workflow_dataset_id))
        if session is None:
            return ()
        return session.result_workspace.saved_results

    def _workspace_active_result(
        self, project: ProjectState, dataset: DatasetState
    ) -> SavedFitResult | None:
        session = self._manual_sessions.get((project, dataset.workflow_dataset_id))
        return (
            None
            if session is None
            else session.result_workspace.associated_saved_result
        )

    def _manual_context(
        self, session: ManualFitSession, group_index: int
    ) -> ManualFitContext | None:
        dataset = next(
            (
                item
                for item in session.owner[0].datasets
                if item.workflow_dataset_id == session.owner[1]
            ),
            None,
        )
        if dataset is None or dataset.dataset.role is not SpectrumRole.SAMPLE:
            return None
        workflow = self._workflow_project_for(session.owner[0])
        return resolve_manual_fit_context(
            workflow, self._workflow_dataset(workflow, dataset), group_index
        ).context

    def _resolve_untitled(self, session: ManualFitSession) -> bool:
        """Obtain explicit consent before replacing unsaved untitled work."""

        item = session.result_workspace.untitled
        if item is None or not (item.has_result or item.has_draft_edits):
            return True
        choices = [DialogChoice("Discard", "discard", destructive=True)]
        if item.has_result:
            choices.insert(0, DialogChoice("Save", "save"))
            message = "Save the Untitled Result before replacing it?"
        else:
            message = (
                "Discard the unfitted edits? Setups without a fit cannot be saved."
            )
        choice = choose_dialog(
            self,
            "Replace Untitled",
            message,
            (*choices, DialogChoice("Cancel", "cancel", default=True)),
            cancel_value="cancel",
        )
        if choice == "cancel":
            return False
        if choice == "save":
            name, accepted = QInputDialog.getText(self, "Save Result As", "Result name")
            if not accepted or not name.strip():
                return False
            session.result_workspace = save_current_result_as(
                session.result_workspace, name.strip()
            )
        return True

    def _resolve_result_close(
        self, sessions: tuple[ManualFitSession, ...], *, quitting: bool = False
    ) -> bool:
        """Resolve all affected in-memory work in one explicit close decision."""
        lines: list[str] = []
        for session in sessions:
            workspace = session.result_workspace
            pending = (*unsaved_result_items(workspace), *draft_edit_items(workspace))
            dataset_name = self._dataset_name_for_workflow_id(
                session.owner[0], session.owner[1]
            )
            seen: set[int] = set()
            for item in pending:
                if id(item) in seen:
                    continue
                seen.add(id(item))
                name = item.name if isinstance(item, SavedFitResult) else "Untitled"
                kinds = []
                if item.modified:
                    kinds.append("unsaved fit")
                if item.has_draft_edits:
                    kinds.append("unfitted edits")
                lines.append(f"{dataset_name} · {name} · {' / '.join(kinds)}")
        any_result = any(
            session.result_workspace.saved_results
            or session.result_workspace.current_result is not None
            for session in sessions
        )
        if not lines and not (quitting and any_result):
            return True
        message = "\n".join(lines)
        if quitting:
            message += (
                "\n\nezQENS does not yet save projects to disk. All Results, "
                "including saved Results, are discarded on quit."
            )
        message += "\nUnfitted Untitled setups cannot be saved and will be discarded."
        choice = choose_dialog(
            self,
            "Close Results",
            message.strip(),
            (
                DialogChoice("Save All", "save"),
                DialogChoice("Discard", "discard", destructive=True),
                DialogChoice("Cancel", "cancel", default=True),
            ),
            cancel_value="cancel",
        )
        if choice != "save":
            return choice == "discard"
        untitled_sessions = tuple(
            session
            for session in sessions
            if session.result_workspace.untitled is not None
            and session.result_workspace.untitled.has_result
        )
        names: tuple[str, ...] = ()
        if untitled_sessions:
            naming = ResultNamesDialog(
                tuple(
                    self._dataset_name_for_workflow_id(*session.owner) + " · Untitled"
                    for session in untitled_sessions
                ),
                self,
            )
            if naming.exec() != QDialog.DialogCode.Accepted:
                return False
            names = naming.names
            if not all(name.strip() for name in names):
                return False
        updates: list[tuple[ManualFitSession, FittingWorkspaceState]] = []
        try:
            for session in sessions:
                workspace = session.result_workspace
                if session in untitled_sessions:
                    workspace = save_current_result_as(
                        workspace, names[untitled_sessions.index(session)].strip()
                    )
                active_index = workspace.current_saved_result_index
                active_item, untitled = workspace.active_item, workspace.untitled
                modified_indices = tuple(
                    index
                    for index, entry in enumerate(workspace.saved_results)
                    if entry.modified_copy is not None
                )
                for index in modified_indices:
                    entry = workspace.saved_results[index]
                    # Save the entry's existing working copy, without loading its
                    # historical Q topology into the current Dataset/editor.
                    workspace = save_current_result(
                        replace(workspace, active_item=entry, untitled=None)
                    )
                workspace = replace(
                    workspace,
                    active_item=(
                        workspace.saved_results[active_index]
                        if active_index is not None
                        else active_item
                    ),
                    untitled=untitled,
                )
                updates.append((session, workspace))
        except ValueError as error:
            show_message_dialog(self, "Save All", str(error))
            return False
        for session, workspace in updates:
            session.result_workspace = workspace
            self.workspace.refresh_dataset_analysis(session.owner[0])
        return True

    def _default_result_view(self, session: ManualFitSession) -> None:
        workspace = session.result_workspace
        if workspace.active_item is not None:
            return
        entry = workspace.best_saved_result or next(iter(workspace.saved_results), None)
        if entry is not None and entry.result.total_count == len(session.draft.setups):
            session.result_workspace, _draft = load_saved_result(
                workspace, session.draft, entry
            )

    def _fresh_workspace_dataset(
        self, project: ProjectState, dataset: DatasetState
    ) -> None:
        if self._fit_all_q_mutation_blocked() or not self.open_dataset(
            project, dataset
        ):
            return
        if dataset.dataset.role is not SpectrumRole.SAMPLE:
            return
        session = self._manual_sessions.get((project, dataset.workflow_dataset_id))
        if session is None:
            self.show_manual_fit(project, dataset)
            session = self._active_manual_session
        if session is None or not self._resolve_untitled(session):
            return
        self._discard_pending_manual_interaction()
        session.result_workspace, _draft = fresh_analysis(
            session.result_workspace, session.draft, discard_untitled=True
        )
        session.execution_by_group.clear()
        self._activate_manual_session(session)

    def _open_saved_result(
        self,
        project: ProjectState,
        dataset: DatasetState,
        entry: SavedFitResult,
    ) -> None:
        """Load the saved entry's editable, cached working copy."""

        if self._fit_all_q_mutation_blocked():
            return
        session = self._manual_sessions.get((project, dataset.workflow_dataset_id))
        if session is None or not any(
            item is entry for item in session.result_workspace.saved_results
        ):
            return
        if dataset.dataset.role is not SpectrumRole.SAMPLE:
            show_message_dialog(
                self,
                "Open Saved Result",
                "Results can only be loaded for a Sample Dataset.",
            )
            return
        if entry.result.total_count != len(dataset.dataset.spectra):
            show_message_dialog(
                self,
                "Open Saved Result",
                "This Result has a different Q-group count and cannot be loaded.",
            )
            return
        if not self.open_dataset(project, dataset):
            return
        if not self._resolve_untitled(session):
            return
        self._discard_pending_manual_interaction()
        session.result_workspace, _draft = load_saved_result(
            session.result_workspace, session.draft, entry, discard_untitled=True
        )
        session.execution_by_group.clear()
        self._activate_manual_session(session)

    def _rename_saved_result(
        self, project: ProjectState, dataset: DatasetState, entry: SavedFitResult
    ) -> None:
        session = self._saved_result_session(project, dataset, entry)
        if session is None:
            return
        name, accepted = QInputDialog.getText(
            self, "Rename Result", "Result name", text=entry.name
        )
        if (
            accepted
            and name.strip()
            and self._saved_result_session(project, dataset, entry) is session
        ):
            session.result_workspace = rename_saved_result(
                session.result_workspace, entry, name.strip()
            )
            self.workspace.refresh_dataset_analysis(project)

    def _best_saved_result(
        self, project: ProjectState, dataset: DatasetState, entry: SavedFitResult
    ) -> None:
        session = self._saved_result_session(project, dataset, entry)
        if session is None:
            return
        session.result_workspace = set_best_saved_result(
            session.result_workspace, None if entry.best else entry
        )
        self.workspace.refresh_dataset_analysis(project)

    def _revert_saved_result(
        self, project: ProjectState, dataset: DatasetState, entry: SavedFitResult
    ) -> None:
        if self._fit_all_q_mutation_blocked():
            return
        session = self._saved_result_session(project, dataset, entry)
        if session is None:
            return
        if session.result_workspace.active_item is entry:
            self._discard_pending_manual_interaction()
        session.result_workspace, _draft = revert_saved_result(
            session.result_workspace, session.draft, entry
        )
        session.execution_by_group.clear()
        self.workspace.refresh_dataset_analysis(project)
        if self._active_manual_session is session:
            self._refresh_manual_fit()

    def _saved_result_session(
        self, project: ProjectState, dataset: DatasetState, entry: SavedFitResult
    ) -> ManualFitSession | None:
        session = self._manual_sessions.get((project, dataset.workflow_dataset_id))
        return (
            session
            if session is not None
            and any(item is entry for item in session.result_workspace.saved_results)
            else None
        )

    def _delete_saved_result(
        self, project: ProjectState, dataset: DatasetState, entry: SavedFitResult
    ) -> None:
        if self._fit_all_q_mutation_blocked():
            return
        session = self._manual_sessions.get((project, dataset.workflow_dataset_id))
        if session is None or not any(
            item is entry for item in session.result_workspace.saved_results
        ):
            return
        if not confirm_dialog(
            self,
            "Delete Result",
            f'Delete Result "{entry.name}"?',
            accept_text="Delete Result",
            destructive=True,
        ):
            return
        # Recheck after the modal confirmation in case the row became stale.
        if not any(item is entry for item in session.result_workspace.saved_results):
            return
        was_active = session.result_workspace.active_item is entry
        session.result_workspace = remove_saved_result(session.result_workspace, entry)
        if was_active:
            session.execution_by_group.clear()
            self._default_result_view(session)
        self.workspace.refresh_dataset_analysis(project)
        if self._active_manual_session is session:
            self._refresh_manual_fit()

    def _begin_manual_component_interaction(self, kind: ManualComponentKind) -> None:
        if self._fit_all_q_mutation_blocked() or self._manual_draft is None:
            return
        group_index = self.dataset_view.current_group_index
        if kind in {ManualComponentKind.ELASTIC, ManualComponentKind.LORENTZIAN}:
            if self._open_project is None or self._open_dataset is None:
                return
            workflow = self._workflow_project_for(self._open_project)
            sample = self._workflow_dataset(workflow, self._open_dataset)
            resolution = resolve_manual_fit_context(workflow, sample, group_index)
            if resolution.context is None:
                show_message_dialog(
                    self,
                    "Add Fitting Function",
                    _manual_interaction_blocker_message(resolution.diagnostics),
                )
                return
        self._pending_manual_interaction = begin_component_interaction(
            self._manual_draft,
            group_index,
            kind,
        )
        self.dataset_view.begin_manual_component_interaction(kind)

    def _cancel_manual_component_interaction(self) -> None:
        """Forget pending geometry before another task can commit it."""

        self._pending_manual_interaction = None
        self.dataset_view.set_manual_pending_preview(None)

    def _discard_pending_manual_interaction(self) -> None:
        """Cancel both canvas geometry and its matching workflow token."""

        self.dataset_view.cancel_manual_component_interaction()
        self._pending_manual_interaction = None

    def _complete_manual_component_interaction(
        self,
        kind: ManualComponentKind,
        hints: dict[str, float],
    ) -> None:
        if self._fit_all_q_mutation_blocked():
            self._discard_pending_manual_interaction()
            return
        pending = self._pending_manual_interaction
        if (
            self._manual_draft is None
            or pending is None
            or pending.component_kind is not kind
        ):
            return
        if self._open_project is None:
            return
        workflow = self._workflow_project_for(self._open_project)
        try:
            if kind is ManualComponentKind.ELASTIC:
                updated = complete_elastic_interaction(
                    workflow,
                    self._manual_draft,
                    pending,
                    component_peak_center=hints["component_peak_center"],
                    component_peak_height=hints["component_peak_height"],
                )
            elif kind is ManualComponentKind.LORENTZIAN:
                updated = complete_lorentzian_interaction(
                    workflow,
                    self._manual_draft,
                    pending,
                    component_peak_center=hints["component_peak_center"],
                    component_peak_height=hints["component_peak_height"],
                    width_endpoint_energy=hints["width_endpoint_energy"],
                )
            else:
                updated = complete_background_interaction(
                    self._manual_draft,
                    pending,
                    first_energy=hints["first_energy"],
                    first_height=hints["first_height"],
                    second_energy=hints["second_energy"],
                    second_height=hints["second_height"],
                )
        except WorkflowError as error:
            self._show_workflow_error("Add Fitting Function", error)
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit()
        finally:
            self._pending_manual_interaction = None
        self._refresh_manual_fit()

    def _preview_manual_component_interaction(
        self,
        kind: ManualComponentKind,
        hints: dict[str, float] | None,
    ) -> None:
        """Render only the public workflow's transient scientific evaluation."""

        pending = self._pending_manual_interaction
        if (
            hints is None
            or self._manual_draft is None
            or pending is None
            or pending.component_kind is not kind
            or self._open_project is None
            or self._open_dataset is None
        ):
            self.dataset_view.set_manual_pending_preview(None)
            return
        workflow = self._workflow_project_for(self._open_project)
        display_energy = self._open_dataset.dataset.spectra[pending.group_index].energy
        try:
            if kind is ManualComponentKind.ELASTIC:
                preview = preview_pending_elastic_interaction(
                    workflow,
                    self._manual_draft,
                    pending,
                    component_peak_center=hints["component_peak_center"],
                    component_peak_height=hints["component_peak_height"],
                    display_energy=display_energy,
                )
            elif kind is ManualComponentKind.LORENTZIAN:
                preview = preview_pending_lorentzian_interaction(
                    workflow,
                    self._manual_draft,
                    pending,
                    component_peak_center=hints["component_peak_center"],
                    component_peak_height=hints["component_peak_height"],
                    width_endpoint_energy=hints["width_endpoint_energy"],
                    display_energy=display_energy,
                )
            else:
                preview = preview_pending_background_interaction(
                    workflow,
                    self._manual_draft,
                    pending,
                    first_energy=hints["first_energy"],
                    first_height=hints["first_height"],
                    second_energy=hints["second_energy"],
                    second_height=hints["second_height"],
                    display_energy=display_energy,
                )
        except WorkflowError:
            self.dataset_view.set_manual_pending_preview(None)
            return
        self.dataset_view.set_manual_pending_preview(preview.evaluation)

    def _update_manual_parameter(
        self,
        reference: ParameterReference,
        edit: ManualParameterEdit,
    ) -> None:
        if self._fit_all_q_mutation_blocked():
            return
        self._discard_pending_manual_interaction()
        if self._manual_draft is None:
            return
        try:
            updated = update_manual_parameter(
                self._manual_draft,
                self.dataset_view.current_group_index,
                reference,
                edit,
            )
        except WorkflowError as error:
            self._show_workflow_error("Fitting Parameters", error)
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit(retain_fit_result=True)
        self._refresh_manual_fit()

    def _handle_manual_chain(self, reference: ParameterReference) -> None:
        if self._fit_all_q_mutation_blocked():
            return
        self._discard_pending_manual_interaction()
        if self._manual_draft is None:
            return
        group_index = self.dataset_view.current_group_index
        model = self._manual_draft.setup(group_index).model
        if model is None:
            return
        try:
            if model.tie_for(reference) is not None:
                updated = untie_parameter(
                    self._manual_draft,
                    group_index,
                    reference,
                )
            elif model.center_group_for(reference) is not None:
                updated = untie_center_group_parameter(
                    self._manual_draft,
                    group_index,
                    reference,
                )
            else:
                same_family = tuple(
                    group
                    for group in model.parameter_ties
                    if group.members[0].family is reference.family
                )
                if same_family:
                    updated = join_parameter_tie(
                        self._manual_draft,
                        group_index,
                        same_family[0].group_id,
                        reference,
                    )
                elif reference.family is ParameterFamily.CENTER and model.center_groups:
                    updated = join_center_group(
                        self._manual_draft,
                        group_index,
                        model.center_groups[0].group_id,
                        reference,
                    )
                else:
                    updated = create_parameter_tie(
                        self._manual_draft,
                        group_index,
                        (reference,),
                        source_member=reference,
                    )
        except WorkflowError as error:
            self._show_workflow_error("Fitting Parameters", error)
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit(retain_fit_result=True)
        self._refresh_manual_fit()

    def _begin_new_manual_tie_group(self, reference: ParameterReference) -> None:
        if self._fit_all_q_mutation_blocked():
            return
        self._discard_pending_manual_interaction()
        if self._manual_draft is None:
            return
        try:
            updated = create_parameter_tie(
                self._manual_draft,
                self.dataset_view.current_group_index,
                (reference,),
                source_member=reference,
            )
        except WorkflowError as error:
            self._show_workflow_error("Fitting Parameters", error)
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit(retain_fit_result=True)
        self._refresh_manual_fit()

    def _join_manual_tie(self, reference: ParameterReference, group_id: str) -> None:
        if self._fit_all_q_mutation_blocked():
            return
        self._discard_pending_manual_interaction()
        if self._manual_draft is None:
            return
        group_index = self.dataset_view.current_group_index
        model = self._manual_draft.setup(group_index).model
        if model is None:
            return
        try:
            if any(group.group_id == group_id for group in model.center_groups):
                updated = join_center_group(
                    self._manual_draft,
                    group_index,
                    group_id,
                    reference,
                )
            else:
                updated = join_parameter_tie(
                    self._manual_draft,
                    group_index,
                    group_id,
                    reference,
                )
        except WorkflowError as error:
            self._show_workflow_error("Fitting Parameters", error)
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit(retain_fit_result=True)
        self._refresh_manual_fit()

    def _remove_manual_component(self, identity: ComponentIdentity) -> None:
        self._apply_manual_component_removal(
            identity,
            error_title="Fitting Parameters",
        )

    def _apply_manual_component_removal(
        self,
        identity: ComponentIdentity,
        *,
        error_title: str,
    ) -> bool:
        """Apply the one authoritative component-removal lifecycle."""

        if self._fit_all_q_mutation_blocked():
            return False
        self._discard_pending_manual_interaction()
        if self._manual_draft is None:
            return False
        try:
            updated = remove_manual_component(
                self._manual_draft,
                self.dataset_view.current_group_index,
                identity,
            )
        except WorkflowError as error:
            self._show_workflow_error(error_title, error)
            return False
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit()
        self._refresh_manual_fit()
        return True

    def clear_manual_model(self, *, confirmed: bool = False) -> bool:
        """Remove every component from only the current group-local Manual model."""

        if self._fit_all_q_mutation_blocked() or self._manual_draft is None:
            return False
        group_index = self.dataset_view.current_group_index
        model = self._manual_draft.setup(group_index).model
        if model is None:
            return False
        if not confirmed:
            accepted = confirm_dialog(
                self,
                "Clear Model",
                "Clear all functions from the current Group?",
                accept_text="Clear Model",
                destructive=True,
            )
            if not accepted:
                return False
        identities: list[ComponentIdentity] = []
        if model.elastic_area is not None:
            identities.append(ELASTIC_COMPONENT)
        identities.extend(item.identity for item in model.lorentzians)
        if model.background is not BackgroundModel.NONE:
            identities.append(BACKGROUND_COMPONENT)
        for identity in identities:
            if not self._apply_manual_component_removal(
                identity,
                error_title="Clear Model",
            ):
                return False
        return True

    def _on_manual_fit_expanded_changed(self, expanded: bool) -> None:
        """Cancel only provisional interaction state when the editor collapses."""

        if self._active_manual_session is not None:
            self._active_manual_session.editor_expanded = expanded
        if not expanded:
            self._discard_pending_manual_interaction()

    def _on_manual_group_changed(self, _group_index: int) -> None:
        """Project one Group's retained draft and execution state without mutation."""

        self.dataset_view.cancel_manual_component_interaction()
        if self._active_manual_session is not None:
            self._active_manual_session.group_index = (
                self.dataset_view.current_group_index
            )
        self._sync_manual_fit_entry()
        self._refresh_manual_fit()

    def _refresh_manual_fit(self) -> None:
        session = self._active_manual_session
        if (
            session is None
            or not session.draft_available
            or self._manual_draft is None
            or self._open_project is None
            or self._open_dataset is None
            or self._manual_owner
            != (self._open_project, self._open_dataset.workflow_dataset_id)
        ):
            return
        workflow = self._workflow_project_for(self._open_project)
        group_index = self.dataset_view.current_group_index
        readiness = manual_workflow_readiness(
            workflow,
            self._manual_draft,
            group_index,
        )
        current_result = session.result_workspace.current_result
        current_point = (
            current_result.point(group_index) if current_result is not None else None
        )
        branch_outcome = current_point.outcome if current_point is not None else None
        display_fit = self._manual_fit_result
        result_outcome = branch_outcome
        if (
            branch_outcome is not None
            and branch_outcome.status is MultiQFitStatus.SUCCESS
            and display_fit is None
        ):
            result_outcome = None
        model = self._manual_draft.setup(group_index).model
        spectrum = self._open_dataset.dataset.spectra[group_index]
        metadata: tuple[ParameterMetadata, ...] = ()
        if model is not None and readiness.context is not None:
            try:
                materialized = materialize_manual_setup(
                    workflow,
                    self._manual_draft,
                    group_index,
                )
            except WorkflowError:
                pass
            else:
                metadata = manual_parameter_metadata(
                    materialized.fit_model,
                    energy_unit=spectrum.energy_unit,
                    intensity_unit=spectrum.intensity_unit,
                )
        result_metadata = (
            manual_parameter_metadata(
                display_fit.configuration,
                energy_unit=spectrum.energy_unit,
                intensity_unit=spectrum.intensity_unit,
            )
            if display_fit is not None
            else metadata
        )
        self.manual_fit_editor.set_state(
            model,
            metadata,
            readiness,
            lifecycle=self._manual_fit_lifecycle,
            execution_diagnostics=self._manual_execution_diagnostics,
            fit_result=display_fit,
            result_outcome=result_outcome,
            result_metadata=result_metadata,
        )
        self._refresh_manual_result_actions()
        if self._active_fit_all_q_generation is None:
            self._refresh_result_status_projection()
        self._update_manual_inspector_minimum_width()
        QTimer.singleShot(0, self._update_manual_inspector_minimum_width)
        try:
            preview = (
                None
                if readiness.context is None or model is None
                else preview_manual_fit(
                    workflow,
                    self._manual_draft,
                    group_index,
                    display_energy=spectrum.energy,
                )
            )
        except WorkflowError:
            preview = None
        self._manual_preview = preview
        self.dataset_view.set_manual_display_state(
            preview,
            display_fit,
        )
        self.workspace.refresh_dataset_analysis(self._open_project)

    def _refresh_manual_result_actions(self) -> None:
        """Project Result actions from lifecycle state and unresolved GUI drafts."""

        session = self._active_manual_session
        if session is None:
            return
        workspace = session.result_workspace
        associated = workspace.associated_saved_result
        self.manual_fit_editor.set_result_actions(
            has_current_result=workspace.current_result is not None,
            associated=associated is not None,
            modified=(
                associated is not None and associated.snapshot is not workspace.current
            ),
            fit_all_q_enabled=self._fit_all_q_available(session),
            running=self._active_fit_all_q_generation is not None,
        )
        for dialog in self.findChildren(AutoFitCandidateDialog):
            context = dialog.outcome.scientific_context
            if (context.project_id, context.sample.dataset_id) == (
                session.draft.project_id,
                session.draft.sample_id,
            ):
                dialog.set_active_saved_result(associated is not None)

    def _result_fit_statuses(
        self,
        session: ManualFitSession,
    ) -> tuple[ResultDisplayStatus | None, ...]:
        count = len(session.draft.setups)
        session.context_for = lambda index: self._manual_context(session, index)
        return tuple(session.display_status(index) for index in range(count))

    def _refresh_result_status_projection(self) -> None:
        session = self._active_manual_session
        if (
            session is None
            and self._open_project is not None
            and self._open_dataset is not None
        ):
            session = self._manual_sessions.get(
                (self._open_project, self._open_dataset.workflow_dataset_id)
            )
        if session is not None and not session.draft_available:
            session = None
        if self.dataset_view.dataset is None:
            return
        statuses = (
            tuple(None for _ in self.dataset_view.dataset.spectra)
            if session is None
            else self._result_fit_statuses(session)
        )
        self.dataset_view.set_fit_statuses(statuses)

    def _update_manual_inspector_minimum_width(self) -> None:
        """Stop the Inspector before its populated three-slot grid can clip."""

        parameter_width = self.manual_fit_editor.populated_parameter_minimum_width()
        if not parameter_width:
            return
        margins = self._inspector_layout.contentsMargins()
        scroll_width = self.inspector_scroll_area.verticalScrollBar().sizeHint().width()
        required = parameter_width + margins.left() + margins.right() + scroll_width
        self._manual_inspector_minimum_width = max(
            self._manual_inspector_minimum_width,
            required,
        )
        self.inspector.setMinimumWidth(self._manual_inspector_minimum_width)
        if not self.inspector.isHidden():
            self._enforce_inspector_minimum_width()

    def _emphasize_manual_parameter(self, reference: ParameterReference) -> None:
        if self._manual_draft is None:
            return
        model = self._manual_draft.setup(self.dataset_view.current_group_index).model
        if model is None:
            return
        tie = model.tie_for(reference)
        references = (reference,) if tie is None else tie.members
        components = frozenset(item.component for item in references)
        self.dataset_view.set_manual_preview(
            self._manual_preview,
            emphasized_components=components,
        )

    def _dataset_name_for_workflow_id(
        self,
        project: ProjectState,
        dataset_id: str,
    ) -> str:
        return next(
            (
                item.name
                for item in project.datasets
                if item.workflow_dataset_id == dataset_id
            ),
            "Resolution",
        )

    def _show_workflow_error(self, title: str, error: WorkflowError) -> None:
        """Present typed workflow failures without parsing them for control flow."""

        show_message_dialog(
            self,
            title,
            "\n".join(item.message for item in error.diagnostics),
        )

    def show_units_editor(
        self,
        project: ProjectState | None = None,
        dataset: DatasetState | None = None,
    ) -> SourceUnitsDialog | None:
        """Open the small source-metadata editor without importing or converting."""

        if self._fit_all_q_mutation_blocked():
            return None
        target_project, target_dataset = self._editing_target(project, dataset)
        if target_project is None or target_dataset is None:
            return None
        self._discard_pending_manual_interaction()
        dialog = SourceUnitsDialog(target_dataset.dataset, self)
        dialog.set_apply_handler(
            lambda energy, intensity: self._apply_source_units(
                target_project,
                target_dataset,
                energy,
                intensity,
            ),
        )
        dialog.open()
        return dialog

    def remove_project(self, project: ProjectState, *, confirmed: bool = False) -> bool:
        """Remove Project state after confirming source files stay intact."""

        if self._fit_all_q_mutation_blocked():
            return False
        if not confirmed:
            accepted = confirm_dialog(
                self,
                "Delete Project",
                "Remove this ezQENS Project from the Workspace? Imported source "
                "files will not be deleted from disk.",
                accept_text="Delete Project",
                destructive=True,
            )
            if not accepted:
                return False
        if self._open_project is project and not self._resolve_mask_task_transition():
            return False
        sessions = tuple(
            session
            for session in self._manual_sessions.values()
            if session.owner[0] is project
        )
        if not self._resolve_result_close(sessions):
            return False
        if self._open_project is project:
            self._clear_open_context()
        self._workflow_projects.pop(project, None)
        for owner in tuple(self._preserved_q_sources):
            if owner[0] is project:
                self._preserved_q_sources.pop(owner)
        for owner in tuple(self._manual_sessions):
            if owner[0] is project:
                self._manual_sessions.pop(owner)
        for owner in tuple(self._sample_view_sessions):
            if owner[0] is project:
                self._sample_view_sessions.pop(owner)
        self.workspace.remove_project(project)
        return True

    def remove_dataset(
        self,
        project: ProjectState,
        dataset: DatasetState,
        *,
        confirmed: bool = False,
    ) -> bool:
        """Remove Dataset application state while preserving its source file."""

        if self._fit_all_q_mutation_blocked():
            return False
        if not confirmed:
            accepted = confirm_dialog(
                self,
                "Remove from Project",
                "Remove this dataset from the ezQENS Project? Its source file will "
                "not be deleted from disk. Its Results will be discarded.",
                accept_text="Remove",
                destructive=True,
            )
            if not accepted:
                return False
        manual_context_affected = self._manual_context_uses_dataset(project, dataset)
        removing_open_dataset = (
            self._open_project is project and self._open_dataset is dataset
        )
        if removing_open_dataset and not self._resolve_mask_task_transition():
            return False
        session = self._manual_sessions.get((project, dataset.workflow_dataset_id))
        if session is not None and not self._resolve_result_close((session,)):
            return False
        if removing_open_dataset and dataset is not self._open_dataset:
            # Saving the resolved task replaces its immutable DatasetState owner.
            if self._open_dataset is None:
                return False
            dataset = self._open_dataset
        if removing_open_dataset:
            self._clear_open_context()
        elif manual_context_affected:
            self._discard_pending_manual_interaction()
        affected_sessions = self._manual_sessions_using_dataset(project, dataset)
        workflow = self._workflow_projects.get(project)
        if workflow is not None:
            self._workflow_projects[project] = remove_project_dataset(
                workflow,
                self._workflow_dataset(workflow, dataset),
            )
        removed_owner = (project, dataset.workflow_dataset_id)
        self._preserved_q_sources.pop(removed_owner, None)
        self._manual_sessions.pop(removed_owner, None)
        self._sample_view_sessions.pop(removed_owner, None)
        for session in affected_sessions:
            if session.owner != removed_owner:
                self._invalidate_manual_session(session)
        self.workspace.remove_dataset(project, dataset)
        if manual_context_affected and not removing_open_dataset:
            self._refresh_inspector_context()
            self._refresh_manual_fit()
        return True

    def remove_q_method(
        self,
        project: ProjectState,
        method: QMethodState,
        *,
        confirmed: bool = False,
    ) -> bool:
        """Remove only the reusable Project method, never its source file."""

        if not confirmed:
            accepted = confirm_dialog(
                self,
                "Remove from Project",
                "Remove this Q Method from the ezQENS Project? Its source file will "
                "not be deleted from disk.",
                accept_text="Remove",
                destructive=True,
            )
            if not accepted:
                return False
        self.workspace.remove_q_method(project, method)
        if project is self._open_project and not self.dataset_view.q_editor.isHidden():
            self.dataset_view.q_editor.set_available_methods(
                tuple(
                    (candidate.name, candidate.q_bins)
                    for candidate in project.q_methods
                ),
            )
        return True

    def _clear_open_context(self) -> None:
        """Clear every view reference before its Workspace object is removed."""

        self._teardown_mask_task()
        self.close_manual_fit(discard_session=True)
        self.dataset_view.set_mask_edit_available(False)
        self.dataset_view.clear_dataset()
        self.workspace.clear_active_dataset()
        self._open_project = None
        self._open_dataset = None
        self.edit_mask_action.setEnabled(False)
        self._sync_manual_fit_entry()
        self.set_scientific_figure_visible(False)
        self._refresh_context_label()
        self._refresh_inspector_context()

    def show_q_editor(
        self,
        project: ProjectState | None = None,
        dataset: DatasetState | None = None,
    ) -> None:
        """Open the shared inline editor below the current overview."""

        if self._fit_all_q_mutation_blocked():
            return
        target_project, target_dataset = self._editing_target(project, dataset)
        if target_project is None or target_dataset is None:
            return
        self._discard_pending_manual_interaction()
        if (
            target_project is not self._open_project
            or target_dataset is not self._open_dataset
        ):
            if not self.open_dataset(target_project, target_dataset):
                return
        if self.dataset_view.q_editor.isHidden():
            self.dataset_view.open_q_editor()
        self.dataset_view.q_editor.set_available_methods(
            tuple((method.name, method.q_bins) for method in target_project.q_methods),
        )

    def show_mask_editor(
        self,
        project: ProjectState | None = None,
        dataset: DatasetState | None = None,
        *,
        mask_task_decision: str | None = None,
    ) -> None:
        """Open the stored Mask baseline for an explicit Workspace dataset."""

        if self._fit_all_q_mutation_blocked():
            return
        target_project, target_dataset = self._editing_target(project, dataset)
        if target_project is None or target_dataset is None:
            return
        if (
            target_project is not self._open_project
            or target_dataset is not self._open_dataset
        ):
            if not self.open_dataset(
                target_project,
                target_dataset,
                mask_task_decision=mask_task_decision,
            ):
                return
        self.enter_mask_task()

    def change_dataset_role(
        self,
        project: ProjectState,
        dataset: DatasetState,
        role: SpectrumRole,
        *,
        mask_task_decision: str | None = None,
    ) -> DatasetState | None:
        """Apply a role transition only after resolving its active Mask task."""

        if self._fit_all_q_mutation_blocked():
            return None
        self.workspace.validate_dataset_membership(project, dataset)
        if dataset.dataset.role is role:
            return dataset
        target = self._resolve_scientific_dataset_replacement(
            project,
            dataset,
            mask_task_decision=mask_task_decision,
        )
        if target is None:
            return None
        return self.workspace.set_dataset_role(project, target, role)

    def _apply_q_bins(
        self,
        q_bins: QBins,
        diagnostics: tuple[ImportDiagnostic, ...],
        *,
        mask_task_decision: str | None = None,
    ) -> bool:
        if (
            self._fit_all_q_mutation_blocked()
            or self._open_project is None
            or self._open_dataset is None
        ):
            return False
        target = self._resolve_scientific_dataset_replacement(
            self._open_project,
            self._open_dataset,
            mask_task_decision=mask_task_decision,
        )
        if target is None:
            return False
        return (
            self.workspace.assign_q_bins(
                self._open_project,
                target,
                q_bins,
                diagnostics=diagnostics,
            )
            is not None
        )

    def _apply_source_units(
        self,
        project: ProjectState,
        dataset: DatasetState,
        energy_unit: str,
        intensity_unit: str,
        *,
        mask_task_decision: str | None = None,
    ) -> bool:
        """Apply metadata replacement only after resolving an active Mask task."""

        if self._fit_all_q_mutation_blocked():
            return False
        target = self._resolve_scientific_dataset_replacement(
            project,
            dataset,
            mask_task_decision=mask_task_decision,
        )
        if target is None:
            return False
        return (
            self.workspace.assign_source_units(
                project,
                target,
                energy_unit=energy_unit,
                intensity_unit=intensity_unit,
            )
            is not None
        )

    def _resolve_scientific_dataset_replacement(
        self,
        project: ProjectState,
        dataset: DatasetState,
        *,
        mask_task_decision: str | None = None,
    ) -> DatasetState | None:
        """Resolve a Mask task before replacing its open scientific dataset."""

        if self._fit_all_q_mutation_blocked():
            return None
        self.workspace.validate_dataset_membership(project, dataset)
        changing_open_dataset = (
            project is self._open_project and dataset is self._open_dataset
        )
        if not changing_open_dataset:
            return dataset
        if not self._resolve_mask_task_transition(mask_task_decision):
            return None
        # A Save replaces the immutable owner. Keep the requested logical target,
        # rather than redirecting a non-open DatasetState to the open one.
        return self._open_dataset

    def _save_q_method(self, q_bins: QBins) -> None:
        if self._open_project is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Save Q Method",
            "Method name:",
            text=f"Q Method {len(self._open_project.q_methods) + 1}",
        )
        if accepted and name.strip():
            self.workspace.save_q_method(self._open_project, q_bins, name=name.strip())
            self.dataset_view.q_editor.set_available_methods(
                (method.name, method.q_bins) for method in self._open_project.q_methods
            )

    def _save_imported_q_method(
        self,
        q_bins: QBins,
        _diagnostics: tuple[ImportDiagnostic, ...],
    ) -> None:
        """Make an editor-imported DAVE assignment reusable in this Project."""

        if self._open_project is None:
            return
        self.workspace.save_q_method(self._open_project, q_bins)
        self.dataset_view.q_editor.set_available_methods(
            tuple(
                (method.name, method.q_bins) for method in self._open_project.q_methods
            ),
        )

    def _apply_dragged_q_method(
        self,
        method_project_index: int,
        method_index: int,
    ) -> None:
        """Apply a method dropped on the open scientific view or editor draft."""

        if self._open_project is None or self._open_dataset is None:
            return
        try:
            source_project = self.workspace.projects[method_project_index]
            method = source_project.q_methods[method_index]
        except IndexError:
            return
        self._apply_q_method(self._open_project, self._open_dataset, method)

    def _apply_q_method(
        self,
        project: ProjectState,
        dataset: DatasetState,
        method: QMethodState,
        *,
        confirmed: bool = False,
    ) -> bool:
        """Apply a compatible reusable Q Method with one explicit overwrite guard."""

        if self._fit_all_q_mutation_blocked():
            return False
        if method.q_bins.group_count != len(dataset.dataset.spectra):
            show_message_dialog(
                self,
                "Q Method",
                "This Q Method does not match the dataset Group count.",
            )
            return False
        editing_open_dataset = (
            project is self._open_project
            and dataset is self._open_dataset
            and not self.dataset_view.q_editor.isHidden()
        )
        if editing_open_dataset:
            self.dataset_view.q_editor.load_q_bins_draft(method.q_bins)
            return True
        if dataset.dataset.q_bins is not None and not confirmed:
            accepted = confirm_dialog(
                self,
                "Replace Q Assignment",
                "Replace this dataset's existing Q assignment with the selected "
                "Q Method?",
            )
            if not accepted:
                return False
        target = self._resolve_scientific_dataset_replacement(project, dataset)
        if target is None:
            return False
        return self.workspace.assign_q_bins(project, target, method.q_bins) is not None

    def enter_mask_task(self) -> None:
        """Enter the familiar viewer-based Mask task with a local draft."""

        if self._fit_all_q_mutation_blocked():
            return
        self._discard_pending_manual_interaction()
        if self._mask_draft is not None:
            return
        if self._open_dataset is None or not self._open_dataset.mask_editable:
            return
        auto_mask = self._open_dataset.auto_mask
        if auto_mask is None:
            return
        try:
            self._mask_draft = MaskTaskDraft(auto_mask, self._open_dataset.dataset)
        except ValueError as error:
            show_message_dialog(self, "Mask", str(error))
            return
        self._mask_draft_owner = self._open_dataset
        self._mask_boundary_preview_active = False
        self.dataset_view.set_mask_boundary_coordinates(
            self._mask_draft.boundary_coordinates
        )
        self.dataset_view.set_selection(self._mask_draft.selection)
        self.dataset_view.set_mask_inspection_mode(True)
        self._set_mask_tool("boundary", True)
        self._set_mask_operation("exclude", True)
        self.mask_task_bar.show()
        self._sync_mask_controls()

    def save_mask_task(self) -> bool:
        """Commit the current local draft into in-memory project state."""

        if (
            self._fit_all_q_mutation_blocked()
            or self._mask_draft is None
            or self._open_project is None
            or self._open_dataset is None
            or self._mask_draft_owner is not self._open_dataset
            or self._mask_draft.selection.dataset is not self._open_dataset.dataset
            or self._mask_boundary_preview_active
        ):
            return False
        current = self.workspace.update_auto_mask(
            self._open_project,
            self._open_dataset,
            self._mask_draft.saved_state(),
        )
        self._open_dataset = current
        if current.auto_mask is None:
            return False
        self._mask_draft = MaskTaskDraft(current.auto_mask)
        self._mask_draft_owner = current
        self.dataset_view.set_mask_boundary_coordinates(
            self._mask_draft.boundary_coordinates
        )
        self._sync_mask_controls()
        self._refresh_manual_result_actions()
        return True

    def close_mask_task(self, decision: str | None = None) -> bool:
        """Leave the Mask task, prompting for task-local unsaved changes."""

        if self._mask_draft is None:
            return True
        if self._mask_draft_owner is not self._open_dataset:
            return False
        choice = decision
        if self._mask_draft.is_dirty and choice is None:
            choice = choose_dialog(
                self,
                "Save Mask Changes",
                "Save mask changes before closing?",
                (
                    DialogChoice("Cancel", "cancel"),
                    DialogChoice("Discard", "discard", destructive=True),
                    DialogChoice("Save", "save", default=True),
                ),
                cancel_value="cancel",
            )
        if choice == "cancel":
            return False
        if choice == "save" and not self.save_mask_task():
            return False
        if self._open_dataset is not None:
            self.dataset_view.set_selection(
                None
                if self._open_dataset.auto_mask is None
                else self._open_dataset.auto_mask.selection,
            )
        self._teardown_mask_task()
        return True

    def _resolve_mask_task_transition(self, decision: str | None = None) -> bool:
        """Resolve a task before detaching it from its current DatasetState owner."""

        if self._mask_draft is None:
            return True
        if self._mask_draft_owner is not self._open_dataset:
            return False
        return self.close_mask_task(decision)

    def _teardown_mask_task(self) -> None:
        """Clear all task-local view state after a Mask task has been resolved."""

        self._mask_boundary_preview_active = False
        self.dataset_view.set_mask_inspection_mode(False)
        self.dataset_view.set_mask_tool(None)
        self.dataset_view.set_mask_boundary_coordinates(None)
        for button in (
            self.mask_boundary_button,
            self.mask_rectangle_button,
            self.mask_lasso_button,
            self.mask_exclude_button,
            self.mask_restore_button,
        ):
            blocker = QSignalBlocker(button)
            button.setChecked(False)
            del blocker
        self._mask_draft = None
        self._mask_draft_owner = None
        self.mask_task_bar.hide()
        self._sync_mask_controls()
        self._refresh_manual_result_actions()

    def rerun_auto_mask(self, confirmed: bool = False) -> bool:
        """Replace the proposal after confirmation clears manual edits."""

        if (
            self._fit_all_q_mutation_blocked()
            or self._open_project is None
            or self._open_dataset is None
            or self._mask_draft is None
        ):
            return False
        if not confirmed:
            accepted = confirm_dialog(
                self,
                "Re-run AutoMask",
                "Replace the existing AutoMask proposal and all manual mask edits "
                "with a newly computed AutoMask proposal?",
                accept_text="Re-run AutoMask",
                destructive=True,
            )
            if not accepted:
                return False
        state = masking.create_auto_mask_state(self._open_dataset.dataset)
        try:
            # Validate the new core proposal before replacing the committed baseline.
            MaskTaskDraft(state, self._open_dataset.dataset)
            current = self.workspace.update_auto_mask(
                self._open_project,
                self._open_dataset,
                state,
            )
            self._open_dataset = current
            if current.auto_mask is None:
                return False
            self._mask_draft = MaskTaskDraft(
                current.auto_mask,
                current.dataset,
            )
            self._mask_draft_owner = current
            self._mask_boundary_preview_active = False
        except ValueError as error:
            show_message_dialog(self, "Re-run AutoMask", str(error))
            return False
        self.dataset_view.set_mask_boundary_coordinates(
            self._mask_draft.boundary_coordinates
        )
        self.dataset_view.set_selection(self._mask_draft.selection)
        self._sync_mask_controls()
        self._refresh_manual_result_actions()
        return True

    def _editing_target(
        self,
        project: ProjectState | None,
        dataset: DatasetState | None,
    ) -> tuple[ProjectState | None, DatasetState | None]:
        return (
            project or self._open_project,
            dataset or self._open_dataset,
        )

    def _manual_fit_capability(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> tuple[bool, str]:
        """Ask the public workflow boundary whether this dataset can own a draft."""

        workflow = self._workflow_project_for(project)
        try:
            open_manual_fit_draft(
                workflow,
                self._workflow_dataset(workflow, dataset),
            )
        except WorkflowError as error:
            return False, "\n".join(item.message for item in error.diagnostics)
        return True, "Open Fitting Parameters for the currently open Sample"

    def _auto_fit_capability(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> tuple[bool, str]:
        """Resolve the authoritative Single-Q context without running AutoFit."""

        workflow = self._workflow_project_for(project)
        sample = self._workflow_dataset(workflow, dataset)
        try:
            open_manual_fit_draft(workflow, sample)
        except WorkflowError as error:
            return False, "\n".join(item.message for item in error.diagnostics)
        group_index = (
            self.dataset_view.current_group_index
            if project is self._open_project and dataset is self._open_dataset
            else 0
        )
        resolution = resolve_manual_fit_context(
            workflow,
            sample,
            group_index,
        )
        if resolution.context is None:
            return False, "\n".join(item.message for item in resolution.diagnostics)
        return True, "Evaluate AutoFit candidates for the current Sample Group"

    def _sync_manual_fit_entry(self) -> None:
        """Keep Manual Fit menu and AutoFit action bound to the open dataset."""

        context_locked = self._fit_all_q_mutation_blocked()
        available = False
        auto_available = False
        auto_tooltip = "Open a prepared Sample to use AutoFit"
        if self._open_project is not None and self._open_dataset is not None:
            available, _detail = self._manual_fit_capability(
                self._open_project,
                self._open_dataset,
            )
            auto_available, auto_tooltip = self._auto_fit_capability(
                self._open_project,
                self._open_dataset,
            )
        self.manual_fit_action.setEnabled(available and not context_locked)
        self.manual_fit_button.setEnabled(available and not context_locked)
        self.manual_fit_button.setToolTip(
            "Open Fitting Parameters for the current Sample"
            if available
            else "Open a prepared Sample to use Manual Fit"
        )
        self.auto_fit_action.setEnabled(auto_available and not context_locked)
        self.q_rebin_action.setEnabled(
            not context_locked
            and self._open_dataset is not None
            and self._open_dataset.dataset.role is SpectrumRole.SAMPLE
        )
        self.auto_fit_button.setEnabled(auto_available and not context_locked)
        self.auto_fit_button.setToolTip(auto_tooltip)

    def _set_mask_tool(self, tool: str, enabled: bool) -> None:
        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return
        selected = tool if enabled else None
        self.dataset_view.set_mask_tool(selected)
        for button, value in (
            (self.mask_boundary_button, "boundary"),
            (self.mask_rectangle_button, "rectangle"),
            (self.mask_lasso_button, "lasso"),
        ):
            blocker = QSignalBlocker(button)
            button.setChecked(selected == value)
            del blocker

    def _set_mask_operation(self, operation: str, checked: bool) -> None:
        """Keep the persistent Rectangle/Lasso operation explicit and visible."""

        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return
        selected = operation if checked else "exclude"
        self.dataset_view.set_mask_operation(selected)
        for button, value in (
            (self.mask_exclude_button, "exclude"),
            (self.mask_restore_button, "restore"),
        ):
            blocker = QSignalBlocker(button)
            button.setChecked(selected == value)
            del blocker

    def _set_mask_operation_feedback(self, text: str) -> None:
        """Reflect the effective temporary operation without adding passive text."""

        operation = "restore" if text.startswith("Restore") else "exclude"
        for button, value in (
            (self.mask_exclude_button, "exclude"),
            (self.mask_restore_button, "restore"),
        ):
            blocker = QSignalBlocker(button)
            button.setChecked(operation == value)
            del blocker

    def _exclude_rectangle(
        self,
        group_index: int,
        lower_energy: float,
        upper_energy: float,
        lower_intensity: float,
        upper_intensity: float,
        operation: str,
    ) -> None:
        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return
        edit = (
            self._mask_draft.restore_rectangle
            if operation == "restore"
            else self._mask_draft.exclude_rectangle
        )
        if edit(
            group_index,
            lower_energy=lower_energy,
            upper_energy=upper_energy,
            lower_intensity=lower_intensity,
            upper_intensity=upper_intensity,
        ):
            self._refresh_mask_preview()

    def _exclude_lasso(
        self,
        group_index: int,
        vertices: object,
        operation: str,
    ) -> None:
        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return
        edit = (
            self._mask_draft.restore_lasso
            if operation == "restore"
            else self._mask_draft.exclude_lasso
        )
        if edit(group_index, vertices):
            self._refresh_mask_preview()

    def _preview_auto_boundary(
        self,
        group_index: int,
        side: object,
        energy: float,
    ) -> None:
        """Show an effective-mask preview while a boundary handle is dragged."""

        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return
        try:
            selection, coordinates = self._mask_draft.preview_auto_boundary(
                group_index,
                side=BoundarySide(str(side)),
                energy=energy,
            )
        except ValueError:
            self._refresh_mask_preview()
            return
        self._mask_boundary_preview_active = True
        self.dataset_view.set_mask_boundary_coordinates(coordinates)
        self.dataset_view.set_selection(selection)
        self._refresh_manual_result_actions()

    def _cancel_mask_boundary_preview(self) -> None:
        """Restore the task draft after its transient Boundary gesture ends."""

        if self._mask_boundary_preview_active:
            self._refresh_mask_preview()

    def _edit_auto_boundary(
        self,
        group_index: int,
        side: object,
        energy: float,
    ) -> None:
        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return
        boundary_side = BoundarySide(str(side))
        self._mask_draft.set_auto_boundary(
            group_index,
            side=boundary_side,
            energy=energy,
        )
        self._refresh_mask_preview()

    def _apply_mask_boundary_to_all_groups(
        self,
        *,
        confirmed: bool = False,
    ) -> bool:
        """Apply the current energy boundary to every Group as one Mask edit."""

        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return False
        if not confirmed:
            accepted = confirm_dialog(
                self,
                "Apply Boundary to All Groups",
                "Apply the current energy boundary to all Groups?",
                accept_text="Apply to All Groups",
            )
            if not accepted:
                return False
        try:
            changed = self._mask_draft.apply_boundary_to_all_groups(
                self.dataset_view.current_group_index,
            )
        except ValueError as error:
            show_message_dialog(self, "Apply Boundary to All Groups", str(error))
            return False
        if changed:
            self._refresh_mask_preview()
        return changed

    def _undo_mask(self) -> None:
        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return
        self._mask_draft.undo()
        self._refresh_mask_preview()

    def _redo_mask(self) -> None:
        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return
        self._mask_draft.redo()
        self._refresh_mask_preview()

    def _disable_auto_mask(self) -> None:
        """Record valid AUTO re-inclusions across all Groups in the local draft."""

        if (
            not self._fit_all_q_mutation_blocked()
            and self._mask_draft is not None
            and self._mask_draft.disable_auto_mask()
        ):
            self._refresh_mask_preview()

    def _reset_mask_group(self) -> None:
        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return
        group_index = self.dataset_view.current_group_index
        if self._mask_draft.reset_group(group_index):
            self._refresh_mask_preview()

    def _reset_mask_all(self, confirmed: bool = False) -> bool:
        """Restore all Groups to the existing proposal after explicit confirmation."""

        if self._fit_all_q_mutation_blocked() or self._mask_draft is None:
            return False
        if self._mask_draft.has_manual_edits and not confirmed:
            accepted = confirm_dialog(
                self,
                "Reset All Groups",
                "Discard manual mask edits in every Group and restore the existing "
                "AutoMask proposal?",
                accept_text="Reset All",
                destructive=True,
            )
            if not accepted:
                return False
        if self._mask_draft.reset_all():
            self._refresh_mask_preview()
            return True
        return False

    def _refresh_mask_preview(self) -> None:
        if self._mask_draft is None:
            return
        self._mask_boundary_preview_active = False
        self.dataset_view.set_mask_boundary_coordinates(
            self._mask_draft.boundary_coordinates
        )
        self.dataset_view.set_selection(self._mask_draft.selection)
        self._sync_mask_controls()
        self._refresh_manual_result_actions()

    def _sync_mask_controls(self, _group_index: int | None = None) -> None:
        draft = self._mask_draft
        self.mask_undo_button.setEnabled(draft is not None and draft.can_undo)
        self.mask_redo_button.setEnabled(draft is not None and draft.can_redo)
        self.mask_disable_auto_button.setEnabled(
            draft is not None and draft.can_disable_auto_mask,
        )
        self.mask_boundary_all_button.setEnabled(
            draft is not None
            and draft.can_apply_boundary_to_all_groups(
                self.dataset_view.current_group_index
            ),
        )

    def _refresh_context_label(self) -> None:
        if self._open_project is not None and self._open_dataset is not None:
            text = f"{self._open_project.name} · {self._open_dataset.name}"
        else:
            self.project_context_label.hide()
            return
        self.project_context_label.setText(text)
        self.project_context_label.setVisible(True)

    def _refresh_inspector_context(self, _group_index: int | None = None) -> None:
        if self._open_project is None or self._open_dataset is None:
            self.inspector_context_label.setText("Open a dataset to view its details.")
            self._inspector_source_details_available = False
            self.inspector_source_title.hide()
            self.inspector_source_metadata_label.hide()
            self.inspector_resolution_title.hide()
            self.inspector_resolution_label.hide()
            self.inspector_resolution_button.hide()
            self._sync_inspector_context_visibility()
            return
        role = (
            "Resolution data"
            if self._open_dataset.dataset.role is SpectrumRole.RESOLUTION
            else "Data"
        )
        self.inspector_context_label.setText(
            "\n".join(
                (
                    self._open_project.name,
                    self._open_dataset.name,
                    _dataset_source_detail(self._open_dataset),
                    f"{role} · {self.dataset_view.current_group_description}",
                )
            )
        )
        self.inspector_context_label.setToolTip(
            _dataset_source_tooltip(self._open_dataset)
        )
        metadata = self._open_dataset.dataset.source_metadata
        if metadata is None:
            self.inspector_source_title.hide()
            self.inspector_source_metadata_label.hide()
            details: tuple[str, ...] = ()
        else:
            details = tuple(
                f"{label}: {_compact_inspector_value(str(value))}"
                for label, value in (
                    ("File", metadata.source_filename),
                    ("Instrument", metadata.instrument),
                    ("Sample", metadata.sample),
                    ("Title", metadata.title),
                    (
                        "Temperature",
                        None
                        if metadata.temperature_kelvin is None
                        else f"{metadata.temperature_kelvin:g} K",
                    ),
                    (
                        "Wavelength",
                        None
                        if metadata.wavelength_angstrom is None
                        else f"{metadata.wavelength_angstrom:g} Å",
                    ),
                )
                if value is not None
            )
        if details:
            self._inspector_source_details_available = True
            self.inspector_source_title.show()
            self.inspector_source_metadata_label.setText("\n".join(details))
            self.inspector_source_metadata_label.show()
        else:
            self._inspector_source_details_available = False
            self.inspector_source_title.hide()
            self.inspector_source_metadata_label.hide()
        if self._open_dataset.dataset.role is not SpectrumRole.SAMPLE:
            self.inspector_resolution_title.hide()
            self.inspector_resolution_label.hide()
            self.inspector_resolution_button.hide()
            self._sync_inspector_context_visibility()
            return
        workflow = self._workflow_project_for(self._open_project)
        sample = self._workflow_dataset(workflow, self._open_dataset)
        association = next(
            (
                item
                for item in workflow.resolution_associations
                if item.sample_id == sample.dataset_id
            ),
            None,
        )
        self.inspector_resolution_title.show()
        self.inspector_resolution_label.show()
        self.inspector_resolution_button.show()
        self.inspector_resolution_button.setEnabled(
            not self._fit_all_q_mutation_blocked()
        )
        if association is None:
            self.inspector_resolution_label.setText("No Resolution applied")
            self.inspector_resolution_button.setText("Apply Resolution…")
        else:
            self.inspector_resolution_label.setText(
                self._dataset_name_for_workflow_id(
                    self._open_project,
                    association.resolution_id,
                ),
            )
            self.inspector_resolution_button.setText("Replace Resolution…")
        self._sync_inspector_context_visibility()

    def _prompt_import_files(self) -> None:
        """Choose candidate Data or standalone Q Method files for a Project."""

        project = self.workspace.current_project
        if project is None:
            self._show_import_target_required()
            return
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Import Data or Q Method Files",
            "",
            "Reduced data (*.dat *.txt *.csv);;All files (*)",
        )
        if not paths:
            return
        result = self._import_paths_with_busy_feedback(
            tuple(Path(path) for path in paths),
            project,
        )
        if len(paths) > 1 or result.failures:
            self._show_import_result(result)

    def _prompt_import_folder(self) -> None:
        """Choose a folder and content-route its supported text files."""

        project = self.workspace.current_project
        if project is None:
            self._show_import_target_required()
            return
        directory = QFileDialog.getExistingDirectory(
            self,
            "Import Reduced Data Folder",
        )
        if not directory:
            return
        try:
            paths = _supported_reduced_data_files(Path(directory))
        except OSError as error:
            self._show_import_result(
                ImportBatchResult(
                    (),
                    (ImportBatchFailure(Path(directory), str(error)),),
                ),
            )
            return
        self._show_import_result(self._import_paths_with_busy_feedback(paths, project))

    def _import_paths_with_busy_feedback(
        self,
        paths: tuple[Path, ...],
        project: ProjectState,
    ) -> ImportBatchResult:
        """Keep synchronous import visibly acknowledged without fake progress."""

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return self.import_paths(paths, project=project)
        finally:
            QApplication.restoreOverrideCursor()

    def _show_import_target_required(self) -> None:
        show_message_dialog(
            self,
            "Select a Project",
            "Select a Project in the Workspace before importing data.",
        )

    def _show_import_result(self, result: ImportBatchResult) -> None:
        """Show one compact summary, with detailed core diagnostics when needed."""

        if not result.failures:
            show_message_dialog(self, "Import", result.summary)
            return
        show_message_dialog(
            self,
            "Import",
            f"{result.summary}\n\nSome files could not be imported.\n\n"
            f"{_format_batch_failures(result.failures)}",
        )

    def _on_theme_changed(self, _scheme: object) -> None:
        self._apply_workspace_state_colors()
        self._apply_dataset_view_theme()
        self._apply_control_icons()
        self.workspace.set_resolution_icon_color(
            tokens_for(self._appearance_controller.current_scheme).accent,
        )
        self._sync_inspector_controls(self.toggle_inspector_action.isChecked())

    def _apply_layout_defaults(self) -> None:
        """Apply the static production dimensions for owned task controls."""

        tokens = DEFAULT_LAYOUT_TOKENS
        for control in (
            self.mask_boundary_button,
            self.mask_boundary_all_button,
            self.mask_rectangle_button,
            self.mask_lasso_button,
            self.mask_exclude_button,
            self.mask_restore_button,
            self.mask_undo_button,
            self.mask_redo_button,
            self.mask_disable_auto_button,
            self.mask_reset_group_button,
            self.mask_reset_all_button,
            self.mask_rerun_button,
            self.mask_save_button,
            self.mask_close_button,
        ):
            control.setMinimumHeight(tokens.control_height)
        self.mask_undo_button.setIconSize(
            QSize(tokens.control_icon_size, tokens.control_icon_size),
        )
        self.mask_redo_button.setIconSize(
            QSize(tokens.control_icon_size, tokens.control_icon_size),
        )
        self.inspector_button.setFixedHeight(tokens.control_height)
        self.inspector_button.setIconSize(
            QSize(tokens.control_icon_size, tokens.control_icon_size),
        )
        self._central_header_layout.setSpacing(tokens.row_spacing)
        self._inspector_layout.setSpacing(tokens.row_spacing)
        self.inspector_source_title.setContentsMargins(
            0,
            tokens.section_spacing,
            0,
            0,
        )
        self._mask_tool_layout.setSpacing(tokens.toolbar_spacing)
        self._mask_action_layout.setSpacing(tokens.toolbar_spacing)
        self._mask_task_layout.setSpacing(tokens.toolbar_spacing)

    def _apply_workspace_state_colors(self) -> None:
        tokens = tokens_for(self._appearance_controller.current_scheme)
        indicators = indicator_tokens_for(
            self._appearance_controller.current_scheme,
        )
        self.workspace.set_tree_state_colors(
            active_background=tokens.surface_selected,
            selection_background=tokens.surface_hover,
        )
        self.workspace.set_dataset_state_colors(
            required=indicators.error,
            ready=indicators.warning,
            partially_fit=indicators.accent,
            fully_fit=indicators.success,
            complete=tokens.text_secondary,
        )

    def _apply_dataset_view_theme(self) -> None:
        tokens = tokens_for(self._appearance_controller.current_scheme)
        indicators = indicator_tokens_for(self._appearance_controller.current_scheme)
        self.dataset_view.set_overview_theme(
            surface=tokens.surface_central,
            text=tokens.text_primary,
            border=tokens.canvas_boundary,
            accent=tokens.accent,
            success=indicators.success,
            warning=indicators.warning,
            danger=indicators.error,
            muted=tokens.text_muted,
        )

    def _apply_control_icons(self) -> None:
        """Apply the small token-colored icon set used alongside action labels."""

        color = tokens_for(self._appearance_controller.current_scheme).text_secondary
        self.workspace.set_control_icon_color(color)
        indicators = indicator_tokens_for(
            self._appearance_controller.current_scheme,
        )
        self.manual_fit_editor.set_icon_colors(
            neutral=color,
            accent=indicators.accent,
            amber=indicators.warning,
            violet=indicators.violet,
        )
        apply_disclosure_icon(
            self.inspector_dataset_toggle,
            expanded=self.inspector_dataset_toggle.isChecked(),
            color=color,
        )
        apply_disclosure_icon(
            self.inspector_source_toggle,
            expanded=self.inspector_source_toggle.isChecked(),
            color=color,
        )
        self.mask_undo_button.setIcon(load_icon(IconName.UNDO, color))
        self.mask_redo_button.setIcon(load_icon(IconName.REDO, color))

    def _on_dataset_role_changed(
        self,
        project: ProjectState,
        previous: DatasetState,
        current: DatasetState,
    ) -> None:
        affected_sessions = self._manual_sessions_using_dataset(project, previous)
        manual_context_affected = any(
            session is self._active_manual_session for session in affected_sessions
        )
        workflow = self._workflow_project_for(project)
        if current.dataset.role is not SpectrumRole.SAMPLE:
            self._preserved_q_sources.pop(
                (project, current.workflow_dataset_id),
                None,
            )
            self._sample_view_sessions.pop(
                (project, current.workflow_dataset_id),
                None,
            )
        if manual_context_affected:
            self._discard_pending_manual_interaction()
        for session in affected_sessions:
            if (
                session.owner[1] == previous.workflow_dataset_id
                and current.dataset.role is not SpectrumRole.SAMPLE
            ):
                self._retire_manual_editing(session)
                if not session.result_workspace.saved_results:
                    self._manual_sessions.pop(session.owner, None)
            elif (
                session.owner[1] == previous.workflow_dataset_id
                and current.dataset.role is SpectrumRole.SAMPLE
                and not session.draft_available
            ):
                session.resume(
                    open_manual_fit_draft(
                        workflow,
                        self._workflow_dataset(workflow, current),
                    )
                )
                session.execution_by_group.clear()
                session.draft_available = True
            else:
                self._invalidate_manual_session(session)
        if self._open_project is project and self._open_dataset is previous:
            self._open_dataset = current
            self.dataset_view.replace_dataset(
                current.dataset,
                overview_source=self._preserved_q_sources.get(
                    (project, current.workflow_dataset_id),
                ),
                allow_group_count_change=(
                    len(previous.dataset.spectra) != len(current.dataset.spectra)
                ),
            )
            self.dataset_view.set_selection(
                None if current.auto_mask is None else current.auto_mask.selection,
            )
            self.dataset_view.set_mask_edit_available(current.mask_editable)
            self.edit_mask_action.setEnabled(
                current.mask_editable and not self._fit_all_q_mutation_blocked()
            )
            available, _tooltip = self._manual_fit_capability(project, current)
            if not available:
                self.close_manual_fit(
                    discard_session=not any(
                        session.result_workspace.saved_results
                        for session in affected_sessions
                        if session.owner[1] == previous.workflow_dataset_id
                    )
                )
            self._sync_manual_fit_entry()
            self._refresh_context_label()
            self._refresh_inspector_context()
            self._refresh_manual_fit()
        elif manual_context_affected:
            self._refresh_inspector_context()
            self._refresh_manual_fit()

    def _on_dataset_added(
        self,
        project: ProjectState,
        _dataset: DatasetState,
    ) -> None:
        """Register imported dataset identity before any later replacement."""

        self._workflow_project_for(project)

    def _on_dataset_updated(
        self,
        project: ProjectState,
        previous: DatasetState,
        current: DatasetState,
    ) -> None:
        affected_sessions = self._manual_sessions_using_dataset(project, previous)
        manual_context_affected = any(
            session is self._active_manual_session for session in affected_sessions
        )
        self._workflow_project_for(project)
        selection_only_update = (
            current.dataset is previous.dataset
            and current.auto_mask is not previous.auto_mask
        )
        safe_unit_replacement = (
            current.dataset.role is previous.dataset.role
            and _source_unit_identity(current.dataset)
            != _source_unit_identity(previous.dataset)
            and measured_point_correspondence_exactly_unchanged(
                previous.dataset,
                current.dataset,
            )
        )
        changed_selection_groups: tuple[int, ...] | None = None
        if selection_only_update:
            self._commit_explicit_fitting_selection(project, current)
            before_selection = (
                None if previous.auto_mask is None else previous.auto_mask.selection
            )
            after_selection = (
                None if current.auto_mask is None else current.auto_mask.selection
            )
            changed_selection_groups = _changed_fitting_selection_groups(
                before_selection,
                after_selection,
                len(current.dataset.spectra),
            )
        elif safe_unit_replacement:
            # Unit correction intentionally replaces dataset identity. The
            # Workspace has already rebound the point-indexed selection after
            # exact measured correspondence was proven, so recommit that fresh
            # selection only on this explicit safe metadata path.
            self._commit_explicit_fitting_selection(project, current)
        if manual_context_affected:
            self._discard_pending_manual_interaction()
        for session in affected_sessions:
            group_indices = (
                changed_selection_groups
                if selection_only_update
                and session.owner[1] == previous.workflow_dataset_id
                else None
            )
            if session.owner[1] == previous.workflow_dataset_id and (
                current.dataset.role is not SpectrumRole.SAMPLE
                or len(session.draft.setups) != len(current.dataset.spectra)
            ):
                self._retire_manual_editing(session)
            else:
                self._invalidate_manual_session(session, group_indices)
        if self._open_project is project and self._open_dataset is previous:
            self._open_dataset = current
            if (
                self._mask_draft_owner is previous
                and current.dataset is previous.dataset
            ):
                self._mask_draft_owner = current
            elif self._mask_draft_owner is previous:
                # Every supported scientific replacement is preflighted above.
                # Do not retain a draft whose selection is bound to old arrays.
                self._teardown_mask_task()
            self.dataset_view.replace_dataset(
                current.dataset,
                overview_source=self._preserved_q_sources.get(
                    (project, current.workflow_dataset_id),
                ),
                allow_group_count_change=(
                    len(previous.dataset.spectra) != len(current.dataset.spectra)
                ),
            )
            self.dataset_view.set_selection(
                None if current.auto_mask is None else current.auto_mask.selection,
            )
            self.dataset_view.set_mask_edit_available(current.mask_editable)
            self.edit_mask_action.setEnabled(
                current.mask_editable and not self._fit_all_q_mutation_blocked()
            )
            available, _tooltip = self._manual_fit_capability(project, current)
            if not available and self._active_manual_session is not None:
                session = self._active_manual_session
                self._retire_manual_editing(session)
                self.close_manual_fit(
                    discard_session=not session.result_workspace.saved_results
                )
            self._refresh_context_label()
            self._refresh_inspector_context()
            self._refresh_manual_fit()
        elif manual_context_affected:
            self._refresh_inspector_context()
            self._refresh_manual_fit()

    def _on_dataset_renamed(
        self,
        project: ProjectState,
        previous: DatasetState,
        current: DatasetState,
    ) -> None:
        if self._open_project is project and self._open_dataset is previous:
            self._open_dataset = current
            if self._mask_draft_owner is previous:
                self._mask_draft_owner = current
            self._refresh_context_label()
            self._refresh_inspector_context()

    def _on_project_renamed(self, project: ProjectState) -> None:
        if self._open_project is project:
            self._refresh_context_label()
            self._refresh_inspector_context()

    def _configure_native_window_chrome(self) -> None:
        """Use Qt's safe native title integration where macOS supports it."""
        if sys.platform == "darwin":
            self.setUnifiedTitleAndToolBarOnMac(True)
            self.setWindowTitle("")
        else:
            self.setWindowTitle("ezQENS")

    def _show_about(self) -> None:
        show_message_dialog(
            self,
            "About ezQENS",
            "ezQENS\nStandardized analysis of reduced QENS data.",
        )


def _source_unit_identity(
    dataset: ReducedDataset,
) -> tuple[tuple[str, str, str], ...]:
    """Return only declared measured-unit identity for safe metadata routing."""

    return tuple(
        (
            spectrum.energy_unit,
            spectrum.intensity_unit,
            spectrum.uncertainty_unit,
        )
        for spectrum in dataset.spectra
    )


def _manual_interaction_blocker_message(
    diagnostics: tuple[WorkflowDiagnostic, ...],
) -> str:
    """Present typed interaction blockers as concise user actions."""

    actions = {
        WorkflowDiagnosticCode.FITTING_SELECTION_UNAVAILABLE: (
            "Prepare or save a usable Mask for this Sample before adding this "
            "fitting function."
        ),
        WorkflowDiagnosticCode.NO_APPLIED_RESOLUTION: (
            "Apply a Resolution to this Sample before adding Elastic or Lorentzian."
        ),
        WorkflowDiagnosticCode.ASSOCIATED_RESOLUTION_INVALID: (
            "Reapply a valid Resolution to this Sample before adding this fitting "
            "function."
        ),
        WorkflowDiagnosticCode.RESOLUTION_PREPARATION_FAILED: (
            "The applied Resolution cannot be prepared for this Sample and Group."
        ),
        WorkflowDiagnosticCode.SAMPLE_ROLE_REQUIRED: (
            "Open a Sample dataset before adding this fitting function."
        ),
        WorkflowDiagnosticCode.INVALID_GROUP: (
            "Select a valid Group before adding this fitting function."
        ),
    }
    messages = tuple(actions.get(item.code, item.message) for item in diagnostics)
    return "\n".join(dict.fromkeys(messages))


def _format_import_diagnostics(diagnostics: tuple[ImportDiagnostic, ...]) -> str:
    lines: list[str] = []
    for diagnostic in diagnostics:
        location = []
        if diagnostic.group is not None:
            location.append(f"group {diagnostic.group}")
        if diagnostic.row is not None:
            location.append(f"row {diagnostic.row}")
        if diagnostic.column is not None:
            location.append(f"column {diagnostic.column}")
        context = f" ({', '.join(location)})" if location else ""
        lines.append(
            f"{diagnostic.severity.value.title()}: {diagnostic.message}{context}"
        )
    return "\n".join(lines)


def _format_batch_failures(failures: tuple[ImportBatchFailure, ...]) -> str:
    """Keep file context attached to the core diagnostic for batch inspection."""

    return "\n\n".join(f"{failure.path.name}\n{failure.detail}" for failure in failures)


def _compact_inspector_value(value: str, limit: int = 72) -> str:
    """Keep optional Source metadata compact in the narrow Inspector column."""

    text = value.strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _supported_reduced_data_files(directory: Path) -> tuple[Path, ...]:
    """Return candidate text files for the parser-backed import router."""

    return tuple(
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.casefold() in SUPPORTED_REDUCED_DATA_SUFFIXES
    )


def _dataset_source_detail(dataset: DatasetState) -> str:
    source = dataset.source_path or dataset.dataset.source_reference
    if source is None:
        return "Source unavailable"
    return f"Source: {Path(source).name}"


def _dataset_source_tooltip(dataset: DatasetState) -> str:
    """Return the unshortened source reference for the Inspector tooltip."""

    source = dataset.source_path or dataset.dataset.source_reference
    return "" if source is None else f"Source: {source}"
