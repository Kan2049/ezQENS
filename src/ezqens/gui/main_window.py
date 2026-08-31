"""Native main-window shell for ezQENS."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRect, QSignalBlocker, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
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
    ComponentIdentity,
    FitResult,
    ManualModelPreview,
    ParameterMetadata,
    ParameterReference,
    manual_parameter_metadata,
)
from ezqens.gui import masking
from ezqens.gui.dataset_view import ReducedDatasetView
from ezqens.gui.dialogs import (
    DialogChoice,
    choose_dialog,
    confirm_dialog,
    show_message_dialog,
)
from ezqens.gui.icons import IconName, apply_disclosure_icon, load_icon
from ezqens.gui.manual_fit import ManualFitEditor, ManualFitLifecycle
from ezqens.gui.masking import MaskTaskDraft
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
    ManualComponentKind,
    ManualFitDraft,
    ManualParameterEdit,
    PendingManualInteraction,
    ProjectDataset,
    WorkflowDiagnostic,
    WorkflowError,
    WorkflowProject,
    add_project_dataset,
    apply_resolution,
    begin_component_interaction,
    commit_fitting_selection,
    complete_background_interaction,
    complete_elastic_interaction,
    complete_lorentzian_interaction,
    create_parameter_tie,
    create_project,
    join_parameter_tie,
    manual_workflow_readiness,
    materialize_manual_setup,
    open_manual_fit_draft,
    preflight_apply_resolution,
    preview_manual_fit,
    preview_pending_background_interaction,
    preview_pending_elastic_interaction,
    preview_pending_lorentzian_interaction,
    remove_manual_component,
    remove_project_dataset,
    replace_project_dataset,
    run_and_adopt_manual_fit,
    untie_parameter,
    update_manual_parameter,
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
    """Execution-only state owned independently by one Sample Group."""

    lifecycle: ManualFitLifecycle = ManualFitLifecycle.READY
    fit_result: FitResult | None = None
    diagnostics: tuple[WorkflowDiagnostic, ...] = ()


@dataclass(eq=False, slots=True)
class ManualFitSession:
    """One authoritative in-session Manual working state for one Sample."""

    owner: tuple[ProjectState, str]
    draft: ManualFitDraft
    group_index: int
    execution_by_group: dict[int, ManualFitExecutionState] = field(default_factory=dict)
    editor_expanded: bool = True
    result_expanded: bool = False

    def execution_state(
        self,
        group_index: int | None = None,
    ) -> ManualFitExecutionState:
        """Return the retained execution state for one validated draft Group."""

        target = self.group_index if group_index is None else group_index
        self.draft.setup(target)
        return self.execution_by_group.setdefault(target, ManualFitExecutionState())

    @property
    def lifecycle(self) -> ManualFitLifecycle:
        return self.execution_state().lifecycle

    @lifecycle.setter
    def lifecycle(self, lifecycle: ManualFitLifecycle) -> None:
        self.execution_state().lifecycle = lifecycle

    @property
    def fit_result(self) -> FitResult | None:
        return self.execution_state().fit_result

    @fit_result.setter
    def fit_result(self, result: FitResult | None) -> None:
        self.execution_state().fit_result = result

    @property
    def execution_diagnostics(self) -> tuple[WorkflowDiagnostic, ...]:
        return self.execution_state().diagnostics

    @execution_diagnostics.setter
    def execution_diagnostics(
        self,
        diagnostics: tuple[WorkflowDiagnostic, ...],
    ) -> None:
        self.execution_state().diagnostics = diagnostics


@dataclass(slots=True)
class SampleViewSession:
    """Presentation-only Spectrum preferences retained for one Sample."""

    y_scale: str = "linear"


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
        self._active_manual_session.draft = draft

    @property
    def _manual_fit_result(self) -> FitResult | None:
        session = self._active_manual_session
        return None if session is None else session.fit_result

    @_manual_fit_result.setter
    def _manual_fit_result(self, result: FitResult | None) -> None:
        if self._active_manual_session is None:
            if result is None:
                return
            raise RuntimeError("Manual result assignment requires an active session")
        self._active_manual_session.fit_result = result

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
        self._manual_sessions: dict[tuple[ProjectState, str], ManualFitSession] = {}
        self._active_manual_session: ManualFitSession | None = None
        self._sample_view_sessions: dict[
            tuple[ProjectState, str], SampleViewSession
        ] = {}
        self._pending_manual_interaction: PendingManualInteraction | None = None
        self._manual_preview: ManualModelPreview | None = None
        self.workspace.set_scientific_replacement_resolver(
            self._resolve_scientific_dataset_replacement,
        )
        self.workspace.set_fitting_started_resolver(self._dataset_fitting_started)
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
        self.workspace.dataset_open_requested.connect(self.open_dataset)
        self.workspace.dataset_added.connect(self._on_dataset_added)
        self.workspace.dataset_role_changed.connect(self._on_dataset_role_changed)
        self.workspace.dataset_role_change_requested.connect(self.change_dataset_role)
        self.workspace.dataset_renamed.connect(self._on_dataset_renamed)
        self.workspace.dataset_updated.connect(self._on_dataset_updated)
        self.workspace.units_requested.connect(self.show_units_editor)
        self.workspace.q_assignment_requested.connect(self.show_q_editor)
        self.workspace.mask_edit_requested.connect(self.show_mask_editor)
        self.workspace.manual_fit_requested.connect(self.show_manual_fit)
        self.workspace.apply_resolution_requested.connect(
            self.apply_resolution_for_sample
        )
        self.workspace.q_method_dropped.connect(self._apply_q_method)
        self.workspace.project_removal_requested.connect(self.remove_project)
        self.workspace.dataset_removal_requested.connect(self.remove_dataset)
        self.workspace.q_method_removal_requested.connect(self.remove_q_method)
        self.dataset_view.group_changed.connect(self._refresh_inspector_context)
        self.dataset_view.group_changed.connect(self._sync_mask_controls)
        self.dataset_view.y_scale_changed.connect(self._on_y_scale_changed)
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
        self.dataset_view.mask_operation_feedback_changed.connect(
            self._set_mask_operation_feedback,
        )
        self.dataset_view.q_editor.set_apply_handler(self._apply_q_bins)
        self.dataset_view.q_editor.save_q_method_requested.connect(self._save_q_method)
        self.dataset_view.q_editor.q_file_imported.connect(
            self._save_imported_q_method,
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
        """Release the application-level appearance listener when closing."""

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
            canvas_width = self.scientific_canvas.width()
            desired_width = max(
                self.inspector.minimumWidth(),
                self._last_inspector_width,
            )
            self._inspector_expansion_width = self._expand_for_inspector()
            self.inspector.show()
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
            self._reverse_inspector_expansion()
        self._sync_inspector_controls(visible)

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

        self.manual_fit_action = QAction("Manual Fit…", self)
        self.manual_fit_action.setEnabled(False)
        self.manual_fit_action.triggered.connect(
            lambda _checked=False: self.show_manual_fit(),
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
        analysis_menu.addAction(self.edit_mask_action)
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
        central_workspace.setMinimumWidth(320)
        central_header = QFrame()
        central_header.setObjectName("centralHeader")
        central_header.setFixedHeight(38)

        self.project_context_label = QLabel()
        self.project_context_label.setObjectName("projectContextLabel")
        self.project_context_label.setVisible(False)

        self.manual_fit_button = QToolButton()
        self.manual_fit_button.setObjectName("centralManualFitButton")
        self.manual_fit_button.setText("Manual Fit")
        self.manual_fit_button.setCheckable(True)
        self.manual_fit_button.setEnabled(False)
        self.manual_fit_button.setToolTip(
            "Open Single-Q Manual Fit for the currently open dataset",
        )
        self.manual_fit_button.clicked.connect(
            lambda _checked=False: self.show_manual_fit(),
        )

        self.inspector_button = QToolButton()
        self.inspector_button.setObjectName("inspectorButton")
        self.inspector_button.setCheckable(True)
        self.inspector_button.setFixedSize(26, 26)
        self.inspector_button.toggled.connect(self.set_inspector_visible)

        header_layout = QHBoxLayout(central_header)
        header_layout.setContentsMargins(12, 5, 8, 5)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.project_context_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.manual_fit_button)
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

    def _reverse_inspector_expansion(self) -> None:
        if not self._inspector_expansion_width or self.isMaximized():
            self._inspector_expansion_width = 0
            return
        restored_width = max(
            self.minimumWidth(),
            self.width() - self._inspector_expansion_width,
        )
        self.resize(restored_width, self.height())
        self._inspector_expansion_width = 0

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
        tooltip = "Hide Inspector" if visible else "Show Inspector"
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
        for reference in tuple(workflow.datasets):
            state = states.pop(reference.dataset_id, None)
            if state is None:
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
        return session is not None and any(
            setup.model is not None for setup in session.draft.setups
        )

    def _store_open_sample_view_state(self) -> None:
        """Retain the current display-only Y scale under stable Sample identity."""

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
        self._sample_view_sessions.setdefault(
            owner, SampleViewSession()
        ).y_scale = self.dataset_view.y_scale

    def _on_y_scale_changed(self, scale: str) -> None:
        """Write through one display-only scale change to the open Sample session."""

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
        self._sample_view_sessions.setdefault(
            owner, SampleViewSession()
        ).y_scale = scale

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
        session = self._manual_sessions.get(owner)
        view_session = (
            self._sample_view_sessions.setdefault(owner, SampleViewSession())
            if dataset.dataset.role is SpectrumRole.SAMPLE
            else None
        )
        self._open_project = project
        self._open_dataset = dataset
        self.dataset_view.open_dataset(
            dataset.dataset,
            group_index=0 if session is None else session.group_index,
            y_scale="linear" if view_session is None else view_session.y_scale,
        )
        self.dataset_view.set_selection(
            None if dataset.auto_mask is None else dataset.auto_mask.selection,
        )
        self.dataset_view.set_mask_edit_available(dataset.mask_editable)
        self.workspace.set_active_dataset(project, dataset)
        self.set_scientific_figure_visible(True)
        self.edit_mask_action.setEnabled(dataset.mask_editable)
        self._sync_manual_fit_entry()
        self._refresh_context_label()
        self._refresh_inspector_context()
        if session is not None:
            self._activate_manual_session(session)
        return True

    def show_manual_fit(
        self,
        project: ProjectState | None = None,
        dataset: DatasetState | None = None,
    ) -> None:
        """Open the group-local Manual Fit draft without requiring a Resolution."""

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
        if session is None:
            try:
                draft = open_manual_fit_draft(
                    workflow,
                    self._workflow_dataset(workflow, target_dataset),
                )
            except WorkflowError as error:
                self._sync_manual_fit_entry()
                self._show_workflow_error("Manual Fit", error)
                return
            session = ManualFitSession(
                owner,
                draft,
                self.dataset_view.current_group_index,
            )
            self._manual_sessions[owner] = session
            self.manual_fit_editor.reset_result_disclosure()
            self._manual_inspector_minimum_width = INSPECTOR_BASE_MINIMUM_WIDTH
        self._activate_manual_session(session, expand=True)

    def _activate_manual_session(
        self,
        session: ManualFitSession,
        *,
        expand: bool = False,
    ) -> None:
        """Bind one retained Sample session to the current editor and plot view."""

        self._active_manual_session = session
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
        self._workflow_projects[target_project] = updated_workflow
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
        self._refresh_inspector_context()
        self._refresh_manual_fit()
        return True

    def _invalidate_manual_session(
        self,
        session: ManualFitSession,
        group_indices: Iterable[int] | None = None,
    ) -> None:
        """Mark the affected Groups in one retained Sample session stale."""

        targets = (
            range(len(session.draft.setups)) if group_indices is None else group_indices
        )
        for group_index in targets:
            execution = session.execution_state(group_index)
            if execution.lifecycle is ManualFitLifecycle.CURRENT:
                execution.lifecycle = ManualFitLifecycle.NEEDS_FIT
            execution.fit_result = None
            execution.diagnostics = ()

    def _invalidate_manual_fit(self) -> None:
        """Mark only the active Group's retained fitted state stale."""

        if self._active_manual_session is not None:
            self._invalidate_manual_session(
                self._active_manual_session,
                (self.dataset_view.current_group_index,),
            )

    def _run_manual_fit(self) -> None:
        """Run and atomically adopt one group through the public workflow seam."""

        if (
            self._manual_fit_lifecycle is ManualFitLifecycle.FITTING
            or self._manual_draft is None
            or self._open_project is None
        ):
            return
        previous_lifecycle = self._manual_fit_lifecycle
        self._discard_pending_manual_interaction()
        self._manual_fit_lifecycle = ManualFitLifecycle.FITTING
        self._manual_fit_result = None
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
            self._manual_draft = outcome.adopted_draft
            self._manual_fit_result = outcome.fit_result
            self._manual_fit_lifecycle = ManualFitLifecycle.CURRENT
        else:
            self._manual_fit_result = None
            self._manual_fit_lifecycle = (
                ManualFitLifecycle.NEEDS_FIT
                if previous_lifecycle is ManualFitLifecycle.NEEDS_FIT
                else ManualFitLifecycle.READY
            )
        self.dataset_view.set_manual_fit_result(self._manual_fit_result)
        self._refresh_manual_fit()

    def _begin_manual_component_interaction(self, kind: ManualComponentKind) -> None:
        if self._manual_draft is None:
            return
        group_index = self.dataset_view.current_group_index
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
            self._show_workflow_error("Add Manual Function", error)
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
            self._show_workflow_error("Manual Fit", error)
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit()
        self._refresh_manual_fit()

    def _handle_manual_chain(self, reference: ParameterReference) -> None:
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
                else:
                    updated = create_parameter_tie(
                        self._manual_draft,
                        group_index,
                        (reference,),
                        source_member=reference,
                    )
        except WorkflowError as error:
            self._show_workflow_error("Manual Fit", error)
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit()
        self._refresh_manual_fit()

    def _begin_new_manual_tie_group(self, reference: ParameterReference) -> None:
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
            self._show_workflow_error("Manual Fit", error)
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit()
        self._refresh_manual_fit()

    def _join_manual_tie(self, reference: ParameterReference, group_id: str) -> None:
        self._discard_pending_manual_interaction()
        if self._manual_draft is None:
            return
        try:
            updated = join_parameter_tie(
                self._manual_draft,
                self.dataset_view.current_group_index,
                group_id,
                reference,
            )
        except WorkflowError as error:
            self._show_workflow_error("Manual Fit", error)
        else:
            self._manual_draft = updated
            self._invalidate_manual_fit()
        self._refresh_manual_fit()

    def _remove_manual_component(self, identity: ComponentIdentity) -> None:
        self._apply_manual_component_removal(identity, error_title="Manual Fit")

    def _apply_manual_component_removal(
        self,
        identity: ComponentIdentity,
        *,
        error_title: str,
    ) -> bool:
        """Apply the one authoritative component-removal lifecycle."""

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

        if self._manual_draft is None:
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
        self._refresh_manual_fit()

    def _refresh_manual_fit(self) -> None:
        if (
            self._manual_draft is None
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
        self.manual_fit_editor.set_state(
            model,
            metadata,
            readiness,
            lifecycle=self._manual_fit_lifecycle,
            execution_diagnostics=self._manual_execution_diagnostics,
            fit_result=self._manual_fit_result,
        )
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
            self._manual_fit_result,
        )
        self.workspace.refresh_dataset_analysis(self._open_project)

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
        if self._open_project is project:
            self._clear_open_context()
        self._workflow_projects.pop(project, None)
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

        if not confirmed:
            accepted = confirm_dialog(
                self,
                "Remove from Project",
                "Remove this dataset from the ezQENS Project? Its source file will "
                "not be deleted from disk.",
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
        if self._open_project is None or self._open_dataset is None:
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
        self.dataset_view.set_selection(self._mask_draft.selection)
        self.dataset_view.set_mask_inspection_mode(True)
        self._set_mask_tool("boundary", True)
        self._set_mask_operation("exclude", True)
        self.mask_task_bar.show()
        self._sync_mask_controls()

    def save_mask_task(self) -> bool:
        """Commit the current local draft into in-memory project state."""

        if (
            self._mask_draft is None
            or self._open_project is None
            or self._open_dataset is None
            or self._mask_draft_owner is not self._open_dataset
            or self._mask_draft.selection.dataset is not self._open_dataset.dataset
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
        self._sync_mask_controls()
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

        self.dataset_view.set_mask_inspection_mode(False)
        self.dataset_view.set_mask_tool(None)
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

    def rerun_auto_mask(self, confirmed: bool = False) -> bool:
        """Replace the proposal after confirmation clears manual edits."""

        if (
            self._open_project is None
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
        except ValueError as error:
            show_message_dialog(self, "Re-run AutoMask", str(error))
            return False
        self.dataset_view.set_selection(self._mask_draft.selection)
        self._sync_mask_controls()
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
        return True, "Open Single-Q Manual Fit for the currently open dataset"

    def _sync_manual_fit_entry(self) -> None:
        """Keep menu and central task entry bound only to the open dataset."""

        available = False
        tooltip = "Open a reduced dataset to use Manual Fit"
        if self._open_project is not None and self._open_dataset is not None:
            available, tooltip = self._manual_fit_capability(
                self._open_project,
                self._open_dataset,
            )
        self.manual_fit_action.setEnabled(available)
        if not hasattr(self, "manual_fit_button"):
            return
        self.manual_fit_button.setEnabled(available)
        self.manual_fit_button.setToolTip(tooltip)
        self.manual_fit_button.setChecked(
            self._manual_draft is not None
            and self._open_project is not None
            and self._open_dataset is not None
            and self._manual_owner
            == (self._open_project, self._open_dataset.workflow_dataset_id),
        )

    def _set_mask_tool(self, tool: str, enabled: bool) -> None:
        if self._mask_draft is None:
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

        if self._mask_draft is None:
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
        if self._mask_draft is None:
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
        if self._mask_draft is None:
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

        if self._mask_draft is None:
            return
        try:
            selection = self._mask_draft.preview_auto_boundary(
                group_index,
                side=BoundarySide(str(side)),
                energy=energy,
            )
        except ValueError:
            return
        self.dataset_view.set_selection(selection)

    def _edit_auto_boundary(
        self,
        group_index: int,
        side: object,
        energy: float,
    ) -> None:
        if self._mask_draft is None:
            return
        boundary_side = BoundarySide(str(side))
        if self._mask_draft.set_auto_boundary(
            group_index,
            side=boundary_side,
            energy=energy,
        ):
            self._refresh_mask_preview()
        else:
            self.dataset_view.set_selection(self._mask_draft.selection)

    def _apply_mask_boundary_to_all_groups(
        self,
        *,
        confirmed: bool = False,
    ) -> bool:
        """Apply the current energy boundary to every Group as one Mask edit."""

        if self._mask_draft is None:
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
        if self._mask_draft is None:
            return
        self._mask_draft.undo()
        self._refresh_mask_preview()

    def _redo_mask(self) -> None:
        if self._mask_draft is None:
            return
        self._mask_draft.redo()
        self._refresh_mask_preview()

    def _disable_auto_mask(self) -> None:
        """Record valid AUTO re-inclusions across all Groups in the local draft."""

        if self._mask_draft is not None and self._mask_draft.disable_auto_mask():
            self._refresh_mask_preview()

    def _reset_mask_group(self) -> None:
        if self._mask_draft is None:
            return
        group_index = self.dataset_view.current_group_index
        if self._mask_draft.reset_group(group_index):
            self._refresh_mask_preview()

    def _reset_mask_all(self, confirmed: bool = False) -> bool:
        """Restore all Groups to the existing proposal after explicit confirmation."""

        if self._mask_draft is None:
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
        self.dataset_view.set_selection(self._mask_draft.selection)
        self._sync_mask_controls()

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
        self.inspector_button.setFixedSize(
            tokens.control_height,
            tokens.control_height,
        )
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
        self.dataset_view.set_overview_theme(
            surface=tokens.surface_central,
            text=tokens.text_primary,
            border=tokens.canvas_boundary,
            accent=tokens.accent,
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
        self._workflow_project_for(project)
        if current.dataset.role is not SpectrumRole.SAMPLE:
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
                if session is not self._active_manual_session:
                    self._manual_sessions.pop(session.owner, None)
            else:
                self._invalidate_manual_session(session)
        if self._open_project is project and self._open_dataset is previous:
            self._open_dataset = current
            self.dataset_view.replace_dataset(current.dataset)
            self.dataset_view.set_selection(
                None if current.auto_mask is None else current.auto_mask.selection,
            )
            self.dataset_view.set_mask_edit_available(current.mask_editable)
            self.edit_mask_action.setEnabled(current.mask_editable)
            available, _tooltip = self._manual_fit_capability(project, current)
            if not available:
                self.close_manual_fit(discard_session=True)
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
        if manual_context_affected:
            self._discard_pending_manual_interaction()
        for session in affected_sessions:
            group_indices = (
                changed_selection_groups
                if selection_only_update
                and session.owner[1] == previous.workflow_dataset_id
                else None
            )
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
            self.dataset_view.replace_dataset(current.dataset)
            self.dataset_view.set_selection(
                None if current.auto_mask is None else current.auto_mask.selection,
            )
            self.dataset_view.set_mask_edit_available(current.mask_editable)
            self.edit_mask_action.setEnabled(current.mask_editable)
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
