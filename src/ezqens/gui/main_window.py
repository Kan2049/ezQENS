"""Native main-window shell for ezQENS."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QRect, QSignalBlocker, QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
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
from ezqens.gui import masking
from ezqens.gui.dataset_view import ReducedDatasetView
from ezqens.gui.dialogs import (
    DialogChoice,
    choose_dialog,
    confirm_dialog,
    show_message_dialog,
)
from ezqens.gui.icons import IconName, load_icon
from ezqens.gui.masking import MaskTaskDraft
from ezqens.gui.theme import (
    DEFAULT_LAYOUT_TOKENS,
    Appearance,
    application_appearance_controller,
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
from ezqens.preprocessing import BoundarySide

INSPECTOR_PREFERRED_WIDTH = 280
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
        self.workspace.set_scientific_replacement_resolver(
            self._resolve_scientific_dataset_replacement,
        )
        self.inspector = self._build_inspector()
        self._create_actions()
        self.central_workspace = self._build_central_workspace()
        self._apply_control_icons()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("workspaceSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.workspace)
        self.splitter.addWidget(self.central_workspace)
        self.splitter.addWidget(self.inspector)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([230, 760, INSPECTOR_PREFERRED_WIDTH])
        self.inspector.hide()

        self._create_menus()
        self.workspace.project_created.connect(self._set_active_project)
        self.workspace.project_selected.connect(self._set_active_project)
        self.workspace.project_renamed.connect(self._on_project_renamed)
        self.workspace.import_files_requested.connect(self._prompt_import_files)
        self.workspace.import_folder_requested.connect(self._prompt_import_folder)
        self.workspace.dataset_open_requested.connect(self.open_dataset)
        self.workspace.dataset_role_changed.connect(self._on_dataset_role_changed)
        self.workspace.dataset_role_change_requested.connect(self.change_dataset_role)
        self.workspace.dataset_renamed.connect(self._on_dataset_renamed)
        self.workspace.dataset_updated.connect(self._on_dataset_updated)
        self.workspace.units_requested.connect(self.show_units_editor)
        self.workspace.q_assignment_requested.connect(self.show_q_editor)
        self.workspace.mask_edit_requested.connect(self.show_mask_editor)
        self.workspace.q_method_dropped.connect(self._apply_q_method)
        self.workspace.project_removal_requested.connect(self.remove_project)
        self.workspace.dataset_removal_requested.connect(self.remove_dataset)
        self.workspace.q_method_removal_requested.connect(self.remove_q_method)
        self.dataset_view.group_changed.connect(self._refresh_inspector_context)
        self.dataset_view.q_assignment_requested.connect(self.show_q_editor)
        self.dataset_view.units_requested.connect(self.show_units_editor)
        self.dataset_view.q_method_dropped.connect(self._apply_dragged_q_method)
        self.dataset_view.mask_edit_requested.connect(self.enter_mask_task)
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
        if visible == self.inspector.isVisible():
            self._sync_inspector_controls(visible)
            return

        if visible:
            canvas_width = self.scientific_canvas.width()
            self._inspector_expansion_width = self._expand_for_inspector()
            self.inspector.show()
            self.splitter.setSizes(
                [
                    max(self.workspace.width(), 200),
                    max(canvas_width, 320),
                    INSPECTOR_PREFERRED_WIDTH,
                ],
            )
        else:
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
        self.edit_mask_action.triggered.connect(self.enter_mask_task)

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
        central_header = QFrame()
        central_header.setObjectName("centralHeader")
        central_header.setFixedHeight(38)

        self.project_context_label = QLabel()
        self.project_context_label.setObjectName("projectContextLabel")
        self.project_context_label.setVisible(False)

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
        self.mask_reset_all_button.clicked.connect(self._reset_mask_all)
        self.mask_rerun_button = QToolButton()
        self.mask_rerun_button.setText("Re-run AutoMask…")
        self.mask_rerun_button.clicked.connect(self.rerun_auto_mask)
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
        inspector.setMinimumWidth(220)
        inspector.setMaximumWidth(480)

        title = QLabel("Inspector")
        title.setObjectName("inspectorTitle")
        self.inspector_context_label = QLabel("Open a dataset to view its details.")
        self.inspector_context_label.setObjectName("inspectorContextLabel")
        self.inspector_context_label.setProperty("secondary", True)
        self.inspector_context_label.setWordWrap(True)
        self.inspector_source_title = QLabel("Source")
        self.inspector_source_title.setObjectName("inspectorSourceTitle")
        self.inspector_source_title.hide()
        self.inspector_source_metadata_label = QLabel()
        self.inspector_source_metadata_label.setObjectName("inspectorSourceMetadata")
        self.inspector_source_metadata_label.setProperty("secondary", True)
        self.inspector_source_metadata_label.setWordWrap(True)
        self.inspector_source_metadata_label.hide()

        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(5)
        layout.addWidget(title)
        layout.addWidget(self.inspector_context_label)
        layout.addWidget(self.inspector_source_title)
        layout.addWidget(self.inspector_source_metadata_label)
        layout.addStretch(1)
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
        """Open an imported Workspace dataset at Group 1."""

        self.workspace.validate_dataset_membership(project, dataset)
        if (
            project is not self._open_project or dataset is not self._open_dataset
        ) and not self._resolve_mask_task_transition(mask_task_decision):
            return False
        self._open_project = project
        self._open_dataset = dataset
        self.dataset_view.open_dataset(dataset.dataset)
        self.dataset_view.set_selection(
            None if dataset.auto_mask is None else dataset.auto_mask.selection,
        )
        self.dataset_view.set_mask_edit_available(dataset.mask_editable)
        self.workspace.set_active_dataset(project, dataset)
        self.set_scientific_figure_visible(True)
        self.edit_mask_action.setEnabled(dataset.mask_editable)
        self._refresh_context_label()
        self._refresh_inspector_context()
        return True

    def show_units_editor(
        self,
        project: ProjectState | None = None,
        dataset: DatasetState | None = None,
    ) -> SourceUnitsDialog | None:
        """Open the small source-metadata editor without importing or converting."""

        target_project, target_dataset = self._editing_target(project, dataset)
        if target_project is None or target_dataset is None:
            return None
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
        self.workspace.remove_dataset(project, dataset)
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
        self.dataset_view.set_mask_edit_available(False)
        self.dataset_view.clear_dataset()
        self.workspace.clear_active_dataset()
        self._open_project = None
        self._open_dataset = None
        self.edit_mask_action.setEnabled(False)
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

    def enter_mask_task(self, checked: bool = True) -> None:
        """Enter the familiar viewer-based Mask task with a local draft."""

        if not checked and self._mask_draft is not None:
            return
        if self._mask_draft is not None:
            return
        if self._open_dataset is None or not self._open_dataset.mask_editable:
            return
        auto_mask = self._open_dataset.auto_mask
        if auto_mask is None:
            return
        try:
            self._mask_draft = MaskTaskDraft(auto_mask)
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
            MaskTaskDraft(state)
            current = self.workspace.update_auto_mask(
                self._open_project,
                self._open_dataset,
                state,
            )
            self._open_dataset = current
            if current.auto_mask is None:
                return False
            self._mask_draft = MaskTaskDraft(current.auto_mask)
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

    def _sync_mask_controls(self) -> None:
        draft = self._mask_draft
        self.mask_undo_button.setEnabled(draft is not None and draft.can_undo)
        self.mask_redo_button.setEnabled(draft is not None and draft.can_redo)
        self.mask_disable_auto_button.setEnabled(
            draft is not None and draft.can_disable_auto_mask,
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
            self.inspector_source_title.hide()
            self.inspector_source_metadata_label.hide()
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
            return
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
            self.inspector_source_title.show()
            self.inspector_source_metadata_label.setText("\n".join(details))
            self.inspector_source_metadata_label.show()
        else:
            self.inspector_source_title.hide()
            self.inspector_source_metadata_label.hide()

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
        self.workspace.set_tree_state_colors(
            active_background=tokens.surface_selected,
            selection_background=tokens.surface_hover,
        )
        self.workspace.set_dataset_state_colors(
            required=tokens.danger,
            ready=tokens.warning,
            partially_fit=tokens.accent,
            fully_fit=tokens.success,
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
        self.mask_undo_button.setIcon(load_icon(IconName.UNDO, color))
        self.mask_redo_button.setIcon(load_icon(IconName.REDO, color))

    def _on_dataset_role_changed(
        self,
        project: ProjectState,
        previous: DatasetState,
        current: DatasetState,
    ) -> None:
        if self._open_project is project and self._open_dataset is previous:
            self._open_dataset = current
            self.dataset_view.replace_dataset(current.dataset)
            self.dataset_view.set_selection(
                None if current.auto_mask is None else current.auto_mask.selection,
            )
            self.dataset_view.set_mask_edit_available(current.mask_editable)
            self.edit_mask_action.setEnabled(current.mask_editable)
            self._refresh_context_label()
            self._refresh_inspector_context()

    def _on_dataset_updated(
        self,
        project: ProjectState,
        previous: DatasetState,
        current: DatasetState,
    ) -> None:
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
