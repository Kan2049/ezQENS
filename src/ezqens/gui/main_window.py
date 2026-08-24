"""Native main-window shell for ezQENS."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QRect, QSignalBlocker, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ezqens.domain import ImportDiagnostic, ImportValidationError, SpectrumRole
from ezqens.gui.dataset_view import ReducedDatasetView
from ezqens.gui.icons import IconName, load_icon
from ezqens.gui.theme import (
    Appearance,
    application_appearance_controller,
    tokens_for,
)
from ezqens.gui.workspace import DatasetState, ProjectState, WorkspaceSidebar
from ezqens.io.importers import import_reduced_data

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

    @property
    def summary(self) -> str:
        """Return a small user-facing result summary."""

        return f"{len(self.imported)} imported · {len(self.failures)} skipped"


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
        self.inspector = self._build_inspector()
        self._create_actions()
        self.central_workspace = self._build_central_workspace()
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
        self.workspace.dataset_renamed.connect(self._on_dataset_renamed)
        self.dataset_view.group_changed.connect(self._refresh_inspector_context)
        self._appearance_controller.theme_changed.connect(self._on_theme_changed)

        shell = QWidget()
        shell.setObjectName("applicationShell")
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.splitter)
        self.setCentralWidget(shell)

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
        canvas_layout.addWidget(canvas_surface)

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

        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(self.inspector_context_label)
        layout.addStretch(1)
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
        dataset = import_reduced_data(path, role=SpectrumRole.SAMPLE)
        return self.workspace.add_dataset(
            target_project,
            dataset,
            source_path=Path(path),
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
        failures: list[ImportBatchFailure] = []
        for source in paths:
            path = Path(source)
            try:
                imported.append(self.import_data(path, project=target_project))
            except ImportValidationError as error:
                failures.append(
                    ImportBatchFailure(
                        path,
                        _format_import_diagnostics(error.diagnostics),
                    ),
                )
            except OSError as error:
                failures.append(ImportBatchFailure(path, str(error)))
        return ImportBatchResult(tuple(imported), tuple(failures))

    def open_dataset(self, project: ProjectState, dataset: DatasetState) -> None:
        """Open an imported Workspace dataset at Group 1."""

        self.workspace.validate_dataset_membership(project, dataset)
        self._open_project = project
        self._open_dataset = dataset
        self.dataset_view.open_dataset(dataset.dataset)
        self.workspace.set_active_dataset(project, dataset)
        self.set_scientific_figure_visible(True)
        self._refresh_context_label()
        self._refresh_inspector_context()

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

    def _prompt_import_files(self) -> None:
        """Choose one or more ordinary Data files for the selected Project."""

        project = self.workspace.current_project
        if project is None:
            self._show_import_target_required()
            return
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Import Reduced Data Files",
            "",
            "Reduced data (*.dat *.txt *.csv);;All files (*)",
        )
        if not paths:
            return
        result = self.import_paths(tuple(paths), project=project)
        if len(paths) > 1 or result.failures:
            self._show_import_result(result)

    def _prompt_import_folder(self) -> None:
        """Choose a folder and import its supported reduced-data files only."""

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
        self._show_import_result(self.import_paths(paths, project=project))

    def _show_import_target_required(self) -> None:
        QMessageBox.information(
            self,
            "Select a Project",
            "Select a Project in the Workspace before importing data.",
        )

    def _show_import_result(self, result: ImportBatchResult) -> None:
        """Show one compact summary, with detailed core diagnostics when needed."""

        if not result.failures:
            QMessageBox.information(self, "Reduced Data Import", result.summary)
            return
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Reduced Data Import")
        message.setText(result.summary)
        message.setInformativeText("Some files could not be imported.")
        message.setDetailedText(_format_batch_failures(result.failures))
        message.exec()

    def _on_theme_changed(self, _scheme: object) -> None:
        self._apply_workspace_state_colors()
        self._apply_dataset_view_theme()
        self.workspace.set_resolution_icon_color(
            tokens_for(self._appearance_controller.current_scheme).accent,
        )
        self._sync_inspector_controls(self.toggle_inspector_action.isChecked())

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
        )

    def _apply_dataset_view_theme(self) -> None:
        tokens = tokens_for(self._appearance_controller.current_scheme)
        self.dataset_view.set_overview_theme(
            surface=tokens.surface_central,
            text=tokens.text_primary,
            border=tokens.canvas_boundary,
            accent=tokens.accent,
        )

    def _on_dataset_role_changed(
        self,
        project: ProjectState,
        previous: DatasetState,
        current: DatasetState,
    ) -> None:
        if self._open_project is project and self._open_dataset is previous:
            self._open_dataset = current
            self.dataset_view.replace_dataset(current.dataset)
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
        QMessageBox.about(
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


def _supported_reduced_data_files(directory: Path) -> tuple[Path, ...]:
    """Return supported reduced-data files directly inside ``directory``."""

    return tuple(
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.casefold() in SUPPORTED_REDUCED_DATA_SUFFIXES
    )


def _dataset_source_detail(dataset: DatasetState) -> str:
    source = dataset.source_path or dataset.dataset.source_reference
    return f"Source: {source}" if source is not None else "Source unavailable"
