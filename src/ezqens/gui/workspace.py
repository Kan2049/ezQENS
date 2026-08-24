"""Workspace sidebar and the minimal in-memory project/data state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from PySide6.QtCore import (
    QModelIndex,
    QPersistentModelIndex,
    QPoint,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QIcon, QMouseEvent, QPainter, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionToolButton,
    QStyleOptionViewItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ezqens.domain import DiagnosticSeverity, ReducedDataset, SpectrumRole
from ezqens.gui.icons import IconName, load_icon

PROJECT_INDEX_ROLE = Qt.ItemDataRole.UserRole
ITEM_KIND_ROLE = Qt.ItemDataRole.UserRole + 1
DATASET_INDEX_ROLE = Qt.ItemDataRole.UserRole + 2
ACTIVE_DATASET_ROLE = Qt.ItemDataRole.UserRole + 3


class DatasetAnalysisState(Enum):
    """Presentation-only coverage state for the normal analysis workflow."""

    REQUIRED = "red"
    READY = "yellow"
    PARTIALLY_FIT = "blue"
    FULLY_FIT = "green"


_STATE_ICON_SIZE = 12
_state_icon_cache: dict[str, QIcon] = {}


class SplitDataButton(QToolButton):
    """A compact split button that anchors its import menu to the right edge."""

    def showMenu(self) -> None:  # noqa: N802 - Qt virtual method spelling.
        menu = self.menu()
        if menu is None:
            return
        menu.popup(self.menu_popup_position(menu.sizeHint()))

    def menu_popup_position(self, menu_size: QSize) -> QPoint:
        """Return the compact menu origin aligned to the split button's right edge."""

        return self.mapToGlobal(
            QPoint(self.width() - menu_size.width(), self.height()),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Own the arrow hit target so Qt cannot substitute a left-aligned popup."""

        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        menu_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ToolButton,
            option,
            QStyle.SubControl.SC_ToolButtonMenu,
            self,
        )
        if event.button() is Qt.MouseButton.LeftButton and menu_rect.contains(
            event.position().toPoint()
        ):
            self.showMenu()
            event.accept()
            return
        super().mousePressEvent(event)


class WorkspaceItemDelegate(QStyledItemDelegate):
    """Draw active/open and selected tree rows as distinct visual states."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._active_background = "#e6edf5"
        self._selection_background = "#ececea"

    def set_colors(
        self,
        *,
        active_background: str,
        selection_background: str,
    ) -> None:
        """Set theme colors without spreading per-widget stylesheets."""

        self._active_background = active_background
        self._selection_background = selection_background

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        """Keep focus subtle while preserving the open dataset's active state."""

        styled_option = QStyleOptionViewItem(option)
        is_active_dataset = bool(index.data(ACTIVE_DATASET_ROLE))
        is_selected = bool(styled_option.state & QStyle.StateFlag.State_Selected)
        if is_active_dataset:
            self._paint_row_background(
                painter,
                styled_option,
                self._active_background,
            )
        elif is_selected:
            self._paint_row_background(
                painter,
                styled_option,
                self._selection_background,
            )
        styled_option.state &= ~QStyle.StateFlag.State_Selected
        styled_option.state &= ~QStyle.StateFlag.State_HasFocus
        if is_active_dataset:
            styled_option.state &= ~QStyle.StateFlag.State_MouseOver
        super().paint(painter, styled_option, index)

    @staticmethod
    def _paint_row_background(
        painter: QPainter,
        option: QStyleOptionViewItem,
        color: str,
    ) -> None:
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(option.rect.adjusted(2, 1, -2, -1), 4, 4)
        painter.restore()


@dataclass(frozen=True, eq=False)
class DatasetState:
    """One imported reduced dataset displayed in a Project's Data branch."""

    name: str
    dataset: ReducedDataset
    source_order: int = 0
    source_path: Path | None = None


@dataclass(eq=False)
class ProjectState:
    """Minimum in-memory Project state used by the current GUI slice."""

    name: str
    datasets: list[DatasetState] = field(default_factory=list)


class WorkspaceSidebar(QWidget):
    """Persistent project navigator with an intentional empty state."""

    project_created = Signal(object)
    project_selected = Signal(object)
    project_renamed = Signal(object)
    import_files_requested = Signal()
    import_folder_requested = Signal()
    dataset_open_requested = Signal(object, object)
    dataset_role_changed = Signal(object, object, object)
    dataset_renamed = Signal(object, object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workspaceSidebar")
        self.setMinimumWidth(180)
        self.setMaximumWidth(420)
        self._projects: list[ProjectState] = []
        self._project_items: list[QTreeWidgetItem] = []
        self._data_items: list[QTreeWidgetItem] = []
        self._resolution_icon_color: str | None = None
        self._analysis_state_colors = {
            DatasetAnalysisState.REQUIRED: "#a54a4a",
            DatasetAnalysisState.READY: "#a56c1d",
            DatasetAnalysisState.PARTIALLY_FIT: "#496d91",
            DatasetAnalysisState.FULLY_FIT: "#47785a",
        }
        self._active_project: ProjectState | None = None
        self._active_dataset: DatasetState | None = None

        title = QLabel("Workspace")
        title.setObjectName("workspaceTitle")

        self.new_project_button = QPushButton("+ Project")
        self.new_project_button.setObjectName("newProjectButton")
        self.new_project_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.new_project_button.clicked.connect(self.new_project)

        self.import_data_button = SplitDataButton()
        self.import_data_button.setObjectName("importDataButton")
        self.import_data_button.setText("+ Data")
        self.import_data_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.import_data_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup,
        )
        self.import_data_button.clicked.connect(self.import_files_requested.emit)
        import_menu = QMenu(self.import_data_button)
        files_action = import_menu.addAction("Files…")
        files_action.triggered.connect(self.import_files_requested.emit)
        folder_action = import_menu.addAction("Folder…")
        folder_action.triggered.connect(self.import_folder_requested.emit)
        self.import_data_button.setMenu(import_menu)
        self.import_data_button.hide()

        command_row = QHBoxLayout()
        command_row.setContentsMargins(0, 0, 0, 0)
        command_row.setSpacing(0)
        command_row.addWidget(self.new_project_button)
        command_row.addWidget(self.import_data_button)

        self.tree = QTreeWidget()
        self.tree.setObjectName("workspaceTree")
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.setColumnWidth(0, 128)
        self.tree.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.tree.setIndentation(16)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.item_delegate = WorkspaceItemDelegate(self.tree)
        self.tree.setItemDelegate(self.item_delegate)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.currentItemChanged.connect(self._on_current_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addLayout(command_row)
        layout.addWidget(self.tree, 1)

    @property
    def projects(self) -> tuple[ProjectState, ...]:
        """Return the current in-memory Projects."""
        return tuple(self._projects)

    @property
    def project_count(self) -> int:
        """Return the number of projects in the workspace."""
        return len(self._projects)

    @property
    def shows_empty_state(self) -> bool:
        """Return whether the workspace has no Project nodes."""
        return not self._projects

    @property
    def current_project(self) -> ProjectState | None:
        """Return the Project containing the current Workspace item."""

        return self._project_for_item(self.tree.currentItem())

    def new_project(self) -> ProjectState:
        """Create and select the smallest valid empty Project node."""
        ordinal = len(self._projects) + 1
        name = "Untitled Project" if ordinal == 1 else f"Untitled Project {ordinal}"
        project = ProjectState(name=name)
        self._projects.append(project)

        project_item = QTreeWidgetItem([project.name, ""])
        project_item.setData(0, PROJECT_INDEX_ROLE, ordinal - 1)
        project_item.setData(0, ITEM_KIND_ROLE, "project")
        data_item = QTreeWidgetItem(["Data", ""])
        data_item.setData(0, PROJECT_INDEX_ROLE, ordinal - 1)
        data_item.setData(0, ITEM_KIND_ROLE, "data")
        project_item.addChild(data_item)
        self._project_items.append(project_item)
        self._data_items.append(data_item)
        self.tree.addTopLevelItem(project_item)
        project_item.setFirstColumnSpanned(True)
        data_item.setFirstColumnSpanned(True)
        project_item.setExpanded(True)
        data_item.setExpanded(True)
        self.tree.setCurrentItem(project_item)
        self._update_command_affordance()
        self.project_created.emit(project)
        return project

    def add_dataset(
        self,
        project: ProjectState,
        dataset: ReducedDataset,
        *,
        source_path: Path | None = None,
    ) -> DatasetState:
        """Add an imported dataset without opening it in the scientific view."""

        project_index = self._project_index(project)
        name = dataset.source_reference or f"Reduced data {len(project.datasets) + 1}"
        state = DatasetState(
            name=name,
            dataset=dataset,
            source_order=self._next_dataset_order(project),
            source_path=source_path,
        )
        project.datasets.append(state)
        project.datasets.sort(key=_dataset_sort_key)
        self._rebuild_dataset_items(project_index)
        return state

    def rename_project(self, project: ProjectState, name: str) -> None:
        """Change only an in-memory Project display name."""

        project_index = self._project_index(project)
        display_name = name.strip()
        if not display_name:
            raise ValueError("Project display name must not be empty")
        if display_name == project.name:
            return
        project.name = display_name
        self._project_items[project_index].setText(0, display_name)
        self.project_renamed.emit(project)

    def rename_dataset(
        self,
        project: ProjectState,
        dataset: DatasetState,
        name: str,
    ) -> DatasetState:
        """Change only the Workspace display name for one imported dataset."""

        self.validate_dataset_membership(project, dataset)
        display_name = name.strip()
        if not display_name:
            raise ValueError("dataset display name must not be empty")
        if display_name == dataset.name:
            return dataset
        project_index = self._project_index(project)
        dataset_index = self._dataset_index(project, dataset)
        renamed = replace(dataset, name=display_name)
        project.datasets[dataset_index] = renamed
        if self._active_project is project and self._active_dataset is dataset:
            self._active_dataset = renamed
        self._rebuild_dataset_items(project_index, select_dataset=renamed)
        self.dataset_renamed.emit(project, dataset, renamed)
        return renamed

    def open_dataset(self, project: ProjectState, dataset: DatasetState) -> None:
        """Request opening an existing Workspace dataset."""

        self.validate_dataset_membership(project, dataset)
        self.dataset_open_requested.emit(project, dataset)

    def validate_dataset_membership(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> None:
        """Raise unless ``dataset`` is an in-memory member of ``project``."""

        self._project_index(project)
        if not any(candidate is dataset for candidate in project.datasets):
            raise ValueError("dataset does not belong to the selected Project")

    def set_dataset_role(
        self,
        project: ProjectState,
        dataset: DatasetState,
        role: SpectrumRole,
    ) -> DatasetState:
        """Re-role an imported dataset with the public validated domain models."""

        self.validate_dataset_membership(project, dataset)
        if dataset.dataset.role is role:
            return dataset
        project_index = self._project_index(project)
        dataset_index = self._dataset_index(project, dataset)
        re_roled = DatasetState(
            name=dataset.name,
            dataset=_dataset_with_role(dataset.dataset, role),
            source_order=dataset.source_order,
            source_path=dataset.source_path,
        )
        project.datasets[dataset_index] = re_roled
        if self._active_project is project and self._active_dataset is dataset:
            self._active_dataset = re_roled
        project.datasets.sort(key=_dataset_sort_key)
        self._rebuild_dataset_items(project_index, select_dataset=re_roled)
        self.dataset_role_changed.emit(project, dataset, re_roled)
        return re_roled

    def set_resolution_icon_color(self, color: str) -> None:
        """Refresh the restrained Resolution icon when application theme changes."""

        self._resolution_icon_color = color
        for project_index in range(len(self._projects)):
            self._rebuild_dataset_items(project_index)

    def set_tree_state_colors(
        self,
        *,
        active_background: str,
        selection_background: str,
    ) -> None:
        """Apply centralized active and neutral-selection visual tokens."""

        self.item_delegate.set_colors(
            active_background=active_background,
            selection_background=selection_background,
        )
        self.tree.viewport().update()

    def set_dataset_state_colors(
        self,
        *,
        required: str,
        ready: str,
        partially_fit: str,
        fully_fit: str,
    ) -> None:
        """Apply the semantic dataset-state colors from the shared tokens."""

        self._analysis_state_colors = {
            DatasetAnalysisState.REQUIRED: required,
            DatasetAnalysisState.READY: ready,
            DatasetAnalysisState.PARTIALLY_FIT: partially_fit,
            DatasetAnalysisState.FULLY_FIT: fully_fit,
        }
        for project_index in range(len(self._projects)):
            self._rebuild_dataset_items(project_index)

    def set_active_dataset(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> None:
        """Mark the explicitly opened dataset independently of tree selection."""

        self.validate_dataset_membership(project, dataset)
        self._active_project = project
        self._active_dataset = dataset
        for project_index in range(len(self._projects)):
            self._rebuild_dataset_items(project_index)

    def _on_current_item_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            return
        project_index = current.data(0, PROJECT_INDEX_ROLE)
        if isinstance(project_index, int):
            self.project_selected.emit(self._projects[project_index])

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, ITEM_KIND_ROLE) != "dataset":
            return
        project_index = item.data(0, PROJECT_INDEX_ROLE)
        dataset_index = item.data(0, DATASET_INDEX_ROLE)
        if isinstance(project_index, int) and isinstance(dataset_index, int):
            project = self._projects[project_index]
            self.dataset_open_requested.emit(project, project.datasets[dataset_index])

    def _show_context_menu(self, position: QPoint) -> None:
        item = self.tree.itemAt(position)
        menu = self._create_context_menu(item)
        if menu is None:
            return
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _create_context_menu(self, item: QTreeWidgetItem | None) -> QMenu | None:
        """Build the smallest context menu appropriate to a Workspace item."""

        if item is None:
            menu = QMenu(self)
            action = menu.addAction("New Project")
            action.triggered.connect(self.new_project)
            return menu
        item_kind = item.data(0, ITEM_KIND_ROLE)
        project_index = item.data(0, PROJECT_INDEX_ROLE)
        if not isinstance(project_index, int):
            return None
        if item_kind == "project":
            self.tree.setCurrentItem(item)
            project = self._projects[project_index]
            menu = QMenu(self)
            action = menu.addAction("Rename")
            action.triggered.connect(lambda: self._prompt_project_rename(project))
            return menu
        if item_kind != "dataset":
            return None
        dataset_index = item.data(0, DATASET_INDEX_ROLE)
        if not isinstance(dataset_index, int):
            return None
        self.tree.setCurrentItem(item)
        project = self._projects[project_index]
        dataset = project.datasets[dataset_index]
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(
            lambda: self._prompt_dataset_rename(project, dataset),
        )
        menu.addSeparator()
        if dataset.dataset.role is SpectrumRole.RESOLUTION:
            action = menu.addAction("Mark as Data")
            action.triggered.connect(
                lambda: self.set_dataset_role(project, dataset, SpectrumRole.SAMPLE),
            )
        else:
            action = menu.addAction("Mark as Resolution")
            action.triggered.connect(
                lambda: self.set_dataset_role(
                    project,
                    dataset,
                    SpectrumRole.RESOLUTION,
                ),
            )
        return menu

    def _prompt_dataset_rename(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> None:
        name, accepted = QInputDialog.getText(
            self,
            "Rename Dataset",
            "Display name:",
            text=dataset.name,
        )
        if not accepted or not name.strip():
            return
        self.rename_dataset(project, dataset, name)

    def _prompt_project_rename(self, project: ProjectState) -> None:
        name, accepted = QInputDialog.getText(
            self,
            "Rename Project",
            "Display name:",
            text=project.name,
        )
        if not accepted or not name.strip():
            return
        self.rename_project(project, name)

    def _update_command_affordance(self) -> None:
        """Show the one structural or high-frequency action for this state."""

        has_projects = bool(self._projects)
        self.new_project_button.setVisible(not has_projects)
        self.import_data_button.setVisible(has_projects)

    def _project_for_item(
        self,
        item: QTreeWidgetItem | None,
    ) -> ProjectState | None:
        if item is None:
            return None
        project_index = item.data(0, PROJECT_INDEX_ROLE)
        if not isinstance(project_index, int):
            return None
        if not 0 <= project_index < len(self._projects):
            return None
        return self._projects[project_index]

    def _project_index(self, project: ProjectState) -> int:
        for index, candidate in enumerate(self._projects):
            if candidate is project:
                return index
        raise ValueError("Project does not belong to this Workspace")

    @staticmethod
    def _dataset_index(project: ProjectState, dataset: DatasetState) -> int:
        for index, candidate in enumerate(project.datasets):
            if candidate is dataset:
                return index
        raise ValueError("dataset does not belong to the selected Project")

    @staticmethod
    def _next_dataset_order(project: ProjectState) -> int:
        previous_order = max(
            (dataset.source_order for dataset in project.datasets),
            default=-1,
        )
        return previous_order + 1

    def _rebuild_dataset_items(
        self,
        project_index: int,
        *,
        select_dataset: DatasetState | None = None,
    ) -> None:
        project = self._projects[project_index]
        data_item = self._data_items[project_index]
        selected_dataset = select_dataset or self._dataset_for_item(
            self.tree.currentItem(),
        )
        data_item.takeChildren()
        for dataset_index, dataset in enumerate(project.datasets):
            item = self._dataset_item(project_index, dataset_index, dataset)
            data_item.addChild(item)
            if dataset is selected_dataset:
                self.tree.setCurrentItem(item)
        data_item.setExpanded(True)

    def _dataset_for_item(
        self,
        item: QTreeWidgetItem | None,
    ) -> DatasetState | None:
        if item is None or item.data(0, ITEM_KIND_ROLE) != "dataset":
            return None
        project_index = item.data(0, PROJECT_INDEX_ROLE)
        dataset_index = item.data(0, DATASET_INDEX_ROLE)
        if not isinstance(project_index, int) or not isinstance(dataset_index, int):
            return None
        return self._projects[project_index].datasets[dataset_index]

    def _dataset_item(
        self,
        project_index: int,
        dataset_index: int,
        dataset: DatasetState,
    ) -> QTreeWidgetItem:
        role_label = (
            "Resolution" if dataset.dataset.role is SpectrumRole.RESOLUTION else ""
        )
        item = QTreeWidgetItem([dataset.name, role_label])
        item.setData(0, PROJECT_INDEX_ROLE, project_index)
        item.setData(0, ITEM_KIND_ROLE, "dataset")
        item.setData(0, DATASET_INDEX_ROLE, dataset_index)
        is_active = self._active_project is self._projects[project_index] and (
            self._active_dataset is dataset
        )
        item.setData(0, ACTIVE_DATASET_ROLE, is_active)
        item.setData(1, ACTIVE_DATASET_ROLE, is_active)
        item.setToolTip(0, _dataset_source_tooltip(dataset))
        analysis_state = dataset_analysis_state(dataset.dataset)
        item.setIcon(
            1,
            _analysis_state_icon(self._analysis_state_colors[analysis_state]),
        )
        item.setToolTip(
            1,
            _dataset_state_tooltip(analysis_state, dataset.dataset),
        )
        if (
            dataset.dataset.role is SpectrumRole.RESOLUTION
            and self._resolution_icon_color is not None
        ):
            font = QFont(item.font(0))
            font.setWeight(QFont.Weight.DemiBold)
            item.setFont(0, font)
            item.setIcon(
                0,
                load_icon(IconName.RESOLUTION, self._resolution_icon_color),
            )
        return item


def _unit_is_unknown(unit: str) -> bool:
    return unit.strip().casefold() == "unknown"


def dataset_analysis_state(
    dataset: ReducedDataset,
    *,
    fitted_group_count: int | None = None,
) -> DatasetAnalysisState:
    """Map known metadata and future fit coverage to the four semantic states.

    Slice 1 has no fit-presence interface, so calls from the Workspace leave
    ``fitted_group_count`` unset and can truthfully display only REQUIRED or
    READY. Fit quality is intentionally not an input to this mapping.
    """

    if _dataset_information_is_incomplete(dataset):
        return DatasetAnalysisState.REQUIRED
    if fitted_group_count is None or fitted_group_count == 0:
        return DatasetAnalysisState.READY
    if not 0 <= fitted_group_count <= len(dataset.spectra):
        raise ValueError("fitted group count is outside the dataset")
    if fitted_group_count == len(dataset.spectra):
        return DatasetAnalysisState.FULLY_FIT
    return DatasetAnalysisState.PARTIALLY_FIT


def _dataset_information_is_incomplete(dataset: ReducedDataset) -> bool:
    return dataset.q_bins is None or _dataset_units_are_unknown(dataset)


def _dataset_units_are_unknown(dataset: ReducedDataset) -> bool:
    return any(
        _unit_is_unknown(unit)
        for spectrum in dataset.spectra
        for unit in (
            spectrum.energy_unit,
            spectrum.intensity_unit,
            spectrum.uncertainty_unit,
        )
    )


def _dataset_status(dataset: ReducedDataset) -> str:
    statuses: list[str] = []
    if dataset.role is SpectrumRole.RESOLUTION:
        statuses.append("Resolution")
    if dataset.q_bins is None:
        statuses.append("Q required")
    if _dataset_units_are_unknown(dataset):
        statuses.append("Units required")
    warning_count = sum(
        diagnostic.severity is DiagnosticSeverity.WARNING
        for diagnostic in dataset.diagnostics
    )
    if warning_count:
        suffix = "warning" if warning_count == 1 else "warnings"
        statuses.append(f"{warning_count} {suffix}")
    return " · ".join(statuses)


def _diagnostics_tooltip(dataset: ReducedDataset) -> str:
    if not dataset.diagnostics:
        return _dataset_status(dataset)
    return "\n".join(
        f"{diagnostic.severity.value.title()}: {diagnostic.message}"
        for diagnostic in dataset.diagnostics
    )


def _dataset_source_tooltip(dataset: DatasetState) -> str:
    lines = [dataset.name]
    source = dataset.source_path or dataset.dataset.source_reference
    if source is not None:
        lines.append(f"Source: {source}")
    details = _dataset_status(dataset.dataset)
    if details:
        lines.append(details)
    diagnostics = _diagnostics_tooltip(dataset.dataset)
    if diagnostics and diagnostics != details:
        lines.append(diagnostics)
    return "\n".join(lines)


def _dataset_state_tooltip(
    state: DatasetAnalysisState,
    dataset: ReducedDataset,
) -> str:
    descriptions = {
        DatasetAnalysisState.REQUIRED: (
            "Information required before normal analysis can be complete."
        ),
        DatasetAnalysisState.READY: "Ready for analysis; no groups have been fit.",
        DatasetAnalysisState.PARTIALLY_FIT: "At least one group has been fit.",
        DatasetAnalysisState.FULLY_FIT: "Every group has been fit.",
    }
    details = _dataset_status(dataset)
    return "\n".join(part for part in (descriptions[state], details) if part)


def _analysis_state_icon(color: str) -> QIcon:
    cached = _state_icon_cache.get(color)
    if cached is not None:
        return cached
    pixmap = QPixmap(_STATE_ICON_SIZE, _STATE_ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(2, 2, _STATE_ICON_SIZE - 4, _STATE_ICON_SIZE - 4)
    painter.end()
    icon = QIcon(pixmap)
    _state_icon_cache[color] = icon
    return icon


def _dataset_sort_key(dataset: DatasetState) -> tuple[bool, int]:
    return (
        dataset.dataset.role is not SpectrumRole.RESOLUTION,
        dataset.source_order,
    )


def _dataset_with_role(dataset: ReducedDataset, role: SpectrumRole) -> ReducedDataset:
    """Return a validated role change without altering the original numeric values."""

    spectra = tuple(replace(spectrum, role=role) for spectrum in dataset.spectra)
    return replace(dataset, role=role, spectra=spectra)
