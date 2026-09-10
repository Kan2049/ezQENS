"""Workspace sidebar and the minimal in-memory project/data state."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import (
    QByteArray,
    QMimeData,
    QModelIndex,
    QPersistentModelIndex,
    QPoint,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QIcon,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStyleOptionToolButton,
    QStyleOptionViewItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ezqens.domain import (
    DiagnosticSeverity,
    ImportDiagnostic,
    QBins,
    ReducedDataset,
    SpectrumRole,
)
from ezqens.gui.icons import IconName, load_icon
from ezqens.gui.masking import (
    AutoMaskState,
    create_auto_mask_state,
    mask_task_available,
    rebind_auto_mask_state,
)
from ezqens.gui.theme import DEFAULT_LAYOUT_TOKENS

PROJECT_INDEX_ROLE = Qt.ItemDataRole.UserRole
ITEM_KIND_ROLE = Qt.ItemDataRole.UserRole + 1
DATASET_INDEX_ROLE = Qt.ItemDataRole.UserRole + 2
ACTIVE_DATASET_ROLE = Qt.ItemDataRole.UserRole + 3
Q_METHOD_INDEX_ROLE = Qt.ItemDataRole.UserRole + 4
Q_METHOD_MIME_TYPE = "application/x-ezqens-q-method"


class DatasetAnalysisState(Enum):
    """Presentation-only coverage state for the normal analysis workflow."""

    REQUIRED = "red"
    READY = "yellow"
    PARTIALLY_FIT = "blue"
    FULLY_FIT = "green"
    COMPLETE = "neutral"


_STATE_ICON_SIZE = 12
_state_icon_cache: dict[str, QIcon] = {}


def _paint_centered_action_content(
    button: QAbstractButton,
    painter: QPainter,
    text_rect: QRect,
    *,
    icon: QIcon,
    text: str,
    palette: QPalette,
    state: QStyle.StateFlag,
) -> None:
    """Paint an anchored icon and independently centered action label."""

    icon_size = button.iconSize()
    icon_inset = DEFAULT_LAYOUT_TOKENS.control_icon_left_inset
    icon_rect = QRect(
        text_rect.left() + icon_inset,
        text_rect.center().y() - icon_size.height() // 2,
        icon_size.width(),
        icon_size.height(),
    )
    if not icon.isNull():
        icon_mode = QIcon.Mode.Normal
        if not button.isEnabled():
            icon_mode = QIcon.Mode.Disabled
        elif state & QStyle.StateFlag.State_Sunken:
            icon_mode = QIcon.Mode.Selected
        elif state & QStyle.StateFlag.State_MouseOver:
            icon_mode = QIcon.Mode.Active
        icon_state = QIcon.State.On if button.isChecked() else QIcon.State.Off
        icon.paint(
            painter,
            icon_rect,
            Qt.AlignmentFlag.AlignCenter,
            icon_mode,
            icon_state,
        )
    button.style().drawItemText(
        painter,
        text_rect,
        Qt.AlignmentFlag.AlignCenter,
        palette,
        button.isEnabled(),
        text,
        QPalette.ColorRole.ButtonText,
    )


class CenteredActionButton(QPushButton):
    """A normal action button with a fixed icon and independently centered text."""

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt virtual.
        option = QStyleOptionButton()
        self.initStyleOption(option)
        icon = self.icon()
        text = self.text()
        option.icon = QIcon()
        option.text = ""

        painter = QPainter(self)
        self.style().drawControl(
            QStyle.ControlElement.CE_PushButton,
            option,
            painter,
            self,
        )
        _paint_centered_action_content(
            self,
            painter,
            self.contentsRect(),
            icon=icon,
            text=text,
            palette=option.palette,
            state=option.state,
        )
        painter.end()


class SplitDataButton(QToolButton):
    """A compact split button with independent icon and label alignment."""

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

    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802 - Qt virtual method.
        """Keep the label centered in the main action region, apart from its icon."""

        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        main_rect, _menu_rect = self._action_rects(option)
        icon = self.icon()
        text = self.text()
        option.icon = QIcon()
        option.text = ""

        painter = QPainter(self)
        self.style().drawComplexControl(
            QStyle.ComplexControl.CC_ToolButton,
            option,
            painter,
            self,
        )
        _paint_centered_action_content(
            self,
            painter,
            main_rect,
            icon=icon,
            text=text,
            palette=option.palette,
            state=option.state,
        )
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Own the arrow hit target so Qt cannot substitute a left-aligned popup."""

        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        _main_rect, menu_rect = self._action_rects(option)
        if event.button() is Qt.MouseButton.LeftButton and menu_rect.contains(
            event.position().toPoint()
        ):
            self.showMenu()
            event.accept()
            return
        super().mousePressEvent(event)

    def _action_rects(self, option: QStyleOptionToolButton) -> tuple[QRect, QRect]:
        """Return the existing main and menu hit regions from the active style."""

        main_rect = self.contentsRect()
        menu_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ToolButton,
            option,
            QStyle.SubControl.SC_ToolButtonMenu,
            self,
        )
        if not menu_rect.isEmpty():
            if self.layoutDirection() is Qt.LayoutDirection.LeftToRight:
                main_rect.setRight(menu_rect.left() - 1)
            else:
                main_rect.setLeft(menu_rect.right() + 1)
        return main_rect, menu_rect


class WorkspaceTree(QTreeWidget):
    """Native tree drag target for applying a reusable Q Method to a dataset."""

    q_method_dropped = Signal(int, int, int, int)
    removal_requested = Signal(object)

    def mimeData(self, items: Sequence[QTreeWidgetItem]) -> QMimeData:  # noqa: N802
        mime = super().mimeData(items)
        if len(items) != 1:
            return mime
        item = items[0]
        if item.data(0, ITEM_KIND_ROLE) != "q_method":
            return mime
        project_index = item.data(0, PROJECT_INDEX_ROLE)
        method_index = item.data(0, Q_METHOD_INDEX_ROLE)
        if isinstance(project_index, int) and isinstance(method_index, int):
            mime.setData(
                Q_METHOD_MIME_TYPE,
                QByteArray(f"{project_index}:{method_index}".encode()),
            )
        return mime

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasFormat(Q_METHOD_MIME_TYPE):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        target = self.itemAt(event.position().toPoint())
        if (
            event.mimeData().hasFormat(Q_METHOD_MIME_TYPE)
            and target is not None
            and target.data(0, ITEM_KIND_ROLE) == "dataset"
        ):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        target = self.itemAt(event.position().toPoint())
        if target is None or not event.mimeData().hasFormat(Q_METHOD_MIME_TYPE):
            super().dropEvent(event)
            return
        source_project, method_index = _q_method_drag_indices(event)
        target_project = target.data(0, PROJECT_INDEX_ROLE)
        target_dataset = target.data(0, DATASET_INDEX_ROLE)
        if (
            target.data(0, ITEM_KIND_ROLE) == "dataset"
            and source_project is not None
            and method_index is not None
            and isinstance(target_project, int)
            and isinstance(target_dataset, int)
        ):
            self.q_method_dropped.emit(
                source_project,
                method_index,
                target_project,
                target_dataset,
            )
            event.acceptProposedAction()
            return
        event.ignore()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Route ordinary removal keys through the same sidebar command path."""

        if self.state() is QAbstractItemView.State.EditingState:
            super().keyPressEvent(event)
            return
        if event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
            item = self.currentItem()
            if item is not None and item.data(0, ITEM_KIND_ROLE) in {
                "project",
                "dataset",
                "q_method",
            }:
                self.removal_requested.emit(item)
                event.accept()
                return
        super().keyPressEvent(event)


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
    auto_mask: AutoMaskState | None = None
    workflow_dataset_id: str = field(
        default_factory=lambda: f"dataset-{uuid4().hex}",
    )

    @property
    def mask_editable(self) -> bool:
        """Return whether this Sample has the stored baseline required by Mask."""

        return (
            self.dataset.role is SpectrumRole.SAMPLE
            and self.auto_mask is not None
            and mask_task_available(self.auto_mask, self.dataset)
        )


@dataclass(frozen=True, eq=False)
class QMethodState:
    """One explicit, reusable in-memory Q assignment."""

    name: str
    q_bins: QBins


@dataclass(eq=False)
class ProjectState:
    """Minimum in-memory Project state used by the current GUI slice."""

    name: str
    datasets: list[DatasetState] = field(default_factory=list)
    q_methods: list[QMethodState] = field(default_factory=list)


class WorkspaceSidebar(QWidget):
    """Persistent project navigator with an intentional empty state."""

    project_created = Signal(object)
    project_selected = Signal(object)
    project_renamed = Signal(object)
    import_files_requested = Signal()
    import_folder_requested = Signal()
    dataset_open_requested = Signal(object, object)
    dataset_added = Signal(object, object)
    dataset_role_changed = Signal(object, object, object)
    dataset_role_change_requested = Signal(object, object, object)
    dataset_renamed = Signal(object, object, object)
    dataset_updated = Signal(object, object, object)
    units_requested = Signal(object, object)
    q_assignment_requested = Signal(object, object)
    mask_edit_requested = Signal(object, object)
    manual_fit_requested = Signal(object, object)
    auto_fit_requested = Signal(object, object)
    apply_resolution_requested = Signal(object, object)
    q_method_dropped = Signal(object, object, object)
    project_removal_requested = Signal(object)
    dataset_removal_requested = Signal(object, object)
    q_method_removal_requested = Signal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workspaceSidebar")
        self.setMinimumWidth(180)
        self.setMaximumWidth(420)
        self._projects: list[ProjectState] = []
        self._project_items: list[QTreeWidgetItem] = []
        self._data_items: list[QTreeWidgetItem] = []
        self._methods_items: list[QTreeWidgetItem] = []
        self._resolution_icon_color: str | None = None
        self._analysis_state_colors = {
            DatasetAnalysisState.REQUIRED: "#a54a4a",
            DatasetAnalysisState.READY: "#a56c1d",
            DatasetAnalysisState.PARTIALLY_FIT: "#496d91",
            DatasetAnalysisState.FULLY_FIT: "#47785a",
            DatasetAnalysisState.COMPLETE: "#676764",
        }
        self._active_project: ProjectState | None = None
        self._active_dataset: DatasetState | None = None
        self._scientific_replacement_resolver: (
            Callable[[ProjectState, DatasetState], DatasetState | None] | None
        ) = None
        self._fitting_started_resolver: (
            Callable[[ProjectState, DatasetState], bool] | None
        ) = None

        title = QLabel("ezQENS")
        title.setObjectName("workspaceTitle")

        self.new_project_button = CenteredActionButton("Project")
        self.new_project_button.setObjectName("newProjectButton")
        self.new_project_button.setToolTip("Create a new Project")
        self.new_project_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.new_project_button.clicked.connect(self.new_project)

        self.import_data_button = SplitDataButton()
        self.import_data_button.setObjectName("importDataButton")
        self.import_data_button.setProperty("controlKind", "split")
        self.import_data_button.setText("Import")
        self.import_data_button.setToolTip("Import reduced data files")
        self.import_data_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
        )
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

        self.tree = WorkspaceTree()
        self.tree.setObjectName("workspaceTree")
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.tree.setIconSize(QSize(14, 14))
        self.tree.setUniformRowHeights(True)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.tree.setIndentation(16)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.item_delegate = WorkspaceItemDelegate(self.tree)
        self.tree.setItemDelegate(self.item_delegate)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.currentItemChanged.connect(self._on_current_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.q_method_dropped.connect(self._on_q_method_dropped)
        self.tree.removal_requested.connect(self._request_item_removal)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addLayout(command_row)
        layout.addWidget(self.tree, 1)
        self._layout = layout
        for control in (self.new_project_button, self.import_data_button):
            control.setMinimumHeight(DEFAULT_LAYOUT_TOKENS.control_height)
            control.setIconSize(
                QSize(
                    DEFAULT_LAYOUT_TOKENS.control_icon_size,
                    DEFAULT_LAYOUT_TOKENS.control_icon_size,
                ),
            )
        self._layout.setSpacing(DEFAULT_LAYOUT_TOKENS.section_spacing)

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
        methods_item = QTreeWidgetItem(["Method", ""])
        methods_item.setData(0, PROJECT_INDEX_ROLE, ordinal - 1)
        methods_item.setData(0, ITEM_KIND_ROLE, "methods")
        project_item.addChild(data_item)
        project_item.addChild(methods_item)
        self._project_items.append(project_item)
        self._data_items.append(data_item)
        self._methods_items.append(methods_item)
        self.tree.addTopLevelItem(project_item)
        project_item.setFirstColumnSpanned(True)
        data_item.setFirstColumnSpanned(True)
        methods_item.setFirstColumnSpanned(True)
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
            auto_mask=create_auto_mask_state(dataset),
        )
        project.datasets.append(state)
        project.datasets.sort(key=_dataset_sort_key)
        self._rebuild_dataset_items(project_index)
        self.dataset_added.emit(project, state)
        return state

    def assign_q_bins(
        self,
        project: ProjectState,
        dataset: DatasetState,
        q_bins: QBins,
        *,
        diagnostics: tuple[ImportDiagnostic, ...] = (),
    ) -> DatasetState | None:
        """Replace validated Q identity while retaining the current mask state."""

        target = self._resolve_scientific_replacement_target(project, dataset)
        if target is None:
            return None
        dataset = target
        replacement = dataset.dataset.assign_q_bins(q_bins)
        if diagnostics:
            replacement = replace(
                replacement,
                diagnostics=(*replacement.diagnostics, *diagnostics),
            )
        return self._replace_dataset(
            project,
            dataset,
            replacement,
        )

    def assign_source_units(
        self,
        project: ProjectState,
        dataset: DatasetState,
        *,
        energy_unit: str,
        intensity_unit: str,
    ) -> DatasetState | None:
        """Assign source metadata without numerical conversion or array changes."""

        target = self._resolve_scientific_replacement_target(project, dataset)
        if target is None:
            return None
        dataset = target
        energy = energy_unit.strip()
        intensity = intensity_unit.strip()
        if not energy or not intensity:
            raise ValueError("source units must not be empty")
        spectra = tuple(
            replace(
                spectrum,
                energy_unit=energy,
                intensity_unit=intensity,
                uncertainty_unit=intensity,
            )
            for spectrum in dataset.dataset.spectra
        )
        return self._replace_dataset(
            project,
            dataset,
            replace(dataset.dataset, spectra=spectra),
        )

    def save_q_method(
        self,
        project: ProjectState,
        q_bins: QBins,
        *,
        name: str | None = None,
    ) -> QMethodState:
        """Store a Q assignment only after an explicit user request."""

        project_index = self._project_index(project)
        method = QMethodState(
            name=name or f"Q Method {len(project.q_methods) + 1}",
            q_bins=q_bins,
        )
        project.q_methods.append(method)
        self._rebuild_method_items(project_index)
        return method

    def remove_project(self, project: ProjectState) -> None:
        """Remove in-memory Project state only; source files remain untouched."""

        project_index = self._project_index(project)
        self._projects.pop(project_index)
        self._project_items.pop(project_index)
        self._data_items.pop(project_index)
        self._methods_items.pop(project_index)
        self.tree.takeTopLevelItem(project_index)
        if self._active_project is project:
            self._active_project = None
            self._active_dataset = None
        self._update_project_indices()
        self._update_command_affordance()

    def remove_dataset(self, project: ProjectState, dataset: DatasetState) -> None:
        """Remove a dataset node/state without touching its external source file."""

        project_index = self._project_index(project)
        dataset_index = self._dataset_index(project, dataset)
        data_item = self._data_items[project_index]
        current_item = self.tree.currentItem()
        if self._project_for_item(current_item) is project:
            self.tree.setCurrentItem(data_item)
        project.datasets.pop(dataset_index)
        if self._active_project is project and self._active_dataset is dataset:
            self._active_dataset = None
        self._rebuild_dataset_items(project_index)
        self.tree.setCurrentItem(data_item)

    def remove_q_method(self, project: ProjectState, method: QMethodState) -> None:
        """Remove a reusable in-memory Q Method without external file deletion."""

        try:
            method_index = project.q_methods.index(method)
        except ValueError as error:
            raise ValueError(
                "Q Method does not belong to the selected Project"
            ) from error
        project.q_methods.pop(method_index)
        self._rebuild_method_items(self._project_index(project))

    def clear_active_dataset(self) -> None:
        """Clear the explicit Open/View marker independently of tree selection."""

        self._active_project = None
        self._active_dataset = None
        for project_index in range(len(self._projects)):
            self._rebuild_dataset_items(project_index)

    def update_auto_mask(
        self,
        project: ProjectState,
        dataset: DatasetState,
        auto_mask: AutoMaskState,
    ) -> DatasetState:
        """Commit a core-derived proposal or a saved task-local mask selection."""

        return self._replace_dataset(
            project,
            dataset,
            dataset.dataset,
            auto_mask=auto_mask,
        )

    def synchronize_replayed_resolution(
        self,
        project: ProjectState,
        previous: DatasetState,
        replacement: ReducedDataset,
    ) -> DatasetState:
        """Prepare the visible replay result before workflow publication/signals."""

        return self._replace_dataset(
            project,
            previous,
            replacement,
            auto_mask=create_auto_mask_state(replacement),
            emit_update=False,
        )

    def apply_q_rebin_transaction(
        self,
        project: ProjectState,
        sample: DatasetState,
        rebinned_sample: ReducedDataset,
        *,
        resolution: DatasetState | None = None,
        rebinned_resolution: ReducedDataset | None = None,
    ) -> tuple[DatasetState, DatasetState | None]:
        """Replace visible rows only after the paired core transaction succeeds."""

        self.validate_dataset_membership(project, sample)
        if (resolution is None) is not (rebinned_resolution is None):
            raise ValueError(
                "Resolution state and replacement must be supplied together"
            )
        if resolution is not None:
            self.validate_dataset_membership(project, resolution)
        current_sample = self._replace_dataset(
            project,
            sample,
            rebinned_sample,
            auto_mask=create_auto_mask_state(rebinned_sample),
            emit_update=False,
        )
        current_resolution = None
        if resolution is not None and rebinned_resolution is not None:
            current_resolution = self._replace_dataset(
                project,
                resolution,
                rebinned_resolution,
                auto_mask=create_auto_mask_state(rebinned_resolution),
                emit_update=False,
            )
        self.dataset_updated.emit(project, sample, current_sample)
        if resolution is not None and current_resolution is not None:
            self.dataset_updated.emit(project, resolution, current_resolution)
        return current_sample, current_resolution

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

    def set_scientific_replacement_resolver(
        self,
        resolver: Callable[[ProjectState, DatasetState], DatasetState | None],
    ) -> None:
        """Set the application preflight used before scientific replacement."""

        self._scientific_replacement_resolver = resolver

    def set_fitting_started_resolver(
        self,
        resolver: Callable[[ProjectState, DatasetState], bool],
    ) -> None:
        """Derive dataset-level fitting progress from application workflow state."""

        self._fitting_started_resolver = resolver

    def refresh_dataset_analysis(self, project: ProjectState) -> None:
        """Refresh only the presentation derived for one Project's Data rows."""

        self._rebuild_dataset_items(self._project_index(project))

    def dataset_analysis_state_for(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> DatasetAnalysisState:
        """Return the current derived Workspace indicator state for one dataset."""

        self.validate_dataset_membership(project, dataset)
        fitting_started = (
            self._fitting_started_resolver(project, dataset)
            if self._fitting_started_resolver is not None
            else False
        )
        return dataset_analysis_state(
            dataset.dataset,
            fitting_started=fitting_started,
        )

    def _resolve_scientific_replacement_target(
        self,
        project: ProjectState,
        dataset: DatasetState,
    ) -> DatasetState | None:
        """Return the current target after an application-level task preflight."""

        self.validate_dataset_membership(project, dataset)
        if self._scientific_replacement_resolver is None:
            return dataset
        target = self._scientific_replacement_resolver(project, dataset)
        if target is not None:
            self.validate_dataset_membership(project, target)
        return target

    def set_dataset_role(
        self,
        project: ProjectState,
        dataset: DatasetState,
        role: SpectrumRole,
    ) -> DatasetState | None:
        """Re-role an imported dataset with the public validated domain models."""

        self.validate_dataset_membership(project, dataset)
        if dataset.dataset.role is role:
            return dataset
        target = self._resolve_scientific_replacement_target(project, dataset)
        if target is None:
            return None
        dataset = target
        re_roled = self._replace_dataset(
            project,
            dataset,
            _dataset_with_role(dataset.dataset, role),
            emit_update=False,
        )
        self.dataset_role_changed.emit(project, dataset, re_roled)
        return re_roled

    def set_resolution_icon_color(self, color: str) -> None:
        """Refresh the restrained Resolution icon when application theme changes."""

        self._resolution_icon_color = color
        for project_index in range(len(self._projects)):
            self._rebuild_dataset_items(project_index)

    def set_control_icon_color(self, color: str) -> None:
        """Refresh the few text-supporting workspace action icons."""

        self.new_project_button.setIcon(load_icon(IconName.PROJECT, color))
        self.import_data_button.setIcon(load_icon(IconName.IMPORT, color))

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
        complete: str | None = None,
    ) -> None:
        """Apply the semantic dataset-state colors from the shared tokens."""

        self._analysis_state_colors = {
            DatasetAnalysisState.REQUIRED: required,
            DatasetAnalysisState.READY: ready,
            DatasetAnalysisState.PARTIALLY_FIT: partially_fit,
            DatasetAnalysisState.FULLY_FIT: fully_fit,
            DatasetAnalysisState.COMPLETE: complete or ready,
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

    def _replace_dataset(
        self,
        project: ProjectState,
        previous: DatasetState,
        replacement_dataset: ReducedDataset,
        *,
        auto_mask: AutoMaskState | None = None,
        emit_update: bool = True,
    ) -> DatasetState:
        """Replace one immutable dataset state and synchronize the Workspace row."""

        self.validate_dataset_membership(project, previous)
        project_index = self._project_index(project)
        dataset_index = self._dataset_index(project, previous)
        mask_state = auto_mask
        if mask_state is None and previous.auto_mask is not None:
            mask_state = rebind_auto_mask_state(previous.auto_mask, replacement_dataset)
        current = replace(
            previous,
            dataset=replacement_dataset,
            auto_mask=mask_state,
        )
        project.datasets[dataset_index] = current
        if self._active_project is project and self._active_dataset is previous:
            self._active_dataset = current
        project.datasets.sort(key=_dataset_sort_key)
        self._rebuild_dataset_items(project_index, select_dataset=current)
        if emit_update:
            self.dataset_updated.emit(project, previous, current)
        return current

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
            delete_action = menu.addAction("Delete Project")
            delete_action.triggered.connect(
                lambda: self._request_item_removal(item),
            )
            return menu
        if item_kind == "q_method":
            method_index = item.data(0, Q_METHOD_INDEX_ROLE)
            if not isinstance(method_index, int):
                return None
            menu = QMenu(self)
            action = menu.addAction("Remove from Project")
            action.triggered.connect(lambda: self._request_item_removal(item))
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
        units_action = menu.addAction("Units…")
        units_action.triggered.connect(
            lambda: self.units_requested.emit(project, dataset),
        )
        q_action = menu.addAction(
            "Edit Q…" if dataset.dataset.q_bins is not None else "Assign Q…",
        )
        q_action.triggered.connect(
            lambda: self.q_assignment_requested.emit(project, dataset),
        )
        if dataset.mask_editable:
            mask_action = menu.addAction("Edit Mask…")
            mask_action.triggered.connect(
                lambda: self.mask_edit_requested.emit(project, dataset),
            )
        if dataset.dataset.role is SpectrumRole.SAMPLE:
            auto_fit_action = menu.addAction("AutoFit…")
            auto_fit_action.triggered.connect(
                lambda: self.auto_fit_requested.emit(project, dataset),
            )
            manual_fit_action = menu.addAction("Fitting Parameters…")
            manual_fit_action.triggered.connect(
                lambda: self.manual_fit_requested.emit(project, dataset),
            )
            resolution_action = menu.addAction("Apply Resolution…")
            resolution_action.triggered.connect(
                lambda: self.apply_resolution_requested.emit(project, dataset),
            )
        menu.addSeparator()
        if dataset.dataset.role is SpectrumRole.RESOLUTION:
            action = menu.addAction("Mark as Data")
            action.triggered.connect(
                lambda: self.dataset_role_change_requested.emit(
                    project,
                    dataset,
                    SpectrumRole.SAMPLE,
                ),
            )
        else:
            action = menu.addAction("Mark as Resolution")
            action.triggered.connect(
                lambda: self.dataset_role_change_requested.emit(
                    project,
                    dataset,
                    SpectrumRole.RESOLUTION,
                ),
            )
        menu.addSeparator()
        remove_action = menu.addAction("Remove from Project")
        remove_action.triggered.connect(lambda: self._request_item_removal(item))
        return menu

    def _request_item_removal(self, item: QTreeWidgetItem) -> None:
        """Emit the object-level removal request shared by menu and keyboard paths."""

        project = self._project_for_item(item)
        if project is None:
            return
        item_kind = item.data(0, ITEM_KIND_ROLE)
        if item_kind == "project":
            self.project_removal_requested.emit(project)
        elif item_kind == "dataset":
            dataset = self._dataset_for_item(item)
            if dataset is not None:
                self.dataset_removal_requested.emit(project, dataset)
        elif item_kind == "q_method":
            method_index = item.data(0, Q_METHOD_INDEX_ROLE)
            if isinstance(method_index, int) and method_index < len(project.q_methods):
                self.q_method_removal_requested.emit(
                    project,
                    project.q_methods[method_index],
                )

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

    def _update_project_indices(self) -> None:
        """Keep tree item identity data aligned after Project removal."""

        for project_index, project_item in enumerate(self._project_items):
            project_item.setData(0, PROJECT_INDEX_ROLE, project_index)
            self._data_items[project_index].setData(
                0,
                PROJECT_INDEX_ROLE,
                project_index,
            )
            self._methods_items[project_index].setData(
                0,
                PROJECT_INDEX_ROLE,
                project_index,
            )
            self._rebuild_dataset_items(project_index)
            self._rebuild_method_items(project_index)

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

    def _rebuild_method_items(self, project_index: int) -> None:
        """Render the small, in-memory Q Method list without a manager layer."""

        methods_item = self._methods_items[project_index]
        methods_item.takeChildren()
        for method_index, method in enumerate(self._projects[project_index].q_methods):
            child = QTreeWidgetItem([method.name, "Q Method"])
            child.setData(0, PROJECT_INDEX_ROLE, project_index)
            child.setData(0, ITEM_KIND_ROLE, "q_method")
            child.setData(0, Q_METHOD_INDEX_ROLE, method_index)
            child.setFlags(
                child.flags() | Qt.ItemFlag.ItemIsDragEnabled,
            )
            methods_item.addChild(child)
        methods_item.setExpanded(True)

    def _on_q_method_dropped(
        self,
        method_project_index: int,
        method_index: int,
        dataset_project_index: int,
        dataset_index: int,
    ) -> None:
        """Resolve native tree drag identifiers to the in-memory state objects."""

        try:
            method_project = self._projects[method_project_index]
            method = method_project.q_methods[method_index]
            dataset_project = self._projects[dataset_project_index]
            dataset = dataset_project.datasets[dataset_index]
        except IndexError:
            return
        self.q_method_dropped.emit(dataset_project, dataset, method)

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
        if not 0 <= project_index < len(self._projects):
            return None
        if not 0 <= dataset_index < len(self._projects[project_index].datasets):
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
        analysis_state = self.dataset_analysis_state_for(
            self._projects[project_index],
            dataset,
        )
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
    fitting_started: bool = False,
) -> DatasetAnalysisState:
    """Map metadata, fitting start, and future fit coverage to semantic states."""

    if _dataset_information_is_incomplete(dataset):
        return DatasetAnalysisState.REQUIRED
    if dataset.role is SpectrumRole.RESOLUTION:
        return DatasetAnalysisState.COMPLETE
    if fitted_group_count is not None:
        if not 0 <= fitted_group_count <= len(dataset.spectra):
            raise ValueError("fitted group count is outside the dataset")
        if fitted_group_count == len(dataset.spectra):
            return DatasetAnalysisState.FULLY_FIT
        if fitted_group_count:
            return DatasetAnalysisState.PARTIALLY_FIT
    if fitting_started:
        return DatasetAnalysisState.PARTIALLY_FIT
    return DatasetAnalysisState.READY


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
        DatasetAnalysisState.READY: "Ready for Fit; fitting has not started.",
        DatasetAnalysisState.PARTIALLY_FIT: (
            "Fitting has started; dataset Groups are not yet fully completed."
        ),
        DatasetAnalysisState.FULLY_FIT: "Every group has been fit.",
        DatasetAnalysisState.COMPLETE: (
            "Required metadata are complete; Resolution has no fit-coverage state."
        ),
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


def q_method_drag_indices_from_mime_data(
    payload: QByteArray,
) -> tuple[int | None, int | None]:
    """Decode the tiny local drag payload without serializing scientific data."""

    try:
        project_text, method_text = bytes(payload.data()).decode().split(":", 1)
        return int(project_text), int(method_text)
    except (UnicodeDecodeError, ValueError):
        return None, None


def _q_method_drag_indices(event: QDropEvent) -> tuple[int | None, int | None]:
    return q_method_drag_indices_from_mime_data(
        event.mimeData().data(Q_METHOD_MIME_TYPE),
    )


def _dataset_with_role(dataset: ReducedDataset, role: SpectrumRole) -> ReducedDataset:
    """Return a validated role change without altering the original numeric values."""

    spectra = tuple(replace(spectrum, role=role) for spectrum in dataset.spectra)
    return replace(dataset, role=role, spectra=spectra)
