"""Focused AutoFit candidate and selected-branch comparison dialog."""

from __future__ import annotations

from threading import Event

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.patches import Rectangle
from PySide6.QtCore import (
    QObject,
    QPoint,
    QSignalBlocker,
    QSize,
    Qt,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QActionGroup, QCloseEvent, QFont, QMouseEvent, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ezqens.batch import MultiQExecutionStatus, MultiQFitStatus
from ezqens.fitting import (
    BackgroundModel,
    CandidateFitResult,
    ComponentFamily,
    ComponentIdentity,
    FitResult,
    StandardModelCandidate,
)
from ezqens.gui.scientific_canvas import (
    SCIENTIFIC_BACKGROUND,
    SCIENTIFIC_BACKGROUND_COLOR,
    SCIENTIFIC_ELASTIC_COLOR,
    SCIENTIFIC_LORENTZIAN_COLORS,
    SCIENTIFIC_MEASURED_COLOR,
    SCIENTIFIC_RESIDUAL_COLOR,
    SCIENTIFIC_TOTAL_FIT_COLOR,
    SCIENTIFIC_UNCERTAINTY_COLOR,
    ScientificCanvas,
    log_display_values,
    symlog_linthresh,
    zoom_limits,
)
from ezqens.workflow import (
    AutoFitCandidateBranchResult,
    ManualFitDraft,
    SelectedAutoFitMultiQResult,
    SingleQAutoFitOutcome,
    WorkflowProject,
    continue_selected_auto_fit_candidates,
)

_ZOOM_DRAG_THRESHOLD_PX = 5.0
_MIN_COMPARISON_PANEL_WIDTH = 320
_MIN_COMPARISON_PANEL_HEIGHT = 300
_MAX_COMPARISON_COLUMNS = 3

type _ComparisonEntry = tuple[CandidateFitResult, FitResult | None, str]


class _CandidateRow(QWidget):
    """One independently selectable and checkable AutoFit candidate row."""

    checked_changed = Signal(bool)
    activated = Signal()

    def __init__(
        self,
        name: str,
        reduced_chi_square: str,
        recommendation: str,
        execution_state: str,
        execution_description: str,
        *,
        eligible: bool,
        recommended: bool,
    ) -> None:
        super().__init__()
        self.checkbox = QCheckBox()
        self.checkbox.setObjectName("autoFitCandidateCheckbox")
        self.checkbox.setEnabled(eligible)
        self.checkbox.setToolTip(
            "Select this candidate for comparison and All-Q execution"
            if eligible
            else "This candidate has no usable successful fit"
        )
        self.status_dot = QLabel()
        self.status_dot.setObjectName("autoFitCandidateStatusDot")
        self.status_dot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.name_label = QLabel(name)
        self.name_label.setObjectName("autoFitCandidateName")
        self.name_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.reduced_chi_square_label = QLabel(reduced_chi_square)
        self.reduced_chi_square_label.setObjectName("autoFitCandidateReducedChiSquare")
        self.reduced_chi_square_label.setProperty("secondary", True)
        self.reduced_chi_square_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.recommendation_label = QLabel(recommendation)
        self.recommendation_label.setObjectName("autoFitCandidateRecommendation")
        self.recommendation_label.setProperty("secondary", True)
        self.recommendation_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        for label in (
            self.status_dot,
            self.name_label,
            self.reduced_chi_square_label,
            self.recommendation_label,
        ):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        if recommended:
            font = QFont(self.name_label.font())
            font.setBold(True)
            self.name_label.setFont(font)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 6, 2)
        layout.setSpacing(4)
        layout.addWidget(self.status_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.checkbox, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.name_label, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(
            self.reduced_chi_square_label,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        layout.addStretch(1)
        layout.addWidget(
            self.recommendation_label,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.set_execution_state(execution_state, execution_description)
        self.checkbox.toggled.connect(self.checked_changed)

    def set_column_widths(self, name_width: int, statistic_width: int) -> None:
        """Align stable row columns while leaving recommendation space flexible."""

        self.name_label.setFixedWidth(name_width)
        self.reduced_chi_square_label.setFixedWidth(statistic_width)

    def set_content(
        self,
        *,
        name: str,
        reduced_chi_square: str,
        recommendation: str,
        execution_state: str,
        execution_description: str,
    ) -> None:
        self.name_label.setText(name)
        self.reduced_chi_square_label.setText(reduced_chi_square)
        self.recommendation_label.setText(recommendation)
        self.set_execution_state(execution_state, execution_description)

    def set_execution_state(self, state: str, description: str) -> None:
        self.status_dot.setProperty("executionState", state)
        self.status_dot.setToolTip(description)
        self.status_dot.setAccessibleName(description)
        self.status_dot.setAccessibleDescription(description)
        style = self.status_dot.style()
        style.unpolish(self.status_dot)
        style.polish(self.status_dot)

    def set_accessible_content(self, name: str, description: str) -> None:
        self.setAccessibleName(name)
        self.setAccessibleDescription(description)
        self.setToolTip(description)

    def item_size_hint(self) -> QSize:
        """Return content-derived row height without demanding pane width."""

        layout = self.layout()
        assert layout is not None
        return QSize(0, layout.sizeHint().height())

    def set_checked(self, checked: bool) -> None:
        blocker = QSignalBlocker(self.checkbox)
        self.checkbox.setChecked(checked)
        del blocker

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self.activated.emit()
        super().mousePressEvent(event)


class _ComparisonScrollArea(QScrollArea):
    """Report viewport changes so the small-multiple grid can reflow."""

    viewport_resized = Signal()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.viewport_resized.emit()


class _SelectedCandidatesWorker(QObject):
    """Run one public selected-candidate continuation away from the UI thread."""

    completed = Signal(object)
    failed = Signal(object)

    def __init__(
        self,
        project: WorkflowProject,
        draft: ManualFitDraft,
        outcome: SingleQAutoFitOutcome,
        candidates: tuple[StandardModelCandidate, ...],
        cancel_event: Event,
    ) -> None:
        super().__init__()
        self._project = project
        self._draft = draft
        self._outcome = outcome
        self._candidates = candidates
        self._cancel_event = cancel_event

    @Slot()
    def run(self) -> None:
        try:
            result = continue_selected_auto_fit_candidates(
                self._project,
                self._draft,
                self._outcome,
                self._candidates,
                cancel_requested=self._cancel_event.is_set,
            )
        except Exception as error:  # noqa: BLE001 - transported to the GUI thread
            self.failed.emit(error)
        else:
            self.completed.emit(result)


class AutoFitCandidateDialog(QDialog):
    """Compare AutoFit candidates and retain explicitly selected Q branches."""

    def __init__(
        self,
        outcome: SingleQAutoFitOutcome,
        parent: QWidget | None = None,
        *,
        project: WorkflowProject | None = None,
        draft: ManualFitDraft | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("autoFitDialog")
        self.setWindowTitle("AutoFit")
        self.setModal(True)
        self.resize(980, 640)
        self.setMinimumSize(760, 500)
        self.outcome = outcome
        self._project = project
        self._draft = draft
        self.accepted_candidate: CandidateFitResult | None = None
        self.accepted_branches: tuple[AutoFitCandidateBranchResult, ...] = ()
        self._candidates = outcome.recommendation.candidate_results
        self._lorentzian_styles_by_candidate = {
            candidate.candidate: self._anchor_lorentzian_styles(candidate)
            for candidate in self._candidates
        }
        self._candidate_items: list[QListWidgetItem] = []
        self._candidate_rows: list[_CandidateRow] = []
        self._executed_branches: dict[
            StandardModelCandidate, AutoFitCandidateBranchResult
        ] = {}
        self.current_group_index = outcome.group_index
        self.y_scale = "linear"
        self.spectrum_axes: Axes | None = None
        self.residual_axes: Axes | None = None
        self.spectrum_axes_by_candidate: dict[StandardModelCandidate, Axes] = {}
        self.residual_axes_by_candidate: dict[StandardModelCandidate, Axes] = {}
        self._x_limits: tuple[float, float] | None = None
        self._y_limits: tuple[float, float] | None = None
        self._zoom_start: tuple[float, float] | None = None
        self._zoom_start_display: tuple[float, float] | None = None
        self._zoom_axes: Axes | None = None
        self._zoom_rectangle: Rectangle | None = None
        self._comparison_columns = 1
        self._drawing_comparison = False
        self._running = False
        self._cancel_event: Event | None = None
        self._worker: _SelectedCandidatesWorker | None = None
        self._worker_thread: QThread | None = None
        self._run_candidates: tuple[StandardModelCandidate, ...] = ()
        self._run_result: SelectedAutoFitMultiQResult | None = None
        self._run_error: Exception | None = None

        title = QLabel("AutoFit Candidates")
        title.setObjectName("ezqensDialogTitle")
        recommendation = outcome.recommendation
        summary = QLabel(
            " · ".join(
                (
                    recommendation.primary_family_support.value.replace("_", " "),
                    recommendation.primary_residual_adequacy.value.replace("_", " "),
                ),
            ).capitalize(),
        )
        summary.setObjectName("autoFitRecommendationSummary")
        summary.setProperty("secondary", True)
        summary.setWordWrap(True)

        self.candidate_list = QListWidget()
        self.candidate_list.setObjectName("autoFitCandidateList")
        self.candidate_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.candidate_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.candidate_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.candidate_list.setSpacing(0)
        self.candidate_list.setUniformItemSizes(True)
        for row, candidate in enumerate(self._candidates):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, row)
            recommended = candidate is recommendation.most_recommended
            accessible_description = self._candidate_accessible_description(candidate)
            item.setData(
                Qt.ItemDataRole.AccessibleTextRole,
                candidate.candidate.name,
            )
            item.setData(
                Qt.ItemDataRole.AccessibleDescriptionRole,
                accessible_description,
            )
            row_widget = _CandidateRow(
                candidate.candidate.name,
                self._candidate_reduced_chi_square_text(candidate),
                self._candidate_recommendation_text(candidate),
                *self._candidate_execution_state(candidate),
                eligible=candidate.success,
                recommended=recommended,
            )
            row_widget.set_accessible_content(
                candidate.candidate.name,
                accessible_description,
            )
            row_widget.checked_changed.connect(
                lambda checked, index=row: self._candidate_check_changed(index, checked)
            )
            row_widget.activated.connect(
                lambda index=row: self.candidate_list.setCurrentRow(index)
            )
            self.candidate_list.addItem(item)
            self.candidate_list.setItemWidget(item, row_widget)
            item.setSizeHint(row_widget.item_size_hint())
            self._candidate_items.append(item)
            self._candidate_rows.append(row_widget)
        if self._candidate_rows:
            name_width = max(
                row.name_label.sizeHint().width() for row in self._candidate_rows
            )
            statistic_width = max(
                row.reduced_chi_square_label.sizeHint().width()
                for row in self._candidate_rows
            )
            for item, candidate_row in zip(
                self._candidate_items,
                self._candidate_rows,
                strict=True,
            ):
                candidate_row.set_column_widths(name_width, statistic_width)
                item.setSizeHint(candidate_row.item_size_hint())

        self.candidate_note_label = QLabel()
        self.candidate_note_label.setObjectName("autoFitRecommendationReason")
        self.candidate_note_label.setProperty("secondary", True)
        self.candidate_note_label.setWordWrap(True)

        self.run_selected_button = QPushButton("Run selected across Q")
        self.run_selected_button.setObjectName("autoFitRunSelectedButton")
        self.run_selected_button.clicked.connect(self.run_selected_across_q)
        self.selection_status_label = QLabel()
        self.selection_status_label.setObjectName("autoFitSelectionStatus")
        self.selection_status_label.setProperty("secondary", True)
        self.selection_status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("autoFitBusyIndicator")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(84)
        self.progress_bar.hide()
        self.progress_label = QLabel()
        self.progress_label.setObjectName("autoFitProgressLabel")
        self.progress_label.setProperty("secondary", True)
        self.progress_label.hide()
        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(7)
        run_row.addWidget(self.run_selected_button)
        run_row.addWidget(self.progress_bar)
        run_row.addStretch(1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(title)
        left_layout.addWidget(summary)
        left_layout.addWidget(self.candidate_list, 1)
        left_layout.addWidget(self.candidate_note_label)
        left_layout.addLayout(run_row)
        left_layout.addWidget(self.progress_label)
        left_layout.addWidget(self.selection_status_label)

        self.preview_title = QLabel()
        self.preview_title.setObjectName("autoFitPreviewTitle")
        self.preview_title.setProperty("secondary", True)
        self.previous_group_button = QPushButton("‹")
        self.previous_group_button.setObjectName("autoFitPreviousGroupButton")
        self.previous_group_button.setToolTip("Previous Q group")
        self.previous_group_button.clicked.connect(
            lambda: self.set_current_group(self.current_group_index - 1)
        )
        self.next_group_button = QPushButton("›")
        self.next_group_button.setObjectName("autoFitNextGroupButton")
        self.next_group_button.setToolTip("Next Q group")
        self.next_group_button.clicked.connect(
            lambda: self.set_current_group(self.current_group_index + 1)
        )
        preview_header = QHBoxLayout()
        preview_header.setContentsMargins(0, 0, 0, 0)
        preview_header.setSpacing(6)
        preview_header.addWidget(self.preview_title, 1)
        preview_header.addWidget(self.previous_group_button)
        preview_header.addWidget(self.next_group_button)

        self.preview_canvas = ScientificCanvas()
        self.preview_canvas.setObjectName("autoFitPreviewCanvas")
        self.preview_canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.preview_canvas.customContextMenuRequested.connect(
            self._show_preview_context_menu
        )
        self.preview_canvas.mpl_connect("scroll_event", self._on_scroll)
        self.preview_canvas.mpl_connect("button_press_event", self._on_button_press)
        self.preview_canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.preview_canvas.mpl_connect("button_release_event", self._on_button_release)
        self.comparison_scroll_area = _ComparisonScrollArea()
        self.comparison_scroll_area.setObjectName("autoFitComparisonScrollArea")
        self.comparison_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.comparison_scroll_area.setWidgetResizable(True)
        self.comparison_scroll_area.setWidget(self.preview_canvas)
        self.comparison_scroll_area.viewport_resized.connect(
            self._comparison_view_resized
        )
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addLayout(preview_header)
        right_layout.addWidget(self.comparison_scroll_area, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("autoFitCandidateSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 640])

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_or_reject)
        self.use_candidate_button = QPushButton("Apply")
        self.use_candidate_button.setObjectName("autoFitUseCandidateButton")
        self.use_candidate_button.setDefault(True)
        self.use_candidate_button.clicked.connect(self._use_focused_candidate)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.use_candidate_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(splitter, 1)
        layout.addLayout(buttons)

        self.candidate_list.currentRowChanged.connect(self._focus_candidate)
        initial_row = self._initial_row()
        if 0 <= initial_row < len(self._candidate_rows):
            self._candidate_rows[initial_row].set_checked(
                self._candidates[initial_row].success
            )
        self.candidate_list.setCurrentRow(initial_row)
        self._update_candidate_rows()
        self._sync_controls()

    @property
    def focused_candidate(self) -> CandidateFitResult | None:
        """Return the active candidate independently of checkbox state."""

        row = self.candidate_list.currentRow()
        return self._candidates[row] if 0 <= row < len(self._candidates) else None

    @property
    def checked_candidates(self) -> tuple[CandidateFitResult, ...]:
        """Return checked usable evidence in canonical candidate-list order."""

        return tuple(
            candidate
            for candidate, row in zip(
                self._candidates, self._candidate_rows, strict=True
            )
            if row.checkbox.isChecked() and candidate.success
        )

    @property
    def executed_branches(
        self,
    ) -> tuple[AutoFitCandidateBranchResult, ...]:
        """Return cached branches in canonical candidate-list order."""

        return tuple(
            self._executed_branches[candidate.candidate]
            for candidate in self._candidates
            if candidate.candidate in self._executed_branches
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def candidate_checkbox(self, row: int) -> QCheckBox:
        """Return one row checkbox for focused interaction tests."""

        return self._candidate_rows[row].checkbox

    def set_candidate_checked(self, row: int, checked: bool) -> bool:
        """Set one eligible checkbox without changing the active row."""

        candidate = self._candidates[row]
        if checked and not candidate.success:
            return False
        self._candidate_rows[row].checkbox.setChecked(checked)
        return True

    def reject(self) -> None:
        """Cancel running work, or close without retaining a final choice."""

        if self._running:
            self.request_cancel()
            return
        self.accepted_candidate = None
        self.accepted_branches = ()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._running:
            self.request_cancel()
            event.ignore()
            return
        super().closeEvent(event)

    def _initial_row(self) -> int:
        recommended = self.outcome.recommendation.most_recommended
        if recommended is not None:
            return next(
                index
                for index, candidate in enumerate(self._candidates)
                if candidate is recommended
            )
        return next(
            (
                index
                for index, candidate in enumerate(self._candidates)
                if candidate.success
            ),
            0,
        )

    def _candidate_accessible_description(self, candidate: CandidateFitResult) -> str:
        _state, execution_description = self._candidate_execution_state(candidate)
        return "\n".join(
            (
                " · ".join(
                    filter(
                        None,
                        (
                            candidate.candidate.name,
                            self._candidate_recommendation_description(candidate),
                        ),
                    )
                ),
                self._candidate_statistics_text(candidate),
                execution_description,
            )
        )

    def _candidate_recommendation_text(self, candidate: CandidateFitResult) -> str:
        if candidate is self.outcome.recommendation.most_recommended:
            return "Most Recommended"
        if candidate is self.outcome.recommendation.best_supported_candidate:
            return "Best Supported"
        return ""

    def _candidate_recommendation_description(
        self, candidate: CandidateFitResult
    ) -> str:
        labels: list[str] = []
        if candidate is self.outcome.recommendation.most_recommended:
            labels.append("Most Recommended")
        if candidate is self.outcome.recommendation.best_supported_candidate:
            labels.append("Best Supported")
        return " · ".join(labels)

    @staticmethod
    def _candidate_statistics_text(candidate: CandidateFitResult) -> str:
        if candidate.fit is None:
            return "Statistics unavailable"
        statistics = candidate.fit.statistics
        return (
            f"AICc {statistics.aicc:.4g}  ·  BIC {statistics.bic:.4g}  ·  "
            f"reduced χ² {statistics.reduced_chi_square:.4g}"
        )

    @staticmethod
    def _candidate_reduced_chi_square_text(candidate: CandidateFitResult) -> str:
        if candidate.fit is None:
            return "χ²/ν —"
        return f"χ²/ν {candidate.fit.statistics.reduced_chi_square:.4g}"

    def _candidate_check_changed(self, row: int, checked: bool) -> None:
        candidate = self._candidates[row]
        if checked and not candidate.success:
            self._candidate_rows[row].set_checked(False)
            return
        self._capture_view_state()
        self._update_candidate_rows()
        self._sync_controls()
        self._draw_comparison()

    def _focus_candidate(self, _row: int) -> None:
        self._capture_view_state()
        self._sync_controls()
        self._draw_comparison()

    def _candidate_execution_state(
        self, candidate: CandidateFitResult
    ) -> tuple[str, str]:
        if not candidate.success:
            return "red", "Anchor unavailable"
        cached = self._executed_branches.get(candidate.candidate)
        if cached is None:
            return "orange", "Anchor ready"
        if cached.branch_result.status is MultiQExecutionStatus.COMPLETED:
            return "green", "All Q groups processed"
        return "orange", "Partial Q coverage"

    def _update_candidate_rows(self) -> None:
        for item, row, candidate in zip(
            self._candidate_items,
            self._candidate_rows,
            self._candidates,
            strict=True,
        ):
            execution_state, execution_description = self._candidate_execution_state(
                candidate
            )
            row.set_content(
                name=candidate.candidate.name,
                reduced_chi_square=self._candidate_reduced_chi_square_text(candidate),
                recommendation=self._candidate_recommendation_text(candidate),
                execution_state=execution_state,
                execution_description=execution_description,
            )
            accessible_description = self._candidate_accessible_description(candidate)
            item.setData(
                Qt.ItemDataRole.AccessibleDescriptionRole,
                accessible_description,
            )
            row.set_accessible_content(
                candidate.candidate.name,
                accessible_description,
            )
            item.setSizeHint(row.item_size_hint())

    def _sync_controls(self) -> None:
        checked = self.checked_candidates
        pending = tuple(
            item for item in checked if item.candidate not in self._executed_branches
        )
        has_execution_context = self._project is not None and self._draft is not None
        self.run_selected_button.setEnabled(
            has_execution_context and bool(pending) and not self._running
        )
        if not has_execution_context:
            candidate = self.focused_candidate
            self.use_candidate_button.setText("Use Candidate")
            self.use_candidate_button.setEnabled(
                not self._running and candidate is not None and candidate.success
            )
        else:
            self.use_candidate_button.setText(
                "Save selected results" if len(checked) > 1 else "Apply"
            )
            self.use_candidate_button.setEnabled(
                bool(checked)
                and not self._running
                and all(item.candidate in self._executed_branches for item in checked)
            )
        executed_checked = sum(
            item.candidate in self._executed_branches for item in checked
        )
        if checked:
            self.selection_status_label.setText(
                f"{len(checked)} selected · {executed_checked} executed · "
                f"{len(pending)} not run"
            )
        else:
            self.selection_status_label.setText("Select candidates to compare")
        self._update_calculated_groups()
        self._update_active_candidate_note()

        navigable = bool(executed_checked) and not self._running
        group_count = self._group_count()
        self.previous_group_button.setEnabled(
            navigable and self.current_group_index > 0
        )
        self.next_group_button.setEnabled(
            navigable and self.current_group_index + 1 < group_count
        )

    def _calculated_group_counts(self) -> tuple[int, int]:
        group_count = self._group_count()
        calculated = 0
        for candidate in self.checked_candidates:
            cached = self._executed_branches.get(candidate.candidate)
            if cached is None:
                calculated += 1
            elif cached.branch_result.status is MultiQExecutionStatus.COMPLETED:
                calculated += group_count
            else:
                calculated += sum(
                    outcome.status is not MultiQFitStatus.NOT_RUN
                    for outcome in cached.branch_result.outcomes
                )
        return calculated, len(self.checked_candidates) * group_count

    def _update_calculated_groups(self) -> None:
        calculated, total = self._calculated_group_counts()
        self.progress_label.setText(f"Calculated groups: {calculated} / {total}")
        self.progress_label.show()

    def _update_active_candidate_note(self) -> None:
        candidate = self.focused_candidate
        self.candidate_note_label.setText(self._candidate_note(candidate))

    def _candidate_note(self, candidate: CandidateFitResult | None) -> str:
        if candidate is None:
            return "No candidate-specific AutoFit note."
        recommendation = self.outcome.recommendation
        messages: list[str] = []
        assessments = (
            *recommendation.transition_assessments,
            recommendation.additional_complexity,
        )
        for assessment in assessments:
            proposed = assessment.proposed_candidate
            if proposed is not None and proposed.candidate == candidate.candidate:
                messages.append(assessment.reason)
        messages.extend(
            limitation.message
            for limitation in recommendation.interpretation_limitations
            if limitation.candidate.candidate == candidate.candidate
        )
        messages.extend(
            warning.message
            for warning in recommendation.scientific_warnings
            if warning.candidate is not None
            and warning.candidate.candidate == candidate.candidate
        )
        unique_messages = tuple(dict.fromkeys(messages))
        return (
            "\n".join(f"• {message}" for message in unique_messages)
            if unique_messages
            else "No candidate-specific AutoFit note."
        )

    def set_y_scale(self, scale: str) -> bool:
        """Change the shared comparison spectrum presentation scale."""

        if scale not in {"linear", "symlog", "log"}:
            raise ValueError(
                "AutoFit preview y scale must be 'linear', 'symlog', or 'log'"
            )
        if scale == self.y_scale:
            return False
        self._capture_view_state()
        self.y_scale = scale
        self._y_limits = None
        self._draw_comparison()
        return True

    def set_current_group(self, group_index: int) -> bool:
        """Move every checked executed branch to one shared Q group."""

        if not 0 <= group_index < self._group_count():
            return False
        if group_index == self.current_group_index:
            return False
        self._capture_view_state()
        self.current_group_index = group_index
        self._sync_controls()
        self._draw_comparison()
        return True

    def reset_view(self) -> None:
        """Restore the full current-comparison extent without refitting."""

        self._x_limits = None
        self._y_limits = None
        self._cancel_zoom(redraw=False)
        self._draw_comparison()

    def _build_preview_context_menu(self) -> QMenu:
        menu = QMenu(self)
        scale_menu = menu.addMenu("Y Scale")
        assert scale_menu is not None
        scale_group = QActionGroup(scale_menu)
        scale_group.setExclusive(True)
        for label, scale in (
            ("Linear", "linear"),
            ("SymLog", "symlog"),
            ("Log", "log"),
        ):
            action = scale_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.y_scale == scale)
            action.triggered.connect(
                lambda _checked, value=scale: self.set_y_scale(value)
            )
            scale_group.addAction(action)
        menu.addSeparator()
        reset_action = menu.addAction("Reset View")
        reset_action.triggered.connect(self.reset_view)
        return menu

    def _show_preview_context_menu(self, position: QPoint) -> None:
        if self.spectrum_axes is None:
            return
        menu = self._build_preview_context_menu()
        menu.exec(self.preview_canvas.mapToGlobal(position))

    def _comparison_entries(self) -> tuple[_ComparisonEntry, ...]:
        selected = self.checked_candidates
        if not selected:
            focused = self.focused_candidate
            selected = (focused,) if focused is not None else ()
        compared: list[_ComparisonEntry] = []
        for candidate in selected:
            fit: FitResult | None = None
            if self.current_group_index == self.outcome.group_index:
                fit = candidate.fit
                status = "Anchor ready" if fit is not None else "Unavailable"
            else:
                cached = self._executed_branches.get(candidate.candidate)
                if cached is None:
                    status = "Not run"
                else:
                    group = cached.branch_result.outcome(self.current_group_index)
                    status = group.status.value.replace("_", " ").title()
                    if group.status is MultiQFitStatus.SUCCESS:
                        fit = group.fit_result
            compared.append((candidate, fit, status))
        return tuple(compared)

    def _draw_comparison(self) -> None:
        compared = self._comparison_entries()
        focused = self.focused_candidate
        active = focused.candidate if focused is not None else None
        selected_count = len(self.checked_candidates)
        self.preview_title.setText(
            f"Group {self.current_group_index + 1} / {self._group_count()} · "
            f"{selected_count} selected"
        )
        self._draw_panels(compared, active_candidate=active)

    def _draw_candidate(self, candidate: CandidateFitResult | None) -> None:
        """Draw one supplied candidate for compatibility with focused plot tests."""

        self.preview_title.setText(
            "No candidate selected" if candidate is None else candidate.candidate.name
        )
        compared: tuple[_ComparisonEntry, ...] = (
            (
                (
                    candidate,
                    candidate.fit,
                    "Anchor ready"
                    if candidate.fit is not None
                    else candidate.error_message or "Unavailable",
                ),
            )
            if candidate is not None
            else ()
        )
        self._draw_panels(
            compared,
            active_candidate=candidate.candidate if candidate is not None else None,
        )

    def _draw_panels(
        self,
        compared: tuple[_ComparisonEntry, ...],
        *,
        active_candidate: StandardModelCandidate | None,
    ) -> None:
        self._drawing_comparison = True
        figure = self.preview_canvas.figure
        self._cancel_zoom(redraw=False)
        figure.clear()
        self.spectrum_axes = None
        self.residual_axes = None
        self.spectrum_axes_by_candidate.clear()
        self.residual_axes_by_candidate.clear()
        figure.set_facecolor(SCIENTIFIC_BACKGROUND)
        if not compared:
            self._comparison_columns = 1
            self._set_comparison_canvas_minimum(1, 1)
            axes = figure.add_subplot(111)
            axes.set_facecolor(SCIENTIFIC_BACKGROUND)
            axes.text(
                0.5,
                0.5,
                "No candidate selected",
                transform=axes.transAxes,
                ha="center",
                va="center",
                wrap=True,
            )
            axes.set_axis_off()
            self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]
            self._drawing_comparison = False
            return

        columns = self._comparison_column_count(len(compared))
        self._comparison_columns = columns
        rows = (len(compared) + columns - 1) // columns
        self._set_comparison_canvas_minimum(columns, rows)
        outer_grid = figure.add_gridspec(rows, columns, hspace=0.2, wspace=0.16)
        energy, intensity, uncertainty = self._measurement_for_group(
            self.current_group_index
        )
        shared_linthresh: float | None = None
        if self.y_scale == "symlog":
            scale_values: list[np.ndarray] = [intensity]
            for _candidate, fit, _status in compared:
                if fit is None:
                    continue
                scale_values.append(fit.evaluation.total)
                scale_values.extend(
                    curve.values for curve in fit.evaluation.component_curves
                )
                scale_values.append(fit.evaluation.background)
            shared_linthresh = symlog_linthresh(np.concatenate(scale_values))

        for index, (candidate, fit, status) in enumerate(compared):
            panel_grid = outer_grid[index // columns, index % columns].subgridspec(
                2, 1, height_ratios=(4.0, 1.0), hspace=0.06
            )
            spectrum_axes = figure.add_subplot(panel_grid[0])
            residual_axes = figure.add_subplot(panel_grid[1], sharex=spectrum_axes)
            is_active = candidate.candidate == active_candidate
            for axes in (spectrum_axes, residual_axes):
                axes.set_facecolor(SCIENTIFIC_BACKGROUND)
                axes.tick_params(colors=SCIENTIFIC_MEASURED_COLOR)
                for spine in axes.spines.values():
                    spine.set_color(SCIENTIFIC_UNCERTAINTY_COLOR)
                    spine.set_linewidth(1.05 if is_active else 0.8)
            spectrum_axes.set_title(
                candidate.candidate.name,
                loc="left",
                color=(
                    SCIENTIFIC_TOTAL_FIT_COLOR
                    if is_active
                    else SCIENTIFIC_MEASURED_COLOR
                ),
                fontweight="bold" if is_active else "normal",
                fontsize=9.5,
            )
            measured_points = np.isfinite(energy) & np.isfinite(intensity)
            if self.y_scale == "log":
                spectrum_axes.set_yscale("log", nonpositive="mask")
                measured_points &= intensity > 0.0
            elif self.y_scale == "symlog":
                assert shared_linthresh is not None
                spectrum_axes.set_yscale("symlog", linthresh=shared_linthresh)
            spectrum_axes.errorbar(
                energy[measured_points],
                intensity[measured_points],
                yerr=uncertainty[measured_points],
                fmt=".",
                color=SCIENTIFIC_MEASURED_COLOR,
                ecolor=SCIENTIFIC_UNCERTAINTY_COLOR,
                elinewidth=0.8,
                capsize=1.5,
                label="Measured",
                zorder=3,
            )
            if fit is not None:
                spectrum_axes.plot(
                    fit.evaluation.energy,
                    self._spectrum_display_values(fit.evaluation.total),
                    color=SCIENTIFIC_TOTAL_FIT_COLOR,
                    linewidth=1.7,
                    label="Total fit",
                    zorder=4,
                )
                for curve in fit.evaluation.component_curves:
                    if curve.component.family is ComponentFamily.BACKGROUND:
                        continue
                    label, color = self._component_style(
                        candidate.candidate,
                        curve.component,
                    )
                    spectrum_axes.plot(
                        fit.evaluation.energy,
                        self._spectrum_display_values(curve.values),
                        color=color,
                        linewidth=0.95,
                        linestyle="--",
                        alpha=0.82,
                        label=label,
                        zorder=2,
                    )
                model = fit.fitted_model or fit.configuration
                if model.background is not BackgroundModel.NONE:
                    spectrum_axes.plot(
                        fit.evaluation.energy,
                        self._spectrum_display_values(fit.evaluation.background),
                        color=SCIENTIFIC_BACKGROUND_COLOR,
                        linewidth=1.0,
                        alpha=0.78,
                        label="Background",
                        zorder=2,
                    )
                residual_axes.plot(
                    fit.evaluation.energy,
                    fit.standardized_residuals,
                    color=SCIENTIFIC_RESIDUAL_COLOR,
                    linewidth=1.0,
                    marker=".",
                    markersize=2.5,
                    label="Std. residual",
                )
            else:
                spectrum_axes.text(
                    0.5,
                    0.5,
                    status,
                    transform=spectrum_axes.transAxes,
                    ha="center",
                    va="center",
                    color="#676764",
                    fontsize=9,
                )
                residual_axes.text(
                    0.5,
                    0.5,
                    "No residual available",
                    transform=residual_axes.transAxes,
                    ha="center",
                    va="center",
                    color="#676764",
                    fontsize=8,
                )
            spectrum_axes.set_ylabel("Intensity")
            spectrum_axes.grid(True, color="#e8e8e8", linewidth=0.6)
            spectrum_axes.tick_params(axis="x", labelbottom=False)
            if spectrum_axes.lines or spectrum_axes.containers:
                spectrum_axes.legend(loc="best", fontsize=7.0)
            if self.y_scale == "log" and not self._has_positive_spectrum_value(fit):
                spectrum_axes.text(
                    0.5,
                    0.5,
                    "No positive values available for Log Y display",
                    transform=spectrum_axes.transAxes,
                    ha="center",
                    va="center",
                    color="#676764",
                    fontsize=9,
                )
            residual_axes.axhline(
                0.0,
                color=SCIENTIFIC_UNCERTAINTY_COLOR,
                linewidth=0.7,
                alpha=0.7,
            )
            energy_unit = (
                fit.provenance.energy_unit
                if fit is not None
                else self.outcome.scientific_context.selection.dataset.spectra[
                    self.current_group_index
                ].energy_unit
            )
            residual_axes.set_xlabel(f"Energy ({energy_unit})")
            residual_axes.set_ylabel("Std. residual")
            residual_axes.grid(True, color="#ededed", linewidth=0.5)
            self.spectrum_axes_by_candidate[candidate.candidate] = spectrum_axes
            self.residual_axes_by_candidate[candidate.candidate] = residual_axes

        self.spectrum_axes = (
            self.spectrum_axes_by_candidate.get(active_candidate)
            if active_candidate is not None
            else None
        ) or next(iter(self.spectrum_axes_by_candidate.values()), None)
        self.residual_axes = (
            self.residual_axes_by_candidate.get(active_candidate)
            if active_candidate is not None
            else None
        ) or next(iter(self.residual_axes_by_candidate.values()), None)
        self._apply_or_establish_shared_limits()
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]
        self._drawing_comparison = False

    def _comparison_column_count(self, candidate_count: int) -> int:
        available_width = max(
            self.comparison_scroll_area.viewport().width(),
            _MIN_COMPARISON_PANEL_WIDTH,
        )
        width_columns = max(1, available_width // _MIN_COMPARISON_PANEL_WIDTH)
        return min(candidate_count, width_columns, _MAX_COMPARISON_COLUMNS)

    def _set_comparison_canvas_minimum(self, columns: int, rows: int) -> None:
        self.preview_canvas.setMinimumSize(
            columns * _MIN_COMPARISON_PANEL_WIDTH,
            rows * _MIN_COMPARISON_PANEL_HEIGHT,
        )

    @Slot()
    def _comparison_view_resized(self) -> None:
        if self._drawing_comparison:
            return
        candidate_count = len(self._comparison_entries())
        if not candidate_count:
            return
        columns = self._comparison_column_count(candidate_count)
        if columns == self._comparison_columns:
            return
        self._capture_view_state()
        self._draw_comparison()

    @staticmethod
    def _anchor_lorentzian_styles(
        candidate: CandidateFitResult,
    ) -> dict[ComponentIdentity, tuple[str, str]]:
        fit = candidate.fit
        if fit is None:
            return {}
        return {
            component.identity: (
                f"L{index + 1}",
                SCIENTIFIC_LORENTZIAN_COLORS[
                    min(index, len(SCIENTIFIC_LORENTZIAN_COLORS) - 1)
                ],
            )
            for index, component in enumerate(fit.configuration.lorentzians)
        }

    def _component_style(
        self,
        candidate: StandardModelCandidate,
        identity: ComponentIdentity,
    ) -> tuple[str, str]:
        if identity.family is ComponentFamily.ELASTIC:
            return "Elastic", SCIENTIFIC_ELASTIC_COLOR
        if identity.family is ComponentFamily.LORENTZIAN:
            return self._lorentzian_styles_by_candidate.get(candidate, {}).get(
                identity,
                ("Lorentzian", SCIENTIFIC_LORENTZIAN_COLORS[0]),
            )
        return "Background", SCIENTIFIC_BACKGROUND_COLOR

    def _apply_or_establish_shared_limits(self) -> None:
        spectrum_axes = tuple(self.spectrum_axes_by_candidate.values())
        residual_axes = tuple(self.residual_axes_by_candidate.values())
        if not spectrum_axes:
            return
        if self._x_limits is None:
            x_limits = tuple(axes.get_xlim() for axes in spectrum_axes)
            self._x_limits = (
                min(float(limits[0]) for limits in x_limits),
                max(float(limits[1]) for limits in x_limits),
            )
        if self._y_limits is None:
            y_limits = tuple(axes.get_ylim() for axes in spectrum_axes)
            self._y_limits = (
                min(float(limits[0]) for limits in y_limits),
                max(float(limits[1]) for limits in y_limits),
            )
        for axes in spectrum_axes:
            axes.set_xlim(self._x_limits)
            if self.y_scale != "log" or min(self._y_limits) > 0.0:
                axes.set_ylim(self._y_limits)
        for axes in residual_axes:
            axes.set_xlim(self._x_limits)

    def _measurement_for_group(
        self, group_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if group_index == self.outcome.group_index:
            return (
                self.outcome.measured_energy,
                self.outcome.measured_intensity,
                self.outcome.measured_uncertainty,
            )
        selection = self.outcome.scientific_context.selection
        spectrum = selection.dataset.spectra[group_index]
        retained = selection.retained_mask(group_index)
        return (
            spectrum.energy[retained],
            spectrum.intensity[retained],
            spectrum.uncertainty[retained],
        )

    def _group_count(self) -> int:
        return len(self.outcome.scientific_context.selection.dataset.spectra)

    def run_selected_across_q(self) -> bool:
        """Run only checked candidates without cached branches."""

        if self._running or self._project is None or self._draft is None:
            return False
        pending = tuple(
            candidate.candidate
            for candidate in self.checked_candidates
            if candidate.candidate not in self._executed_branches
        )
        if not pending:
            return False
        self._running = True
        self._run_candidates = pending
        self._run_result = None
        self._run_error = None
        self._cancel_event = Event()
        self.progress_label.show()
        self.progress_bar.show()
        self.cancel_button.setText("Cancel run")
        self._sync_controls()

        thread = QThread(self)
        worker = _SelectedCandidatesWorker(
            self._project,
            self._draft,
            self.outcome,
            pending,
            self._cancel_event,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._execution_completed)
        worker.failed.connect(self._execution_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._execution_thread_finished)
        self._worker_thread = thread
        self._worker = worker
        thread.start()
        return True

    def request_cancel(self) -> bool:
        """Request cooperative cancellation through the public core contract."""

        if not self._running or self._cancel_event is None:
            return False
        self._cancel_event.set()
        self.cancel_button.setText("Cancelling…")
        self.cancel_button.setEnabled(False)
        return True

    @Slot(object)
    def _execution_completed(self, result: object) -> None:
        if not isinstance(result, SelectedAutoFitMultiQResult):
            self._run_error = TypeError(
                "selected-candidate worker returned an unexpected result"
            )
            return
        self._run_result = result
        for branch in result.branches:
            self._executed_branches[branch.candidate] = branch
        self._update_calculated_groups()
        self.selection_status_label.setText(
            "Run cancelled; completed and partial branches were retained."
            if result.status is MultiQExecutionStatus.CANCELLED
            else "Selected candidates are available across Q."
        )

    @Slot(object)
    def _execution_failed(self, error: object) -> None:
        self._run_error = (
            error if isinstance(error, Exception) else RuntimeError(str(error))
        )
        self.selection_status_label.setText(f"Run failed: {self._run_error}")

    @Slot()
    def _execution_thread_finished(self) -> None:
        self._running = False
        self._worker = None
        self._worker_thread = None
        self._cancel_event = None
        self.progress_bar.hide()
        self.cancel_button.setText("Cancel")
        self.cancel_button.setEnabled(True)
        self._update_candidate_rows()
        self._sync_controls()
        if self._run_error is not None:
            self.selection_status_label.setText(f"Run failed: {self._run_error}")
        elif self._run_result is not None:
            self.selection_status_label.setText(
                "Run cancelled; completed and partial branches were retained."
                if self._run_result.status is MultiQExecutionStatus.CANCELLED
                else "Selected candidates are available across Q."
            )
        self._draw_comparison()

    def _cancel_or_reject(self) -> None:
        if self._running:
            self.request_cancel()
        else:
            self.reject()

    def _selected_executed_branches(
        self,
    ) -> tuple[AutoFitCandidateBranchResult, ...]:
        checked = self.checked_candidates
        if not checked or any(
            item.candidate not in self._executed_branches for item in checked
        ):
            return ()
        return tuple(self._executed_branches[item.candidate] for item in checked)

    def _use_focused_candidate(self) -> None:
        if self._project is None or self._draft is None:
            candidate = self.focused_candidate
            if candidate is None or not candidate.success:
                return
            self.accepted_candidate = candidate
            self.accept()
            return
        branches = self._selected_executed_branches()
        if not branches:
            return
        self.accepted_branches = branches
        self.accepted_candidate = branches[0].anchor_evidence
        self.accept()

    def _capture_view_state(self) -> None:
        if self.spectrum_axes is None:
            return
        x_limits = self.spectrum_axes.get_xlim()
        y_limits = self.spectrum_axes.get_ylim()
        self._x_limits = (float(x_limits[0]), float(x_limits[1]))
        self._y_limits = (float(y_limits[0]), float(y_limits[1]))

    def _on_scroll(self, event: MouseEvent) -> None:
        if event.x is None or event.y is None:
            return
        touched = event.inaxes
        spectrum_axes = tuple(self.spectrum_axes_by_candidate.values())
        residual_axes = tuple(self.residual_axes_by_candidate.values())
        if touched not in (*spectrum_axes, *residual_axes):
            return
        assert touched is not None
        x_cursor, y_cursor = touched.transData.inverted().transform((event.x, event.y))
        scale = 0.8 if event.button == "up" else 1.25
        current_x_limits = self._x_limits or touched.get_xlim()
        self._x_limits = zoom_limits(current_x_limits, float(x_cursor), scale)
        if touched in spectrum_axes:
            current_y_limits = self._y_limits or touched.get_ylim()
            self._y_limits = zoom_limits(current_y_limits, float(y_cursor), scale)
        self._apply_or_establish_shared_limits()
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_button_press(self, event: MouseEvent) -> None:
        touched = event.inaxes
        if (
            event.button is not MouseButton.LEFT
            or touched not in self.spectrum_axes_by_candidate.values()
            or event.xdata is None
            or event.ydata is None
        ):
            return
        if getattr(event, "dblclick", False):
            self.reset_view()
            return
        display_point = self._event_display_point(event)
        if display_point is None:
            return
        start = (float(event.xdata), float(event.ydata))
        self._cancel_zoom(redraw=False)
        self._zoom_start = start
        self._zoom_start_display = display_point
        assert touched is not None
        self._zoom_axes = touched
        self._zoom_rectangle = Rectangle(
            start,
            0.0,
            0.0,
            facecolor=SCIENTIFIC_RESIDUAL_COLOR,
            edgecolor=SCIENTIFIC_TOTAL_FIT_COLOR,
            alpha=0.16,
            linewidth=1.0,
            zorder=7,
        )
        touched.add_patch(self._zoom_rectangle)
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_mouse_motion(self, event: MouseEvent) -> None:
        if (
            self._zoom_start is None
            or self._zoom_rectangle is None
            or event.inaxes is not self._zoom_axes
            or event.xdata is None
            or event.ydata is None
        ):
            return
        start_x, start_y = self._zoom_start
        end_x, end_y = float(event.xdata), float(event.ydata)
        self._zoom_rectangle.set_bounds(
            min(start_x, end_x),
            min(start_y, end_y),
            abs(end_x - start_x),
            abs(end_y - start_y),
        )
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _on_button_release(self, event: MouseEvent) -> None:
        start = self._zoom_start
        start_display = self._zoom_start_display
        if start is None or start_display is None:
            return
        if (
            event.inaxes is not self._zoom_axes
            or event.xdata is None
            or event.ydata is None
        ):
            self._cancel_zoom()
            return
        end = (float(event.xdata), float(event.ydata))
        end_display = self._event_display_point(event)
        self._cancel_zoom(redraw=False)
        if (
            end_display is None
            or max(
                abs(end_display[0] - start_display[0]),
                abs(end_display[1] - start_display[1]),
            )
            < _ZOOM_DRAG_THRESHOLD_PX
        ):
            self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        x_limits = (min(start[0], end[0]), max(start[0], end[0]))
        y_limits = (min(start[1], end[1]), max(start[1], end[1]))
        if x_limits[0] == x_limits[1] or y_limits[0] == y_limits[1]:
            self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]
            return
        self._x_limits = x_limits
        self._y_limits = y_limits
        self._apply_or_establish_shared_limits()
        self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _event_display_point(self, event: MouseEvent) -> tuple[float, float] | None:
        if event.x is not None and event.y is not None:
            return float(event.x), float(event.y)
        if self._zoom_axes is None or event.xdata is None or event.ydata is None:
            return None
        point = self._zoom_axes.transData.transform((event.xdata, event.ydata))
        return float(point[0]), float(point[1])

    def _cancel_zoom(self, *, redraw: bool = True) -> None:
        rectangle = self._zoom_rectangle
        self._zoom_rectangle = None
        self._zoom_start = None
        self._zoom_start_display = None
        self._zoom_axes = None
        if rectangle is not None and rectangle.axes is not None:
            rectangle.remove()
            if redraw:
                self.preview_canvas.draw_idle()  # type: ignore[no-untyped-call]

    def _spectrum_display_values(self, values: np.ndarray) -> np.ndarray:
        return log_display_values(values) if self.y_scale == "log" else values

    def _has_positive_spectrum_value(
        self,
        fit: FitResult | None,
    ) -> bool:
        _energy, intensity, _uncertainty = self._measurement_for_group(
            self.current_group_index
        )
        arrays = [intensity]
        if fit is not None:
            arrays.append(fit.evaluation.total)
            arrays.extend(curve.values for curve in fit.evaluation.component_curves)
        return any(np.any(np.isfinite(values) & (values > 0.0)) for values in arrays)
