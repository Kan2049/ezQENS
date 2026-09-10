"""Compact inline Q assignment editor using public domain constructors."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ezqens.domain import ImportDiagnostic, QBins, ReducedDataset, uniform_q_bins
from ezqens.gui.dialogs import show_message_dialog
from ezqens.gui.theme import DEFAULT_LAYOUT_TOKENS
from ezqens.io import DAVEQBinsResult, parse_dave_q_bins

Q_CENTER = "Q Center"
Q_EDGES = "Q-bin Edge"
REGULAR_GRID = "Regular grid"
EXPLICIT_VALUES = "Explicit values"


def parse_q_values(text: str) -> np.ndarray:
    """Parse ordinary copied numeric rows or columns without repairing values."""

    tokens = text.replace(",", " ").replace(";", " ").split()
    if not tokens:
        raise ValueError("Enter one Q value per Group, or paste a numeric column")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as error:
        raise ValueError("Q values must be numeric") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("Q values must be finite")
    return values


class QAssignmentEditor(QFrame):
    """One compact editing surface for every Q-assignment entry point."""

    q_bins_applied = Signal(object, object)
    save_q_method_requested = Signal(object)
    q_file_imported = Signal(object, object)
    q_center_draft_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("qAssignmentEditor")
        self.dataset: ReducedDataset | None = None
        self._q_values: np.ndarray | None = None
        self._edges: np.ndarray | None = None
        self._edited_representation: str | None = None
        self._text_edited_representation: str | None = None
        self._visible_representation: str | None = None
        self._visible_input_mode = REGULAR_GRID
        self._regular_exact_values: dict[str, float] = {}
        self._setting_regular_numeric_text = False
        self._pending_singleton_field: str | None = None
        self._diagnostics: tuple[ImportDiagnostic, ...] = ()
        self._available_methods: tuple[tuple[str, QBins], ...] = ()
        self._q_center_slots: list[float | None] = []
        self._apply_handler: (
            Callable[[QBins, tuple[ImportDiagnostic, ...]], bool] | None
        ) = None

        self.representation_combo = QComboBox()
        self.representation_combo.setObjectName("qRepresentationCombo")
        self.representation_combo.addItems([Q_CENTER, Q_EDGES])
        self.representation_combo.currentTextChanged.connect(self._show_representation)

        self.start_edit = QLineEdit()
        self.start_edit.setObjectName("qStartEdit")
        self.start_edit.setPlaceholderText("Start")
        self.step_edit = QLineEdit()
        self.step_edit.setObjectName("qStepEdit")
        self.step_edit.setPlaceholderText("Step")
        self.end_edit = QLineEdit()
        self.end_edit.setObjectName("qEndEdit")
        self.end_edit.setPlaceholderText("End")
        self.groups_edit = QLineEdit()
        self.groups_edit.setObjectName("qGroupCountEdit")
        self.groups_edit.setReadOnly(True)
        self.groups_edit.setMaximumWidth(54)
        for key, field in (
            ("start", self.start_edit),
            ("step", self.step_edit),
            ("end", self.end_edit),
        ):
            field.textChanged.connect(
                lambda _text, field_key=key: self._regular_field_changed(field_key)
            )
            field.editingFinished.connect(
                lambda field_key=key: self._regular_field_editing_finished(field_key)
            )

        self.uniform_fields = QWidget()
        uniform_layout = QHBoxLayout(self.uniform_fields)
        uniform_layout.setContentsMargins(0, 0, 0, 0)
        uniform_layout.setSpacing(4)
        for label, field in (
            ("Start", self.start_edit),
            ("Step", self.step_edit),
            ("End", self.end_edit),
            ("Groups", self.groups_edit),
        ):
            uniform_layout.addWidget(QLabel(label))
            field.setMaximumWidth(92)
            uniform_layout.addWidget(field)
        self.groups_edit.setMaximumWidth(54)

        self.input_mode_combo = QComboBox()
        self.input_mode_combo.setObjectName("qInputModeCombo")
        self.input_mode_combo.addItems([REGULAR_GRID, EXPLICIT_VALUES])
        self.input_mode_combo.currentTextChanged.connect(self._input_mode_changed)
        self._uniform_layout = uniform_layout

        self.values_edit = QPlainTextEdit()
        self.values_edit.setObjectName("qValuesEdit")
        self.values_edit.setPlaceholderText(
            "Paste Q values (spaces, commas, tabs, or new lines)",
        )
        self.values_edit.setFixedHeight(30)
        self.values_edit.textChanged.connect(self._mark_edited)
        self.values_edit.textChanged.connect(self._refresh_preview)
        self.values_edit.hide()

        self.preview_label = QLabel()
        self.preview_label.setObjectName("qEditorPreview")
        self.preview_label.setProperty("secondary", True)
        self.preview_label.setWordWrap(True)

        self.status_label = QLabel()
        self.status_label.setObjectName("qEditorStatus")
        self.status_label.setProperty("secondary", True)

        self.import_button = QPushButton("Import…")
        self.import_button.setObjectName("qImportButton")
        self.import_button.setToolTip("Import a Q assignment from a DAVE file")
        self.import_button.clicked.connect(self._prompt_dave_import)

        self.use_method_button = QPushButton("Use Method…")
        self.use_method_button.setObjectName("useQMethodButton")
        self.use_method_button.setToolTip(
            "Load a saved Project Q Method into this draft"
        )
        self._method_menu = QMenu(self.use_method_button)
        self.use_method_button.clicked.connect(self._show_method_menu)
        self._rebuild_method_menu()

        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("applyQButton")
        self.apply_button.setToolTip("Apply this Q assignment to the open dataset")
        self.apply_button.clicked.connect(self.apply)
        self.save_method_button = QPushButton("Save as Q Method…")
        self.save_method_button.setObjectName("saveQMethodButton")
        self.save_method_button.setToolTip("Save this Q assignment for this Project")
        self.save_method_button.clicked.connect(self._save_as_method)
        self.close_button = QToolButton()
        self.close_button.setObjectName("closeQEditorButton")
        self.close_button.setText("Close")
        self.close_button.setToolTip("Close Q assignment")
        self.close_button.clicked.connect(self.hide)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Q assignment")
        title.setObjectName("qAssignmentTitle")
        top.addWidget(title)
        top.addWidget(self.representation_combo)
        top.addWidget(self.input_mode_combo)
        top.addWidget(self.uniform_fields)
        top.addStretch(1)
        top.addWidget(self.apply_button)

        feedback = QHBoxLayout()
        feedback.setContentsMargins(0, 0, 0, 0)
        feedback.setSpacing(10)
        feedback.addWidget(self.preview_label)
        feedback.addWidget(self.status_label)
        feedback.addStretch(1)
        self._feedback_layout = feedback

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)
        actions.addWidget(self.import_button)
        actions.addWidget(self.use_method_button)
        actions.addStretch(1)
        actions.addWidget(self.save_method_button)
        actions.addWidget(self.close_button)
        self._actions_layout = actions

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 5)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self.values_edit)
        layout.addLayout(feedback)
        layout.addLayout(actions)
        self._layout = layout
        for control in (
            self.representation_combo,
            self.start_edit,
            self.step_edit,
            self.end_edit,
            self.groups_edit,
            self.input_mode_combo,
            self.import_button,
            self.use_method_button,
            self.apply_button,
            self.save_method_button,
            self.close_button,
        ):
            control.setMinimumHeight(DEFAULT_LAYOUT_TOKENS.control_height)
        self._uniform_layout.setSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        self._feedback_layout.setSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        self._actions_layout.setSpacing(DEFAULT_LAYOUT_TOKENS.row_spacing)
        self._layout.setSpacing(DEFAULT_LAYOUT_TOKENS.section_spacing)
        self.hide()

    def open_for_dataset(self, dataset: ReducedDataset) -> None:
        """Show an existing assignment without rewriting its representation."""

        self.dataset = dataset
        q_bins = dataset.q_bins
        self._q_values = None if q_bins is None else q_bins.q_values.copy()
        self._q_center_slots = (
            [float(value) for value in q_bins.q_values]
            if q_bins is not None
            else [None] * len(dataset.spectra)
        )
        self._edges = (
            None if q_bins is None or q_bins.edges is None else q_bins.edges.copy()
        )
        self._diagnostics = ()
        self._edited_representation = None
        self._text_edited_representation = None
        self._visible_representation = None
        self._regular_exact_values.clear()
        self._pending_singleton_field = None
        blocker = QSignalBlocker(self.representation_combo)
        self.representation_combo.setCurrentText(Q_CENTER)
        del blocker
        mode = (
            EXPLICIT_VALUES if self._requires_explicit_values(q_bins) else REGULAR_GRID
        )
        self._set_input_mode(mode, capture_current=False)
        self._show_representation(Q_CENTER)
        self.groups_edit.setText(str(len(dataset.spectra)))
        self.show()
        self.q_center_draft_changed.emit()

    def import_dave_file(self, path: str) -> DAVEQBinsResult:
        """Use the validated DAVE reader and leave data untouched on mismatch."""

        result = parse_dave_q_bins(path)
        self._validate_count(result.q_bins)
        self.load_q_bins_draft(result.q_bins, diagnostics=result.diagnostics)
        self.q_file_imported.emit(result.q_bins, result.diagnostics)
        return result

    def set_available_methods(
        self,
        methods: Iterable[tuple[str, QBins]],
    ) -> None:
        """Expose the current Project's reusable Q Methods as editor drafts."""

        self._available_methods = tuple(methods)
        self._rebuild_method_menu()

    def load_q_bins_draft(
        self,
        q_bins: QBins,
        *,
        diagnostics: tuple[ImportDiagnostic, ...] = (),
    ) -> None:
        """Load an already validated assignment without changing a dataset."""

        self._validate_count(q_bins)
        self._q_values = q_bins.q_values.copy()
        self._q_center_slots = [float(value) for value in q_bins.q_values]
        self._edges = None if q_bins.edges is None else q_bins.edges.copy()
        self._diagnostics = diagnostics
        self._edited_representation = None
        self._text_edited_representation = None
        self._visible_representation = None
        self._regular_exact_values.clear()
        self._pending_singleton_field = None
        representation = Q_EDGES if q_bins.edges is not None else Q_CENTER
        blocker = QSignalBlocker(self.representation_combo)
        self.representation_combo.setCurrentText(representation)
        del blocker
        self._set_input_mode(EXPLICIT_VALUES, capture_current=False)
        self._show_representation(representation)
        self.groups_edit.setText(str(self._group_count()))
        self.q_center_draft_changed.emit()

    def q_center_slots(self) -> tuple[float | None, ...]:
        """Return the local center draft, including intentionally blank slots."""

        return tuple(self._q_center_slots)

    def set_q_center_slot(self, group_index: int, text: str) -> None:
        """Edit one Q Center draft cell without assigning data or inferring values."""

        if self.representation_combo.currentText() != Q_CENTER:
            raise ValueError("switch to Q Center mode before editing a Q value")
        if not 0 <= group_index < self._group_count():
            raise ValueError("Q Center group is outside the current dataset")
        value = _parse_q_scalar(text, "Q Center")
        if len(self._q_center_slots) != self._group_count():
            self._q_center_slots = [None] * self._group_count()
        self._q_center_slots[group_index] = value
        self._edited_representation = Q_CENTER
        if all(slot is not None for slot in self._q_center_slots):
            self._q_values = np.asarray(self._q_center_slots, dtype=np.float64)
            blocker = QSignalBlocker(self.values_edit)
            self.values_edit.setPlainText(_format_values(self._q_values))
            del blocker
        else:
            self._q_values = None
        self._set_input_mode(EXPLICIT_VALUES, capture_current=False)
        self._refresh_preview()
        self.q_center_draft_changed.emit()

    def set_apply_handler(
        self,
        handler: Callable[[QBins, tuple[ImportDiagnostic, ...]], bool],
    ) -> None:
        """Set the application owner of a validated Q assignment request."""

        self._apply_handler = handler

    def build_q_bins(self) -> QBins:
        """Validate the visible draft with the domain's Q semantics."""

        if self.input_mode_combo.currentText() == REGULAR_GRID:
            self._commit_pending_singleton_edit()
        return self._build_visible_q_bins()

    def _build_visible_q_bins(self) -> QBins:
        if self.input_mode_combo.currentText() == EXPLICIT_VALUES:
            return self._build_explicit_q_bins()
        return self._build_uniform_q_bins()

    def apply(self) -> bool:
        """Emit one fully validated assignment; never partially mutate a dataset."""

        try:
            q_bins = self.build_q_bins()
        except ValueError as error:
            self.status_label.setText(str(error))
            return False
        if self._apply_handler is not None:
            if not self._apply_handler(q_bins, self._diagnostics):
                self.status_label.setText("Q assignment unchanged")
                return False
        else:
            self.q_bins_applied.emit(q_bins, self._diagnostics)
        self.status_label.setText("Q assignment applied")
        self._q_values = q_bins.q_values.copy()
        self._q_center_slots = [float(value) for value in q_bins.q_values]
        self._edges = None if q_bins.edges is None else q_bins.edges.copy()
        self._edited_representation = None
        self._text_edited_representation = None
        self._refresh_preview()
        return True

    def _build_uniform_q_bins(self, representation: str | None = None) -> QBins:
        """Resolve Start/Step/End through the public Q-bin constructors."""

        start = self._regular_value("start", self.start_edit, "Start")
        step = self._regular_value("step", self.step_edit, "Step")
        end = self._regular_value("end", self.end_edit, "End")
        group_count = self._group_count()
        representation = representation or self.representation_combo.currentText()
        try:
            if representation == Q_CENTER:
                if group_count == 1 and start != end:
                    raise ValueError("Start and End must be identical for one Q Center")
                q_bins = QBins.from_q_values_and_uniform_step(
                    np.linspace(start, end, group_count, dtype=np.float64),
                    step,
                )
            else:
                endpoint_bins = uniform_q_bins(
                    lower_q_edge=start,
                    upper_q_edge=end,
                    group_count=group_count,
                )
                q_bins = QBins.from_q_values_and_uniform_step(
                    endpoint_bins.q_values,
                    step,
                )
        except ValueError as error:
            raise ValueError(
                "Start, Step, and End are inconsistent with the current "
                f"Group count ({group_count}): {error}",
            ) from error
        self._validate_count(q_bins)
        return q_bins

    def _build_explicit_q_bins(self) -> QBins:
        """Keep copied values as the deliberate advanced Q-assignment path."""

        dataset = self._require_dataset()
        representation = self.representation_combo.currentText()
        stored_values = self._q_values if representation == Q_CENTER else self._edges
        values = (
            parse_q_values(self.values_edit.toPlainText())
            if self._text_edited_representation == representation
            or stored_values is None
            else stored_values
        )
        if representation == Q_CENTER:
            if values.size != len(dataset.spectra):
                raise ValueError("Q Center count must match the dataset Group count")
            if self._edited_representation != Q_CENTER and self._edges is not None:
                q_bins = QBins(q_values=values, edges=self._edges)
            else:
                q_bins = QBins.from_q_values(values)
        else:
            if values.size != len(dataset.spectra) + 1:
                raise ValueError("Q-bin Edge count must equal the Group count plus one")
            if self._edited_representation != Q_EDGES and self._q_values is not None:
                q_bins = QBins(q_values=self._q_values, edges=values)
            else:
                q_bins = QBins.from_edges(values)
        self._validate_count(q_bins)
        return q_bins

    def _show_representation(self, representation: str) -> None:
        self._store_visible_draft()
        self._visible_representation = representation
        if self.input_mode_combo.currentText() == EXPLICIT_VALUES:
            self._show_explicit_values(representation)
        else:
            self._show_uniform_values(representation)
        self._refresh_preview()

    def _input_mode_changed(self, mode: str) -> None:
        if self._visible_input_mode == REGULAR_GRID and mode != REGULAR_GRID:
            try:
                self._commit_pending_singleton_edit()
            except ValueError as error:
                blocker = QSignalBlocker(self.input_mode_combo)
                self.input_mode_combo.setCurrentText(REGULAR_GRID)
                del blocker
                self.status_label.setText(str(error))
                self.preview_label.setText("")
                return
        self._set_input_mode(mode, capture_current=True)

    def _set_input_mode(self, mode: str, *, capture_current: bool) -> None:
        if mode not in {REGULAR_GRID, EXPLICIT_VALUES}:
            raise ValueError("unknown Q input mode")
        if capture_current:
            self._store_visible_draft(mode=self._visible_input_mode)
        blocker = QSignalBlocker(self.input_mode_combo)
        self.input_mode_combo.setCurrentText(mode)
        del blocker
        self._visible_input_mode = mode
        regular = mode == REGULAR_GRID
        self.uniform_fields.setVisible(regular)
        self.values_edit.setVisible(not regular)
        if mode == EXPLICIT_VALUES:
            self._show_explicit_values(self.representation_combo.currentText())
        else:
            self._show_uniform_values(self.representation_combo.currentText())
        self._refresh_preview()

    def _show_uniform_values(self, representation: str) -> None:
        self._pending_singleton_field = None
        values = self._q_values if representation == Q_CENTER else self._edges
        start, step, end = _uniform_summary(values)
        if (
            representation == Q_CENTER
            and values is not None
            and values.size == 1
            and self._edges is not None
        ):
            step = float(self._edges[1] - self._edges[0])
        for field, value in (
            (self.start_edit, start),
            (self.step_edit, step),
            (self.end_edit, end),
        ):
            key = {
                self.start_edit: "start",
                self.step_edit: "step",
                self.end_edit: "end",
            }[field]
            if value is None:
                self._clear_regular_value(key, field)
            else:
                self._set_regular_value(key, field, value)
        self.groups_edit.setText(str(self._group_count()))
        if values is None:
            noun = "Q centers" if representation == Q_CENTER else "Q-bin edges"
            self.status_label.setText(
                f"Enter uniform {noun} for {self._group_count()} Groups",
            )
        elif step is None:
            self.status_label.setText(
                "Existing assignment uses non-uniform values; use Explicit values "
                "to preserve or edit it.",
            )
        else:
            self.status_label.setText(f"{self._group_count()} Groups")

    def _show_explicit_values(self, representation: str) -> None:
        values = self._q_values if representation == Q_CENTER else self._edges
        blocker = QSignalBlocker(self.values_edit)
        self.values_edit.setPlainText(_format_values(values))
        del blocker
        if values is None:
            noun = "Q Center" if representation == Q_CENTER else "Q-bin Edge"
            self.status_label.setText(
                f"Paste {noun} values for {self._group_count()} Groups",
            )

    def _store_visible_draft(self, *, mode: str | None = None) -> None:
        if self._visible_representation is None:
            return
        mode = self._visible_input_mode if mode is None else mode
        if mode == REGULAR_GRID:
            try:
                self._commit_pending_singleton_edit()
                q_bins = self._build_uniform_q_bins(self._visible_representation)
            except ValueError:
                return
            self._q_values = q_bins.q_values.copy()
            self._edges = None if q_bins.edges is None else q_bins.edges.copy()
            return
        stored_values = (
            self._q_values if self._visible_representation == Q_CENTER else self._edges
        )
        if self._text_edited_representation == self._visible_representation:
            try:
                draft_values = parse_q_values(self.values_edit.toPlainText())
            except ValueError:
                return
        elif stored_values is not None:
            draft_values = stored_values
        else:
            return
        if self._visible_representation == Q_CENTER:
            self._q_values = draft_values
            if draft_values.size == self._group_count():
                self._q_center_slots = [float(value) for value in draft_values]
        else:
            self._edges = draft_values
        self._text_edited_representation = None

    def _mark_edited(self) -> None:
        if self.input_mode_combo.currentText() == EXPLICIT_VALUES:
            self._edited_representation = self.representation_combo.currentText()
            self._text_edited_representation = self.representation_combo.currentText()

    def _regular_field_changed(self, key: str) -> None:
        if self._setting_regular_numeric_text:
            return
        if self._is_singleton_center_field(key):
            self._pending_singleton_field = key
            self._refresh_preview()
            return
        self._regular_exact_values.pop(key, None)
        try:
            value = _parse_q_scalar(
                {
                    "start": self.start_edit,
                    "step": self.step_edit,
                    "end": self.end_edit,
                }[key].text(),
                key.title(),
            )
            self._regular_exact_values[key] = value
            self._couple_regular_fields(key)
        except ValueError:
            pass
        self._refresh_preview()

    def _regular_field_editing_finished(self, key: str) -> None:
        if self._pending_singleton_field != key:
            return
        try:
            self._commit_pending_singleton_edit()
        except ValueError as error:
            self.status_label.setText(str(error))
            self.preview_label.setText("")
            return
        self._refresh_preview()

    def _commit_pending_singleton_edit(self) -> None:
        key = self._pending_singleton_field
        if key is None:
            return
        field = self.start_edit if key == "start" else self.end_edit
        authoritative = _parse_q_scalar(field.text(), key.title())
        self._pending_singleton_field = None
        self._set_regular_value("start", self.start_edit, authoritative)
        self._set_regular_value("end", self.end_edit, authoritative)

    def _is_singleton_center_field(self, key: str) -> bool:
        return (
            key in {"start", "end"}
            and self.representation_combo.currentText() == Q_CENTER
            and self._group_count() == 1
        )

    def _couple_regular_fields(self, edited: str) -> None:
        group_span = self._regular_group_span()
        if group_span == 0:
            authoritative = (
                self._regular_value("end", self.end_edit, "End")
                if edited == "end"
                else self._regular_value("start", self.start_edit, "Start")
            )
            self._set_regular_value("start", self.start_edit, authoritative)
            self._set_regular_value("end", self.end_edit, authoritative)
            return
        start = self._regular_value("start", self.start_edit, "Start")
        if edited == "end":
            end = self._regular_value("end", self.end_edit, "End")
            self._set_regular_value("step", self.step_edit, (end - start) / group_span)
            return
        step = self._regular_value("step", self.step_edit, "Step")
        self._set_regular_value("end", self.end_edit, start + group_span * step)

    def _regular_group_span(self) -> int:
        if self.representation_combo.currentText() == Q_CENTER:
            return self._group_count() - 1
        return self._group_count()

    def _set_regular_value(self, key: str, field: QLineEdit, value: float) -> None:
        self._regular_exact_values[key] = float(value)
        self._setting_regular_numeric_text = True
        try:
            field.setText(f"{value:.12g}")
        finally:
            self._setting_regular_numeric_text = False

    def _clear_regular_value(self, key: str, field: QLineEdit) -> None:
        self._regular_exact_values.pop(key, None)
        self._setting_regular_numeric_text = True
        try:
            field.clear()
        finally:
            self._setting_regular_numeric_text = False

    def _regular_value(self, key: str, field: QLineEdit, name: str) -> float:
        if key in self._regular_exact_values:
            return self._regular_exact_values[key]
        return _parse_q_scalar(field.text(), name)

    def _refresh_preview(self) -> None:
        if self.dataset is None:
            self.preview_label.setText("")
            return
        try:
            q_bins = self._build_visible_q_bins()
        except (RuntimeError, ValueError):
            self.preview_label.setText("")
            return
        edge_text = (
            "no edges"
            if q_bins.edges is None
            else _range_preview("Edges", q_bins.edges)
        )
        self.preview_label.setText(
            f"{_range_preview('Centers', q_bins.q_values)} · {edge_text}",
        )

    def _prompt_dave_import(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Import DAVE Q-bin Parameters",
            "",
            "Text files (*.txt *.dat);;All files (*)",
        )
        if not path:
            return
        try:
            result = self.import_dave_file(path)
        except ValueError as error:
            show_message_dialog(self, "DAVE Q-bin Parameters", str(error))
            return
        if result.diagnostics:
            self.status_label.setText(result.diagnostics[0].message)

    def _rebuild_method_menu(self) -> None:
        """Keep the small method chooser synchronized with the active Project."""

        self._method_menu.clear()
        for name, q_bins in self._available_methods:
            action = self._method_menu.addAction(name)
            action.triggered.connect(
                lambda _checked=False, assignment=q_bins: self._load_method(assignment),
            )
        self.use_method_button.setEnabled(bool(self._available_methods))

    def _show_method_menu(self) -> None:
        """Open the current Project's compact method chooser on explicit request."""

        if self._available_methods:
            self._method_menu.popup(
                self.use_method_button.mapToGlobal(
                    self.use_method_button.rect().bottomLeft(),
                ),
            )

    def _load_method(self, q_bins: QBins) -> None:
        try:
            self.load_q_bins_draft(q_bins)
        except ValueError as error:
            self.status_label.setText(str(error))

    def _save_as_method(self) -> None:
        try:
            self.save_q_method_requested.emit(self.build_q_bins())
        except ValueError as error:
            self.status_label.setText(str(error))

    def _requires_explicit_values(self, q_bins: QBins | None) -> bool:
        if q_bins is None:
            return False
        if q_bins.edges is None:
            # Representative values from the importer do not assert a bin geometry.
            # Keep them on the explicit path so merely opening this editor never
            # turns a regular-looking sequence into inferred centered edges.
            return True
        values = q_bins.q_values if q_bins.edges is None else q_bins.edges
        _start, step, _end = _uniform_summary(values)
        if step is None:
            return True
        try:
            centered = QBins.from_q_values_and_uniform_step(q_bins.q_values, step)
        except ValueError:
            return True
        assert centered.edges is not None
        assert q_bins.edges is not None
        return not np.allclose(centered.edges, q_bins.edges, rtol=0.0, atol=0.0)

    def _validate_count(self, q_bins: QBins) -> None:
        if q_bins.group_count != self._group_count():
            raise ValueError("Q assignment count must match the dataset Group count")

    def _group_count(self) -> int:
        return len(self._require_dataset().spectra)

    def _require_dataset(self) -> ReducedDataset:
        if self.dataset is None:
            raise RuntimeError("open a dataset before assigning Q")
        return self.dataset


def _parse_q_scalar(text: str, name: str) -> float:
    """Parse one literal Q entry without silently filling or correcting it."""

    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _uniform_summary(
    values: np.ndarray | None,
) -> tuple[float | None, float | None, float | None]:
    """Return display entries only when existing values are uniformly spaced."""

    if values is None or values.size == 0:
        return None, None, None
    start = float(values[0])
    end = float(values[-1])
    if values.size == 1:
        return start, None, end
    step = float(values[1] - values[0])
    if step <= 0.0 or not np.allclose(
        np.diff(values),
        step,
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=8.0 * np.finfo(np.float64).eps * max(1.0, abs(step)),
    ):
        return start, None, end
    return start, step, end


def _range_preview(label: str, values: np.ndarray) -> str:
    """Return a compact, non-tabular preview for a resolved assignment."""

    if values.size == 1:
        return f"{label}: {values[0]:.6g}"
    return f"{label}: {values[0]:.6g} … {values[-1]:.6g} ({values.size})"


def _format_values(values: Iterable[float] | None) -> str:
    return "" if values is None else "  ".join(f"{value:.12g}" for value in values)
