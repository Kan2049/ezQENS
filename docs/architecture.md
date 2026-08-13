# Architecture

## 1. Scope and goals

This document defines the current architectural boundary and near-term
direction. It keeps scientific computation independent of PySide6 and permits
a future internal Python API without committing to a stable public API in
version 1.0. The application targets ordinary CPU-based macOS and Windows
laptops; Linux support is best-effort.

The package remains under `src/ezqens`, uses typed Python, avoids global
mutable state, and passes explicit immutable or validated values between
layers.

Milestones 1–6 implement only minimal typed in-memory domain objects needed for
import through derived QENS results. Version 1.0 adds only the lightweight
reproducibility information needed to identify inputs, selections, settings,
results, warnings, and software version. The final project-container format and
extension are unresolved; the former `.qensfit` proposal is not a contract.

## 2. Architectural principles

1. **Scientific core first.** Import, resolution, convolution, fitting, and
   diagnostics are validated before GUI integration.
2. **Dependency direction is inward.** GUI and reporting call application/core
   services; scientific modules never import PySide6.
3. **Data is explicit.** Units, spectrum role, Q identity, masks, parameter
   meaning, source metadata, and warnings travel with data.
4. **Originals are immutable.** Imported sample and resolution values are
   preserved; processing creates derived records.
5. **Operations are reproducible.** Services accept configurations and return
   results rather than relying on process-wide settings.
6. **Ambiguity is visible.** Invalid inputs and unresolved mappings produce
   diagnostics or require a user decision; no silent deletion or guessing.
7. **Reproducibility follows science.** Minimal immutable in-memory models
   support early validation. Version 1.0 records the small set of information
   needed to identify and review an analysis without predeclaring an archive or
   database architecture.
8. **Private data stays local.** Core code has no external-upload behavior, and
   logs summarize rather than dump numerical arrays.
9. **Design seams, not scaffolding.** Extract a general abstraction only after
   at least two real implementations need it. Future raw HDF/NeXus, instrument,
   or Mantid sources may produce the same reduced domain, but no framework for
   them exists now.

The stable scientific flow is:

```text
source-specific import or reduction
  -> ReducedDataset -> ordered Spectrum values
  -> preprocessing / Q mapping / resolution / fitting
```

Downstream science depends on `ReducedDataset` / `Spectrum`, never the original
file type. The current producer is reduced text import. Future producers are
added from real schemas without changing this boundary.

### 2.1 Core stable, edges extensible

Relatively stable core concepts are `Spectrum`, `ReducedDataset`, scientific
units and conventions, Q identity, masks/selections, resolution semantics,
spectral-model semantics, fit-result semantics, and basic derived quantities.

Frequently evolving edges are input/reduction sources, export formats,
visualization, GUI workflows, and later dynamics models. A new edge capability
should normally be localized behind an existing scientific boundary and should
not require unrelated core changes. This is an extension-seam rule, not
permission to build registries, factories, plugins, or adapter hierarchies
before concrete implementations need them.

## 3. Current and near-term package boundaries

The names below are target boundaries, not files created by this task.

```text
ezqens/
  domain/              typed scientific values and validation
  io/
    importers/         reduced DAVE and generic ASCII inputs
    exporters/         tables, machine-readable output, review reports
  preprocessing/       input validation, ranges, masks, explicit transforms
  resolution/          resolution processing and sample-group association
  spectral/            component definitions and spectral evaluation
  convolution/         grid construction and linear numerical convolution
  fitting/             optimizer adapters and fit orchestration
  diagnostics/         residual statistics, information criteria, warnings
  batch/               sequential independent per-Q execution
  derived/             FWHM-to-tau and experimental EISF
  reporting/           GUI-neutral tables, plots, and export view models
  application/         workflow use cases and cancellation boundaries
  gui/                 PySide6 views/controllers/adapters, added later
```

### 3.1 Domain

`domain` owns the scientific concepts in `data_model.md`, unit-bearing fields,
validation errors, warnings, and simple identifiers where useful. It has no
dependency on files, SciPy optimizers, plotting, or GUI types.

Dataclasses are appropriate for small immutable computational values. Pydantic
or equivalent validated types are appropriate at import boundaries. During
milestones 1–6, numerical arrays may be direct NumPy values held by typed
in-memory objects; UUIDs, entity schema versions, hash-addressed registries,
and migration machinery are not mandatory. No persistence schema is
anticipated before a concrete v1.0 reproducibility workflow requires one.

`Spectrum` contains only its scientific role, ordered group identity, immutable
energy/intensity/uncertainty arrays, units, and invalid-value masks.
`ReducedDataset` holds ordered spectra plus minimal source/import traceability.
Whether spectra share an energy grid and which extra columns were ignored are
derived rather than redundantly stored. Text row/column concepts stay at the
import boundary and do not burden future non-text producers.

### 3.2 Import and export

Importers translate untrusted files into reduced domain data plus diagnostics.
They do not preprocess, fit, or infer unspecified scientific meaning. The
detector examines content and returns proposed layout, concise evidence,
counts, required/extra columns, diagnostics, and plausible alternatives. It
stores no confidence, extension hint, explicit-override flag, or GUI
confirmation state. Detection precedence is explicit user choice, DAVE group
blocks, wide `x/yN/yerrN`, single `x/y/yerr`, then custom mapping. Extensions
do not determine layout; later application workflow owns confirmation.

DAVE group blocks, wide shared-grid tables, and single-spectrum tables convert
to the common spectrum interface. Recognized DAVE fit-result columns are
recorded and excluded from measured data. Generic ASCII uses explicit mapping
when ambiguous. Import preserves order, per-group grids/lengths, original
arrays, and source layout; it never interpolates.

Q identity is dataset-level `QBins`: one ordered representative value per
spectrum plus optional explicit edges. Count-driven `uniform_q_bins()` exactly
covers its inclusive outer edges. The DAVE parser instead reconstructs complete
fixed-width bins from the source lower limit, upper limit, and step, so its
source upper limit may exceed the final actual edge. It retains the stored group
count as source metadata and warns if it differs from the reconstructed count.
Both edge-defined paths produce midpoint representatives; explicit values leave
edges unknown. Source-specific metadata and diagnostics stay in `io`, while the
scientific object contains no parser, confirmation, or GUI state.

Reduced-data import and any future raw reduction remain distinct internal
responsibilities even if a later GUI presents both under “Import Data”:

```text
reduced source -> localized importer -> ReducedDataset -> scientific core
raw source -> source-specific reduction -> ReducedDataset -> scientific core
```

Future reduction learns only the required behavior from Mantid, DAVE, facility
software, and documented algorithms; it does not reproduce their architectures.

Exporters consume result/report models; they do not recompute fits. Every
scientific export includes units, FWHM labels, exclusions, warnings, software
version, and sufficient analysis/result context for traceability.

Export and visualization are first-class evolving boundaries. Export may grow
to cover processed spectra, curves/components, residuals, parameters,
uncertainties, basic QENS quantities, later dynamics results, and lightweight
reproducibility configuration. Visualization consumes scientific results and
never independently recomputes EISF, relaxation time, or other formulas.

### 3.3 Preprocessing and resolution

`preprocessing` expresses analysis-level masks and transformations as pure or
side-effect-free operations returning new data plus minimal trace records.
Milestone 2 distinguishes invalid values, per-group fit ranges, `AUTO` padding,
and `REVIEW` padding; later fitting workflows add manual point, spectral, and
derived-result exclusions as separate state. Sigma is valid only when finite
and strictly positive; every other sigma is automatically invalid-masked without
changing original arrays.

Boundary-padding detection is a separate preprocessing service, not importer
logic. It examines boundary-connected repeated intensity/uncertainty pairs per
spectrum and compares signatures across the dataset. `AUTO` and `REVIEW`
produce distinct, mutually exclusive immutable point masks; `NONE` masks
nothing. Boundary results retain only side, run length, energy bounds, status,
and a compact reason. Internal candidates and calculations are not public
domain state. The service never shifts intensity values or treats an internal
constant segment as edge padding.

`resolution` implements the measured-resolution-only Milestone-3 boundary. It
requires exact ordered sample/resolution Q identity, runs padding detection
independently on both datasets, compares retained boundaries diagnostically,
represents a distinct per-Q resolution support, validates accepted measured
coordinates, and derives unit-area normalization on the original grid.

The prepared result references the immutable source datasets and stores only
support, normalization metadata, Q association by order, and diagnostics.
Normalized energy/intensity/uncertainty and source-grid contribution are
derived read-only properties. Resolution `AUTO` padding is applied by default
but has an independent reversible per-group application state; changing support
alone does not disable it. `REVIEW` remains accepted by default and invalid
energy/intensity remains non-overridable. An internal invalid point inside the
selected support blocks preparation rather than creating a trapezoidal bridge.
Sample fitting ranges do not enter this state. There is no baseline operation,
interpolation, convolution, recentering, nearest-Q association, caching, or
analytic fallback in M3.

### 3.4 Spectral models and convolution

`spectral` defines elastic, Lorentzian, and background components in terms of
integrated areas and FWHM parameters. Components expose numerical evaluation
through a stable interface but know nothing about optimizers or UI.

For unit-area processed resolution `R_Q`, the elastic evaluator returns
`A_elastic * R_Q(E - E0)` directly; it never constructs a discrete numerical
delta. Each quasielastic evaluator returns
`A_i * [R_Q convolved with L_i](E - E0)` for a unit-area Lorentzian with FWHM
linewidth. Convolution preserves integrated areas within reviewed finite-grid
tolerance.

`convolution` implements the Milestone-4 CPU numerical core as a reusable
per-Q `ConvolutionPlan` plus a `ConvolvedProfile`. The plan is built from the
authoritative ordered `PreparedResolution` association, validates canonical
meV coordinates, constructs the automatic S1/4 zero-anchored model lattice,
linearly represents and unit-area-corrects the measured resolution, and
precomputes only its fixed FFT state. It exposes explicit model, resolution,
and full-convolution coordinates rather than relying on implicit centering.

The plan accepts a theoretical density already evaluated on its model grid,
performs power-of-two-padded full FFT linear convolution with one spacing
factor, and linearly evaluates a fixed convolved profile on original sample
coordinates. The analytic cell-integrated Lorentzian is the only M4 spectral
primitive because narrow-line correctness requires it. `ConvolutionPlan` does
not contain fit parameters, model composition, optimizer, backend, registry,
cache, GUI, recentering, analytic-resolution fallback, or source-specific Q
policy. Future `E0` shifts change evaluation coordinates, not the plan/grid.

### 3.5 Fitting, diagnostics, and batch execution

`fitting` represents one elastic component, a variable-length Lorentzian
collection, one shared energy shift, and NONE/B0/B1 background. It adapts this
manual configuration to `scipy.optimize.least_squares(method="trf")`, uses the
existing fitting selection and prepared resolution, constructs weighted
standardized residuals on retained original sample coordinates, honors
fixed/free state and bounds, and returns raw optimizer facts without hiding
failure. The scientific model has no Lorentzian-count ceiling; validated
automatic multistart initialization is currently limited to standard 0L/1L/2L
candidates.

Fit results expose component-resolved model values, raw/standardized residuals,
chi-square, reduced chi-square, unscaled absolute-sigma covariance/error
status, Jacobian singular values/rank/condition, bound activity, residual
structure, multistart outcomes, and threshold-free identifiability metrics.
AIC/AICc/BIC use one declared convention. Candidate generation and evaluation
are separate from the unresolved Auto recommendation/adequacy policy.

`batch` orders spectra and invokes the single-spectrum service repeatedly.
Each invocation returns an independent `FitResult`. Previous-successful-fit
seeding is an explicit option and does not create parameter coupling. A
cancellation token is checked between fits and at safe optimizer boundaries.

### 3.6 Derived quantities

`derived` owns explicit FWHM-to-relaxation-time, experimental-EISF, component-Q,
and supported uncertainty-propagation calculations. It consumes completed fit
results and retains component inputs and warnings. Later approved temporal or
spatial dynamics analysis consumes these quantities without changing spectral
fit semantics. No generic model registry or mandatory catalogue is declared.
Automatic inference from coordinates, symmetry, or molecular structure is a
substantially later extension and does not shape current modules.

### 3.7 Application and GUI

`application` exposes workflow-oriented use cases such as import sample,
process resolution, fit spectrum, run batch, derive EISF, retain lightweight
analysis metadata, and export. It is the seam used by both tests and the GUI.

The later PySide6 GUI contains presentation and interaction only. Proposed
screens are Import, Spectrum Fit, Batch Results, and Export/Analysis Summary.
Controllers submit long operations to worker
infrastructure, receive progress/result messages, and support cancellation.
They never contain formulas, residual construction, convolution, or optimizer
logic.

Interaction exposes complexity progressively. Guided workflows emphasize safe
defaults, understandable choices, warnings, and basic physical outputs;
experienced workflows progressively expose ranges, masks, components,
parameters, bounds, resolution settings, batch behavior, diagnostics,
visualization, and export controls. This does not prescribe two literal UI
modes.

The Import screen eventually presents format proposal/evidence/warnings,
manual override, group or paired-column count, and array mapping preview. It
offers linear range, manual Q list, explicit-list file, DAVE Q-bin parameter
file, and supported imported-metadata modes; shows the resolved list and count
comparison; and requires confirmation before applying detected/generated data.

## 4. Dependency rules

```text
GUI -> application -> core services -> domain
reporting/export -> result/domain models
batch -> single-spectrum fitting -> spectral + convolution + diagnostics
```

Disallowed dependencies include:

- core or domain importing `ezqens.gui` or PySide6;
- importers calling fitters;
- exporters recomputing scientific results;
- spectral components reading files or process-global settings;
- Mantid types appearing in core signatures.

If a concrete interoperability source is implemented later, it stays outside
the core and produces `ezqens` domain values. No adapter base class, registry,
factory, or plugin protocol is predeclared now.

## 5. Core callable interfaces

Interfaces should accept typed values/configurations and return a result plus
structured diagnostics. Exceptions represent programmer errors or unrecoverable
I/O; expected scientific/data problems are validation issues in results.

Conceptual use cases include:

```text
detect_reduced_data_format(source, optional_override) -> FormatDetectionResult
import_reduced_data(source, role, units) -> ReducedDataset
QBins.from_edges(edges) / QBins.from_q_values(values) -> QBins
uniform_q_bins(lower_q_edge, upper_q_edge, group_count) -> QBins
parse_dave_q_bins(source) -> DAVEQBinsResult
dataset.assign_q_bins(q_bins) -> ReducedDataset
FittingSelection.uniform(dataset, padding, energy_bounds) -> FittingSelection
prepare_measured_resolution(sample, resolution, support_overrides)
  -> PreparedResolution
fit_spectrum(spectrum, resolution, configuration) -> FitResult
fit_batch(spectra, resolution_map, configuration, cancellation) -> BatchFitResult
derive_qens(batch_result, configuration) -> DerivedQENSResult
export_analysis(results, reproducibility_summary, destination) -> ExportRecord
```

These signatures describe responsibilities only. Concrete names and error
types can evolve before the internal API stabilizes.

## 6. Data flow and auditability

```text
source-specific import or reduction
  -> immutable ReducedDataset + minimal source traceability
  -> dataset-level QBins + per-group FittingSelection
  -> sample fitting selection + prepared measured resolution
  -> per-Q spectral fit configurations/results
  -> derived FWHM/tau/EISF records
  -> reports/exports + lightweight reproducibility summary
```

During milestones 1–6, each arrow returns immutable typed values plus the
minimal configuration, diagnostics, and simple links required to reproduce the
current calculation in memory. It need not create a UUID-backed entity,
hash-addressed payload, migration schema, or append-only project event.
Manual refits remain separate result objects rather than mutating successful
neighbors. Complete persistent history and migration semantics are not v1.0
requirements.

## 7. Reproducibility and future project-file format

Version 1.0 records enough lightweight information to identify inputs,
selections and masks, configurations, results, warnings, and software version.
It does not require container, archive, migration, attachment, hash-registry,
or database infrastructure.

Any later project-file design begins from stable scientific contracts and must
choose its format and extension explicitly. The former `.qensfit` proposal is
not permanent. If a container is introduced, its data-only security, redaction,
integrity, size, path-safety, and atomic-write requirements are reviewed then;
no current domain object or dependency exists solely to anticipate it.

## 8. Reporting and logging

Structured diagnostics use codes, severity, human-readable text, entity
references, and remediation hints. Logs include identifiers and structural
summaries, not full numerical arrays or confidential content. Reports clearly
separate:

- optimizer convergence;
- statistical diagnostics;
- scientific-quality warnings; and
- user exclusions or overrides.

Diagnostics remain available through import, preprocessing, resolution,
fitting, parameter/covariance validity, derived quantities, and any later
dynamics fitting. Optimizer convergence and scientific validity are distinct
whenever the underlying evidence allows that distinction.

Plots and tables receive prepared view models with units and FWHM semantics,
preventing UI-specific reinterpretation.

## 9. Acceptance criteria

- Core tests run without importing PySide6.
- Module boundaries cover every responsibility required by version 1.0.
- No scientific operation depends on global mutable state.
- Sample and resolution spectra are semantically distinct behind one common
  numerical interface.
- Group-block and wide-table inputs preserve their original grids and expose
  independent spectra without importer interpolation.
- Original and processed data are distinct and linked by processing records.
- Single and batch fitting share one validated single-spectrum path.
- Batch results remain independent per Q and support cancellation.
- Private numerical arrays cannot appear in routine logs.
- Windows-specific concerns are isolated behind filesystem/concurrency/UI
  adapters rather than embedded in scientific calculations.

For milestones 1–6, minimal in-memory typed objects and simple traceability are
sufficient. Before the v1.0 release gate, lightweight reproducibility, export,
GUI delegation, and macOS/Windows packaging must be validated. Complete
persistence is not a v1.0 acceptance criterion.

## 10. Assumptions and explicit non-goals

The architecture assumes in-memory NumPy arrays are practical for typical
reduced v1.0 datasets during early scientific milestones; later storage
references may enable lazy loading. It assumes one local user and one active
project per application window. Neither assumption changes a scientific
convention.

This design does not specify raw-data infrastructure, cloud sync, collaboration,
plugin execution, a public stable Python API, or a final GUI visual design.

## 11. Unresolved decisions and risks

Unresolved engineering choices include later result types, any future
persistence format and limits, worker process versus
thread execution, stable report formats, and concrete raw HDF/NeXus schemas if
raw reduction is ever added.

Risks include large-memory use, cancellation that leaves partial state, Windows
path/process differences, dependency leakage into the core, GUI/core drift, and
overgrown domain objects. Address these through immutable results, bounded
workers, interface tests, real-user validation, and narrow modules.

## 12. Milestone dependencies

Domain primitives and diagnostics vocabulary precede all other core work. The
content detector and DAVE/wide/single importers precede Q mapping and
resolution association. Resolution processing and grid validation precede
convolution. Convolution precedes single-spectrum fitting.
Validated single fits precede batch and derived results. Minimal in-memory
contracts evolve through milestones 1–6. Reporting follows stable result
contracts. GUI work follows scientific validation of the
importer-through-single-fit path, and the guided GUI, export, lightweight
reproducibility, documentation, and macOS/Windows packages precede v1.0 release.
