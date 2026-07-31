# Architecture

## 1. Scope and goals

This document defines a proposed version-1 architecture, not implementation.
It keeps scientific computation independent of PySide6, makes transformations
auditable, supports reproducible persistence, and permits a future internal
Python API without committing to a public API in version 1.

The package remains under `src/qensfit`, uses typed Python, avoids global
mutable state, and passes explicit immutable or validated values between
layers.

Milestones 1–6 implement only minimal typed in-memory domain objects needed for
import through derived QENS results. They do not implement the complete
project-file architecture. The versioned `.qensfit` container, security model,
hash registry, migrations, archive layout, attachments, atomic save, and public
file contract are reviewed and implemented in milestone 8.

## 2. Architectural principles

1. **Scientific core first.** Import, resolution, convolution, fitting, and
   diagnostics are validated before GUI integration.
2. **Dependency direction is inward.** GUI and reporting call application/core
   services; scientific modules never import PySide6.
3. **Data is explicit.** Units, spectrum role, Q identity, masks, parameter
   meaning, source metadata, and warnings travel with data.
4. **Originals are immutable.** Imported sample, resolution, and molecular
   values are preserved; processing creates derived records.
5. **Operations are reproducible.** Services accept configurations and return
   results rather than relying on process-wide settings.
6. **Ambiguity is visible.** Invalid inputs and unresolved mappings produce
   diagnostics or require a user decision; no silent deletion or guessing.
7. **Persistence follows science.** Minimal immutable in-memory models support
   early validation. Milestone 8 persistence is data-only and contains
   validated declarative data, never pickle, Python code, or executable plugins.
8. **Private data stays local.** Core code has no external-upload behavior, and
   logs summarize rather than dump numerical arrays.

## 3. Proposed package boundaries

The names below are target boundaries, not files created by this task.

```text
qensfit/
  domain/              typed scientific values and validation
  io/
    importers/         DAVE, generic ASCII, XYZ
    exporters/         tables, machine-readable output, review reports
  preprocessing/       input validation, ranges, masks, baseline operations
  resolution/          resolution processing and sample-group association
  spectral/            component definitions and spectral evaluation
  convolution/         grid construction and linear numerical convolution
  fitting/             optimizer adapters and fit orchestration
  diagnostics/         residual statistics, information criteria, warnings
  batch/               sequential independent per-Q execution
  derived/             FWHM-to-tau and experimental EISF
  molecular/           XYZ-domain geometry and hydrogen selection
  motion/              candidate interfaces and approved implementations
  comparison/          candidate-model evidence and reporting language
  project/             milestone-8 persistence, migration, hashes, audit
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
and migration machinery are not mandatory. Pydantic persistence schemas and
external array references are reviewed in milestone 8.

The common numerical spectrum interface exposes energy, intensity,
uncertainty, invalid-data mask, group identity, and source metadata. It must
also distinguish sample from measured-resolution role, either directly or
through validated wrappers. A shared-grid matrix view is permitted for true
wide-table grids, while a spectrum list supports unequal grids and lengths.
Storage choice is hidden behind the interface and never causes interpolation.

### 3.2 Import and export

Importers translate untrusted files into an import report plus domain data.
They do not preprocess, fit, or infer unspecified scientific meaning. The
detector examines content and returns proposed layout, confidence, evidence,
counts, required/extra columns, warnings, and plausible alternatives. Detection
precedence is explicit user choice, DAVE group blocks, wide `x/yN/yerrN`,
single `x/y/yerr`, then custom mapping. Extensions are hints only, and an
automatic proposal requires confirmation.

DAVE group blocks, wide shared-grid tables, and single-spectrum tables convert
to the common spectrum interface. Recognized DAVE fit-result columns are
recorded and excluded from measured data. Generic ASCII uses explicit mapping
when ambiguous. Import preserves order, per-group grids/lengths, original
arrays, and source layout; it never interpolates.

Q mapping is a separate boundary that retains source definition and resolved
explicit values. It supports inclusive linear range, manual list, explicit-list
file, approved DAVE Q-bin parameter file, and supported imported metadata.
Every resolved list is previewed, confirmed, and checked exactly against sample
and applicable resolution group counts.

Exporters consume result/report models; they do not recompute fits. Every
scientific export includes units, FWHM labels, exclusions, warnings, software
version, and sufficient project/result identifiers for traceability.

### 3.3 Preprocessing and resolution

`preprocessing` expresses analysis-level masks and transformations as pure or
side-effect-free operations returning new data plus minimal trace records. It
distinguishes invalid values, fit ranges, manual point masks, spectral
exclusions, and motion-fit exclusions. Sigma is valid only when finite and
strictly positive; every other sigma is automatically invalid-masked without
changing original arrays.

Boundary-padding detection is a separate preprocessing service, not importer
logic. It examines boundary-connected repeated intensity/uncertainty pairs per
spectrum, compares signatures across the dataset, and returns immutable typed
evidence plus point-level masks. High-confidence points form a reversible
default-on mask; medium-confidence points form a confirmation-gated suggestion
mask. Both remain distinct from invalid/manual/range/Q masks. The service never
shifts intensity values or treats an internal constant segment as edge padding.

`resolution` retains originals and makes manual valid-range selection
mandatory and authoritative in the first implementation. It applies explicit
optional baseline correction and validated unit-area normalization, validates
grids, builds uniform processed grids, and associates sample and resolution
groups. Exact versus user-selected nearest-Q association is explicit.
High-confidence auto-padding may be default-on but reversible; medium-confidence
range suggestions require confirmation. Neither mutates originals or replaces
manual resolution valid-range selection.

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

`convolution` performs numerical linear convolution on validated compatible
grids. Any FFT implementation pads sufficiently and crops deliberately to
avoid circular wrap-around. Grid, centering, interpolation, padding, and
normalization choices are explicit inputs and provenance.

### 3.5 Fitting, diagnostics, and batch execution

`fitting` adapts a `FitConfiguration` to
`scipy.optimize.least_squares`, constructs weighted standardized residuals,
honors fixed/free state and bounds, and returns raw optimizer facts without
hiding failure.

`diagnostics` derives raw/standardized residuals, chi-square, reduced
chi-square, covariance/standard-error status, bound hits, and scientific
warnings. AIC/AICc are added under one approved declared convention; their
absence does not block the initial single-spectrum prototype. Cross-fit
comparison requires identical points, masks/Q selection, uncertainty
treatment, residual definition, and likelihood convention.

`batch` orders spectra and invokes the single-spectrum service repeatedly.
Each invocation returns an independent `FitResult`. Previous-successful-fit
seeding is an explicit option and does not create parameter coupling. A
cancellation token is checked between fits and at safe optimizer boundaries.

### 3.6 Derived quantities, molecular geometry, and motion models

`derived` owns explicit FWHM-to-relaxation-time and experimental-EISF
calculations. It consumes completed fit results and retains component inputs
and warnings.

`molecular` parses and represents all XYZ atoms, selects hydrogens for the
first incoherent model, and calculates geometry relative to the fixed origin
and selected x/y/z axis.

`motion` defines a candidate interface with identity/version, parameter
definitions, required molecular inputs, predicted EISF over Q, validity checks,
and warnings. C2 and C4 implementations cannot be added until their equations
are approved. Candidates are fit independently.

`comparison` compares compatible candidate results using weighted residuals,
reduced chi-square, AICc, uncertainty, residual structure, parameter validity,
and warnings. Its narrative says “best supported under the evaluated models
and assumptions,” never “proven mechanism.”

### 3.7 Application and GUI

`application` exposes workflow-oriented use cases such as import sample,
process resolution, fit spectrum, run batch, derive EISF, compare candidates,
save project, and export. It is the seam used by both tests and the GUI.

The later PySide6 GUI contains presentation and interaction only. Proposed
screens are Import, Spectrum Fit, Batch Results, Motion Models, and
Export/Project Summary. Controllers submit long operations to worker
infrastructure, receive progress/result messages, and support cancellation.
They never contain formulas, residual construction, convolution, or optimizer
logic.

The Import screen eventually presents format detection/confidence/warnings,
manual override, group or paired-column count, and array mapping preview. It
offers linear range, manual Q list, explicit-list file, DAVE Q-bin parameter
file, and supported imported-metadata modes; shows the resolved list and count
comparison; and requires confirmation before applying detected/generated data.

## 4. Dependency rules

```text
GUI -> application -> core services -> domain
reporting/export -> result/domain models
project persistence -> domain schemas and array storage
batch -> single-spectrum fitting -> spectral + convolution + diagnostics
motion comparison -> motion fits -> derived EISF + molecular geometry
```

Disallowed dependencies include:

- core or domain importing `qensfit.gui` or PySide6;
- importers calling fitters;
- exporters recomputing scientific results;
- spectral components reading files or process-global settings;
- motion models accessing GUI state;
- persistence importing executable user code; and
- Mantid types appearing in core signatures.

Optional interoperability adapters, including Mantid, sit outside the core and
translate into qensfit domain values.

## 5. Core callable interfaces

Interfaces should accept typed values/configurations and return a result plus
structured diagnostics. Exceptions represent programmer errors or unrecoverable
I/O; expected scientific/data problems are validation issues in results.

Conceptual use cases include:

```text
detect_reduced_format(source, optional_override) -> FormatDetectionResult
import_reduced_data(source, import_configuration) -> ImportOutcome[Dataset]
resolve_q_mapping(source_definition, sample_count, resolution_count) -> QMapping
process_resolution(original, configuration) -> ResolutionProcessingOutcome
fit_spectrum(spectrum, resolution, configuration) -> FitResult
fit_batch(spectra, resolution_map, configuration, cancellation) -> BatchFitResult
derive_qens(batch_result, configuration) -> DerivedQENSResult
import_xyz(source) -> ImportOutcome[MolecularStructure]
fit_motion_model(derived, structure, definition) -> MotionModelFitResult
compare_motion_models(results, configuration) -> ModelComparisonResult
save_project(project, destination) -> ExportRecord
load_project(source) -> ProjectLoadOutcome
```

These signatures describe responsibilities only. Concrete names and error
types can evolve before the internal API stabilizes.

## 6. Data flow and auditability

```text
source files
  -> immutable imported datasets + provenance
  -> Q mapping + analysis masks
  -> processed sample/resolution views + processing steps
  -> per-Q spectral fit configurations/results
  -> derived FWHM/tau/EISF records
  -> molecular structure + independent motion fits
  -> model comparison
  -> reports/exports/project snapshot
```

During milestones 1–6, each arrow returns immutable typed values plus the
minimal configuration, diagnostics, and simple links required to reproduce the
current calculation in memory. It need not create a UUID-backed entity,
hash-addressed payload, migration schema, or append-only project event.
Manual refits remain separate result objects rather than mutating successful
neighbors. Complete persistent history and migration semantics begin in
milestone 8.

## 7. Project-file format

This entire section is a long-term version-1 proposal. It is not an
implementation requirement for milestones 1–6 and must not delay their
scientific validation. Milestone 8 reviews the proposal and then implements the
approved persistence format, safety policy, hashes, migrations, archive layout,
attachment rules, atomic saving, and public project-file contract.

### 7.1 Proposed container

Use a versioned `.qensfit` ZIP64 container with:

```text
manifest.json
models/project.json
arrays/<content-hash>.npy
attachments/<approved-data-only-files>
checksums.json
```

- `manifest.json` identifies the format, project-schema version, software
  version, creation metadata, entry inventory, and content hashes.
- `project.json` contains validated JSON-compatible domain records and
  references arrays by content hash.
- `.npy` files store numeric arrays with `allow_pickle=False`.
- Attachments are restricted to explicitly supported data-only media types.
- `checksums.json` enables corruption detection.

Imported numerical data are embedded for reproducibility while original source
paths remain references and are optional on reload. Source-file hashes are
stored where practical. Private source paths and user-identifying path segments
must not appear in public exports unless the user explicitly requests them.

This format is a proposal requiring review before implementation. HDF5 remains
an alternative for array storage, but HDF5 input support is not a prerequisite
for the initial ASCII milestone.

### 7.2 Safety

Loaders treat projects as untrusted:

- reject absolute paths, `..`, links, duplicate entries, and archive traversal;
- impose configurable total, entry-count, and decompressed-size limits;
- validate file magic, schema, dimensions, dtypes, finite/invalid masks, and
  identifier references;
- never use pickle, `eval`, dynamic imports, macros, or embedded Python;
- verify hashes before constructing domain objects;
- do not automatically follow external paths or URLs; and
- report unsupported versions without partially mutating active state.

Saves use a temporary sibling file, flush and validate it, then atomically
replace the destination where the platform permits. Backups and recovery
behavior remain a product decision.

### 7.3 Versioning and migration

The container has a format version; the root project and every persistent
entity have schema/model versions. Migrations are explicit, ordered, pure
transformations:

```text
old bytes -> validated old schema -> migration copy -> validated current schema
```

Never modify the source project in place. Preserve unknown fields only through
a documented extension mechanism; otherwise fail clearly to avoid silent
semantic loss. Each migration records source/target versions, software version,
time, warnings, and a pre-migration hash. Round-trip and fixture tests cover
every supported migration. Downgrade is not assumed.

## 8. Reporting and logging

Structured diagnostics use codes, severity, human-readable text, entity
references, and remediation hints. Logs include identifiers and structural
summaries, not full numerical arrays or confidential content. Reports clearly
separate:

- optimizer convergence;
- statistical diagnostics;
- scientific-quality warnings; and
- user exclusions or overrides.

Plots and tables receive prepared view models with units and FWHM semantics,
preventing UI-specific reinterpretation.

## 9. Acceptance criteria

- Core tests run without importing PySide6.
- Module boundaries cover every responsibility required by version 1.
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
sufficient; complete persistence is not an acceptance criterion. At milestone
8, project saves additionally become data-only, versioned, hash-verifiable,
migration-tested, and safely atomic.

## 10. Assumptions and explicit non-goals

The architecture assumes in-memory NumPy arrays are practical for typical
reduced version-1 datasets during early scientific milestones; later storage
references may enable lazy loading. It assumes one local user and one active
project per application window. Neither assumption changes a scientific
convention.

This design does not specify raw-data infrastructure, cloud sync, collaboration,
plugin execution, a public stable Python API, or a final GUI visual design.

## 11. Unresolved decisions and risks

Unresolved engineering choices include concrete module names, error/result
types, DAVE Q-bin parsing after scientific approval, array chunking, archive
limits, autosave/recovery, worker process versus thread execution, stable
report formats, and whether HDF5 should replace NPY inside the project
container.

Risks include archive abuse, schema drift, large-memory use, cancellation that
leaves partial state, Windows path/process differences, dependency leakage into
the core, and overgrown domain objects. Address these through threat-oriented
load tests, explicit migrations, immutable results, bounded workers, interface
tests, and narrow modules.

## 12. Milestone dependencies

Domain primitives and diagnostics vocabulary precede all other core work. The
content detector and DAVE/wide/single importers precede Q mapping and
resolution association. Resolution processing and grid validation precede
convolution. Convolution precedes single-spectrum fitting.
Validated single fits precede batch and derived results. Derived EISF and
approved equations precede candidate-model fitting. Minimal in-memory contracts
evolve through milestones 1–6; full persistence is reviewed and implemented in
milestone 8. Reporting follows stable result contracts. GUI work follows
scientific validation of the importer-through-single-fit path.
