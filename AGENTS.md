# qensfit project instructions

## Purpose and status

qensfit is an early-stage desktop application and GUI-independent Python core
for standardized analysis of reduced quasielastic neutron scattering (QENS)
data. It is not yet suitable for scientific use. The initial platform is macOS
on Apple Silicon; Windows must be supported before a public release.

Treat the documents under `docs/` as the authoritative product, scientific,
architecture, data-model, validation, and sequencing specifications. If code
and documentation disagree, stop and request scientific-owner review rather
than silently changing a scientific convention.

## Version 1 boundaries

Version 1 starts with reduced sample and measured-resolution data. Its workflow
covers import, inspection, Q mapping, analysis-level masking, resolution
processing, measured-resolution convolution, independent and sequential batch
spectral fits, FWHM linewidths, relaxation times, experimental EISF, XYZ
molecular coordinates, separate C2/C4/isotropic candidate-model fits, model
comparison, exports, and reproducible project persistence.

Do not add raw detector reduction, detector calibration or grouping,
detector-level bad-detector masking, INS analysis, fixed-window scans,
Bayesian/MCMC fitting, global simultaneous spectrum fitting, multi-temperature
Arrhenius fitting, arbitrary user Python motion models, unrestricted molecular
dynamics from crystal structures, web or multi-user features, or automatic
publication-figure editing to version 1.

Mantid interoperability is optional and must stay outside the analysis core.
PySide6 is the intended GUI toolkit, but GUI work must wait until the importer,
resolution path, convolution, and single-spectrum fit have been scientifically
validated.

## Non-negotiable scientific conventions

- Store and report Lorentzian linewidths as FWHM, never silently as HWHM.
- For FWHM `Gamma_meV`, calculate
  `tau_ps = 1.3164239138 / Gamma_meV`, using
  `hbar_meV_ps = 0.6582119569`.
- Use integrated component areas, not peak heights, for experimental EISF:
  `A_elastic / (A_elastic + sum(A_quasielastic_i))`.
- The default fit uses `scipy.optimize.least_squares`, standardized residuals
  `(model - data) / sigma`, and weighted least squares.
- A valid statistical uncertainty is finite and strictly greater than zero.
  Automatically mask every point with NaN, infinite, zero, or negative sigma
  for weighted fitting while preserving the original arrays unchanged.
- Detect padding only as boundary-connected repeated `(intensity, uncertainty)`
  plateaus; never infer padding solely from zero or negative intensity and
  never hard-code a sentinel value.
- Edge-padding status is behavioral, not statistical: `AUTO` points enter a
  reversible default-on mask, `REVIEW` points enter a distinct review mask,
  and `NONE` masks nothing. Neither mask changes imported arrays or performs
  an intensity baseline shift.
- Unweighted and logarithmic relative losses are not defaults.
- Evaluate the elastic term directly as `A_elastic * R_Q(E - E0)` using the
  selected, optionally baseline-corrected, unit-area measured resolution.
  Never construct a grid-dependent numerical delta spike for production fits.
- Evaluate each quasielastic term as a unit-area Lorentzian convolved with the
  measured resolution and multiplied by its integrated area. Preserve area
  semantics within a documented finite-grid tolerance.
- Batch fitting is a sequence of independent per-Q fits, not a global fit.
- Never silently discard possible Bragg contamination, Q spectra, resolution
  ranges, or energy points. Record all warnings, masks, and exclusions.
- Preserve original resolution data separately from processed resolution data.
- Preserve component-level results and covariance information needed for later
  uncertainty propagation.
- C2 and C4 formulas are unresolved pending authoritative scientific input. Do
  not invent or encode final formulas. Any other unspecified model equation or
  diagnostic convention also requires explicit review.

## Import and Q-mapping rules

- Detect reduced-data layouts from content, not file extension. The low-level
  detector returns a proposed layout and diagnostics; later application/GUI
  workflow owns user confirmation.
- Support DAVE group blocks, wide `x/yN/yerrN` tables, and single
  `x/y/yerr` tables through one scientific spectrum interface.
- Give every spectrum an explicit sample or resolution role, or use validated
  role-specific wrappers around a shared numerical structure.
- Preserve per-group grids and row counts. A shared-grid matrix is an optional
  optimization only when grids truly match; import must never interpolate.
- Preserve recognized DAVE fit-result columns as metadata but exclude them
  from energy, measured intensity, and uncertainty.
- Keep invalid uncertainty, automatic edge-padding masks, review-only padding
  masks, and per-group fitting ranges distinct. Manual point masks, whole-Q
  exclusions, and Bragg warnings remain future analysis state.
- Store Q identity once at dataset level as ordered representative values in
  `Å^-1`, with optional explicit edges. Never duplicate Q onto `Spectrum`.
- Edge-defined bins require finite strictly increasing edges. Milestone-2
  representative Q values are edge midpoints; explicit Q-value input does not
  imply or infer edges.
- Uniform manual Q bins are count-driven and exactly cover their inclusive
  outer edges: `delta_Q = (upper_q_edge - lower_q_edge) / group_count`.
- Preserve explicit Q-value order and require exactly one value/bin per
  spectrum. Never sort, deduplicate, pad, truncate, extrapolate, or interpolate.
- The approved four-value DAVE Q-bin parser reconstructs complete fixed-width
  bins from the lower limit, upper limit, and step. The stored group count is a
  consistency field that warns on disagreement; the upper limit may exceed the
  final actual bin edge.

## Data and privacy

`private_data/` and `private_results/` are local-only and Git-ignored. Never
stage, commit, redistribute, upload, copy into tests/docs/examples/packages, or
print private arrays in full. Local validation may report only structural
summaries such as group and row counts, detected columns, energy bounds, and
invalid-value counts. Public tests and examples must use independently
generated synthetic data.

Do not hard-code values from private benchmark files. The known 14-group,
0.46–2.02 inverse-angstrom benchmark is validation context, not a public
fixture or a source of defaults.

## Architecture and implementation rules

- Keep the `src/qensfit` layout and typed Python.
- Keep the scientific core callable and independent of PySide6.
- Keep `ReducedDataset` / `Spectrum` as the stable scientific boundary between
  source-specific import or reduction and downstream analysis. Source formats
  must not leak into future fitting interfaces.
- Design extension seams, not extension scaffolding. Do not add abstract HDF,
  NeXus, instrument, Mantid, registry, factory, adapter, or plugin frameworks
  before at least two real implementations require a shared abstraction.
- Separate domain models, import/export, preprocessing, resolution handling,
  spectral components, convolution, fitting, diagnostics, batch execution,
  derived quantities, molecular geometry, motion models, comparison,
  persistence, reporting, and GUI adapters.
- Put no formulas or optimizer logic in GUI code.
- Avoid global mutable state and hidden unit conversions.
- Use dataclasses or similarly typed immutable objects for early computation
  and Pydantic where boundary validation is useful.
- Retain runtime dependencies only when current production functionality uses
  them; add future numerical, plotting, HDF, persistence, or GUI dependencies
  in the milestone that needs them.
- During milestones 1–6, implement only the minimal typed in-memory models
  required by the scientific workflow. Do not require UUIDs on every temporary
  object, entity migrations, hash-addressed arrays, append-only project
  histories, attachment handling, or archive infrastructure.
- Simple identifiers and immutable values may be used where they improve
  traceability, but persistence work must not delay scientific validation.
- Review and implement persistent schema versions, stable identifiers, hashes,
  migrations, security, atomic save, and the `.qensfit` archive contract in
  milestone 8.
- Preserve source references and hashes where practical, imported data,
  processing history, masks, settings, configurations, results, warnings,
  exclusions, software version, and project-schema version.
- When milestone 8 implements project files, treat them as untrusted input: no
  pickle or executable payloads, validate sizes and schemas, prevent unsafe
  archive paths, verify hashes, and write atomically.
- Long-running fitting must eventually execute outside the GUI thread and
  support cancellation.

## Development sequence and quality

The first implementation milestone after these documents covers a content-based
format detector, DAVE group-block, wide-table, and single-spectrum importers,
the minimal shared spectrum representation with explicit sample/resolution
roles, structural diagnostics, and public synthetic tests. Generic arbitrary
column mapping may wait for a concrete workflow. Do not generalize source formats
or Q definitions into registries.

Milestone 1.2 simplifies this validated core: `ReducedDataset` replaces
`ImportedDataset`, derivable state becomes properties, format confidence and
GUI confirmation state leave the core, and edge padding uses
`AUTO`/`REVIEW`/`NONE`. Preserve numerical arrays, invalid masks, import
behavior, and validated automatic padding masks while narrowing the API.

Milestone 2 adds dataset-level `QBins`, approved uniform edge generation and
DAVE Q-bin parsing, per-group inclusive fitting ranges, derived fitting-point
selection, and small GUI-independent inspection plots. It does not add Q
rebinning, manual single-point masks, whole-Q exclusions, fitting, or GUI state.

Do not begin GUI implementation before the importer, resolution processing,
numerical convolution, and single-spectrum fitting are validated. Full project
persistence begins in milestone 8, not milestones 1–6. AIC/AICc remain planned
diagnostics but do not block the first validated single-spectrum prototype
while their likelihood convention is unresolved.

For each implementation change, add tests appropriate to its scientific risk.
The validation program includes unit, synthetic-spectrum, convolution,
parameter-recovery, regression, DAVE and existing-script comparisons, private
benchmark summaries, cross-platform checks, project round trips, GUI smoke
tests, and explicit convention tests. Never alter a scientific formula merely
to make a test pass.

Run the existing pytest suite, Ruff, and mypy before handoff. Document
assumptions and unresolved decisions; do not resolve scientific ambiguity by
guessing.

## Validation Notebook Maintenance

The project maintains one persistent Jupyter validation notebook as a human-readable integration test of the current production API.

After any milestone that changes public APIs, data models, preprocessing, import behavior, fitting interfaces, diagnostics, or scientific outputs, read and follow:

`docs/test_notebook_maintenance.md`

The current production implementation under `src/qensfit/` is the source of truth.

Do not preserve obsolete production APIs solely for notebook compatibility.

At milestone completion, audit the entire notebook for stale API usage, update it to the current API, integrate a minimal realistic demonstration of the new functionality, and validate it from a clean kernel when validation data are available.
