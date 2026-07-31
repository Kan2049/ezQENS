# Scientific and persistent data model

## 1. Scope

This document proposes scientific in-memory contracts and the later versioned
persistent concepts for version 1. It defines data, validation, units, and links
without creating production model classes.

## 2. Common conventions

### 2.1 Milestones 1–6 in-memory scope

Milestones 1–6 use only the minimal typed in-memory objects needed for
reduced-data import, Q mapping, masks, measured-resolution processing,
convolution, single-spectrum fitting, sequential batch fitting, and derived
linewidth/relaxation-time/EISF results.

These objects may contain NumPy arrays directly. Simple identifiers and
immutable values are encouraged where they improve traceability, but early
objects do not require UUIDs, per-entity schema versions, hash-addressed array
registries, append-only histories, attachments, archive records, or migration
infrastructure. The field lists below describe the eventual persistent contract;
early implementations select only the fields required for their milestone.
Persistence infrastructure must not delay scientific validation.

### 2.2 Milestone 8 persistent scope

When an entity becomes part of the approved milestone-8 project format, its
common persistent fields are:

- `id`: required stable UUID string, unique within the project;
- `model_version`: required positive integer identifying that entity schema;
- `created_at`: required timezone-aware ISO 8601 timestamp;
- `extensions`: optional namespaced JSON object for forward-compatible,
  non-scientific metadata only.

References use identifiers, not nested mutable object graphs. Ordered
collections remain ordered. Enums serialize as stable lowercase strings.
Timestamps serialize in UTC. Missing, invalid, excluded, and zero are distinct
states.

Numerical arrays are stored outside JSON as hash-addressed, dtype/shape/unit
descriptors. Loaders disable pickle and validate shape, dtype, size, and hash.
Every unit-bearing scalar or array has an explicit canonical unit field;
unknown intensity units are represented as `unknown` or an explicit
instrument-supplied label, never guessed.

Model versions describe individual entities. `project_schema_version` on the
root describes the compatible entity set and invariants. Migrations validate
both before and after transformation.

Those array, UUID, schema, and migration requirements begin with milestone 8,
not the early in-memory models.

## 3. AnalysisProject

Root aggregate for one reproducible analysis.

- **Required persistent fields:** milestone-8 common fields;
  `project_schema_version`;
  `software_version`; `title`; ordered `dataset_ids`; `resolution_dataset_ids`;
  `q_mapping_ids`; `processing_step_ids`; `mask_ids`; `fit_configuration_ids`;
  `fit_result_ids`; `batch_fit_result_ids`; `derived_result_ids`;
  `molecular_structure_ids`; `motion_definition_ids`;
  `motion_fit_result_ids`; `comparison_result_ids`; `provenance_ids`;
  `export_record_ids`; `history`; `modified_at`.
- **Optional fields:** description, authors, active/selected result identifiers,
  tags, user notes.
- **Units:** none directly; child entities carry units.
- **Validation:** all identifiers resolve to the expected entity type; selected
  results belong to the project; timestamps are ordered; schema/software
  versions are present; history is append-only in meaning.
- **Serialization and links:** serialized in `models/project.json`; links every
  persistent concept. Large data remain array references.

## 4. Dataset

An immutable imported reduced sample dataset before or alongside derived views.

- **Required scientific fields:** `role` (`sample` or `resolution`); label;
  ordered spectra; source layout; detected layout; detection confidence and
  evidence; detection warnings; source column metadata; energy/intensity/
  uncertainty units; source reference/format; parser version; import summary;
  ignored extra-column metadata; energy-storage mode (`per_spectrum` or
  `shared_grid`).
- **Optional fields:** paired wide-column mapping; explicit user format
  override; alternative plausible layouts; instrument/sample metadata;
  temperature and unit; user notes; privacy-safe original header metadata.
- **Units:** energy normally meV; Q when present in `Å^-1`; intensity and
  uncertainty retain explicit source units.
- **Validation:** at least one spectrum; unique group identities; compatible
  y/yerr units; input order retained; no inferred Q without a `QMapping`;
  sample and resolution roles are never interchangeable; shared-grid mode is
  used only when all spectra truly share the same energy axis; wide pairs are
  complete and unambiguous.
- **Serialization and links:** links `Spectrum`, `ProvenanceRecord`, Q mappings,
  masks, and processing steps. Early objects may hold immutable arrays
  directly; milestone 8 uses approved safe array payloads.

## 5. Spectrum

One reduced energy spectrum associated with one imported group.

- **Required scientific fields:** dataset link; explicit `role` (`sample` or
  `resolution`) or a role-specific wrapper; group index/identity/label; energy,
  intensity, uncertainty, and automatically generated invalid-data mask;
  units; point count; original ordering; source metadata.
- **Optional fields:** mapped `q_value`; Q unit; source group metadata; detected
  extra/ignored fitting columns; paired wide-column names; point-level source
  row identifiers; shared-energy-axis link/view.
- **Units:** energy meV after an explicit unit normalization; Q `Å^-1`;
  intensity and uncertainty as declared by the dataset.
- **Validation:** arrays are one-dimensional, equal length, and nonempty;
  group index is unique within a dataset; monotonicity and invalid values are
  reported rather than silently repaired; sigma is valid only when finite and
  strictly positive; invalid sigma is masked without modifying original arrays;
  role-specific processing is enforced.
- **Serialization and links:** links its `Dataset`, applicable `QMapping`,
  `MaskDefinition`, processing outputs, resolution association, and fit
  results. Early arrays may be direct; milestone-8 arrays use external payload
  references.

## 6. ResolutionDataset

Role-specific wrapper/aggregate for original measured-resolution spectra plus
explicitly derived processed forms. It shares the common numerical spectrum
structure but cannot be mistaken for a sample dataset.

- **Required scientific fields:** role fixed as `resolution`; label;
  `origin_type` (`vanadium` or `low_temperature_sample`); ordered original
  resolution spectra; source provenance; import units/summary; mandatory
  manually selected valid range for each processed group; processed spectra;
  baseline setting; normalization result; grid status; processing warnings.
- **Optional fields:** automatic padding-mask results and advisory suggested
  ranges; interpolation settings; source temperature/unit; association
  mapping; notes.
- **Units:** energy meV; resolution intensity in declared source units before
  processing and unit-area semantics after normalization; Q `Å^-1` when known.
- **Validation:** originals cannot be replaced by processed arrays; every
  processed form links its minimal ordered processing decisions; selected range
  is explicit; normalization requires a finite positive integral; grid validity
  and source type are retained; medium-confidence suggestions cannot become
  selected without confirmation; high-confidence auto-padding remains
  reversible and does not replace the mandatory manually selected range.
- **Serialization and links:** links original/processed `Spectrum` records,
  `ProcessingStep`, `QMapping`, sample-resolution associations, and provenance.

## 7. QMapping

Maps ordered dataset groups to explicit Q values.

- **Required scientific fields:** sample dataset link; `strategy`
  (`linear_range`, `manual_list`, `explicit_list_file`,
  `dave_q_parameter_file`, or `imported_metadata`); unmodified
  `source_definition`; ordered resolved `q_values`; unit fixed as `Å^-1`;
  parser/parser version where applicable; source reference where applicable;
  source format; warnings; confirmation status; ordered sample group identity.
- **Optional fields:** original parsed values/parameters; interpreted DAVE
  field names; `q_start`, `q_end`, `count`; blank-line/comment policy;
  resolution-group comparison; confirmation time; association notes.
- **Units:** Q in `Å^-1`.
- **Validation:** all final values are finite; final count exactly equals sample
  spectrum count; applicable group-by-group resolution count matches; linear
  range has finite endpoints and `N >= 2` and resolves by inclusive
  `numpy.linspace` semantics; manual/imported order is preserved; values are
  never silently sorted, deduplicated, padded, truncated, extrapolated, or
  combined; generated/imported values require confirmation.
- **Serialization and links:** links `Dataset`/`Spectrum`; resolution
  association references this mapping rather than independently inventing Q.
  DAVE parameter parsing is invalid for production until supported field
  meanings are approved.

## 8. ProcessingStep

One auditable transformation or user decision.

- **Required scientific fields:** `operation_type`; ordered input
  references; ordered output references; JSON-compatible parameters; actor
  (`user`, `system_suggestion`, or `confirmed_automatic`); software version;
  status; diagnostics; execution timestamp.
- **Optional fields:** parent step; user reason; numerical summary; algorithm
  version; cancellation information.
- **Units:** every numerical parameter includes a unit when applicable.
- **Validation:** input/output references resolve; operation and algorithm
  version are known; parameters validate against the operation schema; failed
  steps cannot masquerade as selected outputs.
- **Serialization and links:** early models retain only minimal traceable step
  objects; append-only project history is a milestone-8 concern. Links datasets,
  spectra, masks, mappings, and derived arrays and stores no arbitrary code.

## 9. MaskDefinition

Explicit analysis-level inclusion/exclusion state.

- **Required scientific fields:** target dataset/spectrum references;
  invalid-point mask reference; auto-padding mask reference; manual-point mask
  reference; spectral-fit inclusion; EISF-model-fit inclusion; origin;
  reason/history references.
- **Optional fields:** fitting-energy minimum/maximum; suggested range;
  override reason; Bragg-warning annotations; restoration link.
- **Units:** energy limits in meV; Q annotations in `Å^-1`.
- **Validation:** point masks match the target length; range min is below max;
  invalid, auto-padding, and manual masks remain separately recoverable;
  exclusion scopes are distinct; invalid sigma feeds only the invalid mask;
  medium-confidence padding suggestions and Bragg flags do not silently become
  masks; every automatic/manual change is minimally traceable.
- **Serialization and links:** links `Spectrum`, `ProcessingStep`,
  `FitConfiguration`, and history. Prefer bit/boolean array references over
  ambiguous lists of dropped values.

### 9.1 EdgePaddingDetectionResult

Minimal milestone-1.1 in-memory result produced outside importers.

- **Required scientific fields:** algorithm version and explicit tolerance
  configuration; ordered per-spectrum group identity; exact read-only
  high-confidence padding mask; separate medium/high suggestion mask; left and
  right run lengths and energy bounds; plateau intensity/uncertainty values;
  adjacent interior index; quantified intensity/uncertainty transitions;
  per-boundary and aggregate confidence; evidence codes; default-on status; and
  structured diagnostics.
- **Units:** energy bounds retain spectrum energy units; plateau values retain
  spectrum intensity and uncertainty units; masks and confidence are
  dimensionless.
- **Validation:** masks match spectrum length; default-on points are a subset of
  suggestions; only boundary-connected plateaus are eligible; high-confidence
  masks are reversible; medium-confidence points remain confirmation-gated;
  invalid-data masks remain separate; originals are immutable.
- **Serialization and links:** early results hold immutable boolean arrays
  directly and link by ordered group identity. Full persistence and audit
  history remain milestone-8 work.

## 10. SpectralModelDefinition

Declarative composition of a resolution-convolved spectrum.

- **Required scientific fields:** stable model name/version; ordered
  component definitions; exactly one elastic component; one or more
  Lorentzians; background type (`constant` or `linear`);
  `linewidth_convention` fixed as `FWHM`; amplitude semantics fixed as
  `integrated_area`; measured-resolution requirement; elastic evaluation fixed
  as direct scaling of unit-area resolution rather than a numerical delta;
  Lorentzian components fixed as unit-area kernels before convolution.
- **Optional fields:** display labels; description; approved implementation
  version.
- **Units:** component centers and FWHM in meV; integrated areas in the
  spectrum's integrated-intensity unit; linear-background units explicit.
- **Validation:** supported component kinds only; unique component IDs;
  required parameter roles exist; no HWHM persistence; background structure
  matches its type; component-area semantics survive convolution within the
  documented finite-grid tolerance.
- **Serialization and links:** links ordered `ParameterDefinition` records and
  is referenced by fit configurations/results. No callable or Python source is
  serialized.

## 11. ParameterDefinition

One configurable fit parameter.

- **Required persistent fields:** milestone-8 common fields; owner
  component/model ID; stable name;
  scientific role; initial value; lower bound; upper bound; fixed/free state;
  unit; value semantics.
- **Optional fields:** display label; scale; user note; prior successful-fit
  seed reference (not a Bayesian prior).
- **Units:** role-dependent; FWHM/center meV, areas in integrated-intensity
  units, background units explicit, dimensionless parameters labeled `1`.
- **Validation:** finite initial value when required; lower <= initial <= upper;
  fixed values are consistent with bounds; FWHM role always declares FWHM;
  identifiers are unique within the model.
- **Serialization and links:** belongs to `SpectralModelDefinition` or
  `MotionModelDefinition`; copied-by-value into a `FitConfiguration` snapshot
  so later edits cannot rewrite past fits.

## 12. FitConfiguration

Complete reproducible configuration for one or more independent spectral fits.

- **Required persistent fields:** milestone-8 common fields; spectral model
  snapshot/reference;
  ordered parameter snapshots; sample spectrum IDs; resolution dataset and
  association references; mask/range references; optimizer identifier fixed by
  default to `scipy.optimize.least_squares`; residual definition; weighting
  mode; loss; optimizer settings; convolution settings; Q traversal and
  initialization policy when used in batch.
- **Optional fields:** robust-loss settings; manual-refit parent result;
  cancellation settings; user notes.
- **Units:** fitting range and grid values in meV; parameter units retained.
- **Validation:** all references are compatible; at least one valid point beyond
  the required statistical minimum; sigma is finite and positive on every fit
  point; automatically invalid sigma stays masked; the eventual point rule has
  at least positive degrees of freedom; initial values and bounds are valid;
  unweighted/log-relative mode cannot be labeled default; nearest-Q
  association requires explicit selection.
- **Serialization and links:** immutable snapshot links spectra, resolution,
  masks, model and parameter definitions, and resulting fits.

## 13. FitResult

One completed or attempted fit for one Q spectrum.

- **Required scientific fields:** configuration/spectrum links; Q and unit;
  resolution group/processed-array references; status; optimized parameter
  values with units; standard-error status; raw residual array reference;
  standardized residual array reference; chi-square; reduced chi-square;
  fitted-point count; free-parameter count; convergence and termination
  records; bound hits; covariance/Jacobian status; warnings; selected flag.
- **Optional fields:** AIC and AICc with likelihood-convention identifier;
  standard errors; covariance array reference; Jacobian reference or summary;
  optimizer evaluation counts; superseded-by result ID; manual-review notes.
- **Units:** parameters retain definition units; raw residual uses intensity
  unit; standardized residual is dimensionless; FWHM is explicitly meV.
- **Validation:** result dimensions match fitted points/free parameters;
  statistics carry validity states; covariance absence is not zero covariance;
  failed fits preserve diagnostics but cannot be selected as successful;
  AIC/AICc comparison requires identical points, masks/Q selection, uncertainty
  treatment, residual definition, and likelihood convention.
- **Serialization and links:** links configuration, spectrum, processed
  resolution, parameters, batch result, derived result, and provenance.

## 14. BatchFitResult

Ordered outcome of sequential independent fits.

- **Required persistent fields:** milestone-8 common fields; fit configuration
  ID; ordered Q/spectrum
  traversal; direction; previous-fit-seeding policy; ordered per-Q fit-result
  IDs; per-Q statuses/warnings; start/end timestamps; completion/cancellation
  status.
- **Optional fields:** failed/skipped Q identifiers; manual-refit result links;
  selected result per Q; progress summary.
- **Units:** Q in `Å^-1`; child results own other units.
- **Validation:** one independent selected result at most per Q; order matches
  policy; previous-fit seeding is traceable; cancellation does not mark
  unattempted spectra failed; no shared optimized parameter object.
- **Serialization and links:** links `FitConfiguration`, `FitResult`,
  `MaskDefinition`, and `DerivedQENSResult`.

## 15. DerivedQENSResult

Per-Q linewidth, relaxation-time, and experimental-EISF values derived from
selected spectral results.

- **Required persistent fields:** milestone-8 common fields; source fit/batch
  result IDs; ordered Q
  values; Q unit; per-component FWHM values with explicit convention; valid
  relaxation times; elastic and individual quasielastic integrated areas;
  EISF values; formula/version identifiers; validity flags; warnings.
- **Optional fields:** linewidth, relaxation-time, area, and EISF
  uncertainties; covariance references; Q exclusions for motion fitting.
- **Units:** Q `Å^-1`; FWHM meV; relaxation time ps; EISF dimensionless; areas
  in integrated-intensity units.
- **Validation:** relaxation time uses `1.3164239138 / Gamma_meV`; nonpositive
  or invalid FWHM yields invalid tau with warning; EISF denominator/components
  are retained; missing covariance does not create zero uncertainty.
- **Serialization and links:** links selected `FitResult` records,
  `MaskDefinition`, formula versions, and motion-model fits.

## 16. MolecularStructure

Immutable imported molecular coordinates plus explicit selections.

- **Required persistent fields:** milestone-8 common fields; label; source
  provenance; ordered atom
  records containing element symbol and x/y/z; coordinate unit fixed as
  angstrom; all-atom count; selected hydrogen atom IDs; rotation center fixed
  as `(0, 0, 0)`; selected axis (`x`, `y`, or `z`); hydrogen distances.
- **Optional fields:** XYZ comment; atom labels if supplied through an approved
  extension; selection notes.
- **Units:** coordinates and distances in angstrom.
- **Validation:** atom count and arrays agree; element symbols are retained;
  all atoms survive import; selected atoms are hydrogens; distances are finite;
  center and axis satisfy version-1 restrictions.
- **Serialization and links:** atom arrays are safe array payloads; links
  provenance and motion definitions. No crystal-derived dynamics are inferred.

## 17. MotionModelDefinition

Declarative definition of one candidate EISF model.

- **Required persistent fields:** milestone-8 common fields; candidate kind
  (`c2`, `c4`, or
  `isotropic`); display name; scientific formula identifier/version; ordered
  parameter definitions; required molecular inputs; applicable Q domain;
  independent-candidate flag fixed true.
- **Optional fields:** molecular structure ID; rotation-axis selection;
  implementation citation after approval; scientific-owner approval record.
- **Units:** Q `Å^-1`; EISF and most fit fractions dimensionless; geometry
  parameters use explicit angstrom units where approved.
- **Validation:** one alternative per definition; no combined population;
  required approval/version must exist before executable C2/C4 definitions are
  valid; parameters obey bounds and physical validity rules.
- **Serialization and links:** contains data only and links
  `ParameterDefinition`, `MolecularStructure`, and formula provenance.
  Authoritative C2/C4 formulas are unresolved and absent from this specification.

## 18. MotionModelFitResult

Independent fit of one candidate model to experimental EISF.

- **Required persistent fields:** milestone-8 common fields; motion definition
  ID; derived-QENS result
  ID; included/excluded Q identifiers; optimized parameters; standard-error
  status; raw and weighted residual references; reduced chi-square; AICc;
  fitted-point/free-parameter counts; convergence; parameter-validity status;
  warnings.
- **Optional fields:** covariance; standard errors; chi-square/AIC when
  calculated under the approved convention; Jacobian status; reviewer notes.
- **Units:** Q `Å^-1`; EISF/residuals dimensionless; parameters retain declared
  units.
- **Validation:** only motion-included Q values are used; candidate is fit
  independently; statistics use the same approved convention across candidates;
  invalid parameters are flagged even after numerical convergence.
- **Serialization and links:** links motion definition, molecular structure
  where required, experimental EISF, masks, and comparison result.

## 19. ModelComparisonResult

Comparison of compatible independent motion-model fits.

- **Required persistent fields:** milestone-8 common fields; ordered candidate
  fit-result IDs;
  comparison-method version; common data-selection identity; metrics including
  weighted residual evidence, reduced chi-square, AICc, uncertainties,
  residual-structure assessment, validity, and warnings; ranking; narrative
  limitation statement.
- **Optional fields:** best-supported result ID; delta-AICc values; model
  weights if later approved; reviewer notes.
- **Units:** metrics dimensionless unless explicitly declared.
- **Validation:** candidates use identical experimental points and statistical
  conventions; invalid candidates remain visible; ties/weak discrimination are
  represented; text never claims unique microscopic proof.
- **Serialization and links:** links candidate `MotionModelFitResult` records
  and source `DerivedQENSResult`.

## 20. ExportRecord

Audit record of a user-requested project or result export.

- **Required persistent fields:** milestone-8 common fields; export type and
  schema/version; created
  time; selected entity IDs; destination reference sanitized for privacy;
  content hash where practical; units/convention manifest; status; warnings.
- **Optional fields:** file name; reviewer label; failure details; redaction
  summary.
- **Units:** none directly; manifest enumerates exported field units.
- **Validation:** selected entities exist; export schema is known; FWHM is
  explicit; private source content is not included without an allowed local
  project-save policy.
- **Serialization and links:** links every exported result and relevant
  `ProvenanceRecord`; project-save records may be stored in later project
  snapshots to avoid self-reference.

## 21. ProvenanceRecord

Origin and integrity metadata for sources, algorithms, decisions, and software.

- **Required persistent fields:** milestone-8 common fields; provenance kind;
  source display name;
  acquisition/import time; software name/version; actor; related entity IDs;
  privacy classification; structural summary.
- **Optional fields:** local source reference; SHA-256 hash; file size;
  modification time; instrument/sample metadata; parent provenance IDs;
  citation; user note.
- **Units:** structural summary values carry units.
- **Validation:** private classification restricts logging/export; hashes use an
  identified algorithm; references resolve; no full intensity arrays are
  stored in summaries.
- **Serialization and links:** forms a directed provenance graph across
  datasets, processing steps, formulas, fits, migrations, and exports. Local
  paths may be redacted without losing embedded-data identity.

## 22. Entity relationship summary

```text
AnalysisProject
  -> Dataset -> Spectrum
  -> ResolutionDataset -> Spectrum
  -> QMapping -> Dataset/Spectrum
  -> ProcessingStep -> input/output entities
  -> MaskDefinition -> Spectrum
  -> SpectralModelDefinition -> ParameterDefinition
  -> FitConfiguration -> model + spectra + resolution + masks
  -> FitResult -> configuration + one spectrum
  -> BatchFitResult -> ordered FitResult set
  -> DerivedQENSResult -> selected FitResult set
  -> MolecularStructure
  -> MotionModelDefinition -> ParameterDefinition + optional structure
  -> MotionModelFitResult -> derived result + motion definition
  -> ModelComparisonResult -> motion fit results
  -> ExportRecord
  -> ProvenanceRecord -> any entity
```

## 23. Acceptance criteria

- Milestones 1–6 can use minimal immutable typed objects without UUID, archive,
  migration, or hash-registry infrastructure.
- Sample and measured-resolution spectra are semantically distinct while
  sharing one numerical spectrum interface.
- Dataset import state retains source/detected layout, confidence/evidence,
  storage mode, paired/ignored columns, and role.
- Q mapping retains source definition separately from confirmed resolved values
  for every required strategy.
- Units and FWHM semantics remain explicit across every relevant link.
- Imported originals cannot be confused with processed values.
- Every derived result can trace back to source arrays and settings.
- Missing uncertainty/covariance is distinct from zero.
- Masks and Q exclusions retain scope and history.
- High-confidence edge-padding masks are exact, reversible, separately
  recoverable, and never imply an intensity baseline shift.
- At milestone 8, project serialization contains data only, validates all
  references, and migrates scientific meaning or fails visibly.

## 24. Assumptions and explicit non-goals

The early proposal assumes direct in-memory NumPy arrays and simple identifiers
are sufficient for milestones 1–6. UUID identity, UTC timestamps, JSON
metadata, external safe numerical payloads, and migrations are milestone-8
engineering proposals, not early scientific prerequisites.

This document does not define production classes, database storage, a public
API, raw detector entities, global-fit entities, Bayesian posterior objects, or
arbitrary executable model serialization.

## 25. Unresolved decisions, risks, and dependencies

Unresolved details include exact enum names, intensity-unit vocabulary,
explicit-list comment syntax, DAVE group identity details, approved DAVE Q-bin
fields, fit-statistic formulas, covariance storage thresholds, motion
formulas/parameters, project archive limits, and redaction defaults. Inclusive
linear-range endpoints and invalid-sigma handling are approved.

Risks include schema over-complexity, duplicated stale values, dangling
references, array/metadata hash mismatch, unit ambiguity, and migration loss.
Normalize links, validate whole-project invariants, prefer immutable snapshots,
and maintain migration fixtures.

Minimal domain contracts precede import implementation. Dataset/Spectrum roles
precede table import, and QMapping follows importer group identity. Resolution
and processing models precede convolution.
Fit models precede batch/derived records. Approved motion definitions precede
motion results. Full persistent schemas are reviewed in milestone 8 after the
scientific in-memory contracts stabilize.
