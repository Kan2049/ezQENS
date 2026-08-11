# Scientific data model

## 1. Scope

This document defines the small current scientific boundary and the minimum
near-term values needed through milestone 7. It does not prescribe production
classes for unimplemented workflows or the milestone-8 persistence container.

The design rule is to preserve scientific invariants while deriving state that
can be calculated reliably. General abstractions are introduced only after at
least two real implementations require them.

## 2. Current reduced-data boundary

```text
source-specific import or reduction
  -> ReducedDataset -> ordered Spectrum values
  -> downstream scientific operations
```

The current producer is the reduced text importer. A future raw-data reducer or
concrete interoperability adapter can produce the same domain values without
changing downstream analysis. No HDF/NeXus, instrument, Mantid, registry,
factory, adapter-base, or plugin model is declared now.

### 2.1 Spectrum

`Spectrum` is one immutable reduced energy spectrum.

**Required fields**

- `role`: `sample` or `resolution`;
- `group_index`: zero-based ordered index within its dataset;
- `group_label`: nonempty source/group identity;
- one-dimensional `energy`, `intensity`, and `uncertainty` arrays;
- explicit energy, intensity, and uncertainty units; and
- derived invalid-energy, invalid-intensity, and invalid-uncertainty masks.

**Validation and semantics**

- Arrays have equal nonzero length and are read-only after construction.
- Parsed numerical values remain unchanged; invalid values are masked, not
  repaired or deleted.
- Energy and intensity are invalid when nonfinite.
- Uncertainty is invalid unless finite and strictly greater than zero.
- Group order is explicit; no Q value is inferred.
- Unequal spectrum lengths and energy grids are valid.

`Spectrum` contains no text rows, column names, source layout, format-detection
state, generalized provenance object, or persistence identity. Those concepts
do not define the reduced scientific spectrum.

### 2.2 ReducedDataset

`ReducedDataset` is an ordered, immutable collection of `Spectrum` values.
The name intentionally does not imply that direct text import is its only
possible producer.

**Required fields**

- `role`, shared by every contained spectrum; and
- one or more ordered spectra with contiguous group indices.

**Optional import-boundary fields**

- source basename/reference;
- resolved reduced-text layout;
- structured diagnostics; and
- ordered text column/row traceability aligned with spectra.

The import-boundary metadata is optional so future non-text producers do not
fabricate rows or columns. It is deliberately a small value, not a generalized
source/provenance hierarchy.

**Derived properties**

- whether all spectra share exactly the same energy grid;
- mapped required source columns;
- unique ignored/extra source columns; and
- privacy-safe structural summaries.

No shared energy-axis copy or shared-grid boolean is stored. An optimized
matrix representation may be introduced only after profiling and must remain
hidden behind the per-spectrum interface without interpolation or data loss.

### 2.3 Text source-column metadata

For each imported text spectrum, the dataset may retain group identity, mapped
energy/intensity/uncertainty column names, ignored extra columns, and original
source row numbers. This metadata supports basic traceability and importer
tests. It belongs to the dataset import operation, not `Spectrum`, and is not a
general public provenance framework.

### 2.4 FormatDetectionResult

The detector returns:

- proposed layout: `dave_group_blocks`, `wide_qens_table`,
  `single_spectrum_table`, `ambiguous`, or `unknown`;
- concise evidence used by the current structural inspector;
- detected required and extra columns;
- detected group/pair count;
- structured diagnostics; and
- plausible alternatives when ambiguous.

It does not store a confidence tier, confirmation state, explicit-override
state, or extension hint. Explicit override is a function argument and must
validate the selected structure. User confirmation belongs to later
application/GUI workflow.

### 2.5 Import diagnostics and summaries

Diagnostics contain a stable code, severity, privacy-safe message, and optional
group, row, or column locator. They never contain full source rows or scientific
arrays.

Structural summaries may contain layout, spectrum count, row counts, mapped
and ignored columns, finite energy bounds, invalid-value counts, and derived
shared-grid status. They omit source arrays and absolute paths.

## 3. Current edge-padding results

Edge padding is preprocessing output, separate from import and invalid masks.

### 3.1 BoundaryPaddingResult

Each left/right result contains only:

- boundary side;
- repeated-pair run length;
- energy bounds when a usable candidate exists;
- behavioral status: `AUTO`, `REVIEW`, or `NONE`; and
- a compact reason code.

Internal plateau signatures, transition calculations, support counters, and
candidate structures are not durable domain state.

### 3.2 SpectrumPaddingResult

Each spectrum result contains:

- group index and identity;
- exact read-only `auto_mask`;
- exact read-only `review_mask`; and
- left and right boundary results.

The masks have the same length as the spectrum and are mutually exclusive.
Counts, aggregate status, and whether review is needed are derived properties.
`AUTO` is eligible to be default-on; `REVIEW` is not. Neither mask changes the
original arrays or invalid-value masks.

### 3.3 EdgePaddingDetectionResult

The dataset result contains ordered spectrum results and one algorithm version.
It does not copy the version, diagnostics, aggregate evidence, or derived
default-on flags into every spectrum result. Its privacy-safe summary reports
only group identity, side counts, total automatic/review counts, statuses, and
retained finite energy bounds.

## 4. Current and near-term scientific values

These values remain minimal immutable in-memory objects through milestone 7.

### 4.1 QBins — milestone 2

`QBins` belongs to `ReducedDataset`, not `Spectrum`. It contains an ordered
read-only `q_values` array in `Å^-1` and an optional read-only `edges` array.
There is exactly one representative value per spectrum. Known edges contain
one additional value and are finite and strictly increasing.

`QBins.from_edges(...)` derives Milestone-2 midpoint representatives while the
base value permits a future reducer to supply a scientifically justified
weighted/effective representative with known edges. `QBins.from_q_values(...)`
preserves explicit nonlinear order and leaves edges unknown. Count-driven
`uniform_q_bins(...)` uses authoritative inclusive outer edges and group count
and covers the range exactly. No Q state is copied onto `Spectrum`, and no
mapping provenance or GUI confirmation state is stored in the scientific value.

`DAVEQBinsResult` keeps DAVE source metadata outside `QBins`: lower limit,
upper limit, step, reported group count, and diagnostics. The parser rebuilds
all complete fixed-width bins from the limits and step. The upper limit need not
equal the final actual bin edge, and disagreement between the reported and
reconstructed group counts is a warning rather than a parse failure.

### 4.2 FittingRange and FittingSelection — milestone 2

Each ordered group has its own finite inclusive `FittingRange`.
`FittingSelection` associates the unchanged `ReducedDataset`, padding result,
and ordered range tuple. It derives rather than stores invalid, in-range,
excluded, and retained masks. Exclusion is invalid measurement OR `AUTO`
padding OR outside the selected group range. `REVIEW` is not automatically
excluded. A selection fails when a group retains no usable measured points.

Manual point masks, whole-Q fitting exclusions, Bragg warnings, and motion-fit
exclusions remain separate future concepts and are not Milestone-2 state.

### 4.3 ResolutionDataset and processing values — milestone 3

Keep original resolution spectra separate from every processed result. Minimal
processing values record authoritative manual valid range, optional approved
baseline setting, unit-area normalization result, grid/interpolation settings,
group association, and diagnostics. Normalization requires a finite positive
integral. Edge-padding masks remain reversible and do not replace manual range
selection.

### 4.4 Spectral and convolution values — milestones 4–5

`SpectralModelDefinition` minimally describes one elastic component, one or
more unit-area Lorentzians, constant or linear background, integrated-area
amplitudes, and FWHM linewidths. `ParameterDefinition` supplies initial value,
bounds, fixed/free state, unit, and meaning.

`FitConfiguration` captures the selected spectrum/resolution, masks/range,
model parameters, weighted residual definition, optimizer and convolution
settings. `FitResult` captures parameters, residuals, chi-square, reduced
chi-square, point/free-parameter counts, convergence, bound hits,
Jacobian/covariance status, and scientific warnings. AIC/AICc remain optional
until their likelihood convention is approved.

### 4.5 Batch and derived values — milestone 6

`BatchFitResult` is an ordered set of independent per-Q results with traversal,
seeding, failure, exclusion, manual-refit, and selected-result state.
`DerivedQENSResult` retains Q, per-component FWHM, valid relaxation times,
elastic and individual quasielastic integrated areas, EISF, validity, and
warnings. Missing covariance never means zero uncertainty.

### 4.6 Molecular and motion values — milestone 7

`MolecularStructure` retains all XYZ atoms, hydrogen selection, coordinates in
angstrom, fixed origin, selected axis, and hydrogen distances.
`MotionModelDefinition`, `MotionModelFitResult`, and `ModelComparisonResult`
are added only after authoritative equations and statistical conventions are
approved. C2, C4, and isotropic candidates remain separate alternatives. C2
and C4 formulas are unresolved and must not be inferred here.

## 5. Persistence is deferred to milestone 8

The long-term version-1 project must eventually represent `AnalysisProject`,
datasets/spectra, mappings, processing steps, masks, model and parameter
definitions, fit/batch/derived results, molecular and motion values,
comparisons, `ExportRecord`, and `ProvenanceRecord`.

Those persistent schemas are not current production requirements. Milestone 8
reviews the stable scientific values first, then defines identifiers, schema
versions, hashes, migration, archive layout, attachments, security, redaction,
and atomic save behavior. No UUID, hash registry, entity migration, append-only
history, archive object, or persistence dependency is required in milestones
1–7 merely to anticipate that work.

The future project format must remain data-only, reject executable/pickle
payloads and unsafe archive paths, validate dimensions and sizes, protect
private source paths, and preserve units, FWHM meaning, original data,
processing decisions, masks, configurations, results, warnings, exclusions,
software version, and project-schema version.

## 6. Relationship summary

```text
ReducedDataset -> ordered Spectrum
ReducedDataset -> optional text import metadata + diagnostics
EdgePaddingDetectionResult -> ordered SpectrumPaddingResult
SpectrumPaddingResult -> one Spectrum by group order/identity

future milestone sequence:
ReducedDataset + QBins -> FittingSelection -> processed resolution
  -> FitConfiguration -> FitResult -> BatchFitResult
  -> DerivedQENSResult -> molecular/motion results -> comparison
  -> milestone-8 project/export/provenance schemas
```

## 7. Acceptance criteria

- `ReducedDataset` / `Spectrum` is the only current downstream scientific
  boundary.
- Sample and resolution roles are explicit and consistent.
- Arrays and invalid masks are immutable, exact, equal-length, and nonempty.
- Invalid uncertainty is finite-and-positive only and is never repaired.
- Text-specific traceability is outside `Spectrum`.
- Shared-grid and extra-column state are derived without caching.
- Detection has no confidence or GUI workflow state.
- `AUTO` and `REVIEW` masks are exact, read-only, mutually exclusive, and
  independent of invalid masks.
- Q bins are dataset-level, count-aligned, immutable, and absent from `Spectrum`.
- Per-group fitting ranges and derived masks preserve all original arrays.
- No Q rebinning, interpolation, baseline shift, or silent repair occurs.
- Future sources can produce `ReducedDataset` without a predeclared framework.

## 8. Explicit non-goals and unresolved decisions

This model does not define raw reduction, HDF/NeXus schemas, Mantid or
instrument protocols, plugin systems, generic reduction factories, GUI state,
or current persistence classes.

Still unresolved are detailed general DAVE grammar, explicit-list comment
syntax, resolution/convolution policies, covariance and AIC/AICc
conventions, Bragg heuristics, motion equations, and the milestone-8 project
format. None may be resolved from private benchmark values or by adding
premature abstraction.
