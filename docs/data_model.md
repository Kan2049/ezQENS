# Scientific data model

## 1. Scope

This document defines the small current scientific boundary and the minimum
near-term values needed for the v1.0 reduced-data workflow. It does not
prescribe production classes for post-v1.0 workflows or a future persistence
container.

The design rule is to preserve scientific invariants while deriving state that
can be calculated reliably. General abstractions are introduced only after at
least two real implementations require them.

The stable scientific backbone comprises spectra/datasets, units, Q identity,
masks/selections, resolution semantics, spectral and fit-result semantics, and
basic derived quantities. Expanding import/export, visualization, GUI, and
later dynamics edges consume or produce these values without adding
source-specific or presentation-specific state to the stable core.

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

These values remain minimal immutable in-memory objects through the v1.0
scientific-core milestones.

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

Manual point masks, whole-Q fitting exclusions, and Bragg warnings remain
separate future concepts and are not Milestone-2 state.

### 4.3 ResolutionDataset and processing values — milestone 3

`PreparedResolution` is the measured-resolution-only dataset result. It
references the immutable sample and resolution `ReducedDataset` values, their
independently calculated `EdgePaddingDetectionResult` values, one ordered
`PreparedResolutionSpectrum` per resolution group, per-Q padding comparisons,
and privacy-safe diagnostics. Association is by exact existing group/Q order;
Q values remain dataset-level and are not copied into `Spectrum`.

`PreparedResolutionSpectrum` references its original resolution `Spectrum`,
its resolution-specific padding result, one `ResolutionSupport`, the raw
normalization integral, its reciprocal factor, the declared trapezoidal method,
and local diagnostics. It does not store copies of original or normalized
arrays. Accepted masks, accepted original coordinates, normalized intensity,
scaled uncertainty, normalized integral, and a source-grid representation with
zero outside accepted support are derived read-only properties.

`ResolutionSupport` is a finite inclusive energy interval distinct from sample
`FittingRange`. Its source is either the default valid measured bounds or an
explicit override. `PreparedResolutionSpectrum.auto_padding_applied` is a
separate boolean that defaults to true and can explicitly restore otherwise
valid AUTO-marked points inside the support. Changing support alone does not
change this state. `REVIEW` remains accepted by default. Invalid
energy/intensity is non-overridable; invalid uncertainty stays inspectable and
is scaled as supplied rather than replaced or included in normalization-area
covariance propagation.

The accepted measured energy coordinates contain at least two finite, strictly
increasing, unique points. An invalid energy/intensity point between the first
and last coordinates selected by the support is a blocking internal hole; it is
not removed and bridged during trapezoidal integration. Trapezoidal area must
be finite and positive. No baseline, interpolation, FFT/convolution,
recentring, fit, optimizer, cache, or GUI state belongs to these values.

### 4.4 Spectral and convolution values — milestones 4–5

Milestone 4 adds two immutable computational values, not persistence entities.
`ConvolutionPlan` contains copied read-only original sample target coordinates,
the canonical-meV S1/4 spacing, zero-anchored intrinsic-model coordinates,
temporary resolution coordinates and unit-area values, explicit full-output
coordinates, the pre-correction representation area, FFT length, and fixed
resolution transform. It derives the corrected numerical resolution area,
full linear length, energy unit, and approximate temporary working memory.

`ConvolvedProfile` contains one read-only full linear-convolution energy/value
pair and its uniform spacing. It evaluates the fixed profile linearly at
arbitrary finite target coordinates, including future `target - E0` queries,
only when those coordinates remain inside the calculated physical domain.
Neither value stores sample intensity/uncertainty, modifies source arrays,
duplicates M3 normalized source state for persistence, or owns fit/model/Q/GUI
policy.

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

### 4.6 Later dynamics values — provisional

Selected temporal or spatial QENS dynamics analysis may later consume validated
FWHM(Q), relaxation-time, EISF(Q), component-Q, and uncertainty results. If a
model is promoted, its minimal typed inputs/results are specified only after
authoritative equations, real workflows, diagnostics, and statistical
conventions are approved. No current class, registry, or interface exists
merely to anticipate a model catalogue.

Automatic model inference from molecular coordinates, symmetry, or structures
is a substantially later and separate capability. It does not shape current
derived-result values.

## 5. Lightweight v1.0 reproducibility and future persistence

Version 1.0 retains enough lightweight information to identify input sources,
Q assignment, processing selections and masks, model configuration, fit
settings, results, warnings, exclusions, and software version. The exact value
or export records are introduced only when the release workflow needs them.

A complex project container is not a v1.0 requirement. The former `.qensfit`
proposal is not a permanent format or extension. Any post-v1.0 persistence
design begins from then-stable scientific values and addresses data-only
security, identifiers, schemas, hashes, migration, archive layout, redaction,
and atomic writes only if the selected format requires them. No UUID, hash
registry, entity migration, append-only history, archive object, or persistence
dependency is added merely to anticipate that work.

## 6. Relationship summary

```text
ReducedDataset -> ordered Spectrum
ReducedDataset -> optional text import metadata + diagnostics
EdgePaddingDetectionResult -> ordered SpectrumPaddingResult
SpectrumPaddingResult -> one Spectrum by group order/identity

future milestone sequence:
ReducedDataset + QBins -> FittingSelection + PreparedResolution
  -> FitConfiguration -> FitResult -> BatchFitResult
  -> DerivedQENSResult -> reports/exports + lightweight reproducibility

later, if explicitly approved:
DerivedQENSResult -> selected temporal/spatial dynamics analysis

substantially later:
molecular/structure input -> automatic candidate inference
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
- Prepared resolution references immutable sources, applies reversible
  resolution `AUTO` exclusion by default, retains `REVIEW`, and normalizes
  independently per Q.
- Exact ordered sample/resolution Q identity is required; nearest-Q association
  and Q interpolation are absent.
- No Q rebinning, interpolation, baseline shift, or silent repair occurs.
- Future sources can produce `ReducedDataset` without a predeclared framework.

## 8. Explicit non-goals and unresolved decisions

This model does not define raw reduction, HDF/NeXus schemas, Mantid or
instrument protocols, plugin systems, generic reduction factories, GUI state,
or current persistence classes.

Still unresolved are detailed general DAVE grammar, explicit-list comment
syntax, Milestone-4 convolution-grid/interpolation/centering policies, any
future explicit resolution baseline or energy-recentering operation,
normalization-integral covariance propagation, analytic resolution sources,
covariance and AIC/AICc conventions, Bragg heuristics, and any future project
format. Post-v1.0 motion equations also remain unresolved. None may be resolved
from private benchmark values or by adding premature abstraction.
