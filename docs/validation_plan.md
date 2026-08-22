# Validation plan

## 1. Purpose and scope

ezQENS requires both software verification and scientific validation. Passing
tests demonstrates behavior against approved references; it does not by itself
establish scientific correctness. Scientific formulas and conventions must
never be altered merely to make tests pass.

This plan covers the v1.0 reduced-data path through scientific export,
lightweight reproducibility, the guided GUI, and macOS/Windows packaging. Raw
detector reduction, automatic molecular/structure-driven inference, and other
v1.0 non-goals are excluded. Validation for a selected temporal or spatial
dynamics model becomes active only if the owner promotes that concrete model;
no catalogue is assumed.

## 2. Validation principles

- Use independently generated synthetic public fixtures.
- Keep private benchmark data local, ignored, summarized, and absent from test
  output/artifacts.
- Test scientific invariants at module boundaries and across any implemented
  lightweight export/reload boundary.
- Compare against at least one independent implementation or analytic case for
  high-risk numerical operations.
- Separate optimizer convergence from statistical and scientific validity.
- Record tolerances with physical/numerical justification.
- Make stochastic tests deterministic through explicit seeds; avoid dependence
  on random luck.
- Run platform-independent core tests on macOS and Windows.
- Treat warnings and failure behavior as testable outputs.
- Keep structured diagnostics testable across import, preprocessing,
  resolution, fitting, parameter/covariance validity, derived quantities, and
  any later dynamics fitting.

## 3. Test layers

### 3.1 Unit tests

Early unit tests cover validators, units, Q mappings, masks, spectrum roles,
parameter bounds, minimal processing traceability, formula constants, and
result diagnostics. Tests are fast, deterministic, and do not read private
data. Complex archive and migration tests are added only if a post-v1.0
persistence design actually requires them.

Required convention cases include:

- FWHM is persisted and displayed as FWHM, never silently HWHM;
- `tau_ps = 1.3164239138 / Gamma_meV` for representative positive widths;
- zero/negative/nonfinite FWHM yields invalid tau plus warning;
- experimental EISF uses integrated elastic and individual quasielastic areas;
- missing covariance does not produce zero uncertainty;
- default residual sign is `model - data`;
- default weighting divides by valid `sigma`;
- only finite, strictly positive sigma is statistically valid;
- invalid sigma remains in original arrays and is automatically masked;
- the fit fails before nonpositive statistical degrees of freedom;
- direct elastic evaluation scales unit-area measured resolution without a
  discrete delta;
- batch and derived-result exclusions remain distinct; and
- original and processed resolution references cannot alias semantically.

### 3.2 Importer contract tests

Public synthetic text fixtures cover the following contracts.

#### 3.2.1 Format detection

- DAVE group blocks are detected from markers and compatible headers.
- Wide `x/yN/yerrN` tables are detected from paired columns.
- Single `x/y/yerr` tables are detected.
- Explicit user format overrides automatic detection.
- Ambiguous input returns plausible alternatives instead of guessing.
- Identical contents with different extensions classify identically.
- Malformed headers return structural diagnostics.
- Detection results include proposal, concise evidence, counts, required and
  extra columns, diagnostics, and alternatives.
- Detection contains no confidence, extension, override-history, or GUI
  confirmation state; explicit override still validates without fallback.

#### 3.2.2 Wide-table parsing

- One and multiple intensity/uncertainty pairs.
- Missing `yerrN` and missing `yN`.
- Duplicate and nonsequential suffixes.
- Reordered but valid paired columns.
- Inconsistent row width.
- NaN, positive/negative infinity, zero, and negative uncertainties.
- Exact preservation of the shared energy axis and original row order.
- Independent per-spectrum domain views over any matrix representation.
- No implicit generation of physical Q values.

#### 3.2.3 DAVE group-block parsing

- One and multiple groups.
- Unequal row counts, energy grids, and valid ranges.
- Required `x`, `y`, and `yerr`.
- Additional `ModelFit`, `Func1`, `Func2`, and recognized fit-component
  columns recorded and excluded from measured values.
- Malformed group boundaries.
- Preserved group/block and row order.
- Invalid uncertainties handled by an invalid-data mask without array mutation.

#### 3.2.4 Q mapping

- Arbitrary finite strictly increasing edges, including nonuniform bins.
- Edge midpoint representatives without a permanent midpoint invariant.
- Count-driven uniform outer-edge generation preserves both endpoints and uses
  `(upper_q_edge - lower_q_edge) / group_count`.
- Explicit representative values preserve nonlinear order and infer no edges.
- Q-bin/value count must match spectrum count.
- No silent sort, deduplication, truncation, padding, extrapolation, discard,
  interpolation, or combination.
- The supported DAVE four-value parser reconstructs complete bins from lower
  limit, upper limit, and step; it retains an exact-fit final bin despite binary
  rounding and excludes a genuinely incomplete final bin.
- DAVE source upper limits may exceed final actual edges, and disagreement
  between stored and reconstructed group counts warns without failing parsing.
- Regression tests explicitly distinguish DAVE step-driven bins from
  count-driven exact-coverage `uniform_q_bins()` results.

#### 3.2.5 Fitting selection and visualization

- Independent finite inclusive ranges for every group.
- One initial common range with immutable group overrides.
- Effective exclusion combines invalid measurement, `AUTO` padding, explicit
  manual point exclusion, and outside-range points while retaining `REVIEW`
  where otherwise usable.
- Manual masks default to all false, are exact-length/read-only per group, can be
  replaced or cleared immutably, cannot restore invalid or `AUTO` points, and
  fail clearly if no usable point remains.
- Empty usable selections fail clearly and no original array is modified.
- Direct in-memory/source-independent datasets support the full milestone path.
- Plot tests verify figures/axes, plotted scientific values, selected ranges,
  and non-mutation without pixel-perfect comparisons.

#### 3.2.6 Edge-padding detection and custom input

- Long repeated pairs at left, right, and both boundaries.
- Different padding lengths by Q group.
- Matching boundary signatures across multiple groups.
- Two-point boundary runs promoted only by strong long-run cross-group
  evidence.
- Near-equal plateau values exercise the declared tolerances.
- Finite positive padding uncertainty is accepted.
- Invalid uncertainty remains solely in its invalid-data mask.
- Isolated negative interior points and noisy negative physical tails remain
  unmasked.
- Repeated values inside a physical region are not automatically removed.
- Weak transitions are `REVIEW`, not default-on masks.
- Otherwise identical positive, zero, and negative plateaus receive equivalent
  status and mask behavior.
- No sigma-jump, nonpositive-intensity, or relative-run heuristic affects the
  decision.
- Detection has no hard-coded padding value.
- Exact boundary plateau membership includes the point adjacent to the first
  retained interior point.
- `AUTO` and `REVIEW` masks match spectrum length, are read-only, and are
  mutually exclusive.
- Detection does not mutate or baseline-shift imported arrays.
- Privacy-safe summaries contain group identities, statuses, counts, and
  retained energy bounds, not arrays or plateau values.
- Padding, invalid sigma, manual masks, fit range, resolution range, and Bragg
  warnings remain distinct.
- Generic custom mappings identify energy, intensity, uncertainty, optional
  group identity, and optional embedded Q without silent inference.
- Structural reports remain privacy-safe.

Golden fixtures should be short, human-readable, independently generated, and
reviewed. Do not derive their values or layout by copying private files.

### 3.3 Synthetic spectrum tests

Construct spectra from approved component definitions on controlled energy
grids, with known elastic/Lorentzian integrated areas, FWHM values, center
shifts, backgrounds, measured-resolution kernels, and uncertainty arrays.

Maintain tiers:

1. Noise-free analytic/near-analytic cases for exact bookkeeping.
2. Fixed-seed Gaussian-noise cases for realistic recovery.
3. Edge cases near bounds, overlapping Lorentzians, weak components, truncated
   windows, invalid points, and heterogeneous sigma.
4. Deliberate non-identifiability and failure cases that must warn.

Synthetic generation must be independent enough from the fitted evaluation
path to avoid testing code against itself. Where reuse is unavoidable, compare
against analytic integrals or a separately reviewed reference calculation.

### 3.4 Numerical convolution tests

Validate numerical convolution before using it in fitting:

- Gaussian × Gaussian, Lorentzian × Lorentzian, and Gaussian-resolution ×
  Lorentzian cases agree with analytic or independent numerical references;
- unit-area measured resolution multiplied directly by `A_elastic` integrates
  to the elastic area within tolerance;
- elastic evaluation does not construct or depend on a discrete numerical
  delta spike;
- a unit-area Lorentzian multiplied by `A_i` retains its integrated area after
  measured-resolution convolution within reviewed finite-grid tolerance;
- ordinary, sub-cell, and approximately `FWHM/h = 0.05` Lorentzians match their
  analytic cell probabilities and remain narrow-line safe relative to the
  measured resolution;
- direct discrete convolution agrees tightly with the power-of-two FFT path,
  output coordinates begin at the sum of input origins, and padding prevents
  circular signal at the opposite boundary;
- the discrete calculation applies exactly one `h` factor and changing
  numerical refinement does not systematically rescale area or peak;
- nonuniform measured resolution is linearly represented with zero outside
  support and a validated representation-only unit-area correction;
- different sample/resolution grid lengths and spacings, asymmetric sample
  intervals, asymmetric resolution support, and nonzero resolution peaks
  behave without hidden recentering;
- the derived model domain covers the original sample targets and only the
  convolved model is linearly evaluated back on those coordinates;
- a fixed plan supports sub-cell `E0/h` phases `0`, `0.1`, `0.25`, `0.5`,
  `0.75`, and `0.9` with stable centroid, peak shape, FWHM, and area;
- dimensionally equivalent problems remain stable from approximately `1e-3`
  to `1` meV, and explicitly converted `25 µeV` and `0.025 meV` inputs agree;
- repeated evaluation is deterministic and unusable/noncanonical coordinates
  fail without sorting, repair, extrapolation, or unit guessing;
- boundary and padded-tail detection does not modify originals; and
- prepared measured-resolution input retains exact ordered Q association.

Scale-aware profile/peak/FWHM/centroid/area criteria target about `5e-4` where
scientifically meaningful. Pure FFT/direct and normalization identities use
tighter floating-point tolerances without fragile machine-epsilon equality.

### 3.5 Parameter-recovery tests

For approved synthetic cases, fit enough fixed-seed replicates to test recovery
without asserting that every noisy realization equals truth. Evaluate:

- integrated elastic and Lorentzian areas;
- Lorentzian FWHM;
- shared energy-center shift and manual independent Lorentzian centers;
- constant and linear background parameters;
- one and multiple Lorentzians;
- shared-center numerical backward compatibility and one/multiple independently
  centered Lorentzians, including fixed/free/bounded centers;
- parameter names/counts, covariance/correlation dimensions, nominal DOF, and
  FWHM canonicalization with center state kept on its component;
- exactly nine shared-center standard Auto candidates with unchanged nominal
  parameter counts and no independent-center Auto candidate;
- fixed/free states and active bounds;
- per-Q independence and optional previous-fit seeding; and
- derived tau and EISF.

Acceptance ranges are defined before looking at benchmark outcomes. Tests must
also verify bound-hit, covariance/Jacobian, and scientific-quality warnings.
Parameter recovery is not declared successful solely because the optimizer
reports convergence.

### 3.6 Regression tests

After a scientifically approved baseline exists, retain compact public inputs
and machine-readable expected summaries for:

- parsed structure and metadata;
- processed-resolution ranges, normalization, and grid identity;
- convolved values at selected indices plus whole-array hashes where stable;
- fit parameters and diagnostics with justified tolerances;
- batch traversal/result selection;
- derived values;
- lightweight reproducibility/export records; and
- report/export schemas.

Update a scientific baseline only with documented cause, numerical diff,
reviewer approval, and confirmation that no convention changed inadvertently.

## 4. External and benchmark comparisons

### 4.1 DAVE comparison

Use public synthetic files that DAVE can read where practical. Compare group
boundaries, columns, energy values, intensities, uncertainties, and Q identity.
When DAVE produces additional `ModelFit`/`Func` columns, confirm ezQENS detects
but excludes them from primary imported data.

For comparable spectral configurations, compare model curves, component-area
meaning, linewidth convention, and fit summaries. Differences due to optimizer
or statistical convention must be explained rather than forced to equality.

### 4.2 Existing Python scripts

The scientific owner identifies current trusted scripts and their applicable
domains. Compare:

- preprocessed resolution arrays;
- convolution alignment/normalization;
- fitted parameters and model curves;
- FWHM-to-tau values;
- experimental EISF component bookkeeping.

Freeze script version/hash, inputs, environment, settings, and conventions in a
local validation record. Review discrepancies scientifically; neither
implementation is automatically authoritative.

### 4.3 Private benchmark validation

The confidential benchmark remains under `private_data/` and outside committed
test discovery. A local-only validation command may report:

- sample/resolution group counts;
- rows per group;
- detected/ignored column names;
- energy minima/maxima;
- invalid-value and padded-boundary counts;
- supplied Q mapping summary;
- fit success/warning counts by Q without full arrays.

It must not print full intensities, uncertainties, resolution arrays, or source
content. Outputs go to ignored `private_results/`, are
never uploaded automatically, and are manually checked before sharing.

The known benchmark expectations—14 groups, user Q range 0.46–2.02 `Å^-1`,
and extra DAVE fitting columns—are validation assertions only where the owner
supplies them explicitly. They are not code defaults or public fixtures.

Before the v1.0 release gate, validate representative DAVE- and
Mantid-preprocessed real workflows, including PSI FOCUS and ILL IN5/IN16 where
available. Compare structural and scientific results without embedding
instrument-specific policy in the core.

Milestone 1.2 private validation compares the prior and v2 automatic masks by
source basename, group identity, left/right counts, and retained energy bounds.
Any difference requires an explicit structural explanation and scientific-owner
review; no private arrays or absolute paths appear in the comparison.

## 5. Resolution validation

Create independent synthetic resolution inputs with valid central peaks,
nonfinite values, constant offsets, nonuniform grids, and repeated boundary
padding. Validate:

- exact sample/resolution Q count, order, representative-value, and known-edge
  matching within the fixed small tolerance;
- clear count/value/edge failure with no reordering, nearest-Q matching, or
  repair;
- independent sample and resolution padding results;
- matching retained boundaries and warning-only mismatch diagnostics without
  mask mutation;
- resolution `AUTO` exclusion from normalization and source-grid kernel
  contribution by default, independently of support;
- explicit per-group AUTO disabling restores otherwise valid points while a
  support-only change does not alter AUTO application;
- `REVIEW` retention by default;
- explicit support overrides distinct from sample fitting ranges;
- preview of original/accepted support and raw accepted values before use;
- blocking missing or unconfirmed per-Q acceptance decisions;
- confirmed KEEP equivalence to the frozen M3 normalization and downstream
  convolution/fitting path;
- confirmed low-, high-, and both-boundary contiguous exclusion before
  normalization, including unit area, identically scaled uncertainty, and zero
  source-grid contribution outside support;
- rejection of unchanged/out-of-bounds/insufficient exclusion support and any
  internal invalid hole without an arbitrary-mask escape hatch;
- independent decisions, supports, warnings, and signed-area ratios by Q,
  including a ratio above 1 after excluding a negative outer tail without any
  automatic warning, block, clamp, or policy;
- neutral retained-structure warning provenance without hidden correction;
- unit-area trapezoidal normalization on uniform and nonuniform measured grids;
- integrated-area rather than peak-height normalization;
- explicit failure on nonfinite, zero, and negative areas;
- finite, strictly increasing, unique accepted coordinates and enough points;
- blocking internal invalid energy/intensity holes without interpolation or a
  trapezoidal bridge, while support may exclude invalid boundary regions;
- original arrays and masks unchanged and referenced by prepared state;
- accepted energy coordinates unchanged, demonstrating no M3 interpolation;
- original uncertainty retained and normalized by the deterministic factor,
  with invalid uncertainty never replaced by zero; and
- inspection plots consuming the authoritative prepared state.

No fixed padding value is approved. The `edge-padding-v2.0.0` detector uses the
small explicit rule in `scientific_conventions.md`; changes require versioned
tests and scientific-owner review. M3 performs no baseline treatment,
recentring, or interpolation. Milestone 4 validates the common numerical grid,
interpolation, and convolution boundary behavior separately. Normalization-area
covariance propagation and analytic resolution sources remain deferred.

## 6. Fit diagnostics validation

Use hand-calculable residual cases to check raw/standardized residuals,
chi-square, degrees of freedom, and reduced chi-square. Verify that invalid
sigma is masked without modifying originals and that fitting fails when
`n_fitted_points - n_free_parameters <= 0`.

Test the approved absolute-sigma, unscaled `(J.T @ J)^-1` covariance against an
independent hand calculation. Test AIC/AICc/BIC under the declared
Gaussian-absolute-sigma convention, including small-sample/undefined AICc
cases. Cross-fit comparisons must reject differing data points, masks/Q
selection, uncertainty treatment, residual definitions, or likelihood
conventions.

Create Jacobians/covariances that are full rank, rank deficient, ill
conditioned, and unavailable. Verify standard-error status and warnings.
Explicitly test exact/near bound hits, insufficient points, invalid sigma,
optimizer failure, cancellation, and scientifically implausible parameters.

## 7. Batch and manual-review validation

Test low-to-high and high-to-low traversal, with and without previous-fit
seeding. Demonstrate that modifying one Q result does not mutate another.
Failures must not discard successful neighbors. Manual refits create new linked
results and selection history.

Test inclusion in spectral fitting, exclusion from spectral fitting, separate
inclusion in scientifically validated derived quantities, manual refitting, and
restoration after exclusion.

## 8. Later dynamics-model validation — provisional

The owner may schedule a concrete temporal or spatial dynamics model before or
after v1.0. Its numerical validation is blocked until the scientific owner
approves its equations, parameter definitions, applicability, diagnostics, and
independent reference cases. Tests must cover analytic limits, recovery,
parameter validity, uncertainty, residual structure, and cautious
interpretation as appropriate to that model.

Automatic coordinate/symmetry-driven candidate inference is a substantially
later capability. If that specific workflow is promoted, C2, C4, and isotropic
implementations still require approved equations and parameterizations.

For that later structure-driven workflow:

- test analytic limits and symmetry cases;
- compare predictions against independent hand/reference calculations;
- test x/y/z axes and fixed `(0, 0, 0)` center;
- retain all atoms while selecting only hydrogens for calculation;
- recover parameters from independent synthetic EISF;
- use identical experimental points, masks/Q selection, uncertainty treatment,
  residual definition, and approved likelihood convention across candidates;
- test invalid parameter and weak-discrimination warnings; and
- verify comparison language does not claim a unique mechanism.

## 9. Lightweight reproducibility and future persistence

Before v1.0 release, verify that exported reproducibility information identifies
inputs, selections and masks, model and fit settings, results, warnings, and
software version without requiring a complex container.

The remaining validation is provisional post-v1.0 and begins only if a
project-container design is approved. Its format and extension are unresolved;
the former `.qensfit` proposal is not a contract.

Round-trip tests compare scientific meaning, units, exact masks, IDs,
provenance, configurations, diagnostics, warnings, exclusions, and array hashes
before and after reload. Include incomplete/failed fits and manual-refit
history.

Security tests reject traversal paths, links, duplicate entries, unsupported
versions, oversized archives, altered hashes, object/pickle arrays, invalid
JSON, dangling references, and inconsistent units. Simulate interrupted writes
to ensure the prior project is not corrupted.

Each supported schema migration has immutable before/after fixtures and tests
for idempotent current-version loading. Source files are never migrated in
place.

## 10. Cross-platform and packaging matrix

Continuous integration should eventually cover:

- supported CPython 3.12 and 3.13;
- supported macOS laptops;
- supported 64-bit Windows laptops before public release;
- best-effort Linux when it adds little complexity and does not delay release;
  and
- path, line-ending, locale, Unicode, archive, multiprocessing, and floating
  point behavior relevant to each platform.

Core tests must run headlessly. Platform-specific numerical differences use
reviewed tolerances, not blanket skips. Dependency wheels and packaging are
verified on clean systems.

## 11. GUI smoke and workflow tests

GUI tests begin only after core scientific gates pass. Headless/offscreen smoke
tests cover application startup, each planned screen, lightweight analysis
restore where implemented, validation-error presentation, task progress,
cancellation, and results display.

A small number of end-to-end synthetic workflows cover import through export.
GUI tests verify delegation to the core; they do not duplicate formulas in UI
assertions. Manual usability review covers readability, units, warnings,
keyboard navigation, high-DPI scaling, and Windows/macOS behavior.

The v1.0 usability gate includes an external real user, including a non-QENS
expert, completing a guided reduced-data analysis on an ordinary macOS or
Windows laptop, inspecting warnings/results, and exporting the analysis.

## 12. Acceptance gates

### Gate A: import

Content-based detection and strict explicit override; DAVE group-block,
wide-table, and single-spectrum parsing; explicit sample/resolution roles;
preserved shared or
per-spectrum grids; structural diagnostics; privacy-safe summaries; and correct
group identity. Milestone 2 adds count-aligned dataset-level Q bins, the approved
DAVE parameter semantics, per-group fitting ranges, derived point selection,
and scientific inspection plots. Generic custom mapping remains deferred.

### Gate B: resolution and convolution

Milestone 3 first gates exact Q association, independent resolution support and
padding, original preservation, measured-grid validation, positive finite
trapezoidal area, unit normalization, and inspectable diagnostics. The
corrective acceptance gate additionally requires an explicit confirmed per-Q
KEEP or contiguous-EXCLUDE decision before normalization and records neutral
warning and signed-area-ratio provenance. Milestone 4
then separately gates independent direct/FFT agreement, interpolation policy,
area preservation, and no wrap-around.

### Gate C: single-spectrum science

Approved component semantics, weighted residuals, parameter recovery,
diagnostics, warning behavior, and independent benchmark/script comparison.
Direct elastic-resolution evaluation and convolved integrated-area preservation
must pass. Dense narrow/sub-bin observable tests must cover multiple energy-
shift phases and coarse/irregular sample grids. Candidate evidence must remain
separate from the unresolved Auto recommendation rule.

### Gate D: batch and derived quantities

Independent sequential results, manual-refit history, FWHM/tau/EISF convention
tests, and exclusion-scope tests.

### Gate E: ezQENS v1.0 public release

Gates A–D plus guided GUI delegation to the core, useful export, lightweight
reproducibility, user/scientific documentation, representative external-user
validation, and supported macOS/Windows packages.

### Gate F: later capabilities — provisional

Any selected dynamics analysis requires approved equations and independent
references and gates v1.0 only if explicitly promoted into release scope. Any
later project container requires an approved format plus safe
round-trip, migration where necessary, privacy, integrity, and security tests.

## 13. Assumptions and explicit non-goals

This plan assumes the owner can run private DAVE/script comparisons locally and
review numerical tolerances. It does not assume private files can enter CI or
that one optimizer/reference implementation is infallible.

Version 1.0 does not validate raw reduction, detector operations, INS,
Bayesian/global fits, Arrhenius analysis, molecular interpretation,
candidate-motion comparison, or GPU acceleration.

## 14. Unresolved decisions, risks, and milestone dependencies

Unresolved items include detailed general parser grammar, explicit-list comment
policy, future edge-padding refinements, convolution grid
details, fit-quality thresholds, Auto information-criterion weighting/decision
rules, and supported Windows versions. Post-v1.0 motion equations and
persistence format also remain unresolved. Q-bin edge/midpoint and DAVE
four-value semantics, invalid-sigma handling, and padding-v2 are approved.

Risks include circular tests, overfitting tolerances to one benchmark,
platform-specific false failures, leaking private data through artifacts, and
treating convergence as validity. Independent references, predeclared
tolerances, privacy checks, and staged gates mitigate these.

Detector/table-import tests precede Q mapping and resolution association.
Convolution tests precede
parameter recovery. Single-spectrum recovery precedes batch tests. Covariance
validation precedes EISF uncertainty. Core gates A–C precede GUI implementation;
gates A–D plus GUI/export/reproducibility/packaging validation precede the v1.0
release gate. Approved equations precede any post-v1.0 motion tests.
