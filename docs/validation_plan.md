# Validation plan

## 1. Purpose and scope

qensfit requires both software verification and scientific validation. Passing
tests demonstrates behavior against approved references; it does not by itself
establish scientific correctness. Scientific formulas and conventions must
never be altered merely to make tests pass.

This plan covers reduced-data import through project persistence and the later
GUI. Raw detector reduction and all other version-1 non-goals are excluded.

## 2. Validation principles

- Use independently generated synthetic public fixtures.
- Keep private benchmark data local, ignored, summarized, and absent from test
  output/artifacts.
- Test scientific invariants at module boundaries and after serialization.
- Compare against at least one independent implementation or analytic case for
  high-risk numerical operations.
- Separate optimizer convergence from statistical and scientific validity.
- Record tolerances with physical/numerical justification.
- Make stochastic tests deterministic through explicit seeds; avoid dependence
  on random luck.
- Run platform-independent core tests on macOS and Windows.
- Treat warnings and failure behavior as testable outputs.

## 3. Test layers

### 3.1 Unit tests

Early unit tests cover validators, units, Q mappings, masks, spectrum roles,
parameter bounds, minimal processing traceability, formula constants, and
result diagnostics. Archive safety and migration functions are added with
milestone 8; they are not milestones 1–6 prerequisites. Tests are fast,
deterministic, and do not read private data.

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
- batch exclusions and motion-fit exclusions remain distinct; and
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

- Inclusive linear-range mapping agrees with `numpy.linspace`.
- Finite endpoint and `N >= 2` validation.
- Manual-list order preservation for comma/newline input.
- Explicit-list file parsing under the approved blank/comment policy.
- Sample-count mismatch and applicable resolution-count mismatch.
- No silent sort, deduplication, truncation, padding, extrapolation, discard,
  or combination.
- Preview, resolved-list display, and confirmation state.
- DAVE Q-bin parsing only after its field semantics are approved; until then,
  production parsing is rejected with an actionable unresolved diagnostic.

#### 3.2.5 Edge-padding detection and custom input

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
- XYZ files compatible with common VESTA exports retain all atoms.
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

- delta-like kernel reproduces the input within justified tolerance;
- unit-area measured resolution multiplied directly by `A_elastic` integrates
  to the elastic area within tolerance;
- elastic evaluation does not construct or depend on a discrete numerical
  delta spike;
- a unit-area Lorentzian multiplied by `A_i` retains its integrated area after
  measured-resolution convolution within reviewed finite-grid tolerance;
- constant/known signals have expected linear-convolution behavior;
- symmetric kernels preserve approved centering;
- direct discrete convolution agrees with the FFT path;
- padding/cropping prevents signal appearing at the opposite boundary;
- nonuniform grids are rejected or explicitly interpolated;
- kernel unit-area normalization is verified numerically;
- different sample/resolution grid lengths and spacings behave as specified;
- boundary and padded-tail detection does not modify originals; and
- exact versus explicitly selected nearest-Q association is traceable.

Tests cover odd/even lengths and energy-center offsets. Tolerances are expressed
in absolute/relative terms and tied to grid spacing and floating-point error.

### 3.5 Parameter-recovery tests

For approved synthetic cases, fit enough fixed-seed replicates to test recovery
without asserting that every noisy realization equals truth. Evaluate:

- integrated elastic and Lorentzian areas;
- Lorentzian FWHM;
- energy-center shift;
- constant and linear background parameters;
- one and multiple Lorentzians;
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
- project save/reload; and
- report/export schemas.

Update a scientific baseline only with documented cause, numerical diff,
reviewer approval, and confirmation that no convention changed inadvertently.

## 4. External and benchmark comparisons

### 4.1 DAVE comparison

Use public synthetic files that DAVE can read where practical. Compare group
boundaries, columns, energy values, intensities, uncertainties, and Q identity.
When DAVE produces additional `ModelFit`/`Func` columns, confirm qensfit detects
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
- experimental EISF component bookkeeping; and
- approved motion-model predictions/fits.

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
- fit success/warning counts by Q without full arrays; and
- aggregate candidate-model metrics without molecular coordinate dumps.

It must not print full intensities, uncertainties, resolution arrays, molecular
coordinates, or source content. Outputs go to ignored `private_results/`, are
never uploaded automatically, and are manually checked before sharing.

The known benchmark expectations—14 groups, user Q range 0.46–2.02 `Å^-1`,
extra DAVE fitting columns, and separate C2/C4/isotropic comparison—are
validation assertions only where the owner supplies them explicitly. They are
not code defaults or public fixtures.

Milestone 1.2 private validation compares the prior and v2 automatic masks by
source basename, group identity, left/right counts, and retained energy bounds.
Any difference requires an explicit structural explanation and scientific-owner
review; no private arrays or absolute paths appear in the comparison.

## 5. Resolution validation

Create independent synthetic resolution inputs with valid central peaks,
nonfinite values, constant offsets, nonuniform grids, and repeated boundary
padding. Validate:

- detection versus mutation separation;
- `AUTO` padding is reversible and exactly point-masked;
- `REVIEW` padding remains decision-gated;
- mandatory manual valid-range selection and its authoritative use;
- original and selected ranges are both inspectable;
- optional suggested-range rationale and explicit confirmation;
- optional baseline correction;
- unit-area normalization and failure on invalid area;
- uniform-grid interpolation under the approved method;
- original/processed separation;
- exact and user-selected nearest-Q association; and
- full processing audit history.

No fixed padding value is approved. The `edge-padding-v2.0.0` detector uses the
small explicit rule in `scientific_conventions.md`; changes require versioned
tests and scientific-owner review. Interpolation algorithms remain unresolved.

## 6. Fit diagnostics validation

Use hand-calculable residual cases to check raw/standardized residuals,
chi-square, degrees of freedom, and reduced chi-square. Verify that invalid
sigma is masked without modifying originals and that fitting fails when
`n_fitted_points - n_free_parameters <= 0`.

The first single-spectrum gate does not require AIC/AICc. Once their convention
is approved, test AIC/AICc against an independent hand calculation, including
small-sample and undefined cases. Cross-fit comparisons must reject differing
data points, masks/Q selection, uncertainty treatment, residual definitions,
or likelihood conventions.

Create Jacobians/covariances that are full rank, rank deficient, ill
conditioned, and unavailable. Verify standard-error status and warnings.
Explicitly test exact/near bound hits, insufficient points, invalid sigma,
optimizer failure, cancellation, and scientifically implausible parameters.

## 7. Batch and manual-review validation

Test low-to-high and high-to-low traversal, with and without previous-fit
seeding. Demonstrate that modifying one Q result does not mutate another.
Failures must not discard successful neighbors. Manual refits create new linked
results and selection history.

Test the full matrix of:

- included in both spectral and motion fitting;
- excluded from spectral fitting;
- successfully fit but excluded from motion fitting; and
- restored after exclusion.

## 8. Motion-model validation

Motion-model numerical validation is blocked until the scientific owner
provides and approves authoritative equations and parameter definitions,
explicitly including C2 and C4. The isotropic implementation also needs an
approved equation/parameterization.

After approval:

- test analytic limits and symmetry cases;
- compare predictions against independent hand/reference calculations;
- test x/y/z axes and fixed `(0, 0, 0)` center;
- retain all atoms while selecting only hydrogens for calculation;
- recover parameters from independent synthetic EISF;
- use identical experimental points, masks/Q selection, uncertainty treatment,
  residual definition, and approved likelihood convention across candidates;
- test invalid parameter and weak-discrimination warnings; and
- verify comparison language does not claim a unique mechanism.

## 9. Project save/reload and migration

This validation begins with milestone 8 and does not gate milestones 1–6.

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
- macOS Apple Silicon as the initial primary platform;
- supported 64-bit Windows before public release; and
- path, line-ending, locale, Unicode, archive, multiprocessing, and floating
  point behavior relevant to each platform.

Core tests must run headlessly. Platform-specific numerical differences use
reviewed tolerances, not blanket skips. Dependency wheels and packaging are
verified on clean systems.

## 11. GUI smoke and workflow tests

GUI tests begin only after core scientific gates pass. Headless/offscreen smoke
tests cover application startup, each of the five screens, safe project
open/save, validation-error presentation, task progress, cancellation, and
results display.

A small number of end-to-end synthetic workflows cover import through export.
GUI tests verify delegation to the core; they do not duplicate formulas in UI
assertions. Manual usability review covers readability, units, warnings,
keyboard navigation, high-DPI scaling, and Windows/macOS behavior.

## 12. Acceptance gates

### Gate A: import

Content-based detection and strict explicit override; DAVE group-block,
wide-table, and single-spectrum parsing; explicit sample/resolution roles;
preserved shared or
per-spectrum grids; structural diagnostics; privacy-safe summaries; and correct
group identity. Generic custom mapping and the complete Q-mapping suite finish
in milestone 2. DAVE Q-bin parsing remains gated on approved semantics.

### Gate B: resolution and convolution

Approved processing policies; independent direct/FFT agreement; no wrap-around;
original preservation; auditable group association.

### Gate C: single-spectrum science

Approved component semantics, weighted residuals, parameter recovery,
diagnostics, warning behavior, and independent benchmark/script comparison.
Direct elastic-resolution evaluation and convolved integrated-area preservation
must pass. AIC/AICc are not required until their likelihood convention is
approved.

### Gate D: batch and derived quantities

Independent sequential results, manual-refit history, FWHM/tau/EISF convention
tests, and exclusion-scope tests.

### Gate E: motion comparison

Scientifically approved equations, independent references, candidate-fit
recovery, comparable metrics, and cautious interpretation.

### Gate F: persistence and GUI

After milestone 8, safe round trips/migrations and privacy/export review; after
GUI implementation, cross-platform core and GUI smoke/workflow passes.

## 13. Assumptions and explicit non-goals

This plan assumes the owner can run private DAVE/script comparisons locally and
review numerical tolerances. It does not assume private files can enter CI or
that one optimizer/reference implementation is infallible.

It does not validate raw reduction, detector operations, INS, Bayesian/global
fits, Arrhenius analysis, or non-version-1 motion features.

## 14. Unresolved decisions, risks, and milestone dependencies

Unresolved items include detailed parser grammar, DAVE Q-bin field semantics,
explicit-list comment policy, future edge-padding refinements, convolution grid
details, fit-quality thresholds, AIC/AICc/covariance conventions, approved
motion equations, and supported Windows versions. Inclusive linear Q endpoints,
invalid-sigma handling, and the versioned padding-v2 behavior are approved.

Risks include circular tests, overfitting tolerances to one benchmark,
platform-specific false failures, leaking private data through artifacts, and
treating convergence as validity. Independent references, predeclared
tolerances, privacy checks, and staged gates mitigate these.

Detector/table-import tests precede Q mapping and resolution association.
Convolution tests precede
parameter recovery. Single-spectrum recovery precedes batch tests. Covariance
validation precedes EISF uncertainty. Approved equations precede motion tests.
Stable scientific contracts precede milestone-8 persistence fixtures. Core
gates A–C precede GUI implementation.
