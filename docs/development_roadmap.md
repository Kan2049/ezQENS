# Development roadmap

## 1. Scope and sequencing rule

This roadmap incrementally delivers version 1 while placing scientific
validation before interface breadth. Each milestone produces typed,
GUI-independent core behavior, synthetic tests, documentation updates, and
explicit diagnostics.

The first implementation milestone after this documentation is the
content-based detector plus DAVE group-block, wide-table, and single-spectrum
import path with synthetic tests. GUI implementation cannot start before the
core importer, resolution processing, convolution, and single-spectrum fitting
pass their scientific validation gates.

Milestones 1–6 use only minimal typed in-memory models. They do not implement
the final `.qensfit` ZIP64 container, UUIDs for every temporary object,
entity-level migrations, hash-addressed array registries, append-only project
history, attachments, complete security/migration infrastructure, or atomic
project saves. Simple identifiers and immutable values may improve
traceability, but persistence cannot delay scientific validation. Milestone 8
reviews and implements the complete persistence contract.

## 2. Milestone 0 — requirements and architecture baseline

**Deliverables**

- Product, scientific, architecture, data-model, validation, and roadmap
  documents.
- Project instructions and privacy boundaries.
- Initial package/tooling skeleton.
- Register of unresolved scientific decisions.

**Acceptance**

- Required workflow and non-goals are traceable.
- FWHM, relaxation-time, EISF, fitting, batch, masking, and privacy conventions
  are explicit.
- C2/C4 formulas remain marked unresolved.
- No production scientific or GUI code is introduced.

**Dependencies:** none.

## 3. Milestone 1 — core reduced-data detection and import

This is the first implementation milestone.

**Deliverables**

- Content-based format-detector interface returning proposal, confidence,
  evidence, counts, required/extra columns, warnings, and alternatives.
- DAVE group-block importer.
- Wide QENS `x/yN/yerrN` table importer.
- Single-spectrum `x/y/yerr` table importer.
- Minimal shared scientific spectrum representation with explicit sample or
  resolution role and support for shared or per-spectrum energy grids.
- Recognition of DAVE fitting columns such as `ModelFit`, `Func1`, and `Func2`
  as ignored source metadata rather than measured data.
- Structural summaries that do not print full arrays.
- Independently generated synthetic fixtures and malformed-input tests.
- Confirmation state for automatically proposed formats.

**Acceptance**

- Group-block, wide, and single-spectrum fixtures are correctly detected and
  imported with order, grids, row counts, arrays, units, role, columns, and
  diagnostics intact.
- Wide shared-grid storage still exposes independent spectra; unequal
  group-block grids are not interpolated.
- Invalid sigma is masked without modifying original arrays.
- Missing/ambiguous content fails clearly without partial silent import.
- File extension alone cannot determine classification.
- Public tests contain no private benchmark values/files.
- Validation-plan Gate A passes for the milestone-1 subset.

Generic arbitrary column mapping may wait for milestone 2, but milestone 1
interfaces must leave room for it. DAVE Q-bin parameter semantics are not
implemented unless already approved.

**Dependencies:** milestone 0 and scientific-owner approval of the initial
DAVE group-block grammar.

## 4. Milestone 2 — generic ASCII, Q mapping, and analysis masks

**Deliverables**

- Configurable ASCII/TXT/DAT/CSV column mapping.
- Inclusive linear Q range using `numpy.linspace` semantics.
- Manual ordered Q list with comma/newline input contract.
- Explicit Q-list file with approved blank/comment policy.
- DAVE Q-bin parameter-file parser only after field semantics are approved.
- Imported-metadata mapping where explicitly supported.
- Resolved explicit-list preview, confirmation, and exact sample/applicable
  resolution group-count diagnostics.
- Distinct invalid-point, fitting-range, manual-point, spectral-Q-exclusion,
  and motion-Q-exclusion concepts.
- Inspection-ready, GUI-neutral summaries/view models.

**Acceptance**

- Ambiguous generic inputs require explicit mapping.
- Linear mapping uses finite inclusive endpoints, `N >= 2`, and one value per
  sample group.
- Manual/imported order is preserved without sort, deduplication, padding,
  truncation, extrapolation, discard, or combination.
- Every mapping is previewed/confirmed and every mask/exclusion has distinct
  scope and minimal traceability.
- Count mismatches report mapping mode and sample/Q/applicable-resolution
  counts.
- Possible contamination annotations never delete data.

**Dependencies:** milestone 1; approval of explicit-list comment syntax and
every DAVE Q-bin field supported by its parser. Linear endpoints and
invalid-uncertainty policy are already approved.

## 5. Milestone 3 — measured-resolution import and processing

**Deliverables**

- Resolution datasets from vanadium or low-temperature samples using the
  validated ASCII path.
- Original/processed separation.
- Invalid/NaN/infinity handling.
- Mandatory, authoritative manual valid-range selection.
- Optional explicit constant baseline, validated unit-area normalization, grid
  validation, and uniform-grid interpolation.
- Exact group and explicitly selected nearest-Q association.
- Minimal traceable processing decisions.
- Optional advisory repeated-padding detection/range suggestion, never
  automatically applied.

**Acceptance**

- Original arrays remain unchanged.
- Original and manually selected ranges are inspectable.
- Every selected processed array is reproducible from recorded decisions.
- Invalid normalization/grid/range states fail with actionable diagnostics.
- Normalization fails without a finite positive integral.
- Advisory padding/range flags do not mutate arrays or select a range.
- Synthetic resolution tests and relevant Gate B prerequisites pass.

**Dependencies:** milestones 1–2; approval of baseline, normalization,
interpolation, and association policies. Automatic suggestion algorithms are
optional and do not block the mandatory manual path.

## 6. Milestone 4 — numerical convolution

**Deliverables**

- GUI-independent linear numerical convolution on validated grids.
- Direct elastic evaluation as `A_elastic * R_Q(E - E0)`, without a discrete
  grid delta.
- Unit-area Lorentzian convolution preserving integrated-area semantics within
  reviewed finite-grid tolerance.
- Explicit centering, interpolation, padding, and crop configuration.
- Direct reference implementation for tests and optimized path as justified.
- Protection against circular FFT wrap-around.

**Acceptance**

- Elastic-area, no-discrete-delta, Lorentzian-area, symmetry,
  direct-versus-FFT, odd/even, boundary, normalization, and no-wrap tests pass
  with reviewed tolerances.
- No implicit grid or FWHM/HWHM conversion occurs.
- Validation-plan Gate B passes.

**Dependencies:** milestone 3 and approval of convolution-grid conventions.

## 7. Milestone 5 — validated single-spectrum fitting

**Deliverables**

- Elastic, one/multiple Lorentzian, constant/linear background definitions
  using integrated areas and FWHM.
- Parameter initial values, bounds, and fixed/free state.
- `scipy.optimize.least_squares` adapter with weighted standardized residuals.
- Optimized values, residual arrays, chi-square/reduced chi-square,
  covariance/standard errors where estimable, convergence, bound,
  Jacobian/covariance, and scientific warnings.
- AIC/AICc after their likelihood convention is approved; their absence does
  not block the initial prototype.
- Synthetic parameter-recovery and independent script/DAVE comparisons.

**Acceptance**

- One selected Q can be configured, fit, reviewed, and refit through a callable
  core API.
- Recovery and failure cases distinguish convergence from scientific quality.
- Required diagnostics survive typed in-memory domain conversion.
- Insufficient valid points fail clearly and at least prevent nonpositive
  statistical degrees of freedom.
- Validation-plan Gate C passes without requiring unresolved AIC/AICc.

**Dependencies:** milestone 4; approved invalid-sigma policy (already fixed)
and approval of covariance and quality-warning policies. AIC/AICc approval may
follow without blocking the prototype.

## 8. Milestone 6 — sequential batch fitting and derived QENS quantities

**Deliverables**

- Low-to-high and optional high-to-low independent per-Q fits.
- Optional previous-successful-fit seeding.
- Per-Q failure, warning, exclusion, manual-refit, and selected-result
  traceability in memory.
- FWHM linewidth tables, relaxation time, experimental EISF from integrated
  component areas, and component retention.

**Acceptance**

- No parameter is globally shared or simultaneously optimized across Q.
- A failed or excluded Q does not erase or invalidate unrelated results.
- Manual refits are new linked results.
- FWHM/tau/EISF and exclusion convention tests pass.
- Validation-plan Gate D passes, except deferred covariance-propagated EISF
  uncertainty if its policy is not yet approved.

**Dependencies:** milestone 5; approved derived-uncertainty policy for any
uncertainties claimed.

## 9. Milestone 7 — XYZ geometry and candidate motion models

**Deliverables**

- XYZ import retaining all atoms.
- Hydrogen selection, coordinates/distances, fixed origin, and x/y/z axis.
- Stable motion-model interface and independent candidate-fit orchestration.
- Approved C2, C4, and isotropic implementations only after scientific-owner
  equation/parameter review.
- Weighted comparison using reduced chi-square, AICc, uncertainty, residual
  structure, validity, and warnings.

**Acceptance**

- No combined candidate populations.
- Candidate comparisons use identical included experimental points and
  compatible statistics.
- Report wording distinguishes best support from proof of mechanism.
- Independent reference calculations and validation-plan Gate E pass.

**Dependencies:** milestone 6, an approved AIC/AICc likelihood convention for
candidate comparison, and a hard scientific gate approving each model's
equation, parameters, limits, and reference cases. Interface work may precede
formula approval; executable model claims may not.

## 10. Milestone 8 — reproducible project persistence and exports

This milestone reviews the long-term proposal against the now-validated
scientific contracts, finalizes it, and implements persistence for the first
time. Earlier milestones may inform the design but do not build partial archive
or migration infrastructure.

**Deliverables**

- Reviewed/finalized data-only `.qensfit` project container and public file
  contract.
- Archive layout, attachment policy, hashes, atomic save, safe load, and
  schema/entity versioning.
- Explicit migrations and round-trip fixtures.
- Project/reference source data, mappings, units, histories, masks, resolution
  settings, fit configurations/results, exclusions, EISF, motion results,
  software version, and schema version.
- Machine-readable tables and scientific-review exports with units, FWHM
  labels, warnings, and provenance.

**Acceptance**

- Complete synthetic project round trips without scientific-semantic loss.
- Corrupt, malicious, oversized, and unsupported containers fail safely.
- Export records identify contents and conventions.
- Private paths/content obey redaction and no-upload rules.

**Dependencies:** stable schemas from milestones 1–7 and approval of the
container/redaction policy.

## 11. Milestone 9 — PySide6 desktop workflow

PySide6 is introduced here, not earlier.

**Deliverables**

- Import, Spectrum Fit, Batch Results, Motion Models, and Export/Project Summary
  screens.
- Thin controllers over application/core services.
- Background long-running operations with progress and cancellation.
- Clear units, masks, warnings, diagnostics, and manual-refit interactions.
- macOS Apple Silicon packaging prototype.

**Acceptance**

- GUI imports no formulas or optimizer implementation.
- Core tests run without PySide6.
- A synthetic workflow completes import through save/export.
- Cancellation leaves a valid project state.
- Headless smoke and manual usability checks pass.

**Dependencies:** milestones 1–5 scientifically validated; in practice
milestones 6–8 provide the full five-screen version-1 workflow.

## 12. Milestone 10 — Windows and public-release readiness

**Deliverables**

- Supported Windows build, installer, and clean-machine verification.
- Cross-platform CI for supported Python/platform combinations.
- Performance, accessibility, security, privacy, licensing, documentation, and
  release review.
- Public synthetic tutorials/examples only.

**Acceptance**

- Core and GUI validation gates pass on macOS Apple Silicon and supported
  Windows.
- Project files interoperate across platforms.
- No private data or identifying artifacts enter packages, logs, docs, tests,
  or release assets.
- Known scientific limitations and unresolved items are user-visible.

**Dependencies:** milestones 1–9 and product decisions on licensing, installer,
support policy, and release criteria.

## 13. Cross-cutting work

At every milestone:

- update typed contracts and documentation before changing scientific meaning;
- add unit/type validation and synthetic tests;
- preserve milestone-appropriate traceability and structured diagnostics;
- keep logs privacy-safe;
- run pytest, Ruff, and mypy;
- assess Windows portability even before Windows packaging; and
- record scientific-owner approvals for conventions and baselines.

## 14. Assumptions and explicit non-goals

The roadmap assumes incremental owner review and access to private comparison
workflows locally. Dates and staffing are intentionally not estimated.

No milestone adds raw reduction, INS/fixed-window analysis, Bayesian/MCMC or
global fits, Arrhenius analysis, arbitrary model-code execution, web/multi-user
features, or automatic publication figure editing.

## 15. Unresolved decisions and risks

Scientific gates remain for detailed DAVE grammar, DAVE Q-bin field semantics,
explicit-list comment syntax, resolution/convolution numerical policy,
covariance/AIC conventions, Bragg heuristics, fit-quality thresholds, and all
candidate-motion equations and parameterizations. Inclusive linear Q mapping
and invalid-sigma handling are approved.

Product/engineering decisions remain for the project container, recovery,
export formats, GUI worker model, macOS/Windows packaging, licensing, and
support matrix.

The critical-path risk is discovering a scientific convention problem after UI
or persistence depends on it. The sequence mitigates that risk by validating
import, resolution, convolution, and one-spectrum fitting before batch,
derived, motion, or GUI expansion.
