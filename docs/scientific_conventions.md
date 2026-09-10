# Scientific conventions

## 1. Scope

This document fixes the scientific conventions that are authoritative for
version 1.0. Unspecified details are marked unresolved and must not be inferred
from private data or chosen merely to make tests pass.

Reduced inputs are assumed to have completed detector reduction and
detector-level bad-detector masking. ezQENS performs analysis-level operations
only.

## 2. Axes, units, and identity

- Energy transfer is represented canonically in meV. Physical convolution and
  fitting require an explicitly known/canonicalized unit; an unknown or
  unresolved energy unit is blocking rather than guessed.
- Momentum transfer Q is represented in inverse angstrom (`Å^-1`).
- Molecular Cartesian coordinates and radii are represented in angstrom.
- Relaxation time is represented in picoseconds.
- Intensity and uncertainty units must be retained explicitly from input or
  labeled as unknown/arbitrary; they must never be guessed.
- Every numerical array must retain its unit, source spectrum, Q association,
  processing state, and mask association.

The sign convention for energy transfer must be preserved from the imported
file and displayed. Any future sign transformation requires an explicit,
recorded processing step; no default transformation is defined here.

### 2.1 Spectrum roles and representations

Sample spectra and measured-resolution spectra must be semantically distinct
through an explicit role or validated role-specific wrapper. They may share
energy, intensity, uncertainty, invalid-value masks, source metadata, and group
identity, but their validation and allowed processing differ.

DAVE group blocks with unequal grids and wide tables with a truly shared grid
both map to one ordered per-spectrum scientific interface. Matrix storage is an
optional optimization only for genuinely shared grids. Import never
interpolates, changes row order, or discards values to force a representation.

### 2.2 Q mapping

Q identity is stored once at dataset/group level as `QBins`, never copied onto
each `Spectrum`. It always contains one finite representative `q_value` in
`Å^-1` per ordered spectrum. Explicit Q-value input preserves the supplied
order exactly and does not imply known bin edges.

Explicit ordered edges are the preferred unambiguous Q-bin definition. For
`N` groups, edges are finite, strictly increasing, and satisfy:

```text
len(edges) = N + 1
q_value[i] = (edge[i] + edge[i + 1]) / 2
```

The midpoint is the approved Milestone-2 convention for values generated from
edges, not a permanent assertion about future weighted/effective Q values.
Nonuniform edges are valid.

Uniform manual bins are count-driven, use authoritative inclusive outer edges
and group count, and exactly cover the specified range:

```text
delta_Q = (upper_q_edge - lower_q_edge) / N
edges = lower_q_edge + i * delta_Q, i = 0, ..., N
```

The outer edges are not first/last group centers. Fixed-width construction uses
an authoritative lower edge, requested upper limit, and step. It includes every
complete equal-width bin whose upper edge remains within the requested limit
under the validated local floating-point tolerance. It never creates a partial
final bin, so a remainder smaller than one step remains uncovered and the
requested upper limit may exceed the final actual edge. Construction fails when
float64 cannot represent finite, strictly increasing edges whose represented
widths match the requested step within the same tightly capped local tolerance.

The known four-value DAVE Q-bin parameter format uses this same complete-bin
construction. Its reported group count is retained as source metadata and
compared with the reconstructed count; disagreement produces a warning and does
not replace the step-derived bins.

Q-bin/value count must equal spectrum count. No mismatch is repaired by sorting,
deduplication, truncation, padding, extrapolation, interpolation, or nearest-Q
matching.

Fractional Q rebinning of reduced data requires explicit source Q-bin edges and
per-point fractional coverage `F(Q,E)`. A user-confirmed source that has not
previously been Q-rebinned starts with `F(Q,E) = 1`; imported data do not acquire
that assertion implicitly. Rebin output propagates existing coverage, so `F` is
finite and nonnegative but may exceed one after source bins are combined.

For source Q bin `i`, target Q bin `j`, and aligned energy point `k`, let `g_ij`
be the unexcluded geometric overlap length divided by the full source-bin width.
Mantid-style reduced intensity `Y`, uncertainty `E`, and coverage `F` propagate
as:

```text
S_jk    = sum_i Y_ik F_ik g_ij
V_jk    = sum_i (E_ik F_ik)^2 g_ij
Fnew_jk = sum_i F_ik g_ij
Ynew_jk = S_jk / Fnew_jk
Enew_jk = sqrt(V_jk) / Fnew_jk
```

If any positively weighted contributor has invalid uncertainty, the output
uncertainty at that point remains invalid; its intensity and coverage still
contribute normally. Q exclusions are explicit absolute-Q intervals applied to
overlap geometry, including partial source-bin overlap, and overlapping
intervals are unioned. Target bins must lie inside current source coverage and
must not refine any contributing current source bin. Repeated rebinning uses the
current propagated `F`; it never resets coverage to one. Q rebinning performs no
energy interpolation, Q extrapolation, nearest-Q repair, or source-array
mutation.

## 3. Lorentzian linewidth and relaxation time

The internal and reported Lorentzian linewidth is full width at half maximum
(FWHM), denoted `Gamma`.

For `Gamma_meV`:

```text
tau_ps = 2 * hbar_meV_ps / Gamma_meV
hbar_meV_ps = 0.6582119569
tau_ps = 1.3164239138 / Gamma_meV
```

Every relevant parameter name, table heading, axis label, plot annotation,
export field, and reproducibility field must identify the linewidth as FWHM. No
layer may silently convert FWHM to HWHM. If a numerical routine uses HWHM
internally, that conversion must be local, explicit, tested, and absent from
persisted meaning.

A finite relaxation time requires a finite positive FWHM. Zero, negative, NaN,
or infinite FWHM must produce no valid `tau_ps` and an explicit warning rather
than a misleading number.

## 4. Spectral model semantics

The measured-resolution-convolved model supports:

- an optional elastic component;
- zero or more Lorentzian components, with no software-level count ceiling;
- a constant or linear background.

At least one component is required. Background-only, Lorentzian-only,
elastic-only, and supported combinations are valid; absent elastic is not
represented as a hidden fixed-zero component.

Automatic initialization and recommendation above the currently validated
standard candidate scope are not implied by this arbitrary-N scientific-model
capability.

Manual equality ties use stable typed parameter references: owning function
identity plus AREA, CENTER, FWHM, OFFSET, or SLOPE family. A tie may connect only
members of the same family and owns exactly one `ParameterConfiguration`—value,
bounds, and fixed/free state—so it contributes one optimizer/covariance/DOF
parameter. Valid examples include elastic/Lorentzian Area ties, elastic/Lorentzian
Center ties, and Lorentzian FWHM ties. No expression/formula constraints or
scientific GUI-master state are implied. Legacy stable named `CenterGroup`
center ties, shared `E0`, and per-Lorentzian independent centers remain supported.
Reusing one `ParameterConfiguration` instance across otherwise independent
references never creates an implicit tie. FWHM canonicalization preserves
Lorentzian and tie identity. Production AutoFit
remains elastic-containing, single-shared-center-only, and accepts no general
Manual tie state as standard candidate evidence.

Manual interaction geometry is initial-guess information only. Background
press-release interaction always initializes B1. A nondegenerate gesture creates
the unique line through both supplied points; a horizontal gesture or coincident
click seeds `b1 = 0`. Zero initial slope is neither fixed slope nor B0: explicit
later Manual model/parameter action is required to select B0 or constrain slope.
Click-versus-drag classification belongs to GUI/application interaction state; the
scientific helper uses exact supplied coordinates and no snapping tolerance. No
positivity constraint or physical baseline interpretation is introduced. Elastic component height is converted to integrated area
using the selected unit-area measured resolution. A Lorentzian horizontal-width
hint is approximate observed/convolved FWHM geometry, not an intrinsic linewidth.
A deterministic forward-convolution search
selects a positive intrinsic-FWHM initial seed whose internally represented
convolved main-peak width is a useful approximate match. Discrete, asymmetric, or
structured measured resolution can make this inverse non-unique or discontinuous;
exact reproduction of the interaction hint is neither required nor claimed. The
optimizer-fitted FWHM is the scientific linewidth. No physical deconvolution, INS
classification, or mechanism inference follows from these seeds. User limits
remain distinct from current preview values and core numerical/scientific bounds.

Elastic and Lorentzian amplitude parameters used for EISF are integrated areas,
not peak heights. Component identifiers and individual quasielastic areas must
be retained even when an aggregate is displayed.

Let `R_Q(E)` be the measured resolution after accepted-support selection and
unit-area normalization. Milestone 3 applies no baseline subtraction, clipping,
or energy shift.
The elastic contribution is evaluated directly as:

```text
A_elastic * R_Q(E - E0)
```

This is mathematically equivalent to convolution of
`A_elastic * delta(E - E0)` with `R_Q`, but production fitting must not depend
on constructing a grid-dependent numerical delta spike. Unit-area
normalization makes `A_elastic` the integrated elastic area.

For each quasielastic component:

```text
A_i * [R_Q convolved with L_i](E - E_i)
```

`L_i` is a unit-area normalized Lorentzian, `A_i` is its integrated
quasielastic area, and its linewidth parameter is FWHM. The default true tie is
`E_i = E0`; a manual expert configuration may instead fit or fix `E_i`
independently. Independent centers and their uncertainties are retained as
neutral component parameters so elastic-relative offsets remain inspectable.
They do not by themselves identify INS or prove a microscopic assignment, and
no automatic center-offset threshold is defined. Numerical convolution must
preserve integrated-area meaning within a reviewed, documented finite-grid
tolerance.

## 5. Experimental EISF

For free spectral fitting, the initial definition is:

```text
EISF(Q) = A_elastic(Q)
          / [A_elastic(Q) + sum(A_quasielastic_i(Q))]
```

All amplitudes are integrated component areas. The denominator includes every
quasielastic Lorentzian selected by the spectral model. The result must retain
the source fit, component areas, included component identifiers, Q value,
validity status, and warnings so that a future EISF definition can be evaluated
without discarding information.

EISF uncertainty will later use covariance propagation when the necessary
covariance is available. The exact propagation and fallback policy are not yet
authoritative and are unresolved. Missing covariance must not be presented as
zero uncertainty.

## 6. Default fitting method

The default optimizer is `scipy.optimize.least_squares`. For data value
`data_i`, model value `model_i`, and supplied uncertainty `sigma_i`, the
standardized residual is:

```text
r_i = (model_i - data_i) / sigma_i
```

The default objective is weighted least squares over valid, in-range, unmasked
points. Unweighted least squares and logarithmic relative losses are not
defaults. A later optional robust weighted mode may use soft-L1 or Huber loss,
but its selection and parameters must be explicit in the fit configuration.

A statistical uncertainty is valid only when `sigma` is finite and strictly
greater than zero. NaN, positive/negative infinity, zero, and negative sigma
are automatically flagged and excluded from weighted fitting by an
invalid-data mask. They must not be replaced with one or neighboring values,
made absolute, estimated from intensity, inferred from other groups, or deleted
from original arrays.

Single-Q fitting requires at least two retained sample energy coordinates. It
also requires positive nominal statistical degrees of freedom:
`n_fitted_points - n_free_parameters > 0`. The fitting service fails clearly
when either precondition is not met.

A mature result records:

- optimized values and estimable standard errors;
- raw residuals (`model - data`);
- standardized residuals;
- chi-square and reduced chi-square;
- AIC, AICc, and BIC under the declared absolute-sigma convention;
- fitted-point and free-parameter counts;
- convergence status and optimizer termination information;
- bound-hit warnings;
- invalid covariance or Jacobian warnings; and
- scientific-quality warnings.

Supplied valid `sigma` values are treated as absolute experimental standard
deviations. For a regular identifiable local fit, covariance is
`(J.T @ J)^-1`; it is not multiplied by reduced chi-square. A missing or
rank-deficient covariance produces unavailable standard errors, never zero
errors. Any active fitted parameter bound also makes the ordinary local
covariance/correlation unavailable for the complete fit in Phase A. The
Phase-A information-criterion convention is Gaussian
absolute-sigma likelihood with the data-only constant omitted: `AIC = chi2 +
2k`, with its small-sample correction for AICc, and `BIC = chi2 + k ln(n)`.
These values may be compared only when the data points, masks/Q selection,
uncertainty treatment, residual definition, and likelihood convention match.

AIC/AICc/BIC are supporting candidate evidence. No minimum-information-
criterion result alone establishes an adequate model or a physically resolved
component decomposition.

### 6.1 Production AutoFit recommendation

Production AutoFit recommends the simplest adequately supported **observable**
spectral model. It does not establish microscopic truth, exclude arbitrarily
weak unresolved physics, prove that each fitted Lorentzian is an independent
physical process, or establish that complexity above the searched scope is
absent. Manual arbitrary-N fitting remains separate.

The fixed automatic search scope is 0L/1L/2L × NONE/B0/B1. Candidate fitting
and recommendation are separate operations. The result distinguishes the best
candidate supported by the traversed diagonal-sigma AICc/BIC evidence from the
stricter Most Recommended candidate whose residual adequacy is endorsed. The
best-supported reference is retained even when residual adequacy prevents a
Most Recommended result. The deterministic policy applies,
in order:

    numerical validity
      -> standardized-residual adequacy
      -> family-envelope AICc/BIC evidence
      -> matched-background robustness
      -> component identifiability
      -> recommendation and interpretation information

Every standard candidate uses one common shared-`E0` search interval. Let
`E_R,peak` be the peak coordinate on the accepted measured-resolution grid and
`[E_min, E_max]` the retained fitting window. The physical
alignment interval is `[E_min - E_R,peak, E_max - E_R,peak]`; AutoFit intersects
it with the existing legal fixed-convolution center coverage. An empty or
nonfinite intersection fails explicitly. The provisional elastic seed aligns
the measured-resolution template over this complete legal interval. No absolute
meV cap, zero-centering assumption, profile extrapolation, or convolution-plan
recentering is introduced.

Family-level AICc/BIC envelopes and concrete recommendation eligibility are
distinct. Envelope evidence continues to use the minimum relevant information
criterion across every numerically usable allowed background. Within each
Lorentzian-count family, however, the concrete candidate considered for Most
Recommended is the minimum-AICc candidate, with deterministic background
tie-breaking, among candidates whose residual adequacy is ADEQUATE or
QUESTIONABLE. An INADEQUATE candidate may contribute to descriptive IC
evidence but cannot become Most Recommended or displace an eligible simpler
primary.

The result separately reports (a) support for the selected primary family and
(b) the evidence for moving from that family to the next searched Lorentzian
family. The best-supported candidate is the minimum-AICc family-envelope
candidate in the last adjacent family reached using only clear/strong AICc/BIC
evidence with existing matched-background support. This traversal does not use
residual-dependent transition dispositions or residual adequacy; it retains the
existing deterministic background tie-break. Consequently, a supported primary can coexist with marginal additional
complexity, a background-confounded rejected transition, or statistically
supported but uninterpretable additional structure. A strong alternative and a
comparator are references to existing candidate results rather than duplicated
fits. Numerical unavailability and no adequate model within the current search
scope are distinct outcomes. If no usable baseline fit exists, residual
adequacy is reported as NOT_EVALUABLE rather than INADEQUATE; INADEQUATE is
reserved for an evaluated fit with inadequate residual behavior.

Strong or clear descriptive IC evidence may therefore coexist with a
SUPPORTED_BUT_RESIDUALLY_INADEQUATE higher-family assessment. That state
retains the higher-family evidence and comparator without treating the
transition as a recommendation. SUPPORTED_TRANSITION means that descriptive
evidence supports traversing to the next observable family; it does not prove
that another physical relaxation process exists.

Reaching the 2L automatic boundary records that the fixed Auto search was
exhausted; it does not by itself establish either adequacy or inadequacy. If
the selected 2L candidate remains inadequate, the result has no Most
Recommended candidate, reports NO_ADEQUATE_MODEL, retains the evaluated 2L
comparator, and exposes that user-configured higher-N manual fitting is
available. The same capability remains technically available after a supported
2L result, but availability is not a scientific recommendation to add another
Lorentzian. Any 3L-or-higher result remains a manual result and is not
retroactively part of the 0L–2L Auto recommendation.

For an added-L transition, family-envelope evidence compares the best AICc in
each family and, separately, the best BIC in each family. Evidence is strong
for delta_AICc >= 10 and delta_BIC >= 6, clear for delta_AICc >= 6 and
delta_BIC >= 2, and marginal/conflicting when delta_AICc >= 2,
delta_BIC >= 0, or the signs disagree. The same evidence classification is
evaluated for matched NONE, B0, and B1 backgrounds. If an added Lorentzian
appears supported without background but that evidence disappears with allowed
B0/B1 backgrounds, only that transition is rejected as background-confounded;
an otherwise supported simpler primary is not demoted.

Standardized-residual calibration uses moderate thresholds at RMS > 1.12,
absolute lag-1 correlation > 0.12, same-sign run >= 12, absolute linear trend
> 0.35, and maximum absolute residual > 3.8. Strong thresholds use,
respectively, > 1.35, > 0.25, >= 18, > 0.8, and > 5.0. Lag-1 correlation,
same-sign run, and trend are related views of one serial-structure evidence
family rather than three independent confirmations. Adequacy counting therefore
uses three evidence families: RMS scale, serial structure, and maximum absolute
residual. A fit is inadequate with at least two strong evidence families or RMS
> 1.6; it is questionable with at least one strong family or two moderate
families. With pointwise sigma but no residual covariance/whitening model,
serial structure alone can make a result questionable but cannot impose an
absolute recommendation veto. These values and the information-criterion
boundaries are reviewed M5 policy calibration choices, not universal physical
constants.

Residual diagnostics retain lag-1 correlation, longest same-sign run, and
linear trend individually. The longest run also records its start energy, end
energy, and physical-energy span on the retained measured coordinates. No
central, QENS, or outer-window interpretation is inferred from that location.

Severe component-interpretation limitations are covariance/rank failure,
unavailable free-parameter uncertainty, a materially relevant active bound,
Lorentzian area/standard-error below 1 or unavailable, component-relevant
absolute parameter correlation at least 0.995, no successful multistart result,
or relative Lorentzian-decomposition span at least 0.5 among successful starts
within delta_chi_square <= 2. The multistart comparison canonicalizes components
by increasing FWHM and evaluates Lorentzian integrated areas, total
quasielastic area, and Lorentzian FWHMs. Variation confined to E0, elastic area,
or background parameters does not create a severe decomposition limitation.
Correlation is assessed with parameter identity; high
correlation between unrelated nuisance parameters does not reject a component
decomposition. A finite raw condition number at least 1e8 and adjacent
intrinsic FWHM ratio at most 1.2 are advisory only and never standalone hard
gates.

Warnings are structured advisory results attached to candidate references.
They can report accepted comparisons already present in fit diagnostics,
background sensitivity, background-confounded rejected transitions, and
within-family background ambiguity without becoming confidence percentages.
No warning score or registry exists.

Current fit provenance identifies accepted resolution support, acceptance
origin/authorization,
signed-area diagnostics, and neutral acceptance warnings, but it has no
structured scientific assessment of whether relevant measured-resolution
structure was truncated. AutoFit records this provenance capability gap and
applies no resolution-containment gate. It does not derive one from
signed_area_ratio, a support/FWHM ratio, user confirmation, or free text.

D_unique is not part of production M5 AutoFit: it is neither a result field,
recommendation input, warning input, nor threshold.

## 7. Batch fitting

Milestone-6 Multi-Q fitting applies one fixed branch topology through ordinary,
independent Single-Q fits. It is not global or simultaneous fitting and creates
no parameter coupling between Q groups.

The current Q is the anchor. Each branch traverses outward independently:

```text
anchor -> lower neighbor -> next lower Q -> ...
anchor -> higher neighbor -> next higher Q -> ...
```

Lower and higher are determined from the dataset's physical representative
`q_values`, not from group-index direction. Explicit Q input order remains
unchanged: the executor derives only a temporary visit order, while group
identity and stored/result ordering remain the original dataset order. A branch
requires unique representative Q values because duplicate values cannot define
an unambiguous lower or higher continuation side.

At an edge anchor, only the available direction runs. A target uses fitted
values from the nearest successful predecessor in its direction solely as
execution-time optimizer starts. Failure at one Q does not stop the direction,
does not seed the next Q, and does not erase completed neighboring results.

Continuation is Result-to-Result: ordinary fitted/current free-parameter values
come from a concrete successful Result, while a Method remains a reusable
configuration system. Every production Result is bound to the exact dataset,
Q assignment, retained fitting selection, prepared measured Resolution, and
group context that produced it. The S1 executor rejects a foreign or stale
anchor before traversal, even when group index, representative Q, and model
topology happen to match. Switching datasets therefore cannot reinterpret the
previous dataset's Current Result as the new dataset's Current Fit.

`Fit All Q from Current Fit` accepts a Current Fit originating from either a
Manual Run Fit or an applied AutoFit candidate; after that point both origins
use the same Multi-Q semantics. `AutoFit All Q` instead performs fresh
candidate discovery only at the anchor. It uses the frozen nine-candidate
search there, never reruns that search independently at every Q, and does not
silently replace an absent Most Recommended result with Best Supported. A user
may explicitly select any anchor candidate with a usable fit for a separate
comparison branch while retaining its warnings and limitations. Branches share
neither results nor seeds.

The direct AutoFit B0 topology remains CONSTANT. An applied B0 Current Fit
retains the frozen editable Manual representation: LINEAR with the fitted
Offset and an exact fixed-zero Slope. No Multi-Q-specific B0 topology is added.

Model topology is mandatory for a branch. User-selected Method categories may
also be transferred where applicable, including user bounds, Free/Fixed state
and fixed values, parameter relationships, Resolution, Q-bin configuration,
and fitting selection/context. When user-bound transfer is off, a target retains
its own user-bound state; when on, the source Method's user bounds overwrite the
target state. Group-local scientific/core constraints are always recomputed and
remain authoritative.

Component identities and Lorentzian numbering follow the anchor model through
the branch. The workflow does not automatically relabel components by FWHM,
area, or another matching heuristic. Ambiguous decomposition retains identity
and exposes existing fit/identifiability limitations.

Per-Q execution states are `SUCCESS`, `FAILED`, `BLOCKED`, `EXCLUDED`, and
`NOT_RUN`; branch states include `COMPLETED` and `CANCELLED`. Completed results
remain available after cancellation. `EXCLUDED` is an explicit user decision,
not a synonym for failed or blocked execution. Exclusion from spectral fitting
and exclusion from later derived-result use remain distinct and recorded.

## 8. Analysis-level masks and warnings

The following analysis-level concepts remain distinct:

- automatically detected invalid values, including invalid sigma;
- `AUTO` boundary-padding masks;
- `REVIEW` boundary-padding masks;
- explicit per-group manual point-exclusion masks;
- explicit per-group manual `AUTO`-reinclusion masks;
- the inclusive fitting-energy range selected independently for each group; and
- the derived effective fitting-point mask.

The derived exclusion is exactly invalid measurement data OR (`AUTO` padding
AND NOT manual `AUTO` re-inclusion) OR manual exclusion OR points outside that
group's inclusive fitting range. `AUTO` is therefore a reversible default
analysis proposal, while invalid data remain non-overridable. `REVIEW` padding
remains retained unless excluded by another rule. Manual exclusion and manual
`AUTO` re-inclusion are mutually exclusive at each point and never modify the
stored padding proposal or original arrays. Whole-Q fitting exclusions and
derived-result exclusions remain separate later-workflow state.

Possible Bragg contamination, including contamination near the elastic line at
specific Q, may generate a warning. ezQENS must not silently delete the Q
spectrum or affected points. The detection heuristic is unresolved.

The importer preserves every repeated value. Edge-padding detection runs
separately and primarily examines boundary-connected constant
`(intensity, uncertainty)` runs. It does not mask an identical plateau occurring
solely inside the spectrum or hard-code a sentinel. The sole sign-specific
version-2.1 extension is the explicitly bounded singleton rule below; intensity
sign alone is otherwise insufficient.

The approved `edge-padding-v2.1.0` rule is deliberately behavior-oriented:

- repeated-pair equality uses relative tolerance `1e-7` and absolute tolerance
  `1e-12` for intensity and uncertainty;
- repeated-run candidates require at least two boundary-connected pairs;
- a repeated-run `AUTO` candidate requires a finite adjacent interior point with
  strictly positive uncertainty and an intensity or uncertainty transition
  greater than `1e-12 + 0.05 * max(abs(plateau), abs(interior))`;
- independently of repeated-run classification, one outermost point is `AUTO`
  with reason `singleton_negative_edge_drop` only when its energy, intensity,
  and strictly positive uncertainty are valid, its intensity is strictly
  negative, the immediately inward point is valid, the inward intensity is
  greater, and that intensity rise passes the same transition criterion; no
  interior point or second point is included by this singleton rule;
- a clear run of at least five points is `AUTO`;
- a shorter clear run is `AUTO` only when the same boundary signature is
  corroborated by clear runs of at least five points in at least two other
  spectra;
- other usable repeated boundary runs are `REVIEW`; and
- a boundary with neither a usable repeated-pair candidate nor a qualifying
  singleton is `NONE`.

These labels are actions, not statistical confidence. `AUTO` points enter an
exact reversible default-on mask; `REVIEW` points enter a mutually exclusive
review mask; `NONE` masks nothing. The repeated-run rule has no sigma-jump test, intensity-sign support,
relative-length criterion, or hard-coded sentinel; the singleton exception adds
no generalized sign heuristic. Otherwise identical positive, zero, and negative
plateau values are classified by the same run and transition rules.

Padding masks never modify or baseline-shift intensity arrays. All plateau
points are included exactly, including the point immediately adjacent to the
first retained interior point. Invalid-data masks, manual masks, ranges, Q
exclusions, Bragg warnings, and padding masks remain independently recoverable.
Phase-A production backgrounds are NONE, B0 (constant `b0`), and B1 (linear
`b0 + b1 * E`). They are additive spectral-model components and do not alter
the source intensity arrays.

Every automatic flag, user mask, range change, Q exclusion, restoration, and
reason remains traceable in the analysis state. A complete append-only project
history is not required for v1.0.

## 9. Resolution processing and convolution

Resolution may originate from vanadium or a low-temperature sample. Original
arrays are immutable. A Milestone-3 prepared resolution references the original
`Spectrum`/`ReducedDataset` and stores minimal support, normalization,
association, and diagnostic state rather than duplicate original arrays.

Sample and resolution datasets require exact ordered one-to-one Q association:
equal group counts and representative values, plus equal edges when explicitly
known on both. Fixed relative tolerance `1e-10` and absolute tolerance `1e-12`
in `Å^-1` allow harmless representation noise only. No reordering, nearest-Q
matching, Q interpolation, or repair occurs.

Padding detection runs independently on sample and resolution. Resolution
`AUTO` points do not enter a proposed kernel or its normalization and have
conceptual zero kernel contribution by default. AUTO application is an
independent per-group boolean state and is reversible: explicitly disabling it
restores otherwise valid AUTO-marked points inside the selected support. A
support change alone does not disable AUTO. `REVIEW` remains accepted by
default. Invalid energy/intensity is non-overridable; invalid uncertainty
remains visible and is never replaced with zero. The original/pre-QC support
spans the valid measured bounds, independently of AUTO application. Resolution
support is distinct from the sample fitting range.

A normal resolution Q group that passes existing structural/preparation QC is
automatically accepted as default `KEEP`, with default support and default AUTO
padding application, before normalization. Provenance records automatic QC rather
than user confirmation. Explicit user-reviewed `KEEP`, AUTO-padding override, or
`EXCLUDE_BY_CONTIGUOUS_SUPPORT` remains available; explicit review decisions must
be confirmed. The `confirmed` flag authorizes scientific use but does not alone prove
user confirmation; `source` carries the automatic-versus-user origin. KEEP is
scientifically neutral and never identifies the origin of
measured structure. A neutral
`suspicious_structure_retained_by_user` warning may accompany KEEP when expert
judgement accepts unresolved or overlapping structure without correction.
`EXCLUDE_BY_CONTIGUOUS_SUPPORT` requires a strictly narrower inclusive interval
inside the original/pre-QC support and can remove only low-energy, high-energy,
or both outer boundary regions. It cannot represent an internal hole.

The review preview exposes the original and proposed accepted supports, raw
accepted energy/intensity/uncertainty, the proposed normalization factor,
warning, authorization, and acceptance-origin states, and the accepted
pre-normalization area. It
also reports

```text
signed_area_ratio = accepted_signed_trapezoidal_area / pre_QC_signed_trapezoidal_area
```

when the pre-QC signed area is finite and positive. Both numerator and
denominator retain the measured intensity signs after existing invalid/AUTO
handling, so this dimensionless ratio is not constrained to `[0, 1]` and may
legitimately exceed 1. It is not a probability, a physical resolution-
containment fraction, or a measure of how much of the true instrumental
response remains. It is diagnostic/provenance information only: it has no
approval or rejection threshold, cannot certify that the accepted support is
scientifically correct, and cannot determine the scientific origin of a
feature.
Pending explicit-review groups cannot become prepared kernels. Automatically
accepted normal groups require no second confirmation. No automatic peak
detection, classification, subtraction, clipping, reconstruction, or
interpolation across suspicious internal structure is performed. If the user
cannot justify retaining or boundary-excluding the measured structure, an
independently expert-prepared resolution should be supplied instead.

Sample and resolution AUTO-retained boundaries are compared per associated Q
as a warning-only consistency diagnostic using fixed relative tolerance
`1e-10` and absolute tolerance `1e-12` in the energy unit. A disagreement does
not modify either mask.

After automatic or explicit acceptance, for measured coordinates `(E_j, I_j)`,
the
existing Milestone-3 calculation uses trapezoidal
integration on the actual, possibly nonuniform grid:

```text
area_Q = trapezoid(I_j, E_j)
R_Q(E_j) = I_j / area_Q
sigma_R_Q(E_j) = sigma_j / area_Q
```

The area must be finite and strictly positive. Accepted coordinates must be
finite, strictly increasing, unique, and contain at least two usable points.
If invalid energy/intensity occurs between the first and last measured indices
selected by the support, preparation fails with a blocking diagnostic. The
invalid point is not deleted and valid neighbors are not connected across it;
M3 performs no interpolation or implicit trapezoidal bridge. Invalid boundary
regions may instead be excluded explicitly by the support.
Normalization is by integrated area, never peak height. Sample spectra are not
normalized. Uncertainty in `area_Q` is not propagated into a covariance matrix;
M3/M4/M5 initially treat the measured kernel as fixed.

M3 preserves the accepted original measured grid and does not interpolate
sample or resolution data. It performs no baseline subtraction or estimate,
negative-intensity clipping, curve shift, automatic elastic-peak detection, or
energy recentering. Alignment is inspected against the unchanged `E = 0`
reference. A future explicit common energy transformation remains unresolved.

Milestone 4 builds a temporary fixed physical grid for each exactly associated
sample/resolution group. Its automatic spacing is:

```text
h = min(median positive sample spacing,
        median positive prepared-resolution spacing) / 4
```

Both coordinate sequences must be finite and strictly increasing; they are not
sorted, deduplicated, or repaired. For sample evaluation bounds
`[E_sample_min, E_sample_max]` and accepted resolution support
`[E_res_min, E_res_max]`, the required intrinsic-model domain is:

```text
model_min = E_sample_min - E_res_max
model_max = E_sample_max - E_res_min
```

The zero-centered intrinsic-model lattice covers these bounds with no more
than the deterministic spacing-alignment extension required at either side.
It is fixed by physical coordinates and never moves with a future `E0` fit
parameter. A future shift is evaluated as the fixed convolved profile at
`E_sample - E0`.

The accepted M3 resolution is linearly interpolated onto its temporary uniform
representation, is exactly zero outside accepted support, and receives a small
representation-only correction back to unit numerical area. This does not
alter or reinterpret the M3 measurement. Resolution, theory, and sample energy
coordinates are never recentered, and no missing tails are extrapolated.

For Lorentzian FWHM `Gamma > 0`, M4 uses `gamma = Gamma / 2` locally and the
analytic average probability density in each model-grid cell:

```text
L_bar_i = [atan((E_i + h/2 - E_c) / gamma)
           - atan((E_i - h/2 - E_c) / gamma)] / (pi * h)
```

The intrinsic primitive defaults to `E_c = 0`. This cell-integrated form keeps
the integrated-area meaning for lines much narrower than a grid cell; intrinsic
FWHM is not bounded below by instrumental width. The implementation performs
full FFT linear convolution with at least `N_model + N_resolution - 1` samples,
multiplies by `h` exactly once, constructs output coordinates from the sum of
the two input origins, and linearly evaluates only the convolved model on the
unchanged original sample coordinates. Scale-aware profile, peak, FWHM,
centroid, and area validation uses the provisional relative target of about
`5e-4`; exact FFT/direct and normalization identities use tighter tolerances.

Asymmetric accepted support is a supported normal condition and is not a
warning by itself. Possible incomplete resolution-peak containment remains a
future warning candidate informed by boundary signal, local rise, edge-region
fraction, and related sample coverage. Future sample/resolution boundary
comparisons use physical energy coordinates rather than matching array or bin
indices across different grids. No universal threshold, baseline subtraction,
tail reconstruction, recentering, or automatic Q change is approved. A future
analytic Gaussian, Lorentzian, or ideal resolution source also remains deferred
and has no M3/M4 registry or fallback.

Finite model domains capture less than the infinite area of sufficiently broad
Lorentzian tails. Future Milestone-5 amplitude semantics must account for this
explicitly and must not silently renormalize a truncated numerical model array.

## 10. Post-v1.0 molecular coordinates — provisional

If this post-v1.0 capability is promoted, XYZ input consists of element symbols
and Cartesian coordinates in angstrom. Import retains all atoms. The first
incoherent motion-model calculation selects hydrogen atoms only.

The initially proposed rotation center is exactly `(0, 0, 0)`. The user selects x, y, or
z as rotation axis. The system exposes all atoms, selected hydrogens, hydrogen
coordinates, and hydrogen distances from the rotation center.

Atom-selection decisions and the XYZ source/provenance must persist. No
automatic derivation of unrestricted dynamics from arbitrary crystal
structures is in scope.

## 11. Post-v1.0 candidate motion models — provisional

C2 rotational motion, C4 rotational motion, and isotropic reorientation are
separate alternative candidates, not combined populations. The analysis flow
is:

```text
free spectral fitting
  -> experimental EISF(Q)
  -> independent candidate-model fits
  -> model comparison
```

Comparison uses weighted residuals, reduced chi-square, AICc, parameter
uncertainties, residual structure, parameter validity, and scientific
warnings. Reports identify the best-supported model and alternatives but must
state that numerical preference does not prove a unique microscopic mechanism.

This capability is not required for v1.0. The authoritative C2 and C4 equations
and parameter definitions are explicitly unresolved pending scientific-owner
input and review. They must not be invented
in specifications or production code. The exact isotropic equation and
parameterization also require written approval because they are not supplied
by the current requirements.

## 12. Acceptance criteria

- FWHM meaning and units survive every calculation and serialization round trip.
- Known positive FWHM inputs reproduce the stated relaxation-time constant.
- EISF uses integrated areas and preserves component-level inputs.
- Weighted standardized residuals are the default and use valid uncertainties.
- Invalid sigma values are preserved and automatically masked, never repaired.
- Elastic area is obtained by direct scaling of unit-area measured resolution,
  without a discrete numerical delta.
- Lorentzian convolution preserves integrated area within approved tolerance.
- Original resolution data remain byte-for-byte logically distinct from
  processed data.
- All masks, range choices, mappings, exclusions, and warnings are traceable.
- Batch results are demonstrably independent per Q.
- Any later candidate comparison language avoids claims of unique mechanism.
- Tests encode authoritative conventions rather than undocumented defaults.

## 13. Explicit non-goals

Version 1.0 does not define raw-data reduction, INS or fixed-window analysis,
Bayesian/MCMC inference, global spectrum fitting, Arrhenius analysis, molecular
interpretation, XYZ-driven motion selection, candidate-motion comparison,
combined motion-model populations, GPU acceleration, or arbitrary user-defined
motion equations.

## 14. Risks, unresolved decisions, and dependencies

Key risks are FWHM/HWHM confusion, area/height confusion, invalid sigma values,
resolution-tail artifacts, FFT wrap-around, unreliable covariance, and
overinterpretation of candidate models. Explicit metadata, warnings, synthetic
tests, and scientific review mitigate them.

Unresolved items are the formulas and numerical policies identified above,
explicit-list comment syntax, fit-quality thresholds, and Bragg-warning
criteria. Q-bin edge/midpoint rules, the approved DAVE four-value semantics,
and invalid-uncertainty handling are fixed conventions.

ASCII parsing and Q identity must be validated before resolution association.
Resolution processing and convolution must be validated before spectral fits.
Fit covariance and component areas must be validated before EISF uncertainty.
Experimental EISF and approved candidate equations remain prerequisites for any
post-v1.0 motion-model comparison.
