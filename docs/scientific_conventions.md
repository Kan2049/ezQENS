# Scientific conventions

## 1. Scope

This document fixes the scientific conventions that are authoritative for
version 1. Unspecified details are marked unresolved and must not be inferred
from private data or chosen merely to make tests pass.

Reduced inputs are assumed to have completed detector reduction and
detector-level bad-detector masking. qensfit performs analysis-level operations
only.

## 2. Axes, units, and identity

- Energy transfer is represented in meV.
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

Every applied Q mapping resolves to an ordered explicit list in `Å^-1`, with
exactly one value per sample spectrum. Manual-list order is authoritative and
must not be sorted or repaired.

Linear-range mapping has inclusive endpoints:

```text
Q_i = Q_start + i * (Q_end - Q_start) / (N - 1)
i = 0, ..., N - 1
```

Both endpoints are finite, `N >= 2`, and the generated values are equivalent to
`numpy.linspace(Q_start, Q_end, N)`. The resolved list is reviewed and
confirmed before use. Group-by-group resolution association also validates the
applicable resolution count. No mismatch is fixed by truncation, padding,
discarding, or combining values or spectra.

Manual lists, explicit-list files, approved DAVE Q-bin parameter files, and
supported imported metadata are additional sources of resolved lists. DAVE
parameter semantics remain unresolved until every supported field is approved.

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
export field, and project field must identify the linewidth as FWHM. No layer
may silently convert FWHM to HWHM. If a numerical routine uses HWHM internally,
that conversion must be local, explicit, tested, and absent from persisted
meaning.

A finite relaxation time requires a finite positive FWHM. Zero, negative, NaN,
or infinite FWHM must produce no valid `tau_ps` and an explicit warning rather
than a misleading number.

## 4. Spectral model semantics

The measured-resolution-convolved model supports:

- an elastic component;
- one Lorentzian or multiple Lorentzians;
- a constant or linear background.

For each Q spectrum, free fitting permits independent elastic integrated area,
Lorentzian integrated area, Lorentzian FWHM, energy-center shift, and
background parameters. Each parameter has an initial value, lower and upper
bounds, and fixed/free status.

Elastic and Lorentzian amplitude parameters used for EISF are integrated areas,
not peak heights. Component identifiers and individual quasielastic areas must
be retained even when an aggregate is displayed.

Let `R_Q(E)` be the measured resolution after manual valid-range selection,
optional approved constant-baseline correction, and unit-area normalization.
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
A_i * [R_Q convolved with L_i](E - E0)
```

`L_i` is a unit-area normalized Lorentzian, `A_i` is its integrated
quasielastic area, and its linewidth parameter is FWHM. Numerical convolution
must preserve this integrated-area meaning within a reviewed, documented
finite-grid tolerance.

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

The fitting service fails clearly when too few valid, in-range, unmasked points
remain. The final minimum-point rule is resolved during fitting implementation,
but it must at least ensure positive statistical degrees of freedom:
`n_fitted_points - n_free_parameters > 0`.

A mature result records:

- optimized values and estimable standard errors;
- raw residuals (`model - data`);
- standardized residuals;
- chi-square and reduced chi-square;
- AIC and AICc;
- fitted-point and free-parameter counts;
- convergence status and optimizer termination information;
- bound-hit warnings;
- invalid covariance or Jacobian warnings; and
- scientific-quality warnings.

The exact likelihood assumptions used for AIC/AICc and the rules for deriving
standard errors from the Jacobian are unresolved pending scientific review.
Until fixed, implementations must expose the chosen convention and never
compare values calculated under different conventions.

AIC/AICc do not block the first validated single-spectrum prototype. That
prototype may pass using convergence/termination, raw and standardized
residuals, chi-square, reduced chi-square, parameter recovery, bound hits,
Jacobian/covariance status, and scientific warnings. When implemented,
AIC/AICc are comparable only for fits with identical data points, masks and Q
selection, uncertainty treatment, residual definition, and likelihood
convention.

## 7. Batch fitting

Batch fitting is sequential application of a shared model configuration to
multiple Q spectra. Every Q retains independent parameters and an independent
result. It is not a global or simultaneous fit.

Required traversal is low Q to high Q, with optional high-to-low traversal.
Using the previous successful fit as the next initial guess is optional and
recorded. Failure at one Q does not erase other results. A failed Q can be
manually refitted. Exclusion from spectral fitting and exclusion from later
EISF-model fitting are distinct states with recorded reasons.

## 8. Analysis-level masks and warnings

The following concepts remain distinct:

- automatically detected invalid values, including invalid sigma;
- `AUTO` boundary-padding masks;
- `REVIEW` boundary-padding masks;
- the fitting-energy range;
- manually masked energy points;
- Q spectra excluded from spectral fitting; and
- fitted Q spectra excluded from EISF-model fitting.

Possible Bragg contamination, including contamination near the elastic line at
specific Q, may generate a warning. qensfit must not silently delete the Q
spectrum or affected points. The detection heuristic is unresolved.

The importer preserves every repeated value. Edge-padding detection runs
separately and examines boundary-connected constant `(intensity, uncertainty)`
runs only. It does not mask an identical plateau occurring solely inside the
spectrum, infer padding from intensity sign alone, or hard-code a sentinel.

The approved `edge-padding-v2.0.0` rule is deliberately behavior-oriented:

- repeated-pair equality uses relative tolerance `1e-7` and absolute tolerance
  `1e-12` for intensity and uncertainty;
- a candidate requires at least two boundary-connected repeated pairs;
- `AUTO` requires a finite adjacent interior point with strictly positive
  uncertainty and an intensity or uncertainty transition greater than
  `1e-12 + 0.05 * max(abs(plateau), abs(interior))`;
- a clear run of at least five points is `AUTO`;
- a shorter clear run is `AUTO` only when the same boundary signature is
  corroborated by clear runs of at least five points in at least two other
  spectra;
- other usable repeated boundary runs are `REVIEW`; and
- a boundary without a usable repeated-pair candidate is `NONE`.

These labels are actions, not statistical confidence. `AUTO` points enter an
exact reversible default-on mask; `REVIEW` points enter a mutually exclusive
review mask; `NONE` masks nothing. The rule has no sigma-jump test,
nonpositive-intensity support, relative-length criterion, or hard-coded
sentinel. Otherwise identical positive, zero, and negative plateau values are
classified by the same run and transition rules.

Padding masks never modify or baseline-shift intensity arrays. All plateau
points are included exactly, including the point immediately adjacent to the
first retained interior point. Invalid-data masks, manual masks, ranges, Q
exclusions, Bragg warnings, and padding masks remain independently recoverable.
Constant or linear backgrounds remain future spectral-model components.

Every automatic flag, user mask, range change, Q exclusion, restoration, and
reason remains traceable in the analysis state; the complete append-only
project history is deferred to milestone 8.

## 9. Resolution processing and convolution

Resolution may originate from vanadium or a low-temperature sample. Original
arrays are immutable and stored separately from processed arrays.

The first implementation supports:

1. Detection and flagging of invalid values, NaNs, and infinities.
2. Mandatory, authoritative manual valid-range selection.
3. Optional explicit constant-baseline correction.
4. Explicit unit-area normalization.
5. Energy-grid validation.
6. Interpolation onto a uniform internal grid.
7. Explicit association with sample Q groups.
8. Exact group matching or explicitly selected nearest-Q matching.
9. Numerical convolution with protection against circular FFT wrap-around.

The full imported resolution range is not presumed physically valid. Processing
order, parameters, array references, warnings, and user choices are auditable.
Normalization must fail clearly when a finite positive normalization area
cannot be established.

`AUTO` boundary-padding masks may be default-on while remaining reversible.
`REVIEW` masks still require a recorded later decision. Neither changes
original values nor replaces
mandatory manual resolution valid-range selection. The processing preview
exposes original range, auto-padding mask, optional suggestion, selected range,
invalid points, baseline setting, normalization result, and warnings.

Uniform-grid spacing, interpolation method, baseline-estimation method,
convolution boundary treatment, kernel centering, and normalization integration
rule remain unresolved. These must be validated independently before
single-spectrum fitting is accepted.

## 10. Molecular coordinates

Version 1 XYZ input consists of element symbols and Cartesian coordinates in
angstrom. Import retains all atoms. The first incoherent motion-model
calculation selects hydrogen atoms only.

The version-1 rotation center is exactly `(0, 0, 0)`. The user selects x, y, or
z as rotation axis. The system exposes all atoms, selected hydrogens, hydrogen
coordinates, and hydrogen distances from the rotation center.

Atom-selection decisions and the XYZ source/provenance must persist. No
automatic derivation of unrestricted dynamics from arbitrary crystal
structures is in scope.

## 11. Candidate motion models

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

The authoritative C2 and C4 equations and parameter definitions are explicitly
unresolved pending scientific-owner input and review. They must not be invented
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
- Candidate comparison language avoids claims of unique mechanism.
- Tests encode authoritative conventions rather than undocumented defaults.

## 13. Explicit non-goals

This document does not define raw-data reduction, INS or fixed-window analysis,
Bayesian/MCMC inference, global spectrum fitting, Arrhenius analysis, combined
motion-model populations, or arbitrary user-defined motion equations.

## 14. Risks, unresolved decisions, and dependencies

Key risks are FWHM/HWHM confusion, area/height confusion, invalid sigma values,
resolution-tail artifacts, FFT wrap-around, unreliable covariance, and
overinterpretation of candidate models. Explicit metadata, warnings, synthetic
tests, and scientific review mitigate them.

Unresolved items are the formulas and numerical policies identified above,
DAVE Q-bin parameter semantics, explicit-list comment syntax, fit-quality
thresholds, and Bragg-warning criteria. Inclusive linear Q mapping and
invalid-uncertainty handling are now fixed conventions.

ASCII parsing and Q identity must be validated before resolution association.
Resolution processing and convolution must be validated before spectral fits.
Fit covariance and component areas must be validated before EISF uncertainty.
Experimental EISF and approved candidate equations are prerequisites for motion
model comparison.
