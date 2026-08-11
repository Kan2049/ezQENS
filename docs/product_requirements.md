# Product requirements

## 1. Purpose

ezQENS is a lightweight, scientifically rigorous desktop application for
analysis of reduced quasielastic neutron scattering (QENS) data. It is designed
for both QENS experts and non-experts. A non-expert should eventually be able to
complete a trustworthy basic analysis through a guided GUI without knowing the
Python implementation or every advanced fitting option. The Python distribution
and import package are named `ezqens`.

Design priorities are scientific correctness, reliability, simplicity,
usability, performance, and feature count, in that order. For equally correct
solutions, prefer fewer concepts, less code, fewer dependencies, lower CPU and
memory use, and easier maintenance. The application targets ordinary personal
laptops and is CPU-first; GPU or HPC resources are not assumed.

The product comprises a desktop GUI, a GUI-independent Python analysis core,
standardized fitting workflows, independent batch analysis, lightweight
reproducibility information, and exports suitable for scientific review. A
public Python API is not a v1.0 user-facing deliverable, although the core must
be callable cleanly from the start.

ezQENS creates value through scientifically safe defaults, compression of
repetitive expert workflows, and progressive analytical depth from spectra to
basic physical quantities and later approved dynamics analysis.

## 2. Scope and assumptions

Version 1.0 operates on reduced sample data and corresponding measured
resolution data. Raw detector-data reduction is assumed complete before data
reach ezQENS. Detector-level calibration, grouping, and masking therefore
remain the responsibility of upstream reduction software.

The primary public-release platforms are macOS and Windows laptops. Linux may
be supported on a best-effort basis when this adds little complexity, but must
not delay the first public release. Initial reduced-input support is centered
on DAVE- and Mantid-preprocessed workflows. Real validation should include
workflows such as PSI FOCUS and ILL IN5/IN16 without hard-coded
instrument-specific assumptions. Mantid must not be a required core dependency.

### 2.1 Product and scientific backbone

The primary experience is a GUI with progressive depth. Non-experts receive a
guided path, understandable choices, trustworthy basic outputs, and meaningful
warnings. Experienced users can progressively access ranges, masks, model
components, parameters, bounds, resolution settings, batch behavior,
diagnostics, visualization, and exports. Both use the same scientific core.

The long-term scientific workflow is:

```text
reduced QENS data
  -> preprocessing and fitting-data selection
  -> measured-resolution treatment
  -> spectral fitting
  -> basic QENS quantities
  -> later approved physical dynamics analysis
```

Spectral analysis yields integrated elastic/quasielastic intensities, FWHM,
uncertainties, convergence information, diagnostics, and residuals. Basic QENS
quantities include FWHM(Q), relaxation time, EISF(Q), component intensity versus
Q, and supported uncertainty propagation. Later dynamics analysis may fit their
Q dependence to scientifically approved temporal or spatial models. Automatic
model inference from arbitrary coordinates, symmetry, or molecular structure is
a substantially later capability and must not drive current architecture.

## 3. Required version-1.0 workflow

A user must be able to:

1. Import reduced sample data.
2. Import corresponding measured resolution data.
3. Inspect `S(Q,E)`.
4. Define Q mapping and fitting ranges.
5. Preprocess sample and resolution data.
6. Configure a spectral model.
7. Fit one selected Q spectrum.
8. Sequentially batch-fit all selected Q spectra.
9. Review quality and manually refit problematic spectra.
10. Extract FWHM linewidths.
11. Calculate relaxation times.
12. Calculate experimental EISF.
13. Inspect diagnostics, warnings, residuals, and failure states.
14. Export useful scientific results.
15. Retain lightweight reproducibility information for the analysis.

Each stage must preserve sufficient provenance to explain inputs, settings,
transformations, exclusions, warnings, and results.

## 4. Supported inputs

Version 1.0 must support:

- DAVE group-block ASCII output;
- wide QENS tables with shared `x` and paired `yN`/`yerrN` columns;
- single-spectrum `x`/`y`/`yerr` tables;
- generic ASCII, TXT, DAT, and CSV files through explicit custom mapping;
- measured resolution from vanadium or a low-temperature sample;
- uniform outer-edge bins, explicit representative values, approved DAVE
  Q-parameter files, and later validated explicit-list/imported-metadata Q
  mappings.

HDF5 input may follow only after the ASCII path is validated.

### 4.1 Shared spectrum representation and roles

All reduced-data layouts convert to `ReducedDataset`, an ordered collection of
`Spectrum` values exposing original energy/intensity/uncertainty arrays,
invalid-data masks, source metadata, and format provenance. Each spectrum is
unambiguously a sample or measured-resolution spectrum, either through a
required role or a validated role-specific wrapper around a shared numerical
structure. Role-specific validation and processing remain separate.

Downstream analysis depends on `ReducedDataset` / `Spectrum`, not the source
file layout. A future raw-data reducer or interoperability adapter may produce
the same boundary, but no raw HDF/NeXus, instrument, Mantid, registry, factory,
or plugin framework is part of the current milestone.

A shared-grid wide table may use energy shape `(n_energy,)` and intensity and
uncertainty shapes `(n_q, n_energy)` internally, while still exposing every
spectrum independently. Group-block data may retain unequal grids and lengths.
Import never interpolates or discards values to select a storage layout.

### 4.2 Reduced-data layouts

**DAVE group-block format:** repeated markers such as `# Group N` identify
blocks containing at least `x`, `y`, and `yerr`. Preserve block/group order,
row order, arrays, source columns, and invalid-value diagnostics. Groups may
have unequal grids, row counts, and valid ranges. Recognized `ModelFit`,
`Func1`, `Func2`, and other fit-component columns are recorded as ignored
source metadata, never interpreted as measured intensity or uncertainty.

**Wide QENS table:** exactly one shared energy column and one or more complete
`yN`/`yerrN` pairs. Suffixes identify and order groups unless an explicit
mapping is supplied. Missing, duplicate, or ambiguous suffixes and inconsistent
row widths produce diagnostics. The layout does not itself define physical Q
unless explicit Q metadata are present.

**Single-spectrum table:** one energy, intensity, and uncertainty triplet
without group markers represents one spectrum. Q-dependent analysis still
requires one explicit Q value.

**Generic custom mapping:** when classification is unsafe, the product lets the
user identify energy, intensity, uncertainty, optional group identity, and
optional embedded-Q columns. No scientific column meaning is guessed when
multiple interpretations are plausible.

### 4.3 Content-based format detection

Detection precedence is:

1. explicit user-selected format;
2. recognized DAVE group markers and compatible block headers;
3. recognized wide-table energy plus `yN`/`yerrN` pairs;
4. recognized single-spectrum `x`/`y`/`yerr`;
5. custom mapping.

File extensions do not determine layout. The low-level detector returns the
proposed format, concise evidence, group/column count, required and extra
columns, structural diagnostics, and plausible alternatives. It has no
confidence score or confirmation state. The later application/GUI workflow
shows an automatic proposal for review and owns confirmation. Ambiguity must
not be resolved silently.

### 4.4 Q-mapping modes

Every current mode produces dataset-level `QBins` with exactly one finite
representative value per spectrum in `Å^-1`.

**Uniform edge range:** finite lower/upper Q values are authoritative inclusive
outer edges, the existing spectrum count `N` is authoritative, and the bins
exactly cover the range:

```text
delta_Q = (upper_q_edge - lower_q_edge) / N
q_value[i] = (edge[i] + edge[i + 1]) / 2
```

Generate `N + 1` contiguous edges and `N` midpoint representatives. The outer
edges are not first/last group centers. No private range is an application
default.

**Explicit representative values:** preserve arbitrary nonlinear supplied order
and leave edges unknown. Never infer edges, sort, deduplicate, interpolate, pad,
truncate, extrapolate, or replace entries.

**DAVE Q-bin parameter file:** support the approved four numeric values only:
lower limit, upper limit, reported group count, and step. Reconstruct all
complete fixed-width bins from the limits and step without extending to the
upper limit, merging the remainder, or creating a partial final bin. Retain the
source upper limit even when it exceeds the final actual edge. A reported-count
mismatch warns but does not replace the reconstructed bins.

For every mode, Q count must equal spectrum count. Explicit-list syntax,
imported-Q metadata, resolution association, and application/GUI confirmation
remain future workflow concerns.

### 4.5 Boundary-padding detection

Sample and resolution imports preserve all original values. A separate
preprocessing service inspects only boundary-connected repeated
`(intensity, uncertainty)` plateaus and may use matching signatures across
groups as supporting evidence. It never identifies padding solely from zero or
negative intensity and never hard-codes a sentinel value.

Padding status is behavioral rather than statistical. `AUTO` produces a
point-level reversible default-on mask, `REVIEW` produces a separate review
mask that is not default-on, and `NONE` produces no mask. A boundary result
retains only its side, run length, energy bounds, status, and compact reason.
The dataset-level result records the algorithm version once.

No padding operation shifts, subtracts, or adds an intensity baseline. Original
arrays remain immutable, internal constant segments are not edge padding, and
physical negative or near-zero interior values remain valid. Auto-padding masks
and suggestions remain distinct from invalid uncertainty, manual masks, general
fit range, resolution valid range, and Bragg warnings.

## 5. Spectral analysis capabilities

Measured-resolution convolution must support an elastic component, one or more
Lorentzians, and constant or linear background. Each Q spectrum has independent
free-fit values for elastic integrated area, each Lorentzian integrated area
and FWHM, energy-center shift, and background parameters. Every parameter must
have an initial value, lower and upper bounds, and a fixed/free state.

After valid-range selection, optional approved constant-baseline correction,
and unit-area normalization, let the measured resolution be `R_Q(E)`. Evaluate
the elastic contribution directly as:

```text
A_elastic * R_Q(E - E0)
```

This is mathematically equivalent to convolving
`A_elastic * delta(E - E0)` with the measured resolution, but production fitting
must not construct a grid-dependent numerical delta spike. Because `R_Q` has
unit integrated area, `A_elastic` is the integrated elastic area.

Each quasielastic term is:

```text
A_i * [R_Q convolved with L_i](E - E0)
```

where `L_i` is a unit-area Lorentzian, `A_i` is its integrated area, and its
linewidth is FWHM. Numerical convolution preserves integrated-area semantics
within a documented finite-grid tolerance.

The default optimizer is `scipy.optimize.least_squares`. The default residual
is `(model - data) / sigma`, making weighted least squares the default
objective. A future optional robust weighted mode may use soft-L1 or Huber
loss. Unweighted and logarithmic relative losses are not defaults.

Every mature fit result records optimized values, estimable standard errors,
raw and standardized residuals, chi-square, reduced chi-square, AIC, AICc,
fitted-point count, free-parameter count, convergence, bound hits, invalid
covariance/Jacobian warnings, and scientific-quality warnings. AIC/AICc remain
planned but do not block the first validated single-spectrum prototype while
the likelihood convention is unresolved. They may be compared only for fits
using identical points, masks/Q selection, uncertainty treatment, residual
definition, and likelihood convention.

## 6. Batch analysis and masking

Batch fitting applies one configuration sequentially to independent Q spectra;
it is not simultaneous or global fitting. It must support low-to-high Q,
optional high-to-low Q, optional propagation of the previous successful fit as
the next initial guess, manual refitting after failure, per-Q results, and
per-Q warnings.

The system distinguishes invalid values, fitting-energy ranges, manually
masked energy points, Q spectra excluded from spectral fitting, and fitted Q
spectra excluded from later validated derived quantities. Possible Bragg
contamination may trigger a warning but never silent deletion. Every exclusion
is recorded in the analysis state. A complete append-only persistent history is
not a v1.0 requirement.

Automatic edge-padding masks are a further distinct point-level mask. `AUTO`
may be default-on but remains reversible; `REVIEW` remains separate and
requires a later user decision. Padding is not background subtraction and does
not modify intensity values.

A statistically valid uncertainty is finite and strictly greater than zero.
NaN, positive/negative infinity, zero, and negative sigma values are flagged
and excluded from weighted fitting by an automatic invalid-data mask. They are
not replaced, made absolute, estimated, inferred from other groups, or deleted
from original arrays. The fitting service fails clearly when valid, in-range,
unmasked points are insufficient for the free-parameter count; the eventual
rule must at minimum require positive statistical degrees of freedom.

## 7. Resolution processing

The system retains original and processed resolution data separately. The first
implementation makes manual valid-range selection mandatory and authoritative.
It supports invalid-value detection, optional explicit constant-baseline
correction, explicit validated unit-area normalization, energy-grid validation,
interpolation to a uniform internal grid, sample-to-resolution Q-group
association, exact or explicitly selected nearest-Q matching, numerical
convolution protected against circular FFT wrap-around, and recorded processing
decisions.

`AUTO` edge-padding masks from the preprocessing service may be default-on and
remain reversible. `REVIEW` masks are never applied without an explicit later
user decision. Neither replaces mandatory, authoritative manual resolution
valid-range selection. The processing preview should show original range,
automatic and review masks, selected range, invalid points, baseline setting,
normalization result, and warnings.
Normalization fails when no finite positive integral exists.

The complete imported energy range must never be assumed physically valid.

## 8. Derived quantities and later dynamics analysis

Spectral outputs use FWHM linewidths. Relaxation time and experimental EISF,
where scientifically validated, follow the authoritative formulas in
`scientific_conventions.md`. Component intensity versus Q and supported
uncertainty propagation are also part of progressive basic analysis.

Scientifically approved dynamics analysis may later fit the Q dependence of
these outputs to diffusion/jump-diffusion, characteristic/residence-time,
rotational/reorientational, EISF-geometry, confined/localized-motion, or other
validated QENS models. The owner may schedule selected models before or after
the first public release; this document does not create a mandatory catalogue.
Automatic molecular structure interpretation, XYZ-driven model selection, and
the currently unresolved C2/C4/isotropic candidate workflow remain outside
v1.0 unless explicitly promoted.

## 9. Reproducibility, export, visualization, and GUI requirements

Version 1.0 retains enough lightweight reproducibility information to identify
inputs, selections and masks, model configuration, fit settings, results,
warnings, and software version. This does not require a complex archive or
database, UUIDs everywhere, hash-addressed arrays, migration frameworks,
append-only histories, or attachment infrastructure.

The former `.qensfit` project-container proposal is not a permanent contract.
The final container format and extension remain unresolved until persistence
work is actually designed. Heavy persistence, migration, security, and archive
architecture are provisional post-v1.0 work and must not delay scientific or
release validation.

Scientific export is a first-class evolving boundary. It may progressively
include processed spectra, fitted curves and components, residuals, parameters,
uncertainties, FWHM(Q), relaxation time, EISF(Q), later dynamics results, and
the configuration needed for lightweight reproducibility.

Visualization likewise consumes prepared scientific results and may grow from
spectra/resolution/fit inspection to derived-Q and later dynamics views,
comparison controls, and publication-oriented vector output. It must never
recompute EISF, relaxation times, or other authoritative quantities.

The final GUI uses PySide6 and initially provides:

1. Import
2. Spectrum Fit
3. Batch Results
4. Export or Analysis Summary

It must be small, elegant, scientific, guided for non-experts, free of formulas
and optimizer logic, and responsive during cancellable long-running fits
executed outside the GUI thread. It calls the same scientific core used by
Python workflows; no scientific operation is duplicated in GUI code.

The GUI exposes complexity progressively rather than requiring literal
“beginner” and “advanced” modes. Safe defaults, clear choices, basic outputs,
and warnings come first; detailed ranges, masks, components, parameters,
bounds, resolution settings, diagnostics, visualization, and export controls
appear when needed.

The planned Import screen includes a data-format section showing the automatic
proposal, evidence, warnings, manual override, group/pair count, and a preview
of energy/intensity/uncertainty mapping. Its Q-mapping section offers
linear range, manual list, explicit-list file, DAVE Q-bin parameter file, and
supported imported metadata. It shows the resolved explicit Q list and compares
its count with sample and applicable resolution spectra. Automatically
detected formats and generated/imported mappings require confirmation.

## 10. Explicit v1.0 non-goals

Version 1.0 excludes:

- raw detector-data reduction;
- detector calibration or grouping;
- detector-level bad-detector masking;
- INS analysis and fixed-window scans;
- Bayesian fitting and MCMC;
- web deployment and multi-user collaboration;
- automatic publication-quality figure editing;
- multi-temperature Arrhenius fitting;
- simultaneous global spectrum fitting;
- arbitrary user-defined Python motion models;
- automatic molecular-model interpretation;
- XYZ-driven automatic motion-model selection;
- C2/C4/isotropic candidate-motion comparison;
- general crystal or symmetry analysis;
- GPU acceleration; and
- automatic derivation of unrestricted molecular dynamics from arbitrary
  crystal structures.

## 11. Privacy and benchmark constraints

The initial private benchmark has a reduced DAVE sample, matching measured
resolution, 14 Q groups, a user-supplied range of 0.46–2.02 inverse angstrom,
extra DAVE `ModelFit` and `Func` columns to ignore. These details validate
configurable behavior; none is a default.

Files in `private_data/` are confidential. They cannot be committed, staged,
copied to tests/docs/examples/packages, printed as full arrays, redistributed,
or uploaded. Local tools may expose structural summaries only. Public tests
must use independently generated synthetic data.

## 12. Acceptance criteria

Version 1.0 is acceptable when:

- the required workflow is usable end to end on reduced data;
- all mandatory formats and Q-mapping modes have validated import paths;
- original data and every user-visible transformation remain traceable;
- spectral fits expose the required diagnostics and warnings;
- linewidth/EISF conventions are explicit in UI, tables, plots,
  reproducibility records, and exports;
- sequential batch fits remain independent per Q;
- lightweight reproducibility information identifies inputs, selections,
  settings, results, warnings, and software version;
- private-data rules are enforced by design and test;
- the scientific validation plan passes on macOS Apple Silicon and supported
  Windows targets; and
- documentation makes limitations and unresolved scientific decisions visible;
  and
- an external real user, including a non-QENS expert, can complete a guided
  reduced-data analysis end to end on an ordinary macOS or Windows laptop,
  inspect results and warnings, and export the analysis.

## 13. Unresolved decisions

- Detailed accepted syntax within DAVE and generic ASCII layout families.
- Blank-line/comment policy for explicit Q-list files.
- Default uniform convolution grid and interpolation method.
- Exact AIC/AICc likelihood convention and covariance-estimation policy.
- Bragg-contamination warning heuristic.
- Final public packaging, licensing, and Windows installer technology.
- Final project-container format and extension, if a container is later needed.
- Post-v1.0 molecular/motion-model equations and parameterizations.

## 14. Risks and milestone dependencies

The largest risks are silent format misinterpretation, invalid resolution
tails, convolution artifacts, parameter non-identifiability, loss of
traceability, GUI/core coupling, non-expert misuse, and leakage of private data.
The mitigation is staged scientific validation, guided interaction, explicit
warnings, and immutable originals.

The content detector and core table importers precede Q mapping and resolution
work. Resolution-grid processing precedes convolution. Convolution precedes
validated single-spectrum fitting.
Single-spectrum validation precedes batch fitting and derived quantities.
Minimal in-memory contracts evolve through the scientific milestones. GUI
implementation starts only after the importer, resolution, convolution, and
single-spectrum path are scientifically validated. GUI, export, lightweight
reproducibility, documentation, and macOS/Windows packaging all precede the
special v1.0 public-release gate.
