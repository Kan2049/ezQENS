# Product requirements

## 1. Purpose

qensfit is a desktop application for standardized analysis of reduced
quasielastic neutron scattering (QENS) data. Its intended users are
neutron-scattering researchers. The product may eventually be released
publicly, but the current project is early-stage and not suitable for
scientific use.

The long-term product comprises a desktop GUI, a Python analysis core,
standardized fitting workflows, batch analysis, motion-model evaluation,
reproducible project files, and exports suitable for scientific review. A
public Python API is not a version-1 user-facing deliverable, although the core
must be callable cleanly from the start.

## 2. Scope and assumptions

Version 1 operates on reduced sample data and corresponding measured
resolution data. Raw detector-data reduction is assumed complete before data
reach qensfit. Detector-level calibration, grouping, and masking therefore
remain the responsibility of upstream reduction software.

Initial development and testing target macOS on Apple Silicon. Windows support
is required before public release. Mantid interoperability is optional and
must not be required by the analysis core.

## 3. Required version-1 workflow

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
13. Import a molecular XYZ model.
14. Calculate and fit separate candidate EISF models.
15. Compare candidate motion models.
16. Export results and save a reproducible project.

Each stage must preserve sufficient provenance to explain inputs, settings,
transformations, exclusions, warnings, and results.

## 4. Supported inputs

Version 1 must support:

- DAVE group-block ASCII output;
- wide QENS tables with shared `x` and paired `yN`/`yerrN` columns;
- single-spectrum `x`/`y`/`yerr` tables;
- generic ASCII, TXT, DAT, and CSV files through explicit custom mapping;
- measured resolution from vanadium or a low-temperature sample;
- inclusive linear-range, manual-list, explicit-list-file, approved DAVE
  Q-parameter-file, and imported-metadata Q mappings; and
- XYZ files with element symbols and Cartesian coordinates in angstrom,
  compatible with common VESTA exports.

HDF5 input may follow only after the ASCII path is validated.

### 4.1 Shared spectrum representation and roles

All reduced-data layouts convert to one scientific interface exposing ordered
spectra, explicit Q mapping, original energy/intensity/uncertainty arrays,
invalid-data masks, source metadata, and format provenance. Each spectrum is
unambiguously a sample or measured-resolution spectrum, either through a
required role or a validated role-specific wrapper around a shared numerical
structure. Role-specific validation and processing remain separate.

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

File extensions are hints only. The detector returns proposed format,
confidence, evidence, group/column count, required and extra columns,
structural warnings, and plausible alternatives. An automatic proposal
requires user review and confirmation. Ambiguity must not be resolved silently.

### 4.4 Q-mapping modes

Every mode resolves to an ordered explicit Q list in `Å^-1`, retained in the
analysis state and confirmed before application.

**Linear range:** for finite `Q_start`, finite `Q_end`, and `N >= 2`, generate
inclusive endpoints:

```text
Q_i = Q_start + i * (Q_end - Q_start) / (N - 1)
for i = 0, ..., N - 1
```

This is equivalent to `numpy.linspace(Q_start, Q_end, N)`. Show the generated
list for review. No private range or count is an application default.

**Manual list:** accept at least newline- and comma-separated numbers in the
future GUI. Preserve supplied order; never sort, deduplicate, interpolate, pad,
truncate, extrapolate, or replace entries.

**Explicit-list file:** support one numerical value per entry under an approved
blank-line/comment policy. Retain source format/reference, original parsed
values, unit, resolved list, warnings, parser version, and confirmation.

**DAVE Q-bin parameter file:** treat this as distinct from a value list. A
documented, tested file definition and scientific-owner approval of every
supported field are prerequisites for production parsing. Preview detected
format, original parameters, interpreted names, generated list/count, warnings,
and comparison with sample/resolution group counts. Until approval, the format
is supported in principle but scientifically unresolved.

**Imported metadata:** use only explicitly supported and validated Q metadata,
showing the resolved list for confirmation.

For every mode, final Q count must equal sample-spectrum count. When resolution
is associated group by group, validate its count too. A mismatch reports sample
count, applicable resolution count, Q count, and mapping mode. Never silently
truncate, pad, discard, or combine Q values or spectra.

### 4.5 Boundary-padding diagnostics

Sample and resolution imports preserve all original values. They may
advisorially flag long repeated `(intensity, uncertainty)` pairs, repeated
constant boundary segments, nonfinite points, and suspiciously identical
padding across rows or groups. No numerical padding value is hard-coded.

Suggested boundaries or valid ranges require explicit recorded confirmation.
Padding diagnostics remain distinct from invalid uncertainty, manual masks,
general fit range, resolution valid range, and Bragg warnings.

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
spectra excluded from later EISF-model fitting. Possible Bragg contamination
may trigger a warning but never silent deletion. Every exclusion is recorded
in the analysis state; the complete persistent project history is a
milestone-8 responsibility.

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

Repeated-boundary detection and automatic valid-range suggestions are optional
advisory enhancements. A suggestion is never applied without explicit user
confirmation. The processing preview should show original range, any suggested
range, selected range, invalid points, baseline setting, normalization result,
and warnings. Normalization fails when no finite positive integral exists.

The complete imported energy range must never be assumed physically valid.

## 8. Derived quantities and motion-model comparison

Spectral outputs use FWHM linewidths. Relaxation time and experimental EISF
follow the authoritative formulas in `scientific_conventions.md`.

XYZ import retains every atom. The first incoherent motion calculation uses
hydrogen coordinates only, a fixed rotation center `(0, 0, 0)`, and a
user-selected x, y, or z axis. Users can inspect all atoms, selected hydrogens,
their coordinates, and distances from the rotation center.

C2 rotation, C4 rotation, and isotropic reorientation are fitted as separate
candidate alternatives, not combined populations. Comparison uses weighted
residuals, reduced chi-square, AICc, parameter uncertainties, residual
structure, parameter validity, and scientific warnings. The result identifies
the best-supported and alternative models without claiming proof of a unique
microscopic mechanism.

## 9. Project, export, and GUI requirements

A saved project preserves source references and practical hashes, imported
data, Q mapping, units, processing history, masks, resolution settings,
spectral definitions, initial values, constraints, fit ranges, optimizer
settings, results, diagnostics, excluded Q values, EISF and motion-model
results, software version, and project-schema version.

The versioned `.qensfit` container remains a long-term version-1 proposal. Its
complete persistence infrastructure is not implemented in milestones 1–6.
Those milestones use only minimal typed in-memory models for import, Q mapping,
masks, resolution, convolution, fitting, batch, and derived results. They do
not require UUIDs for every temporary value, entity migrations, hash-addressed
array registries, append-only histories, final ZIP64 layout, attachments, or
atomic project saving. Milestone 8 reviews and implements the final format,
security model, hashes, migrations, archive layout, and public contract.

The final GUI uses PySide6 and initially provides:

1. Import
2. Spectrum Fit
3. Batch Results
4. Motion Models
5. Export or Project Summary

It must be small, elegant, scientific, free of formulas and optimizer logic,
and responsive during cancellable long-running fits executed outside the GUI
thread.

The planned Import screen includes a data-format section showing automatic
detection, confidence, warnings, manual override, group/pair count, and a
preview of energy/intensity/uncertainty mapping. Its Q-mapping section offers
linear range, manual list, explicit-list file, DAVE Q-bin parameter file, and
supported imported metadata. It shows the resolved explicit Q list and compares
its count with sample and applicable resolution spectra. Automatically
detected formats and generated/imported mappings require confirmation.

## 10. Explicit non-goals

Version 1 excludes:

- raw detector-data reduction;
- detector calibration or grouping;
- detector-level bad-detector masking;
- INS analysis and fixed-window scans;
- Bayesian fitting and MCMC;
- web deployment and multi-user collaboration;
- automatic publication-quality figure editing;
- multi-temperature Arrhenius fitting;
- simultaneous global spectrum fitting;
- arbitrary user-defined Python motion models; and
- automatic derivation of unrestricted molecular dynamics from arbitrary
  crystal structures.

## 11. Privacy and benchmark constraints

The initial private benchmark has a reduced DAVE sample, matching measured
resolution, 14 Q groups, a user-supplied range of 0.46–2.02 inverse angstrom,
extra DAVE `ModelFit` and `Func` columns to ignore, and separate C2/C4/isotropic
comparisons. These details validate configurable behavior; none is a default.

Files in `private_data/` are confidential. They cannot be committed, staged,
copied to tests/docs/examples/packages, printed as full arrays, redistributed,
or uploaded. Local tools may expose structural summaries only. Public tests
must use independently generated synthetic data.

## 12. Acceptance criteria

Version 1 is acceptable when:

- the required workflow is usable end to end on reduced data;
- all mandatory formats and Q-mapping modes have validated import paths;
- original data and every user-visible transformation remain traceable;
- spectral and motion fits expose the required diagnostics and warnings;
- linewidth/EISF conventions are explicit in UI, tables, plots, projects, and
  exports;
- sequential batch fits remain independent per Q;
- project save/reload reproduces the analysis state without executable data;
- private-data rules are enforced by design and test;
- the scientific validation plan passes on macOS Apple Silicon and supported
  Windows targets; and
- documentation makes limitations and unresolved scientific decisions visible.

## 13. Unresolved decisions

- Exact authoritative C2 and C4 equations and parameter definitions.
- Exact isotropic-reorientation equation and parameterization.
- Detailed accepted syntax within DAVE and generic ASCII layout families.
- Supported DAVE Q-bin parameter-file fields and their authoritative meanings.
- Blank-line/comment policy for explicit Q-list files.
- Default uniform convolution grid and interpolation method.
- Exact AIC/AICc likelihood convention and covariance-estimation policy.
- Bragg-contamination warning heuristic.
- Final public packaging, licensing, and Windows installer technology.
- Final review and approval of the proposed project container format.

## 14. Risks and milestone dependencies

The largest risks are silent format misinterpretation, invalid resolution
tails, convolution artifacts, parameter non-identifiability, overstated motion
interpretation, loss of provenance, GUI/core coupling, and leakage of private
data. The mitigation is staged scientific validation with explicit warnings
and immutable originals.

The content detector and core table importers precede Q mapping and resolution
work. Resolution-grid processing precedes convolution. Convolution precedes
validated single-spectrum fitting.
Single-spectrum validation precedes batch fitting and derived quantities.
Experimental EISF precedes motion-model fitting. Minimal in-memory contracts
evolve through the scientific milestones; full persistence is reviewed and
implemented in milestone 8 and must not delay milestones 1–6. GUI
implementation starts only after the importer, resolution, convolution, and
single-spectrum path are scientifically validated.
