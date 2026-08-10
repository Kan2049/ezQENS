# Development Roadmap

## 1. Roadmap philosophy

QENSfit is developed incrementally as a small, focused, scientifically transparent QENS analysis toolkit.

The roadmap follows **rolling refinement**:

* completed milestones retain a concise record of validated behavior;
* the next active milestone is specified in enough detail to implement safely;
* near-term milestones describe scientific goals and validation boundaries;
* distant capabilities remain intentionally concise and provisional.

Detailed implementation decisions belong in the milestone-specific specification written immediately before implementation, not in the long-term roadmap.

The guiding principles are:

1. Validate scientific behavior before adding interface breadth.
2. Preserve original numerical data and make analysis decisions explicit.
3. Keep the scientific core independent of GUI and source-specific infrastructure.
4. Add abstractions only when required by real implementations.
5. Prefer a small conceptual surface over speculative extensibility.
6. Design extension seams, not extension scaffolding.
7. Future data sources should converge on the common `ReducedDataset` / `Spectrum` boundary.
8. Scientific visualization should evolve alongside the analysis core rather than being postponed until the final GUI.
9. Desktop GUI development should begin only after a complete single-Q scientific fitting path has been validated.

When a milestone is completed, it is scientifically validated and frozen before development proceeds. The roadmap is then reviewed and may be adjusted according to newly discovered research needs.

---

## 2. Milestone 0 — Scientific and architectural baseline ✅

Established the initial:

* product requirements;
* scientific conventions;
* architecture principles;
* validation strategy;
* privacy rules;
* Python package and development tooling.

Scientific conventions that remain unresolved stay explicitly unresolved rather than being guessed or encoded.

---

## 3. Milestone 1 — Reduced-data core ✅

Completed and validated:

* immutable reduced `Spectrum` representation;
* source-independent `ReducedDataset` scientific boundary;
* explicit sample/resolution roles;
* DAVE group-block import;
* wide `x/yN/yerrN` import;
* single `x/y/yerr` import;
* content-based format detection;
* invalid-value classification without modifying originals;
* privacy-safe structural summaries;
* boundary-padding detection with `AUTO`, `REVIEW`, and `NONE`;
* reversible point masks.

### Milestone 1.2 — Core simplification ✅

The initial implementation was simplified before further feature growth.

The simplification:

* removed duplicated and derivable state;
* narrowed the public API;
* removed premature confidence and confirmation machinery;
* simplified boundary-padding heuristics;
* replaced confidence terminology with behavioral `AUTO`, `REVIEW`, and `NONE` states;
* removed unsupported sigma-transition, intensity-sign, relative-run, and regular-run heuristics;
* reduced runtime dependencies to those required by current production functionality;
* preserved validated automatic padding masks on available private benchmark data;
* retained a clean `ReducedDataset` / `Spectrum` boundary for future data sources and reduction workflows.

Milestone 1 forms the stable reduced-data baseline for later analysis.

---

## 4. Milestone 2 — Q assignment and fitting-data selection

This is the next implementation milestone.

### Goal

Convert an ordered `ReducedDataset` into analysis-ready QENS spectra with:

* explicit Q identity;
* explicit point-level fitting selection;
* explicit whole-spectrum inclusion/exclusion.

The conceptual workflow is:

```text
ReducedDataset
    ↓
Q assignment
    ↓
point-level fitting selection
    ↓
whole-Q inclusion/exclusion
    ↓
analysis-ready QENS data
```

### 4.1 Q assignment

Initially support only the simplest validated Q assignment methods:

* explicit ordered Q values;
* inclusive linear Q generation equivalent to `numpy.linspace`.

Each sample spectrum must resolve to exactly one finite Q value.

Never silently:

* sort;
* deduplicate;
* truncate;
* pad;
* extrapolate;
* interpolate;
* combine;
* or otherwise repair a Q-count mismatch.

Q assignment should remain logically separable from the numerical `Spectrum` representation so that future data sources or Q-rebin transformations can supply new Q mappings without redesigning the spectral core.

Do not implement DAVE Q-parameter parsing, detector geometry, angle-to-Q conversion, imported-Q heuristics, or generalized Q-source frameworks unless a real supported input requires them.

### 4.2 Point-level fitting selection

Combine only the analysis decisions needed for later spectral fitting:

* existing invalid-value masks;
* existing automatic padding mask;
* existing padding review mask;
* manual point masks;
* fitting-energy range.

Each source of exclusion remains separately inspectable.

An effective fitting mask may be derived from these states, but the original reasons must not be collapsed into one irreversible mask.

Original numerical arrays remain unchanged.

### 4.3 Whole-Q spectral-fit selection

Support manual inclusion or exclusion of complete Q spectra from spectral fitting.

This is required for real QENS situations where a particular Q may be unsuitable for quasielastic analysis, for example because of strong diffraction or Bragg-related elastic intensity.

Whole-Q exclusion:

* must not delete the spectrum;
* must be reversible;
* must remain distinct from point-level masks;
* initially remains user-controlled rather than automatically inferred.

Automatic Bragg/diffraction detection is not part of this milestone.

### 4.4 Q rebinning

Generic Q rebinning is not implemented in the initial reduced-text workflow.

For final reduced `S(Q,E)` spectra, scientifically correct rebinning may require information that is no longer available, such as:

* original detector contributions;
* Q-bin boundaries;
* detector coverage;
* normalization weights;
* geometry or event information.

Future intermediate or raw-data workflows may retain enough information to construct new Q bins correctly.

The architecture should therefore allow a future transformation conceptually like:

```text
source/intermediate/raw data
        ↓
Q grouping or Q rebinning
        ↓
ReducedDataset + Q assignment
        ↓
existing analysis core
```

Do not introduce `QRebinner`, detector-weight, Q-edge, registry, or instrument abstractions before a concrete supported use case exists.

### 4.5 Scientific visualization

Milestone 2 begins the reusable scientific visualization layer.

Initial visualization should support inspection of:

* spectra as a function of Q and energy;
* individual Q spectra;
* fitting-energy ranges;
* invalid points;
* automatic padding;
* review padding;
* manually excluded points;
* whole-Q inclusion/exclusion.

Where practical, an `S(Q,E)` overview or contour representation should make anomalous Q regions visually identifiable.

Visualization must remain usable from Python/Jupyter and must not depend on a desktop GUI.

### Validation

Validate:

* exact Q/spectrum count matching;
* Q-order preservation;
* finite-Q requirements;
* inclusive linear mapping;
* independent mask sources;
* effective fitting-point selection;
* whole-Q reversible exclusion;
* immutable original arrays;
* compatibility with manually constructed/source-independent `ReducedDataset` values;
* visualization consistency with the underlying analysis state.

---

## 5. Milestone 3 — Measured-resolution preparation

### Goal

Prepare measured resolution data for numerical convolution while preserving the original measurement.

Expected capabilities include:

* resolution valid-range selection;
* integration with existing invalid and padding masks;
* optional explicitly requested baseline handling;
* unit-area normalization;
* energy-grid validation;
* association between sample and resolution spectra;
* preparation of the resolution representation needed by convolution.

Original and processed resolution data remain distinct.

Exact normalization, interpolation, association, and valid-range policies will be specified and scientifically validated immediately before this milestone is implemented.

Do not build speculative instrument-specific resolution frameworks.

### Scientific visualization

Add resolution-specific inspection including:

* original measured resolution;
* selected valid range;
* invalid/padding regions;
* optional baseline treatment;
* normalized resolution;
* sample-resolution association.

These plots form part of scientific validation and must remain accessible outside the GUI.

---

## 6. Milestone 4 — Resolution convolution

### Goal

Provide a validated GUI-independent numerical convolution core suitable for QENS fitting.

The convolution path must support:

* measured resolution functions;
* explicit treatment of sample/resolution grid differences;
* appropriate internal numerical grids;
* interpolation of the resolution and theoretical model where required;
* protection against circular FFT wrap-around;
* preservation of integrated-area semantics;
* evaluation of the final model on the original sample measurement grid.

The original sample data are never interpolated merely for residual evaluation.

The elastic component follows the approved measured-resolution convention rather than using an artificial grid-dependent numerical delta spike.

Detailed grid construction, interpolation, centering, padding, cropping, and numerical tolerances will be specified and independently validated at implementation time.

### Scientific visualization

Add numerical diagnostics capable of comparing:

* measured resolution;
* theoretical unconvolved model;
* convolved model;
* model evaluated on the original sample grid.

Visualization should help detect grid, normalization, centering, or convolution-boundary problems.

---

## 7. Milestone 5 — Single-Q free fitting

### Goal

Fit one selected QENS spectrum reliably using the validated measured-resolution and convolution path.

The first useful free-fit model should support:

* elastic contribution;
* one or more quasielastic Lorentzian components;
* simple background;
* explicit parameter initial values;
* bounds;
* fixed/free parameter state;
* weighted fitting using valid uncertainties.

The first accepted implementation prioritizes:

* reliable parameter recovery;
* residual inspection;
* convergence information;
* physically meaningful FWHM output;
* transparent failure behavior.

Additional statistical diagnostics such as advanced covariance treatment, information criteria, or model-quality heuristics are added only after their conventions are reviewed and a real workflow requires them.

### Scientific visualization

Provide reusable plots for:

* measured spectrum;
* total fitted model;
* individual model components where useful;
* measured resolution;
* residuals;
* fitted range and excluded points;
* parameter/result summary.

This milestone establishes the first complete scientific workflow:

```text
load reduced data
    ↓
assign Q
    ↓
select fitting data
    ↓
prepare resolution
    ↓
resolution convolution
    ↓
single-Q free fit
    ↓
inspect result
```

---

## 8. Desktop GUI prototype begins after Milestone 5

Once the single-Q fitting path passes scientific validation, development of the interactive desktop workflow may begin.

The GUI must remain a thin interface over the validated scientific core.

The first prototype should focus on the existing complete workflow rather than implementing future features.

Expected interaction areas include:

* sample and resolution loading;
* spectrum/Q overview;
* fitting-range and mask inspection;
* whole-Q inclusion/exclusion;
* resolution inspection;
* single-Q model configuration;
* parameter editing;
* fit execution;
* fitted-curve and residual visualization.

Scientific formulas, numerical convolution, masking rules, fitting logic, and data transformations must remain outside GUI controllers.

Core analysis must continue to function without the desktop GUI.

---

## 9. Milestone 6 — Multi-Q free fitting and derived QENS quantities

### Goal

Apply the validated single-Q free fit independently across Q.

Expected outputs include:

* per-Q spectral fits;
* FWHM versus Q;
* relaxation times derived from the approved linewidth convention;
* experimental EISF from fitted integrated component areas;
* transparent handling of failed, excluded, or manually refitted Q points.

Batch fitting remains a sequence of independent free fits rather than an implicit global fit.

Detailed fit seeding, refit state, exclusion state, fit-quality summaries, and derived-uncertainty policies will be specified when this milestone becomes active.

### Scientific visualization

Add analysis-level views such as:

* Q-by-Q fit navigation;
* fit-quality overview;
* FWHM(Q);
* relaxation-time trends;
* EISF(Q);
* residual overview;
* included/excluded Q states.

After Milestone 6, the desktop GUI may become the primary interactive workflow while Python/Jupyter usage remains fully supported.

---

## 10. Later scientific capabilities — provisional

The following directions are expected but intentionally not fully specified.

Their order may change according to scientific need.

### Motion-model analysis

Potential capabilities include:

* XYZ molecular structure input;
* hydrogen-coordinate handling;
* scientifically approved rotational/reorientational models;
* fitting theoretical EISF models against experimental EISF;
* comparison of candidate motion models.

Candidate models such as C2, C4, isotropic reorientation, or later alternatives are implemented only after their equations, parameter meanings, and reference cases are scientifically reviewed.

### Additional QENS analysis tools

Possible future additions may include:

* temperature-dependent analysis;
* Arrhenius analysis;
* additional linewidth/diffusion models;
* additional EISF or dynamics models;
* scientifically justified auxiliary QENS analysis.

These capabilities are added only when concrete research use cases require them.

---

## 11. Future data-source and reduction capabilities — provisional

QENSfit should remain capable of expanding upstream from reduced-data analysis toward raw-data workflows.

Possible future sources include:

* HDF5-based facility data;
* NeXus;
* Mantid workspaces;
* intermediate reduced detector spectra;
* instrument-specific raw or partially reduced formats.

The architectural target is:

```text
raw/source-specific data
        ↓
source-specific reader / reduction
        ↓
optional detector grouping / Q rebinning
        ↓
ReducedDataset
        ↓
existing QENS analysis core
```

Raw HDF/HDF5 is not treated as one universal scientific format.

Future support should be based on concrete documented schemas and instrument workflows rather than a speculative generic HDF importer.

No raw-data adapter framework, instrument registry, plugin system, universal reduction abstraction, or Mantid abstraction is implemented before a real supported source requires it.

The existing reduced-data workflow must remain usable independently of future raw-data capabilities.

---

## 12. Future application capabilities — provisional

Possible later product work includes:

* richer scientific plotting;
* publication-oriented export;
* project save/reload;
* reproducible analysis records;
* expanded desktop workflows;
* macOS and Windows packaging;
* public release and documentation.

These capabilities consume the validated scientific core rather than redefine scientific behavior.

Project-container, migration, packaging, security, and long-term persistence architecture are designed only when those capabilities become active.

They are not current scientific-core requirements.

---

## 13. Cross-cutting requirements

At every active milestone:

* preserve original imported or reduced numerical values;
* make scientific transformations explicit;
* keep invalid data distinct from analysis exclusions;
* avoid silent repair or unsupported inference;
* maintain explicit units and scientific conventions;
* use synthetic tests for public validation;
* keep private experimental data out of committed tests and logs;
* run pytest, Ruff, and strict mypy;
* use coverage as a development-quality check where appropriate;
* perform private scientific regression validation when relevant;
* provide an owner-facing validation procedure for real data;
* avoid adding dependencies before corresponding functionality exists.

When a milestone reveals that an earlier architectural assumption is unnecessary or over-engineered, simplify before building additional layers on top of it.

Scientific visualization should be treated as part of validation and usability, not merely final presentation.

---

## 14. Milestone completion and freeze procedure

Every milestone follows the same completion cycle:

```text
implementation
    ↓
automated tests
    ↓
synthetic/numerical validation
    ↓
owner validation with real data
    ↓
scientific review
    ↓
freeze
    ↓
commit and push stable baseline
    ↓
roadmap review
    ↓
next milestone specification
```

Before a milestone is frozen:

1. automated tests and static checks must pass;
2. milestone-specific scientific tests must pass;
3. relevant real/private data should be inspected when available;
4. the owner receives explicit instructions for manual/Jupyter validation;
5. unexpected scientific differences must be understood rather than silently accepted.

After freezing a milestone, review this roadmap with the scientific owner before beginning the next milestone.

Newly discovered research needs may:

* change later milestone order;
* add or remove planned capabilities;
* split or merge future milestones;
* promote a provisional capability into the near-term roadmap.

A frozen milestone should not be reopened merely to anticipate distant functionality unless a genuine architectural or scientific blocker is discovered.

---

## 15. Roadmap maintenance

This roadmap is intentionally not a complete version-1 implementation specification.

When a milestone becomes the next active milestone:

1. review current scientific requirements;
2. inspect the actual validated code;
3. collect relevant real workflow requirements;
4. resolve only the scientific decisions needed immediately;
5. write a short milestone-specific implementation specification;
6. implement;
7. test;
8. validate with real data;
9. simplify if necessary;
10. freeze the milestone;
11. review and update the roadmap.

Distant milestone details are provisional and must not be treated as frozen API, dependency, file-format, or architecture contracts.
