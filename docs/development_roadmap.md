# Development Roadmap

## 1. Roadmap philosophy

ezQENS is developed incrementally as a lightweight, focused, scientifically
transparent QENS analysis application for experts and non-experts. It targets
ordinary CPU-based macOS and Windows laptops; Linux is best-effort.

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
10. For equally correct solutions, prefer fewer concepts, less code, fewer dependencies, lower resource use, and easier maintenance.

When a milestone is completed, it is scientifically validated and frozen before development proceeds. The roadmap is then reviewed and may be adjusted according to newly discovered research needs.

### 1.1 High-level progression

The detailed milestone numbers retain their historical meaning. At product
level, work progresses through four overlapping phases:

1. **Phase I — Scientific foundation:** import, scientific data model, Q
   assignment, selection/masking, resolution, convolution, single-Q fitting,
   independent batch fitting, and basic derived quantities. The scientific core
   becomes trustworthy.
2. **Phase II — Complete QENS analysis:** validated FWHM, relaxation time, EISF,
   component-Q outputs, selected temporal/spatial dynamics models, diagnostics,
   visualization, and scientific output. Individual Phase-II dynamics models
   may be scheduled before or after the first public release by later owner
   decision; no large catalogue is implied.
3. **Phase III — ezQENS application and v1.0 readiness:** progressively guided
   and advanced GUI workflows, usability, export, lightweight reproducibility,
   documentation, and macOS/Windows packaging. This culminates in the special
   v1.0 public-release gate.
4. **Phase IV — Expansion:** additional formats/instruments, richer output,
   additional dynamics, raw reduction, temperature workflows, molecular
   structure input, and advanced automation. These do not burden earlier
   architecture without a concrete requirement.

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

## 4. Milestone 2 — Q-bin assignment and fitting-data selection ✅

Completed and validated:

* immutable dataset-level `QBins` with optional explicit edges;
* midpoint representative Q values for edge-created bins;
* explicit ordered representative Q values without inferred edges;
* uniform bins from inclusive outer edges and authoritative group count;
* the approved four-value DAVE Q-bin parameter parser;
* independent inclusive fitting-energy ranges for every spectrum;
* derived invalid/`AUTO`/outside-range fitting-point exclusion;
* reusable Q-bin and individual-spectrum inspection plots.

The conceptual workflow is:

```text
ReducedDataset + QBins
    ↓
per-group FittingRange values
    ↓
derived FittingSelection masks
    ↓
scientific inspection
```

### 4.1 Q assignment

Q identity is stored once on `ReducedDataset`, never duplicated on each
`Spectrum`. Edge-defined bins are preferred: `N` spectra require `N + 1`
finite strictly increasing edges, and Milestone-2 representative values are
the adjacent-edge midpoints. Explicit Q values preserve arbitrary nonlinear
order and leave edges unknown.

Count-driven uniform construction treats lower/upper inputs as inclusive outer
edges and derives
`delta_Q = (upper_q_edge - lower_q_edge) / group_count`, exactly covering the
range. The DAVE four-value parser intentionally differs: it reconstructs
complete fixed-width bins from lower limit, upper limit, and step, retains the
stored group count for a warning-producing consistency check, and permits an
unused remainder below the source upper limit.

Never silently:

* sort;
* deduplicate;
* truncate;
* pad;
* extrapolate;
* interpolate;
* combine;
* or otherwise repair a Q-count mismatch.

Q assignment remains logically separable from `Spectrum`, allowing future
reduction workflows to produce the same `ReducedDataset + QBins` boundary.

### 4.2 Point-level fitting selection

Each group has an independent inclusive fitting-energy range. One range may be
used as the initial value for all groups and then overridden group by group.
The effective exclusion is derived from invalid values, `AUTO` padding, and
outside-range points. `REVIEW` remains visible but is not automatically
excluded. Each reason remains separately inspectable.

Original numerical arrays remain unchanged.

### 4.3 Deferred masking

Manual single-point masking, whole-Q spectral exclusion, and automatic
Bragg/diffraction detection are deferred until a concrete fitting or batch
workflow requires them.

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

Do not introduce `QRebinner`, detector-weight, geometry, registry, or instrument
abstractions before a concrete supported use case exists.

### 4.5 Scientific visualization

Milestone 2 begins the reusable scientific visualization layer.

Initial visualization supports Q edges/intervals/representatives and individual
spectra with uncertainties, fitting range, invalid points, `AUTO`/`REVIEW`
padding, outside-range points, and retained points.

Visualization must remain usable from Python/Jupyter and must not depend on a desktop GUI.

### Validation

Validate:

* exact Q/spectrum count matching;
* Q-order preservation;
* finite-Q requirements;
* uniform edge generation and midpoint representatives;
* independent mask sources;
* effective fitting-point selection;
* immutable original arrays;
* compatibility with manually constructed/source-independent `ReducedDataset` values;
* visualization consistency with the underlying analysis state.

---

## 5. Milestone 3 — Measured-resolution preparation (completed)

### Goal

Prepare measured resolution data for later numerical convolution while
preserving and referencing the immutable original measurement.

The completed implementation covers:

* exact ordered sample/resolution Q association, including known-edge checks;
* independent sample and resolution edge-padding detection;
* warning-only comparison of AUTO-retained energy boundaries;
* default resolution support derived from valid measured bounds;
* explicit immutable per-group support overrides distinct from sample ranges;
* default-on, reversible per-group `AUTO` application independent of support;
* default retention of `REVIEW` and non-overridable invalid energy/intensity;
* strict accepted-grid validation without interpolation or reordering;
* blocking diagnostics for internal invalid energy/intensity holes, with no
  deletion or trapezoidal bridging;
* independent per-Q unit-area trapezoidal normalization on actual coordinates;
* derived read-only normalized intensity and scaled uncertainty;
* explicit failure for unusable grids and nonfinite/nonpositive area; and
* minimal Python/Jupyter inspection of support, masks, Q association,
  normalization, retained-boundary comparison, and unchanged energy alignment.

The prepared value references source datasets/spectra and stores only support,
normalization, association, and diagnostic state. Normalized arrays are derived,
not persisted as duplicate scientific state. Sample spectra are not normalized.

M3 does not perform baseline subtraction, energy recentering, interpolation,
convolution, fitting, GUI work, or analytic-resolution fallback. Uniform-grid
construction and interpolation remain Milestone 4 responsibilities. A future
explicit baseline operation and analytic Gaussian/Lorentzian/ideal resolution
source remain deferred. Do not build speculative instrument or resolution-source
frameworks.

### Scientific visualization

Add resolution-specific inspection including:

* original measured resolution;
* accepted support and invalid/`AUTO`/`REVIEW` states;
* normalized resolution;
* exact sample-resolution Q association and retained-boundary comparison; and
* unchanged `E = 0` alignment inspection.

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

## 10. Special Milestone — ezQENS v1.0 First Public Release

This release gate follows validation of the reduced-data scientific core. It
passes only when an external real user, including a non-QENS expert, can use a
guided workflow to complete a reduced-data analysis end to end on an ordinary
macOS or Windows laptop, inspect warnings and results, and export the analysis.

Required release capabilities are:

* a guided, responsive desktop GUI that delegates all science to the same
  GUI-independent Python core;
* useful scientific export;
* lightweight reproducibility information identifying inputs, selections and
  masks, model and fit settings, results, warnings, and software version;
* clear user and scientific documentation;
* validated macOS and Windows packages; and
* real-workflow validation centered on DAVE- and Mantid-preprocessed data,
  including representative PSI FOCUS and ILL IN5/IN16 workflows where
  available, without instrument-specific core assumptions.

Detailed GUI, export, reproducibility, and packaging milestones are defined
through rolling refinement when they become active. A complex project archive,
raw-data reduction, automatic molecular interpretation, and a large mandatory
dynamics-model catalogue do not gate v1.0. A selected dynamics model gates the
release only if the owner explicitly promotes it into release scope.

---

## 11. Later scientific capabilities — provisional

The following directions are expected but intentionally not fully specified.

Their order relative to the first public release may change according to
scientific need and explicit owner decisions.

### Selected QENS dynamics analysis

Potential scientifically approved temporal or spatial analyses include:

* diffusion and jump-diffusion models;
* characteristic or residence times;
* rotational or reorientational dynamics;
* EISF geometry models;
* confined or localized motion; and
* other concrete QENS dynamics models justified by real workflows.

Each model is added locally only after its equations, parameters, references,
diagnostics, and validation cases are approved. Do not create a generic model
registry or mandatory catalogue.

### Automatic molecular or structure-driven analysis

Potential capabilities include:

* XYZ molecular structure input;
* hydrogen-coordinate handling;
* scientifically approved rotational/reorientational models;
* fitting theoretical EISF models against experimental EISF;
* comparison of candidate motion models.

Automatic generation or inference of candidates from coordinates, symmetry, or
molecular structure is a substantially later capability. Candidate models such
as C2, C4, isotropic reorientation, or later alternatives are implemented only
after their equations, parameter meanings, and reference cases are
scientifically reviewed.

### Additional QENS analysis tools

Possible future additions may include:

* temperature-dependent analysis;
* Arrhenius analysis;
* additional linewidth/diffusion models;
* additional EISF or dynamics models;
* scientifically justified auxiliary QENS analysis.

These capabilities are added only when concrete research use cases require them.

---

## 12. Post-v1.0 data-source and reduction capabilities — provisional

ezQENS may later expand upstream from reduced-data analysis toward raw-data
workflows. Raw reduction is not required for v1.0.

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

## 13. Post-v1.0 application capabilities — provisional

Possible later product work includes:

* richer scientific plotting;
* publication-oriented export;
* richer project save/reload beyond v1.0 lightweight reproducibility;
* expanded desktop workflows;
* additional platform packaging.

These capabilities consume the validated scientific core rather than redefine scientific behavior.

The final project-container format and extension remain unresolved. Container,
migration, security, and long-term persistence architecture are designed only
when those capabilities become active; the former `.qensfit` proposal is not a
permanent contract.

They are not current scientific-core requirements.

---

## 14. Cross-cutting requirements

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

## 15. Milestone completion and freeze procedure

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

## 16. Roadmap maintenance

This roadmap is intentionally not a complete v1.0 implementation specification.

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
