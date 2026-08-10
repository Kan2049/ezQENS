# Validation Notebook Maintenance After Each Milestone

The project's Jupyter validation notebook is a persistent human-facing integration test and scientific demonstration of the current `qensfit` API.

It must evolve together with production code.

At the completion of every development milestone or sub-milestone that changes public behavior, perform the maintenance workflow below.

The notebook must never be assumed to remain valid merely because unit tests pass.

---

# 1. Purpose of the Validation Notebook

The validation notebook serves as a human-readable integration layer between:

* production API
* reduced experimental data
* preprocessing
* scientific diagnostics
* intermediate results
* plots
* expected scientific behavior

It is not the authoritative implementation.

Production code under `src/qensfit/` is the source of truth.

The notebook must call production APIs rather than reproduce production algorithms.

The notebook is allowed to contain:

* configuration
* orchestration
* privacy-safe summaries
* sanity checks
* assertions
* plots
* scientific interpretation of outputs

The notebook should not contain duplicate implementations of algorithms that belong in the package.

---

# 2. Mandatory Trigger

Run this maintenance workflow after every milestone or sub-milestone that changes any of the following:

* public dataclasses
* result objects
* enums
* function signatures
* importer behavior
* diagnostics
* preprocessing
* masking
* spectrum representation
* Q metadata
* resolution representation
* convolution interfaces
* fitting interfaces
* model parameters
* result tables
* plotting inputs
* serialization
* validation semantics

Also run it whenever repository tests pass but the validation notebook has not been executed since the latest API change.

---

# 3. Source-of-Truth Rule

Current production implementation is authoritative.

When the notebook conflicts with production API:

1. inspect the production implementation
2. inspect architecture/data-model/scientific-convention documents
3. inspect relevant tests
4. determine whether the API change was intentional
5. update the notebook when production behavior is intentional

Do NOT:

* restore deleted fields merely for notebook compatibility
* add deprecated aliases solely for old notebook cells
* weaken production validation
* duplicate old behavior in notebook helper functions
* modify scientific semantics merely to preserve old notebook output

Production code may be changed only when an independent genuine bug is identified.

---

# 4. API Compatibility Audit

Before changing the notebook, compare every API used by the notebook with its current production definition.

Check for:

* removed fields
* renamed fields
* new fields
* changed properties
* changed methods
* changed return objects
* changed tuple/list structures
* changed enum names or values
* changed optionality
* changed diagnostics
* changed mask semantics
* changed function parameters
* changed defaults
* changed array shape assumptions
* changed group/Q handling
* changed error behavior

Build an internal map:

| Notebook API usage | Current API | Compatibility | Action |
| ------------------ | ----------- | ------------- | ------ |

Do not limit this audit to APIs changed explicitly in the milestone description.

A milestone may indirectly invalidate downstream notebook cells.

---

# 5. Semantic Compatibility

Do not perform blind name replacement.

For every changed API determine the scientific or diagnostic meaning of the old notebook operation.

Preserve the intent using the new model.

Example principle:

Old notebook:

```python
old_result.some_mask
```

Current production:

```python
new_result.some_other_representation
```

Do not assume these are equivalent solely because both involve masks.

Inspect their semantics and construct the current correct validation.

If behavior changed intentionally, update both code and Markdown explanation.

---

# 6. Notebook Structure

Keep the notebook organized approximately as:

## A. Environment / import validation

Confirm the notebook is using the intended development package.

Do not expose full absolute private paths in displayed output.

## B. Configuration

Define:

* input data
* role
* optional format override
* Q metadata
* relevant milestone settings

## C. Import / detection

Exercise the real package importer and detection APIs.

## D. Privacy-safe diagnostics

Display concise structural information without dumping full arrays.

## E. Preprocessing

Exercise current preprocessing APIs.

## F. Scientific sanity checks

Check physically/scientifically meaningful behavior.

## G. Visual validation

Plot selected spectra/intermediate/final results.

## H. Milestone-specific validation

Add a clearly identified section for newly completed functionality.

Avoid accumulating multiple obsolete versions of the same demonstration.

Update or replace the old section when the public API evolves.

---

# 7. Privacy-Safe Output

Displayed notebook output must avoid:

* absolute local file paths
* usernames
* unrelated directory structure
* full raw experimental arrays
* unnecessary file metadata

Prefer:

* basename
* spectrum/group count
* Q values when scientifically relevant
* shape
* finite-point counts
* masked-point counts
* energy range
* diagnostic codes
* current public result fields
* aggregate scientific statistics

---

# 8. Milestone Integration Rule

For every newly completed milestone, determine:

1. What new user-visible capability exists?
2. What current public API exposes it?
3. What minimal realistic notebook example demonstrates it?
4. What scientific sanity check should pass?
5. What plot or concise diagnostic makes the behavior inspectable?
6. Which older notebook cells are now redundant or obsolete?

Add validation for the new milestone without turning the notebook into a second test suite.

Unit tests remain responsible for exhaustive edge cases.

The notebook should emphasize integrated realistic usage.

---

# 9. Regression Check

Every milestone update must also verify existing notebook sections still work.

At minimum re-check:

* imports
* format detection
* sample import
* resolution import where applicable
* Q/group mapping
* preprocessing
* diagnostics
* plots
* all previously integrated milestone outputs

Do not validate only the newly added feature.

---

# 10. Clean-Kernel Requirement

The notebook must be validated from a clean kernel.

Preferred procedure:

1. restart kernel
2. Run All
3. confirm cells do not rely on stale variables
4. confirm cells do not depend on manual out-of-order execution

If package modules changed during development, restart the kernel before final validation.

An API error caused by stale imports must not be "fixed" by altering production code.

---

# 11. Package Location Check

Retain a lightweight development-package check near the beginning of the notebook.

It should establish that the notebook imports the intended repository version of `qensfit`.

Displayed output must remain privacy-safe.

If the notebook is importing an unrelated installed `site-packages` copy rather than the current development package, fix the development environment/import configuration rather than modifying APIs.

---

# 12. Relationship to Unit Tests

The notebook does not replace unit tests.

After updating the notebook:

* run relevant tests
* run the full test suite when practical
* run lint/type checks where part of the normal project workflow

If the notebook exposes a production defect not covered by tests:

1. fix the production defect
2. add or update a unit/regression test
3. then update the notebook

Do not rely only on the notebook to guard production behavior.

---

# 13. Markdown Documentation

Update explanatory Markdown whenever APIs or behavior change.

Remove descriptions of obsolete concepts.

The notebook should never teach a user to use an API that no longer exists.

For milestone-specific sections, explain:

* what is being tested
* what API is being used
* what scientific behavior is expected
* what output should be inspected

Keep explanations concise.

---

# 14. Failure Handling

If notebook execution fails:

First classify the failure as one of:

* stale notebook API
* stale Jupyter import/kernel
* wrong Python environment
* genuine production bug
* missing/private validation data
* optional dependency issue
* expected behavior change
* notebook-only error

Do not modify code before identifying the category.

For API failures:

inspect current production source first.

For scientific failures:

do not force output to match historical behavior without checking current scientific conventions and requirements.

---

# 15. Required Maintenance Report

After each milestone, report:

## Milestone

* milestone name
* user-visible functionality added or changed

## API changes

* new public APIs
* changed APIs
* removed APIs
* notebook usages affected

## Notebook changes

* cells added
* cells changed
* cells removed
* Markdown updated
* plots changed

## Compatibility

* obsolete API references found
* stale assumptions corrected

## Validation

* clean-kernel Run All result
* scientific sanity checks
* visual checks
* unresolved issues

## Tests

* relevant tests
* full suite if run
* lint/type checks if run

## Production changes caused by notebook audit

Normally: none.

If not none, explain the genuine production bug and corresponding regression test.

---

# 16. Completion Criteria

A milestone is not considered fully integrated into the validation notebook until:

* the notebook uses the current production API
* obsolete API references have been removed
* the new milestone has an appropriate integration demonstration
* previous notebook sections still execute
* explanatory Markdown matches current behavior
* the notebook runs from a clean kernel, when validation data are available
* relevant tests pass
* unresolved limitations are documented

The goal is:

**one maintained notebook representing the current integrated state of the project, not a historical collection of milestone-specific code fragments.**
