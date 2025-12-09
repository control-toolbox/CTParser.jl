# 📊 Status Report: Issue #185 - [Dev] Tests

**Date**: 2025-12-09 | **State**: Open | **Repo**: control-toolbox/CTParser.jl  
**PR**: None (no PR referencing `#185` found; local branch `185-dev-tests` in progress)

---

## 📋 Summary
Internal dev issue to reinforce and extend the test suite of `CTParser.jl`.
The goal is to add new unit and integration tests without touching existing ones, by creating parallel `_bis` test files and enforcing a clear separation of concerns and struct naming conventions.

**Created**: 2025-12-09T14:13:41Z  
**Updated**: 2025-12-09T14:13:57Z  
**Labels**: `internal dev`

---

## 💬 Discussion

The issue description (no comments yet) states:

- Make a pass on **unit and integration tests**.
- **Do not replace** existing tests.
- For each existing test file, **create a new file with `_bis` suffix** and add new tests there, avoiding duplication.
- Ensure a **clear separation** between unit and integration tests in the files.
- For tests of the logic, when **fake structs** are needed:
  - Define them at **top level** in the test file.
  - Place them at the **top of the file, before the test function**.
  - Use **unique struct names per file** to avoid ambiguity.

There are currently **no comments** or extra constraints beyond this description.

**Key Decisions (from issue text)**:
- Existing tests are kept as-is; new work is additive via `_bis` files.
- File-level separation (original vs `_bis`) is the main mechanism to distinguish old vs new tests.
- Struct-based test fixtures must be explicit, top-level and uniquely named.

**References**:
- Issue: https://github.com/control-toolbox/CTParser.jl/issues/185

---

## ✅ Completed (Current State, Before Issue #185 Work)

- ✓ **Baseline tests for utilities**  
  - `src/utils.jl` ↔ `test/test_utils.jl`  
  - Covers `subs`, `subs2`–`subs5`, `replace_call`, `has`, `concat`, `constraint_type` with many syntactic variants.

- ✓ **Baseline tests for defaults & prefixes**  
  - `src/defaults.jl` and prefix helpers in `src/onepass.jl` ↔ `test/test_prefix.jl`.  
  - Check default backend / scheme / grid size / init / base type / prefixes, and runtime setters `prefix_fun!`, `prefix_exa!`, `e_prefix!`.

- ✓ **Extensive functional backend tests for `@def` → CTModels**  
  - `src/onepass.jl` (functional backend) ↔ `test/test_onepass_fun.jl`.  
  - Large suite of tests: syntax, aliases, variables, time handling, state/control declarations, dynamics, constraint classification, error paths, etc.  
  - These are mostly **integration/functional tests** through `CTModels.Model` and getters (e.g. `initial_time`, `state_dimension`, etc.).

- ✓ **Extensive ExaModels backend tests for `@def` / `@def_exa`**  
  - `src/onepass.jl` (ExaModels backend) ↔ `test/test_onepass_exa.jl`.  
  - Integration tests for `activate_backend(:exa)`, mock `discretise_exa` / `discretise_exa_full`, constraints, bounds, error paths, and several realistic use cases (Mayer, Lagrange, Bolza, Goddard, quadrotor, etc.).

- ✓ **Meta-tests via Aqua.jl**  
  - `test/test_aqua.jl` runs `Aqua.test_all(CTParser; ...)` and ambiguity checks, providing a quality gate on the module as a whole.

**Conclusion**: The project already has **substantial coverage**, especially at the level of parser+backend integration, but there is **no dedicated `_bis` layer** and little explicit separation between “unit-style” and “integration-style” tests inside individual files.

---

## 📝 Pending Actions

### 🔴 Critical

**1. Define the test strategy and layout for `_bis` files**
- **Why**: Issue #185 mandates a non-destructive extension of tests using `_bis` files. A consistent convention is needed before adding concrete tests.
- **Where**: Test tree & documentation (no code change yet):
  - `test/test_utils_bis.jl`
  - `test/test_prefix_bis.jl`
  - `test/test_onepass_fun_bis.jl`
  - `test/test_onepass_exa_bis.jl`
- **Complexity**: Moderate (requires design agreement more than raw coding).

**2. Create `_bis` files per existing test file and route new tests there**
- **Why**: Satisfies the “do not replace existing ones” requirement and keeps legacy tests intact. `_bis` files become the canonical place for **new** tests.
- **Where**:
  - For each existing file:
    - `test/test_utils.jl` → `test/test_utils_bis.jl`
    - `test/test_prefix.jl` → `test/test_prefix_bis.jl`
    - `test/test_onepass_fun.jl` → `test/test_onepass_fun_bis.jl`
    - `test/test_onepass_exa.jl` → `test/test_onepass_exa_bis.jl`
    - (Optionally) `test/test_aqua.jl` → `test/test_aqua_bis.jl` only if additional Aqua-related tests are desired.
- **Complexity**: Moderate (mechanical to set up, but requires carefully chosen new cases to avoid duplication).

**3. Enforce struct/fixture conventions in new `_bis` tests**
- **Why**: The issue explicitly requires fake structs to be defined at top level with unique names to avoid ambiguity.
- **Where**: Top of each `_bis` file, before the main `test_*` function.
  - Example pattern (conceptual, not yet implemented):
    - Top of `test_onepass_fun_bis.jl`: `struct OnepassFunTestCase1 ... end`
    - Top of `test_onepass_exa_bis.jl`: `struct ExaBackendCase1 ... end`
- **Complexity**: Simple (convention and naming discipline) but critical for long-term maintainability.

### 🟡 High

**4. Strengthen unit-style coverage for internal parsing helpers**
- **Why**: Many internal helpers in `src/onepass.jl` (e.g. `p_time!`, `p_state!`, `p_control!`, `p_dynamics!`, `p_lagrange!`, `p_mayer!`, `p_pragma!`, `p_constraint!`, and their `_fun` / `_exa` variants) are indirectly exercised by integration tests, but not isolated in small, focused unit tests.
- **Where**:
  - `test/test_onepass_fun_bis.jl`: unit-style tests calling parser entry points with minimal examples and checking the resulting `ParsingInfo` / intermediate structures.
  - `test/test_onepass_exa_bis.jl`: unit-style checks focusing on ExaModels-specific branches and error messages.
- **Complexity**: Complex (these helpers are tightly coupled to the parser DSL and backends; good unit tests need carefully chosen minimal OCPs).

**5. Clarify and encode the split “unit vs integration” in test organization**
- **Why**: Currently the distinction is implicit (e.g. `test_onepass_exa.jl` is clearly integration-level). For newcomers, explicit naming or comments would be helpful.
- **Where**:
  - At the top of each `_bis` file: short comment indicating whether it contains **unit-only**, **integration-only**, or a deliberately mixed set.
  - Optionally, grouping `@testset` names to reflect this split.
- **Complexity**: Simple to moderate.

### 🟢 Medium

**6. Extend `test_utils_bis.jl` with edge cases and regression tests**
- **Why**: `test_utils.jl` already covers many expression-transform cases, but `_bis` can accumulate targeted regression tests for bugs discovered later and tricky syntactic patterns (nested calls, exotic Unicode symbols, etc.).
- **Where**: `test/test_utils_bis.jl`.
- **Complexity**: Simple (tests are pure and local).

**7. Add tests focusing specifically on prefix / backend activation lifecycle**
- **Why**: `test_prefix.jl` covers defaults and setters but `_bis` can check interaction with `activate_backend`, `deactivate_backend`, and `is_active_backend` (e.g. multiple backends, resetting state between tests).
- **Where**: `test/test_prefix_bis.jl`.
- **Complexity**: Moderate (requires careful isolation so that backend activation does not leak between tests).

### 🔵 Low

**8. Optional: add more Aqua / meta tests**
- **Why**: Ensure that new `_bis` files do not introduce piracies or dependency issues and that code quality remains high.
- **Where**: `test/test_aqua_bis.jl` (if needed) or simply keep everything in `test_aqua.jl`.
- **Complexity**: Simple.

---

## 🔧 Technical Analysis

**Code Findings (tests vs sources)**

- **Project structure**:
  - `src/CTParser.jl` (main module)
  - `src/defaults.jl` (defaults for backends, schemes, prefixes)
  - `src/onepass.jl` (core parser for `@def` / `@def_exa` and backends)
  - `src/utils.jl` (expression utilities and constraint classification)

- **Existing tests**:
  - `test/test_utils.jl` → **unit-style** tests for utilities.
  - `test/test_prefix.jl` → small **unit-style** tests for defaults and prefixes.
  - `test/test_onepass_fun.jl` → large **functional/integration** suite for `@def` with CTModels backend.
  - `test/test_onepass_exa.jl` → large **integration** suite for ExaModels backend, including real OCPs solved with MadNLP/MadNLPGPU.
  - `test/test_aqua.jl` → meta-tests via Aqua.

- **Mapping (1 code file → N tests)**:
  - `src/utils.jl` → `test_utils.jl` (unit).
  - `src/defaults.jl` + prefix helpers in `src/onepass.jl` → `test_prefix.jl` (unit).
  - `src/onepass.jl` → `test_onepass_fun.jl`, `test_onepass_exa.jl` (integration + partial unit behavior).

**Julia Standards**

- ✅ **Documentation structure**: `docs/` uses Documenter + CTBase’s `automatic_reference_documentation`; internal API docs are generated automatically.
- ✅ **Testing**:
  - Tests are wired in `test/runtests.jl` via a top-level `@testset` iterating over test names (`:aqua`, `:utils`, `:prefix`, `:onepass_fun`, `:onepass_exa`).
  - Structure is conventional and easy to extend with new `_bis` files.
- ✅ **Package structure**: `src/` + `test/` + `Project.toml` follow standard Julia package layout.
- ⚠️ **Type Stability / Performance**: Not explicitly analyzed here; existing integration tests already exercise realistic workloads (e.g. ExaModels + MadNLP), which gives some empirical confidence but not formal stability checks.

**Performance**

- Integration tests (especially in `test_onepass_exa.jl`) may be relatively heavy (MadNLP solves, GPU backends). `_bis` tests should be designed to **avoid unnecessary extra heavy workloads**, e.g. prefer light-weight unit tests where possible.

---

## 🚧 Blockers

At ce stade, il n’y a pas de blocage technique identifié dans le code ou l’infrastructure de tests. Les points suivants nécessitent toutefois une décision ou un accord de style :

1. ❓ **Granularité souhaitée pour les `_bis`**  
   - Strictement « unit » dans certains fichiers et « integration » dans d’autres, ou bien mélange contrôlé dans un même `_bis` ?

2. ❓ **Priorisation des zones à tester en premier**  
   - Faut‑il commencer par les helpers internes de parsing (`p_*` dans `onepass.jl`) ou par des cas d’usage supplémentaires (nouvelles OCPs) ?

Ces points ne bloquent pas l’écriture des premiers `_bis`, mais il est préférable de se mettre d’accord sur le style avant de multiplier les fichiers.

---

## 💡 Recommendations

**Immediate**
1. Valider la **convention `_bis`** proposée (un fichier `_bis` par fichier de test existant) et l’appliquer en créant des squelettes vides avec seulement un `@testset` structuré et quelques commentaires.
2. Commencer par `test_utils_bis.jl` et `test_prefix_bis.jl` (tests légers, purement unitaires) pour roder la convention et la revue.
3. Ajouter dans chaque `_bis` un en-tête clair indiquant :
   - le type de tests (unit vs integration),
   - les conventions de nommage pour structs factices.

**Long-term**
- Introduire progressivement des tests plus fins sur les chemins d’erreur du parser (`ParsingError`, `UnauthorizedCall`, combinaisons de contraintes invalides, etc.), en les isolant dans les `_bis` afin de ne pas alourdir les tests existants.
- Surveiller la durée totale de la suite de tests (en particulier `test_onepass_exa` et ses futurs `_bis`) et, si nécessaire, regrouper les tests les plus lourds dans un sous-ensemble marquable (par exemple via un flag ou un environnement de CI spécifique).

**Julia Alignment**
- La stratégie `_bis` permet de garder une **traçabilité historique** des tests tout en respectant la convention Julia « 1 module = plusieurs niveaux de tests », sans casser la compatibilité avec les outils existants (`Test`, `Aqua`, CI).

---

**Status**: Needs attention (design + implementation of `_bis` suite)  
**Effort**: Medium (structuration + ajout progressif de tests unitaires et d’intégration)
