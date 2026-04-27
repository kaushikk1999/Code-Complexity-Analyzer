# Complexity Lab

Complexity Lab is a Streamlit application for Python data science coding interview preparation. Paste a Python solution, estimate its time and space complexity, run guarded benchmarks, inspect bottlenecks, and generate interview-ready optimization feedback.

The app is designed as a polished portfolio-grade dashboard rather than a plain Streamlit prototype. It uses Streamlit for the app shell, custom CSS for premium dashboard styling, Plotly for charts, and small inline Vanilla JavaScript blocks for copyable report/interview summaries.

## Features

- AST-based static analysis for estimated time and space complexity.
- Per-function estimates with confidence, evidence, caveats, call graph, line-level findings, and detected anti-patterns.
- Interview-pattern recognition for brute force pair search, hash-map lookups, binary search, sliding window, prefix sums, heaps, recursion, memoized DP, sorting + two pointers, graph traversal, and data-science loop patterns.
- Confidence breakdown across syntax, structural signals, data-flow hints, pattern recognition, and caveat pressure.
- Best-effort benchmark execution with repeat timing, `perf_counter_ns`, `tracemalloc`, and timeout protection.
- Benchmark profiles, warmup runs, static-only mode, top-level execution opt-in, generated input-size scaling experiments, empirical curve fitting, and runtime/memory charts.
- Deterministic optimization quality score out of 100.
- Efficiency percentage, severity label, improvement potential, and ranked bottlenecks.
- Step-by-step optimization guidance grouped into quick wins, medium refactors, and advanced improvements, with rewrite confidence and validation tests where safe.
- Interview mode with follow-up questions, answer grading rubric, model answer, suggested explanation wording, and trade-off notes.
- Optional Gemini-enhanced narrative coaching grounded in local JSON facts.
- SQLite-backed practice history, named sessions, and progress dashboard.
- Downloadable markdown and HTML reports plus copyable LinkedIn/GitHub summary.
- Dark-mode-friendly custom theme, gradient hero, metric cards, badges, tabs, and responsive layout.

## Architecture

```text
app.py
analyzer/
  advanced_patterns.py
  anti_patterns.py
  ast_analyzer.py
  complexity_rules.py
  models.py
benchmarking/
  metrics.py
  runner.py
  sandbox.py
interview/
  coaching.py
llm/
  gemini_helper.py
optimization/
  planner.py
scoring/
  optimizer_score.py
visualization/
  charts.py
  styles.py
  ui_helpers.py
utils/
  constants.py
  examples.py
  history_store.py
  report_export.py
tests/
  test_analyzer.py
  test_benchmarking.py
  test_history_store.py
  test_reports.py
  test_scoring.py
streamlit_app.py
.github/workflows/ci.yml
pyproject.toml
```

Business logic is kept outside the Streamlit view layer:

- `analyzer/` owns AST feature detection and heuristic complexity estimates.
- `benchmarking/` owns guarded execution, timing, memory tracing, generated scaling experiments, and structured benchmark results.
- `scoring/` owns the deterministic optimization score.
- `optimization/` owns coaching, guidance, and safe pattern-based optimized suggestions.
- `interview/` owns local follow-up questions and answer grading.
- `visualization/` owns CSS, HTML helpers, and Plotly chart construction.
- `utils/history_store.py` owns SQLite-backed practice history.
- `llm/` contains optional Gemini narrative enhancement only.

## UI Design Approach

The UI uses Streamlit primitives plus a custom CSS layer:

- Gradient hero header.
- Glass-style cards and metric panels.
- Color-coded complexity badges and severity pills.
- Tabbed workflow for Static Analysis, Benchmarks, Optimization Plan, Interview Mode, Report, and Progress.
- Plotly charts with a dark dashboard theme.
- Inline Vanilla JavaScript copy buttons through `streamlit.components.v1.html`.

No separate frontend server is required, which keeps the project deployable on Streamlit Community Cloud.

## Frontend Component Approach

This project intentionally avoids a React/Vite build step. The frontend enhancement is implemented with:

- Streamlit layout primitives for structure.
- Custom CSS in `visualization/styles.py`.
- Small inline Vanilla JavaScript snippets for copy-to-clipboard blocks.

This is the most deployable option for Streamlit Community Cloud because the app runs with only Python dependencies.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Optional local quality checks:

```bash
ruff check .
mypy .
```

## Local Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints in your terminal.

## Streamlit Community Cloud Deployment

1. Push this project to GitHub.
2. Create a new Streamlit app from the repository.
3. Set the main file path to `streamlit_app.py` or `app.py`.
4. Ensure `requirements.txt` is in the repository root.
5. In Advanced settings, choose Python 3.11 for the closest match to CI.
6. Deploy.

Deployment-ready files included in this repository:

- `streamlit_app.py`: conventional Streamlit Cloud entrypoint.
- `app.py`: main application.
- `requirements.txt`: Python dependencies.
- `.streamlit/config.toml`: Streamlit theme configuration.
- `data/.gitkeep`: keeps the local history directory available while excluding runtime SQLite data.

No Node.js build, frontend server, paid API key, or Linux `packages.txt` file is required.

For public deployments, enable `Static-only public mode` in the sidebar when demonstrating with untrusted code. This disables benchmark execution while preserving static analysis, scoring, reports, and interview coaching.

## Gemini Setup

Gemini is optional. The app works fully without it.

To use Gemini-enhanced coaching:

1. Install dependencies from `requirements.txt`.
2. Paste a Gemini API key into the sidebar field.
3. Click `Generate Optimization Plan`.

Gemini receives a compact JSON object of local facts and is used only to improve natural-language explanations and interview coaching. It is not used for:

- benchmark timing,
- memory measurement,
- static complexity estimates,
- optimization scoring.

The default model name is read from `GEMINI_MODEL`, then falls back through `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-2.5-flash-lite`, `gemini-2.0-flash-lite`, `gemini-2.5-flash`, and `gemini-2.0-flash`.

Free-tier Gemini API projects can still hit per-model or per-project RPM, TPM, or daily limits. When Algorithm Planner receives a Gemini quota or rate-limit response, Complexity Lab returns a local fallback optimization plan instead of blocking the workflow with a hard error.

## Estimated vs Measured Metrics

The app deliberately separates two kinds of truth:

- Static estimates: AST-derived heuristics such as loop depth, recursion, sorting, slicing, comprehensions, and repeated scans.
- Measured benchmarks: empirical timing and Python memory tracing for the provided entrypoint and input.
- Scaling experiments: generated inputs of multiple sizes with curve fitting against common growth models. These are empirical fits, not proof.

Static complexity is not a proof for arbitrary Python code. The app shows confidence, evidence, and caveats so users can explain uncertainty honestly in interviews.

## Scoring Methodology

The optimization score is deterministic and explainable. It starts with 100 possible points distributed across seven categories:

- Algorithmic efficiency: 30 points.
- Nested-loop burden: 18 points.
- Data structure appropriateness: 14 points.
- Redundant computation: 12 points.
- Memory overhead: 10 points.
- Maintainability/readability: 10 points.
- Benchmark performance signal: 6 points.

Penalties are applied based on static-analysis signals and benchmark summaries:

- Higher estimated time complexity reduces algorithmic points.
- Nested loops reduce nested-loop points.
- Membership checks inside loops reduce data-structure points.
- repeated `sum`, `min`, `max`, `sorted`, repeated traversal, or sorting inside loops reduces redundant-work points.
- O(n) or O(n^2) auxiliary memory, temporary objects, and slicing inside loops reduce memory points.
- while-loops, recursion, low confidence, and many caveats reduce maintainability points.
- Slow benchmark runs or high traced peak memory reduce benchmark-signal points.

The final score is clamped to 0-100 and also shown as an efficiency percentage. This percentage is a product score, not a universal performance theorem.

## Security Notes

Benchmark execution is best-effort guarded, not a perfect sandbox.

Protections include:

- Syntax parsing before execution.
- AST validation for denied imports and dangerous calls.
- Import allowlist for common interview/data utilities.
- Denied filesystem, process, network, introspection, and dynamic execution operations.
- Restricted builtins.
- Isolated globals and locals.
- Separate worker process with timeout termination.
- Warmup runs and top-level script execution disabled by default.
- Basic checks for suspicious infinite loops, very large literal allocations, monkey-patching builtins, denied imports, and dunder access.
- `tracemalloc` memory tracing.

Limitations:

- Python cannot be safely sandboxed inside the same trust boundary with simple AST checks.
- CPU and memory behavior depends on the deployment environment.
- Third-party libraries may perform work outside Python memory tracing.
- The app is intended for your own interview-practice snippets, not hostile code.

## Known Caveats

- Complexity estimates are heuristic and input-shape dependent.
- Recursive branching estimates are conservative.
- The optimized code suggestion is generated only for recognizable safe patterns.
- Benchmarking a top-level script is disabled by default and must be explicitly enabled.
- Benchmark inputs must be JSON or Python literals.
- Scaling input generation is heuristic and works best when the first argument represents the main input size.
- SQLite history is local to the project runtime in `data/complexity_lab.sqlite3`.

## Future Enhancements

- Add optional Docker-based execution isolation for stronger sandboxing.
- Add radon maintainability metrics directly into the score.
- Add chart images into exported HTML reports.
- Add Monaco editor support through a prebuilt optional component.
- Add original-vs-optimized automatic benchmark validation for more patterns.
