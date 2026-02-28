# AGENTS.md

## Cursor Cloud specific instructions

This is a Python CLI stock trading AI application ("10Day-Earn-3%"). There is no build step, no test framework, and no linting configuration.

### Running the application

- **Interactive CLI**: `python3 main.py` — presents a numbered menu; requires TTY input. To test non-interactively, pipe choices: `echo "12" | python3 main.py` (option 12 exits).
- **Programmatic usage**: import analyzers directly in Python scripts. See `examples/basic_usage.py` for patterns.
- Core analyzers: `EnhancedStockAnalyzer` (US stocks), `ChineseStockAnalyzer` / `ChineseStockRecommender` (Chinese A/H-shares), `DynamicModelAnalyzer`.

### Key caveats

- **`requirements.txt` pins `numpy==1.24.3`** which is incompatible with the system Python 3.12. The update script installs packages without version pins so pip resolves compatible versions. Do not run `pip install -r requirements.txt` directly.
- **Internet access required**: all analysis fetches live data from Yahoo Finance via `yfinance`. No offline/mock mode exists.
- **No tests or linting**: the repo has no test suite (`pytest`, `unittest`, etc.) and no linter config (`flake8`, `ruff`, `mypy`). Validation is done by importing modules and running analysis functions.
- **Model artifacts**: trained ML models are saved under `us_models/` and `chinese_models/`. These directories are created at runtime.
- **Cache**: Chinese stock data is cached in `chinese_cache/` (auto-created).
