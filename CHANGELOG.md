# Changelog

All notable changes to `polymarket-paper-trader` are documented here.

## Unreleased

### Added
- skills.sh support — skill discoverable via `npx skills add agent-next/polymarket-paper-trader`

### Security
- Add data trust boundaries to SKILL.md — mark Polymarket API data as untrusted, prevent indirect prompt injection via market content

## [0.1.6] - 2026-03-01

### Fixed
- Stop shipping `examples` as top-level package in PyPI wheel — was polluting user site-packages with a generic name

### Changed
- Publish workflow now runs full test suite (Python 3.10–3.13, 100% coverage) before publishing
- ClawHub publish automated in GitHub Actions
- GitHub Release auto-created with generated notes on tag push

### Added
- MCP Registry `server.json` for future registry submission
- README ownership verification tag for MCP Registry

## [0.1.5] - 2026-03-01

### Added
- PyPI publishing via GitHub Actions (`publish.yml`)
- LICENSE file (MIT)
- PyPI badge in README
- `py.typed` marker (PEP 561)
- `[project.urls]` in pyproject.toml (Homepage, Repository, Issues)
- Package metadata: authors, keywords, classifiers

### Changed
- Install section in README now shows `pip install` first

## [0.1.4] - 2026-03-01

### Added
- GitHub Actions CI test workflow (Python 3.10–3.13, 100% coverage gate)
- PK battle system — head-to-head strategy comparison
- Leaderboard card generator and MCP tool
- `share_content` MCP tool with platform-specific templates
- Strategy examples: momentum, mean reversion, limit grid
- Stats cards with top positions and template variants

### Changed
- Rewrite all user-facing copy for viral distribution
- Rename `pm_sim` → `pm_trader` package
- Extract `card.py` constants/helpers, eliminate duplication

### Fixed
- Negative P&L copy semantics
- False 280-char claim in stats cards
- `pragma: no cover` on `__main__.py` for CI

## [0.1.3] - 2026-03-01

### Added
- ClawHub skill for OpenClaw integration
- SKILL.md agent behavior program
- 26 MCP tools (up from 20)

## [0.1.2] - 2026-03-01

### Added
- GTC/GTD limit order management
- Backtesting engine with historical price replay
- Multi-outcome (neg-risk) market support
- Performance analytics (Sharpe, win rate, drawdown)
- CSV/JSON export for trades and positions
- Benchmarking harness for strategies
- Multi-account support

### Fixed
- Limit orders enforce price ceiling/floor
- GTD expiry validation
- `watch_prices` raises ValueError for invalid outcomes
- `check_orders` rejects permanently unfillable orders

## [0.1.1] - 2026-03-01

### Added
- MCP server exposing 20 tools for AI agents
- Click CLI with JSON envelope output
- 1:1 faithful order book fill engine (FOK/FAK)
- Polymarket HTTP client (Gamma + CLOB API)
- Trade execution engine (buy/sell/resolve)
- SQLite database with WAL mode
- E2E tests against live Polymarket API
