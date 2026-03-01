# pm-sim — Polymarket Paper Trading Simulator

**Date:** 2026-02-28
**Status:** Approved
**Author:** Claude + Robert

## 1. Overview

A CLI paper trading simulator for Polymarket, designed specifically for AI agents.
Agents call `pm-sim` via shell commands and receive JSON responses. No wallet, no
real money, no blockchain — just real market data + local simulated trades in SQLite.

### Core Principles

- **Agent-first**: All output is JSON by default, machine-parseable
- **Zero wallet**: Only uses Polymarket public APIs (Gamma + CLOB), no authentication
- **Real data**: Trades execute at live midpoint prices from Polymarket order books
- **Auditable**: Every trade logged in SQLite with full provenance
- **Minimal deps**: Only `click`, `httpx`, `sqlite3` (stdlib)

## 2. Architecture

```
Agent (LLM / script / OpenClaw / Claude Code)
    │  shell subprocess
    ▼
pm-sim CLI (Click)
    │
    ├── cli.py       ─── Command definitions, JSON formatting
    ├── engine.py    ─── Trade execution, P&L, portfolio logic
    ├── api.py       ─── Polymarket HTTP client (Gamma + CLOB)
    ├── db.py        ─── SQLite operations, schema, migrations
    └── models.py    ─── Dataclasses (Market, Trade, Position)
    │
    ├── Data flow: read  ─→ HTTP GET to Polymarket API (with 5min cache)
    └── Data flow: write ─→ SQLite at ~/.pm-sim/paper.db
```

### Data Source: Direct HTTP to Polymarket REST API

- **Gamma API** (`https://gamma-api.polymarket.com`): Market discovery, metadata
- **CLOB API** (`https://clob.polymarket.com`): Prices, order books, spreads
- All endpoints are public, no authentication required
- Market data cached in SQLite for 5 minutes to reduce API calls
- Prices for buy/sell always fetched real-time (never cached)

### Why not pmxt / Polymarket CLI?

| Factor | Direct HTTP | pmxt SDK | Polymarket CLI |
|--------|:-----------:|:--------:|:--------------:|
| Dependencies | httpx only | Node.js sidecar | Rust binary |
| Install | `pip install` | `pip install` + npm | `brew install` |
| Latency | ~200ms | ~300ms (sidecar hop) | ~500ms (subprocess) |
| Stability | Years of API | 3 months old | 4 days old |
| Control | Full | Framework-mediated | stdout parsing |

## 3. CLI Interface

### Global Flags

```
pm-sim [--data-dir PATH] [--output json|table] <command>
```

- `--data-dir`: SQLite location (default: `~/.pm-sim/`)
- `--output`: Output format (default: `json` for agent-friendliness)

### Commands

#### Account Management

```bash
pm-sim init [--balance 10000] [--fee-bps 0]   # Initialize paper account
pm-sim balance                                  # Show current cash balance
pm-sim reset [--confirm]                        # Wipe all data, start fresh
```

#### Market Data (cached, read-only from Polymarket)

```bash
pm-sim markets list [--limit 20] [--active] [--sort volume|liquidity]
pm-sim markets search <query> [--limit 10]
pm-sim markets get <slug-or-id>
pm-sim price <slug-or-id>                       # YES/NO midpoints
pm-sim book <slug-or-id>                        # Top 5 bids/asks
pm-sim sync                                     # Pre-warm cache
```

#### Trading (local simulation)

```bash
pm-sim buy <slug-or-id> <yes|no> <amount-usd>
pm-sim sell <slug-or-id> <yes|no> <amount-usd>
```

- `buy`: Spend $amount to buy shares at current midpoint price
- `sell`: Sell $amount worth of shares you own (no shorting)
- Attempting to sell without a position → `NO_POSITION` error
- Attempting to buy in a closed market → `MARKET_CLOSED` error

#### Portfolio & History

```bash
pm-sim portfolio                                # All positions + unrealized P&L
pm-sim history [--limit 50]                     # Trade log
pm-sim performance                              # Phase 2: stats
```

#### Market Resolution

```bash
pm-sim resolve <slug-or-id>                     # Check & settle one market
pm-sim resolve --all                            # Check & settle all positions
```

### JSON Output Format

All commands return a consistent envelope:

```json
// Success
{"ok": true, "data": { ... }}

// Error
{"ok": false, "error": "Insufficient balance", "code": "INSUFFICIENT_BALANCE"}
```

**Error codes:**
- `NOT_INITIALIZED` — Account not yet created (run `pm-sim init`)
- `INSUFFICIENT_BALANCE` — Not enough cash for this trade
- `MARKET_NOT_FOUND` — Slug/ID doesn't match any market
- `MARKET_CLOSED` — Market is no longer active
- `NO_POSITION` — Trying to sell shares you don't own
- `INVALID_OUTCOME` — Outcome must be "yes" or "no"
- `API_ERROR` — Polymarket API returned an error or is unreachable

## 4. Data Model

### SQLite Schema

```sql
CREATE TABLE account (
    id INTEGER PRIMARY KEY DEFAULT 1,
    starting_balance REAL NOT NULL DEFAULT 10000,
    cash REAL NOT NULL DEFAULT 10000,
    fee_bps INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (id = 1)
);

CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_condition_id TEXT NOT NULL,
    market_slug TEXT NOT NULL,
    market_question TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('yes', 'no')),
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    price REAL NOT NULL,
    amount_usd REAL NOT NULL,
    shares REAL NOT NULL,
    fee REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE positions (
    market_condition_id TEXT NOT NULL,
    market_slug TEXT NOT NULL,
    market_question TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('yes', 'no')),
    shares REAL NOT NULL DEFAULT 0,
    avg_entry_price REAL NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0,
    realized_pnl REAL NOT NULL DEFAULT 0,
    is_resolved INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT,
    PRIMARY KEY (market_condition_id, outcome)
);

CREATE TABLE market_cache (
    cache_key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### Key Design Decisions

**Internal ID: `condition_id`** — Market slugs are user-facing convenience.
Internally everything keys on `condition_id` (stable, on-chain identifier).
First use of a slug auto-resolves to `condition_id` via Gamma API and caches the mapping.

**Positions table as materialized view** — Could be computed from trades,
but caching in a table avoids O(n) scans on every `portfolio` call.
Updated transactionally with each trade.

**Prices as floats** — Polymarket prices are 0.00–1.00 with 4 decimal precision.
Python float64 has 15+ significant digits, more than sufficient.
We do NOT use Decimal to keep the code simple.

## 5. Execution Model

### Buy Flow

```
Input: pm-sim buy "bitcoin-100k" yes 100

1. Validate account exists
2. Resolve "bitcoin-100k" → condition_id + token_ids (cached)
3. Check market is active (not closed)
4. Fetch real-time midpoint for YES token via CLOB API
   → midpoint = 0.65
5. Calculate:
   shares = amount / price = 100 / 0.65 = 153.846...
   fee = fee_bps/10000 * min(price, 1-price) * amount
       = 0 (default 0 bps)
   total_cost = amount + fee = 100.00
6. Check cash >= total_cost
7. SQLite transaction:
   - UPDATE account SET cash = cash - 100.00
   - INSERT INTO trades (...)
   - UPSERT positions: shares += 153.85, recalc avg_entry_price
8. Return JSON:
   {"ok": true, "data": {
     "trade_id": 1,
     "market": "bitcoin-100k",
     "outcome": "yes",
     "side": "buy",
     "price": 0.65,
     "shares": 153.85,
     "amount_usd": 100.00,
     "fee": 0.00,
     "cash_remaining": 9900.00
   }}
```

### Sell Flow

```
Input: pm-sim sell "bitcoin-100k" yes 50

1-4. Same as buy (resolve, validate, fetch price)
   → midpoint = 0.70 (price went up)
5. Check position exists and has enough shares
   shares_to_sell = amount / price = 50 / 0.70 = 71.43
   position has 153.85 shares → OK
6. Calculate realized P&L for sold portion:
   realized = shares_to_sell * (sell_price - avg_entry_price)
            = 71.43 * (0.70 - 0.65) = 3.57
7. SQLite transaction:
   - UPDATE account SET cash = cash + 50.00
   - INSERT INTO trades (side='sell', ...)
   - UPDATE positions: shares -= 71.43, realized_pnl += 3.57
8. Return JSON with trade details
```

### Resolution Flow

```
Input: pm-sim resolve "bitcoin-100k"

1. Fetch market data from Gamma API
2. Check if market.closed == true AND outcomePrices settled
   → outcomePrices = ["1.00", "0.00"] means YES won
3. For each position in this market:
   - YES shares: payout = shares * 1.00
   - NO shares: payout = shares * 0.00
4. SQLite transaction:
   - UPDATE account SET cash += total_payout
   - UPDATE positions SET is_resolved = 1, realized_pnl = payout - total_cost
5. Return JSON with settlement details
```

### Portfolio Calculation

```
Input: pm-sim portfolio

1. Fetch all open positions (is_resolved = 0)
2. For each position, fetch current midpoint price (batch if possible)
3. Calculate:
   current_value = shares * current_price
   unrealized_pnl = current_value - total_cost
   percent_pnl = unrealized_pnl / total_cost * 100
4. Return JSON array of positions + summary
```

## 6. Polymarket API Integration

### Endpoints Used

| Endpoint | Purpose | Cache? |
|----------|---------|--------|
| `GET /markets?slug=X` | Resolve slug → market data | 5 min |
| `GET /markets?limit=N` | List markets | 5 min |
| `GET /markets?_q=X` | Search markets | 5 min |
| `GET /midpoint?token_id=X` | Current price | Never |
| `GET /book?token_id=X` | Order book | Never |
| `GET /price-history?...` | Historical prices | 5 min |

### Rate Limits

- Free tier: ~1000 calls/hour
- With 5-minute cache, typical agent session uses <50 calls/hour
- `sync` command pre-warms cache in bulk

### Error Handling

- Network errors → retry once with 2s backoff, then `API_ERROR`
- 429 Too Many Requests → `API_ERROR` with rate limit message
- Invalid slug → `MARKET_NOT_FOUND`

## 7. Fee Model

Default: 0 bps (no fees). Configurable at init time.

When `fee_bps > 0`, fees follow Polymarket's formula:
```
fee = (fee_bps / 10000) * min(price, 1 - price) * amount
```

This means:
- At price 0.50 (maximum uncertainty): fee is highest
- At price 0.01 or 0.99 (near-certain): fee is near zero
- Fees deducted from cash on buy, deducted from proceeds on sell

## 8. Roadmap

### Phase 1 — MVP (Current)

Core paper trading for a single agent.

- [ ] Project scaffolding (pyproject.toml, structure)
- [ ] SQLite schema + db.py operations
- [ ] Polymarket API client (api.py) with caching
- [ ] Trade engine (engine.py): buy, sell, portfolio, resolve
- [ ] CLI commands (cli.py): all Phase 1 commands
- [ ] JSON output envelope
- [ ] Error handling with codes
- [ ] Unit tests for engine + API mocking
- [ ] README with agent usage examples

### Phase 2 — Analytics & Export

Strategy evaluation tools.

- [ ] `performance` command: win rate, total P&L, Sharpe ratio, max drawdown
- [ ] `history --export csv` for external analysis
- [ ] `resolve --auto` periodic resolution check
- [ ] Position age tracking (days held)

### Phase 3 — Realism

More accurate simulation.

- [ ] Order book execution mode (VWAP slippage)
- [ ] Dynamic fee rates from market's actual `fee_rate_bps`
- [ ] Limit orders (price trigger, pending until matched)
- [ ] `watch` command for real-time price monitoring

### Phase 4 — Platform Expansion

Multi-platform and advanced integrations.

- [ ] Multi-account support (`--account strategy-a`)
- [ ] Kalshi adapter (direct API or via pmxt)
- [ ] Historical data backtesting engine
- [ ] MCP Server mode (tool for Claude/AI agents to call directly)
- [ ] WebSocket real-time price feeds

## 9. Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "click>=8.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-httpx>=0.30",
]

[project.scripts]
pm-sim = "pm_sim.cli:main"
```

Only 2 runtime dependencies: `click` (CLI framework) and `httpx` (async-capable HTTP).
`sqlite3` is Python stdlib.
