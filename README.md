# Covered Call Agentic AI

A locally-hosted, multi-agent LangGraph system that screens stocks for covered-call
income, enforces deterministic safety guardrails, and surfaces graded candidates to a
human for approval (no autonomous trading).

Pipeline: **Scout → Quant → News/Sentiment → Risk Manager → Discord (HITL) → human feedback → SQL/Chroma memory.**

---

## 1. Setup

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
cp .env.example .env          # then fill in real values (see below)
./venv/bin/python -m app.memory.sql.init_db   # create the SQLite DB
```

### Required `.env` values
| Variable | Needed for | How to get it |
|---|---|---|
| `MASSIVE_API_KEY` | **Prices, option chains, history, indicators & news** (default provider) | massive.com account (needs the Stocks + Options plans for chains/greeks) |
| `MARKET_DATA_PROVIDER` | Which market-data client to use: `massive` (default) or `schwab` | optional — defaults to `massive` |
| `SCHWAB_APP_KEY` / `SCHWAB_APP_SECRET` | Prices & option chains **only if** `MARKET_DATA_PROVIDER=schwab` | Schwab developer portal (see §2) |
| `SCHWAB_REDIRECT_URI` | OAuth (Schwab provider only) | Must match the redirect registered on your Schwab app |
| `FINNHUB_API_KEY` | Earnings guardrail (primary) | finnhub.io free key (optional — without it, earnings falls back to Google-search, then "unknown → flagged") |
| `DISCORD_WEBHOOK_URL` | HITL alerts | Discord → Server Settings → Integrations → Webhooks |

> **Market-data provider:** the system defaults to **massive.com** (Polygon-compatible) for prices, option chains, historical bars, and technical indicators. Schwab remains available as a drop-in fallback via `MARKET_DATA_PROVIDER=schwab` (requires the OAuth setup in §2). Earnings uses Finnhub first, then Massive/Benzinga (if entitled), then a Google-search LLM engine.

The local LLM (`model/qwen2.5-coder-14b-instruct-q4_k_m.gguf`) and `LLM_*` settings are
already wired; install `llama-cpp-python` (in requirements) and it loads on first use.

---

## 2. Obtaining the Charles Schwab API token *(optional — only for `MARKET_DATA_PROVIDER=schwab`)*

> **You do not need this for normal use.** The system defaults to massive.com for all
> market data. This section applies only if you explicitly set `MARKET_DATA_PROVIDER=schwab`
> to use the legacy provider.

Schwab uses OAuth2. You mint an initial token set once interactively; after that the app
auto-refreshes silently. **Schwab refresh tokens expire after 7 days**, so you re-run the
one-time step weekly (or whenever calls start failing with `invalid_grant`).

> **`invalid_grant` on `auth.py`?** Schwab authorization codes are single-use and expire in
> ~30 seconds. Re-run `auth.py`, approve, and paste the full redirect URL within a few
> seconds using a **fresh** code each time; also confirm your app is **Ready For Use** and the
> registered callback exactly matches `SCHWAB_REDIRECT_URI`. The script now prints the parsed
> code, warns if you were too slow, and gives a plain-English diagnosis on failure.

### One-time setup on the Schwab side
1. Create a developer account at **https://developer.schwab.com** and log in.
2. Create an **App** under the *Market Data Production* product.
3. Note the **App Key** (client id) and **Secret** → put them in `.env` as
   `SCHWAB_APP_KEY` / `SCHWAB_APP_SECRET`.
4. Register a **Callback URL (redirect URI)**, e.g. `https://127.0.0.1`. Put the exact same
   value in `.env` as `SCHWAB_REDIRECT_URI`. *(It must match exactly or the token
   exchange fails.)*
5. Wait until the app status is **Ready/Approved**.

### Mint the first token set
```bash
./venv/bin/python charles_schwab_mcp/auth.py
```
This will:
1. Open your browser to Schwab's login/consent page (prints the URL too).
2. After you approve, Schwab redirects to your callback URL (e.g.
   `https://127.0.0.1/?code=XXXX%40&session=...`). The page may show a "can't connect"
   error — that's fine; **copy the full URL from the address bar**.
3. Paste that full URL at the prompt. The script exchanges the `code` for tokens and
   **saves them to `charles_schwab_mcp/schwab_tokens.json`** (gitignored).

### Verify / refresh anytime
```bash
./venv/bin/python charles_schwab_mcp/refresh_token.py
```
Prints ✅ if the token is valid, or tells you to re-run `auth.py` if the 7-day refresh
window has lapsed. During normal operation you never call this manually — the app's
`token_manager.get_valid_access_token()` refreshes on demand.

> **Security note:** `.env` and `schwab_tokens.json` are gitignored. The app key/secret
> were previously committed in plaintext and have since been moved to `.env` — rotate them
> in the Schwab portal if they may have leaked.

---

## 3. Running the screener

```bash
# set your account cash (used for position sizing)
./venv/bin/python -c "from app.memory import set_cash_balance; set_cash_balance(50000)"

# run one screening pass (loads app/watchlist.json + cash from SQL)
./venv/bin/python -c "from app.graphs import run_entry_screener; run_entry_screener()"
```
The Risk Manager sends the top candidates to Discord, echoes the full summary to the log,
and saves it to `runs/<run_id>.md` + `.json` (status `PENDING_APPROVAL`).

> **Runtime / progress.** The Quant node makes ~2 rate-limited market-data calls per symbol,
> so a 200+ watchlist run takes a few minutes. All massive.com clients (prices, news,
> earnings) share **one** rate limiter against your account quota. Progress is logged per
> symbol (`Quant [12/210] KO ...`). This is normal — it is not stuck. For a quick first test,
> trim `app/watchlist.json` to a handful of symbols, or set `StrategyRules.max_quant_candidates`
> in `app/config.py` to e.g. `15` to analyze only the first N.
>
> Stocks you **already hold** (any OPEN position) are automatically excluded from each
> screener run.

### Approve / deny (Human-In-The-Loop)
```bash
./venv/bin/python -m app.feedback <run_id>
```
The CLI auto-detects the workflow from the run:

- **Entry-screener run** → each candidate shows its **current stock price**, contract, yield,
  sentiment, and a rationale that cites the concrete technicals (trend/RSI/SMA/Bollinger) and
  news finding. You enter **A**pprove / **D**eny / **S**kip, notes, and on approval the actual
  fill (shares, price, premium, contracts) → opens a position.
- **Defense run** (`defense_<symbol>_<ts>`) → the prompt shows your **existing holding +
  short-call details** and the **suggested roll contract**; you choose which branch to execute:
  **A** Hard Eject (stock sale + call buyback → liquidates), **B** Roll Down (buyback + new
  lower-strike call → rolls it, stays open), **C** Hold — or **D**eny / **S**kip.

**Recording outcomes** (via the CLI or the management agent) — the status you use decides
whether the *stock* is sold:
- **EXPIRED** — the call expired worthless: the **premium is kept as realized gain and you
  retain the 100 shares** (no stock sale; the position stays open). Safe to run twice
  (idempotent).
- **ASSIGNED** — called away at the strike: shares sold at the strike, call exercised.
- **LIQUIDATED** — you sold the shares on the market: requires a positive sale price.

Option **premiums are realized income immediately** when the call is sold (the cash is in
hand and never double-counted on a later close). Only this feedback step writes to the SQL
ledger (`positions`, `transactions`, `decision_logs`) and the ChromaDB lesson memory; cash is
adjusted on every executed leg.

---

## 3b. Downside defense (Tree-of-Thoughts)

For an open position that has dropped below its threshold, evaluate the three
escape branches (Hard Eject / Roll Down / Hold) and alert the human:
```bash
./venv/bin/python -c "
from app.graphs import run_defense_monitor
pos = {'position_id':'KO_x','symbol':'KO','stock_purchase_price':100.0,'shares':100,
       'short_call_strike':105.0,'short_call_expiration':'2026-07-28',
       'original_premium':2.0,'historical_premiums_collected':2.0}
run_defense_monitor(pos)  # fetches current price/option asks from the market-data provider
"
```
The Quant node computes exact branch P&L (against the **raw** cost basis), News
checks for catastrophic risk, and the Risk Manager recommends a branch (never
rolls for a non-positive credit) and sends it to Discord for manual execution.

The **Roll Down** branch is earnings-aware and basis-aware: the proposed new
contract is capped to expire **before the next earnings date** (same guardrail the
entry screener uses), and when the new strike is **below your cost basis** the
summary shows the total P&L you'd lock in if called away there — so a roll that
looks like a credit but merely defers a comparable loss is surfaced, not hidden.
Positions whose call has already expired (shares held, no active call) are skipped
by the scan — there's no covered call to defend.

**Scan ALL open holdings at once** (no manual per-position loop):
```bash
./venv/bin/python -c "from app.graphs import run_defense_scan; run_defense_scan()"
```
Loads every OPEN position from the SQL ledger, fetches live market data per name,
and evaluates each — breached positions get a Discord alert.

> **News depth:** the News and Defense nodes fetch the **full article body** (not
> just the headline) and have the LLM judge sentiment from the content, plus a
> deterministic catastrophic-keyword scan (`StrategyRules.catastrophic_keywords`)
> that forces a catastrophic flag regardless of the LLM. Toggle body-fetching with
> `NEWS_FETCH_FULL_CONTENT` (best-effort; paywalled pages fall back to the summary).

## 3c. Weekly / monthly performance report (feedback loop)
```bash
./venv/bin/python -m app.reporting weekly     # or: monthly
```
Aggregates the SQL ledger (P&L, win rate, premium harvested, worst losers, recent
denials), has the LLM write an analytical review, saves it to `runs/`, and stores
the lesson in ChromaDB for future decision cycles.

## 3d. Web UI (recommended) — the Command Center

```bash
./venv/bin/python -m app.web.server      # → http://127.0.0.1:8765
```
A clean local **ReactJS** dashboard (React via CDN — no build step) with:
- **Workflow buttons** — each opens a **dedicated window** that runs the entry
  screener / downside-defense scan / performance report and streams its live,
  color-coded logs + a final summary.
- **My Holdings** — every position with its covered-call contract, plus **live
  current price**, unrealized stock P&L, and realized premium income; portfolio
  totals for cost basis, current value, and unrealized P&L. Positions whose call
  expired show "call expired — shares held".
- **Transaction History** — every ledger transaction (newest first): date,
  symbol, buy/sell action, type, qty, price, option contract, and the signed cash
  effect, with a symbol/action filter, a Stock/Option toggle, and running totals
  (premiums collected, net cash flow).
- **Agent chat** — talk to the multi-step covered-call management expert (it can
  also trigger the workflows itself via tools).
- **Theme switcher** — five color themes (Midnight / Slate / Emerald / Rose /
  Light), remembered across sessions and shared with the popup windows.

## 3e. The management agent (CLI or via the UI)

The agent is a multi-step (ReAct) expert that **chains tools**: manage cash &
holdings, run exact quantitative analysis (yields/Greeks/POP/defense branches),
recall past lessons, and report performance. It never does mental math or
invents data — it calls tools.
```bash
./venv/bin/python -m app.manage
> analyze a 105 call on a $100 stock, $2 premium, 35 DTE, 25% IV
> set cash to 50000 and show my open holdings
> what have we learned about utility stocks?
```

### MCP servers (every engine + every API as a tool)

The whole system's capabilities are exposed over MCP for external clients (Claude
Desktop, IDE agents):
```bash
./venv/bin/python -m math_mcp.math_mcp          # all 20 math-engine functions
./venv/bin/python -m app.data.data_mcp_server   # all market-data + news + earnings API methods
./venv/bin/python -m app.agent.mcp_server       # management + analysis (cash, holdings, reports)
```
- **math_mcp** — every function in the deterministic engine (yields, Greeks,
  probabilities, technicals, scoring, defense branches, …).
- **data_mcp_server** — every market-data method via the active provider
  (massive.com by default): quotes/fundamentals/price history/option
  chains/expirations/optionable, plus massive.com news + article fetch, and the
  earnings chain (Finnhub → Massive/Benzinga → Google-search).
- **agent.mcp_server** — account management + portfolio analysis.

The management agent also has **technical-indicator tools** backed by massive.com's
`/v1/indicators/*` endpoints — `get_sma`, `get_ema`, `get_rsi`, `get_macd` — so you
can ask e.g. "what's AAPL's 14-day RSI?" and it fetches the value directly.

## Cash accounting (single source of truth)
Account cash lives in the SQL `account` table and is adjusted automatically by
**every transaction**: buying shares and buying back calls reduce cash; selling
shares and collecting premium increase it. So approving a trade (buy stock + sell
call) and closing one (`close_position`, e.g. assigned/liquidated) keep the cash
balance correct without manual edits. The screener always reads cash from SQL.

Realized P&L is always **recomputed from the ledger** (never incrementally added),
so a premium is counted exactly once no matter how the position later closes.
Option premiums are realized the moment the call is sold; the stock's gain/loss is
unrealized until the shares are actually sold. Active-call quantity is tracked as
`SELL_TO_OPEN − BUY_TO_CLOSE` contracts (so rolls, partial buybacks, and
expirations stay correct), and a $0 stock "sale" is rejected outright.

## 4. Tests
```bash
for t in tests/test_*.py; do ./venv/bin/python "$t"; done
```
All offline (mocked APIs, fake LLM/embedder), except the real-ChromaDB test which needs
the sandbox off.

---

## 5. Project layout
```
app/
  config.py        settings (.env) + tunable StrategyRules (incl. MARKET_DATA_PROVIDER)
  state.py         LangGraph state schemas
  llm.py           local Qwen2.5 (llama-cpp-python) + structured() JSON helper
  engine/          deterministic math engine (the LLM never does math)
  data/            market-data clients: massive_client (default) + schwab_client (fallback),
                   market_data.py (provider factory), news, Finnhub + massive earnings,
                   rate_limiter (shared), data_mcp_server
  nodes/           scout, quant, news, risk_manager, defense
  notify/          Discord HITL webhook (chunked)
  memory/          sql/ (schema, db) + chroma/ + decision/positions/account stores + vector_db
  graphs/          entry_screener, defense_monitor
  agent/           tools registry, ReAct agent, MCP server
  web/             Flask Command Center + React static UI (index / holdings /
                   transactions / workflow) + styles
  reporting.py     performance report + feedback loop
  runlog.py        per-run artifacts
  feedback.py      human approval CLI (entry + defense)
charles_schwab_mcp/  OAuth + token manager (only for the Schwab provider)
math_mcp/            FastMCP wrapper over the math engine
```
