PRAGMA foreign_keys = ON;

-- ===================================================
-- TABLE 0: ACCOUNT (Single-row, updatable cash balance)
-- ===================================================
-- The CHECK(id = 1) constraint enforces exactly one account row, so the cash
-- balance is a singleton the Quant/Risk nodes read for position sizing.
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash_balance REAL NOT NULL DEFAULT 0.0,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
INSERT OR IGNORE INTO account (id, cash_balance) VALUES (1, 0.0);

-- ===================================================
-- TABLE 1: POSITIONS (Preserves Pure Purchase Cost)
-- ===================================================
CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,           -- e.g., 'NVDA_20260611_1043'
    symbol TEXT NOT NULL,                   -- e.g., 'NVDA'
    status TEXT NOT NULL,                   -- 'OPEN', 'ASSIGNED', 'LIQUIDATED'
    entry_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    close_date DATETIME,
    stock_purchase_price REAL NOT NULL,     -- Pure, unmutated stock purchase price
    total_realized_pnl REAL DEFAULT 0.0,    -- Final dollar P&L upon position closure
    downside_buffer_percent REAL            -- Premium cushion % at entry → dynamic breach threshold
);

-- ===================================================
-- TABLE 2: TRANSACTIONS (Tracks All Income/Expense Legs)
-- ===================================================
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    asset_type TEXT NOT NULL,               -- 'STOCK' or 'OPTION'
    action TEXT NOT NULL,                   -- 'BUY_TO_OPEN', 'SELL_TO_OPEN', 'BUY_TO_CLOSE', 'SELL_TO_CLOSE'
    quantity INTEGER NOT NULL,              -- e.g., 100 for stock, 1 for option
    price REAL NOT NULL,                    -- Premium collected or stock price paid
    fees REAL DEFAULT 0.0,
    strike_price REAL,
    expiration_date DATE,
    
    FOREIGN KEY (position_id) REFERENCES positions(position_id)
);

-- ===================================================
-- TABLE 3: DECISION LOGS (The Complete HITL Cognitive Ledger)
-- ===================================================
CREATE TABLE IF NOT EXISTS decision_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id TEXT,                       -- Can be NULL if an entry proposal is DENIED
    symbol TEXT NOT NULL,                   -- e.g., 'NVDA'
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    workflow_stage TEXT NOT NULL,           -- 'ENTRY_SCREENER', 'DEFENSE_MONITOR', 'REVIEW_SUBAGENT'
    
    -- Agent's Brain State
    agent_recommendation_json TEXT NOT NULL, -- The specific strike/expiry/indicator data proposed
    tot_branches_json TEXT,                 -- Used in defense/review to show evaluated alternatives
    agent_rationale TEXT NOT NULL,          -- The LLM's text justification for its proposal
    
    -- Human-In-The-Loop Feedback Loop
    is_human_approved INTEGER NOT NULL,     -- 1 for Approved, 0 for Denied
    human_feedback_notes TEXT,              -- Your exact reason for approval/denial (The learning signal)
    
    FOREIGN KEY (position_id) REFERENCES positions(position_id)
);