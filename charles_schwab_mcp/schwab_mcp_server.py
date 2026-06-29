import os
import logging
from typing import Optional, List
import httpx
from mcp.server.fastmcp import FastMCP
from token_manager import get_valid_access_token

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("schwab-mcp-server")

# Define Server Name
mcp = FastMCP("Schwab Market Data Server")

BASE_URL = "https://api.schwabapi.com/marketdata/v1"

def get_auth_headers() -> dict:
    """Dynamically retrieves a valid token, refreshing if necessary."""
    # The agent pauses here, checks the JSON file, and fetches a new token if 30 mins have passed
    valid_token = get_valid_access_token() 
    
    return {
        "Authorization": f"Bearer {valid_token}",
        "Accept": "application/json"
    }

@mcp.tool()
async def get_price_history(
    symbol: str,
    period_type: Optional[str] = "day",
    period: Optional[int] = 1,
    frequency_type: Optional[str] = "minute",
    frequency: Optional[int] = 1,
    start_date: Optional[int] = None,
    end_date: Optional[int] = None,
    need_extended_hours_data: Optional[bool] = True,
    need_previous_close: Optional[bool] = True
) -> dict:
    """
    Fetch historical candlestick data (CandleList) for technical screening (e.g., SMA, RSI, Bollinger Bands).
    
    :param symbol: Stock symbol (e.g., AMD, NVDA)
    :param period_type: The type of period to show (day, month, year, ytd)
    :param period: The number of periods to show
    :param frequency_type: The frequency interval type (minute, daily, weekly, monthly)
    :param frequency: The frequency interval numeric value
    """
    url = f"{BASE_URL}/pricehistory"
    params = {
        "symbol": symbol.upper(),
        "periodType": period_type,
        "period": period,
        "frequencyType": frequency_type,
        "frequency": frequency,
        "needExtendedHoursData": str(need_extended_hours_data).lower(),
        "needPreviousClose": str(need_previous_close).lower()
    }
    if start_date: params["startDate"] = start_date
    if end_date: params["endDate"] = end_date

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=get_auth_headers(), params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_option_chain(
    symbol: str,
    contract_type: Optional[str] = "ALL",
    strike_count: Optional[int] = 10,
    include_quotes: Optional[bool] = True,
    strategy: Optional[str] = "SINGLE",
    strike: Optional[float] = None,
    range_filter: Optional[str] = "ALL",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> dict:
    """
    Fetch comprehensive option chain contracts for quantitative Greek checking and matrix evaluation.
    
    :param symbol: Underlying optionable symbol (e.g., AAPL)
    :param contract_type: CALL, PUT, or ALL
    :param strike_count: Number of strikes to return above/below the current price
    :param strategy: Option strategy (SINGLE, ANALYTICAL, COVERED, etc.)
    :param range_filter: Strike range filter (ITM, OTM, ATM, ALL)
    :param from_date: Expiration start range (format: YYYY-MM-DD)
    :param to_date: Expiration end range (format: YYYY-MM-DD)
    """
    url = f"{BASE_URL}/chains"
    params = {
        "symbol": symbol.upper(),
        "contractType": contract_type,
        "strikeCount": strike_count,
        "includeQuotes": str(include_quotes).lower(),
        "strategy": strategy,
        "range": range_filter
    }
    if strike: params["strike"] = strike
    if from_date: params["fromDate"] = from_date
    if to_date: params["toDate"] = to_date

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=get_auth_headers(), params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_single_quote(symbol: str) -> dict:
    """
    Fetch detailed live real-time or delayed market quote for a single asset symbol.
    """
    url = f"{BASE_URL}/{symbol.upper()}/quotes"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=get_auth_headers())
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_multiple_quotes(symbols: List[str]) -> dict:
    """
    Fetch real-time quotes for a batch list of symbols.
    """
    url = f"{BASE_URL}/quotes"
    params = {"symbols": ",".join([s.upper() for s in symbols])}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=get_auth_headers(), params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_option_expirations(symbol: str) -> dict:
    """
    Fetch the expiration chain timeline for a specific optionable equity target.
    """
    url = f"{BASE_URL}/expirationchain"
    params = {"symbol": symbol.upper()}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=get_auth_headers(), params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def lookup_instrument(symbol: str, projection: str = "symbol-search") -> dict:
    """
    Query asset instruments and underlying fundamental definitions via symbol patterns.
    
    :param projection: Type of lookup (symbol-search, security-desc, cusip, symbol-regex)
    """
    url = f"{BASE_URL}/instruments"
    params = {"symbol": symbol.upper(), "projection": projection}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=get_auth_headers(), params=params)
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    # Launching server via stdio transport link
    mcp.run()