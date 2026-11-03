"""
Default scan universe.

Chosen on one criterion that matters more than any other for options: can you
actually get filled at a fair price. These are the most liquid optionable US
names and ETFs -- penny-wide or near-penny markets, deep open interest, weekly
expirations. A great setup on an illiquid underlying is not a tradable setup;
the bid/ask alone will eat the edge.

Replace freely via `tickers:` in your config or --tickers on the command line.
"""

# Index and sector ETFs: the most liquid options in existence.
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLU",
    "XLP", "XLY", "SMH", "GLD", "SLV", "TLT", "HYG", "EEM", "FXI", "ARKK",
]

# Mega-cap and high-liquidity single names.
MEGA_CAP = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD",
    "NFLX", "JPM", "BAC", "WFC", "GS", "V", "MA", "UNH", "JNJ", "LLY", "PFE",
    "XOM", "CVX", "COP", "WMT", "COST", "HD", "MCD", "KO", "PEP", "PG",
    "DIS", "CRM", "ORCL", "ADBE", "INTC", "MU", "QCOM", "TXN", "CAT", "BA",
    "GE", "UBER", "ABNB", "SHOP", "PYPL", "SQ", "COIN", "PLTR", "SNOW", "MRNA",
]

# Higher-volatility names: richer premium, bigger gaps. Worth scanning, worth
# sizing smaller.
HIGH_VOL = [
    "SMCI", "MSTR", "RIVN", "LCID", "SOFI", "RIOT", "MARA", "AFRM",
    "ROKU", "DKNG", "CVNA", "GME", "AMC", "HOOD", "TSM", "ASML",
]

DEFAULT_UNIVERSE = ETFS + MEGA_CAP + HIGH_VOL

PRESETS = {
    "default": DEFAULT_UNIVERSE,
    "etf": ETFS,
    "mega": MEGA_CAP,
    "highvol": HIGH_VOL,
    "liquid": ETFS + MEGA_CAP,
    "small": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA",
              "AMZN", "META", "GOOGL", "JPM", "XOM", "WMT", "COIN"],
}


def resolve(spec) -> list:
    """Accepts a preset name, a comma-separated string, or a list."""
    if not spec:
        return list(DEFAULT_UNIVERSE)
    if isinstance(spec, str):
        if spec in PRESETS:
            return list(PRESETS[spec])
        return [s.strip().upper() for s in spec.split(",") if s.strip()]
    return [str(s).strip().upper() for s in spec if str(s).strip()]
