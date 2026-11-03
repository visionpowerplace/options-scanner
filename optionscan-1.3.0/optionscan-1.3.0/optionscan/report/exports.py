"""CSV and JSON exports, plus the suggestion log that makes the scanner auditable."""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def _default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
    except Exception:
        pass
    return str(o)


def write_csv(res, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = res.to_frame()
    if not df.empty:
        for col in ("components", "penalties", "reasons", "warnings", "breakevens"):
            if col in df.columns:
                df[col] = df[col].apply(lambda v: json.dumps(v, default=_default))
    df.to_csv(p, index=False)
    return p


def write_json(res, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": res.as_of,
        "provider": res.provider,
        "universe_size": res.universe_size,
        "scanned": res.scanned,
        "heat": res.heat,
        "advisories": res.advisories,
        "errors": res.errors,
        "ideas": [
            {**i.as_dict(),
             "legs": [l.as_dict() for l in i.trade.legs],
             "plan": i.plan.as_dict(),
             "size": i.size.as_dict(),
             "trend": i.regime.trend.as_dict(),
             "vol": i.regime.vol.as_dict()}
            for i in res.ideas
        ],
        "diagnostics": [{k: v for k, v in d.items() if k != "ideas"}
                        for d in res.diagnostics],
    }
    p.write_text(json.dumps(payload, indent=2, default=_default))
    return p


def write_html(res, cfg, path: str | Path) -> Path:
    from .html import render
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(res, cfg), encoding="utf-8")
    return p


def write_text(res, cfg, path: str | Path) -> Path:
    from .console import format_report
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(format_report(res, cfg), encoding="utf-8")
    return p


def log_suggestions(res, store) -> int:
    """
    Append today's ideas to the suggestion log.

    This is what lets you grade the scanner later instead of trusting it. Every
    idea is recorded with its score, its probabilities and its entry price, so
    in three months you can ask the only question that matters -- did the
    high-scoring ideas actually do better than the low-scoring ones?
    """
    recs = []
    for i in res.ideas:
        recs.append({
            "logged_at": res.as_of.isoformat(),
            "provider": res.provider,
            "ticker": i.trade.ticker,
            "strategy": i.trade.strategy,
            "score": i.score.total,
            "spot_at_scan": i.trade.spot,
            "expiry": i.trade.expiry.isoformat(),
            "dte": i.trade.dte,
            "order": i.trade.order_ticket(i.size.contracts),
            "entry_limit": i.plan.entry_limit,
            "entry_type": i.plan.entry_type,
            "net_cash": i.trade.net_cash,
            "max_profit": i.trade.max_profit,
            "max_loss": i.trade.max_loss,
            "breakevens": i.trade.breakevens,
            "pop": i.trade.pop,
            "pop_at_rv": i.trade.pop_at_rv,
            "breakeven_win_rate": i.trade.breakeven_win_rate,
            "ev_at_rv": i.trade.ev_at_rv,
            "iv_rank": i.regime.vol.iv_rank,
            "iv_rv_ratio": i.regime.vol.iv_rv_ratio,
            "trend_label": i.regime.trend.label,
            "contracts": i.size.contracts,
        })
    if recs:
        store.append(recs)
    return len(recs)
