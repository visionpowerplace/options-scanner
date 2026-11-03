"""
Alpaca adapter -- PAPER TRADING ONLY.

Three safety properties, in order of importance:

1. LIVE TRADING IS REFUSED. The constructor rejects any base URL that is not
   Alpaca's paper endpoint. There is no flag to turn this off. Going live is a
   deliberate code change you make after reading your own track record, not a
   config value that a typo can flip.

2. CREDENTIALS ARE NEVER LOGGED. Keys live only in request headers, never in
   exception text, never in the order journal, never echoed. This matters
   doubly when the repository and its CI logs are public.

3. EVERY ORDER CAN BE DRY-RUN. `dry_run=True` builds and returns the exact
   payload without sending it. Because this adapter could not be tested against
   the live API while it was written, the first real session should be run with
   --dry-run so the payload can be eyeballed against Alpaca's docs before
   anything is submitted.

Documented assumption to verify on that first run: for a multi-leg order,
`limit_price` is sent as a POSITIVE number for a net debit and a NEGATIVE
number for a net credit. If Alpaca rejects a credit spread, invert the sign in
`_net_limit()` -- it is the one place that convention is applied.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import requests

from .base import Account, Broker, Order, OrderLeg, Position, Quote

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"


class AlpacaPaperBroker(Broker):
    name = "alpaca-paper"
    is_paper = True

    def __init__(self, key_id: str | None = None, secret: str | None = None,
                 base_url: str = PAPER_BASE, dry_run: bool = False,
                 timeout: float = 20.0):
        key_id = key_id or os.environ.get("ALPACA_API_KEY_ID", "")
        secret = secret or os.environ.get("ALPACA_API_SECRET_KEY", "")
        if not key_id or not secret:
            raise RuntimeError(
                "Alpaca credentials missing. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY (as GitHub repository Secrets, never in a "
                "file). Generate PAPER keys at https://app.alpaca.markets .")
        if base_url.rstrip("/") != PAPER_BASE:
            raise RuntimeError(
                f"Refusing to run against '{base_url}'. This adapter is paper-only "
                f"by design; live trading is a deliberate code change, not a setting.")

        self.base = base_url.rstrip("/")
        self.dry_run = dry_run
        self.timeout = timeout
        self._s = requests.Session()
        # Keys live here and nowhere else.
        self._s.headers.update({
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
            "accept": "application/json",
        })

    # ------------------------------------------------------------------
    def _req(self, method: str, url: str, **kw):
        """
        One HTTP call. Error text is truncated and never includes headers, so a
        credential cannot reach a public CI log through an exception.
        """
        for attempt in range(3):
            try:
                r = self._s.request(method, url, timeout=self.timeout, **kw)
            except requests.RequestException as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Alpaca request failed ({type(exc).__name__}). Network or "
                        f"endpoint problem.") from None
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise RuntimeError(
                    f"Alpaca {method} {url.rsplit('/', 1)[-1]} -> HTTP "
                    f"{r.status_code}: {r.text[:300]}")
            return r.json() if r.text else {}
        raise RuntimeError("Alpaca request retried out")

    # ------------------------------------------------------------------
    def get_account(self) -> Account:
        d = self._req("GET", f"{self.base}/v2/account")
        return Account(
            equity=float(d.get("equity", 0) or 0),
            cash=float(d.get("cash", 0) or 0),
            buying_power=float(d.get("buying_power", 0) or 0),
            options_level=int(d.get("options_trading_level", 0) or 0),
            is_paper=True, raw=d)

    def get_positions(self) -> list:
        out = []
        for d in self._req("GET", f"{self.base}/v2/positions") or []:
            if str(d.get("asset_class", "")) not in ("us_option", "option"):
                continue
            out.append(Position(
                symbol=d.get("symbol", ""),
                qty=int(float(d.get("qty", 0) or 0)),
                avg_price=float(d.get("avg_entry_price", 0) or 0),
                market_price=float(d.get("current_price", 0) or 0),
                unrealized_pl=float(d.get("unrealized_pl", 0) or 0),
                raw=d))
        return out

    def get_orders(self, status: str = "open") -> list:
        d = self._req("GET", f"{self.base}/v2/orders",
                      params={"status": status, "limit": 200, "nested": "true"})
        return [self._to_order(o) for o in (d or [])]

    def _to_order(self, d: dict) -> Order:
        legs = []
        for l in (d.get("legs") or []):
            legs.append(OrderLeg(symbol=l.get("symbol", ""), side=l.get("side", ""),
                                 ratio=int(float(l.get("ratio_qty", 1) or 1))))
        if not legs and d.get("symbol"):
            legs = [OrderLeg(symbol=d["symbol"], side=d.get("side", ""), ratio=1)]
        ts = d.get("submitted_at") or d.get("created_at")
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")) if ts else None
        except ValueError:
            dt = None
        fap = d.get("filled_avg_price")
        return Order(
            id=str(d.get("id", "")), status=str(d.get("status", "")), legs=legs,
            qty=int(float(d.get("qty", 0) or 0)),
            filled_qty=int(float(d.get("filled_qty", 0) or 0)),
            limit_price=float(d.get("limit_price") or 0),
            filled_avg_price=float(fap) if fap not in (None, "") else None,
            submitted_at=dt, tag=str(d.get("client_order_id", "")), raw=d)

    # ------------------------------------------------------------------
    def get_quotes(self, symbols: list) -> dict:
        """
        Live option quotes. This is the check that a scan from last night still
        describes the market this morning -- without it the trader would be
        acting on prices that are many hours stale.
        """
        if not symbols:
            return {}
        out: dict[str, Quote] = {}
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i + 100]
            d = self._req("GET", f"{DATA_BASE}/v1beta1/options/quotes/latest",
                          params={"symbols": ",".join(chunk)})
            for sym, q in (d.get("quotes") or {}).items():
                out[sym] = Quote(symbol=sym,
                                 bid=float(q.get("bp", 0) or 0),
                                 ask=float(q.get("ap", 0) or 0))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _net_limit(limit_price: float) -> str:
        """
        Net package price. Positive = debit paid, negative = credit received.
        The single place this convention is applied -- invert here if Alpaca
        rejects credit orders on the first live run.
        """
        return f"{limit_price:.2f}"

    def submit_multileg(self, legs: list, qty: int, limit_price: float,
                        tag: str = "") -> Order:
        payload = {
            "order_class": "mleg" if len(legs) > 1 else "simple",
            "qty": str(int(qty)),
            "type": "limit",
            "limit_price": self._net_limit(limit_price),
            "time_in_force": "day",
        }
        if len(legs) > 1:
            payload["legs"] = [
                {"symbol": l.symbol, "side": l.side,
                 "ratio_qty": str(int(l.ratio)),
                 "position_intent": ("buy_to_open" if l.side == "buy" else "sell_to_open")}
                for l in legs]
        else:
            payload["symbol"] = legs[0].symbol
            payload["side"] = legs[0].side
        if tag:
            payload["client_order_id"] = tag[:48]

        if self.dry_run:
            print("DRY RUN -- would POST /v2/orders:")
            print(json.dumps(payload, indent=2))
            return Order(id=f"dry-{tag or 'order'}", status="new", legs=legs,
                         qty=qty, limit_price=limit_price, tag=tag, raw=payload)

        return self._to_order(self._req("POST", f"{self.base}/v2/orders", json=payload))

    def replace_order(self, order_id: str, limit_price: float) -> Order:
        if self.dry_run:
            print(f"DRY RUN -- would PATCH /v2/orders/{order_id} "
                  f"limit_price={self._net_limit(limit_price)}")
            return Order(id=order_id, status="new", limit_price=limit_price)
        return self._to_order(self._req(
            "PATCH", f"{self.base}/v2/orders/{order_id}",
            json={"limit_price": self._net_limit(limit_price)}))

    def get_order(self, order_id: str) -> Order | None:
        try:
            return self._to_order(self._req("GET", f"{self.base}/v2/orders/{order_id}",
                                            params={"nested": "true"}))
        except RuntimeError:
            return None

    def cancel_order(self, order_id: str) -> bool:
        if self.dry_run:
            print(f"DRY RUN -- would DELETE /v2/orders/{order_id}")
            return True
        try:
            self._req("DELETE", f"{self.base}/v2/orders/{order_id}")
            return True
        except RuntimeError:
            return False

    def market_is_open(self) -> bool:
        try:
            return bool(self._req("GET", f"{self.base}/v2/clock").get("is_open", False))
        except RuntimeError:
            return False
