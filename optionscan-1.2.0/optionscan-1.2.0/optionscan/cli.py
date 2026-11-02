"""
Command-line entry point.

    python -m optionscan scan                       # live scan, default universe
    python -m optionscan scan --provider synthetic   # offline demo, no network
    python -m optionscan scan --tickers AAPL,MSFT,SPY
    python -m optionscan scan --account 50000 --risk 1.5
    python -m optionscan init-config my.yaml
    python -m optionscan backtest --tickers SPY,QQQ
    python -m optionscan history
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import Config
from .data.cache import DiskCache, IVHistoryStore, PositionLog
from .universe import PRESETS, resolve


def _provider(name: str, cfg: Config):
    if name == "synthetic":
        from .data.synthetic import SyntheticProvider
        return SyntheticProvider()
    if name == "yahoo":
        from .data.yahoo import YahooProvider
        return YahooProvider(cache=DiskCache(enabled=cfg.use_cache))
    raise SystemExit(f"Unknown provider '{name}'. Choose: yahoo, synthetic")


def _build_cfg(args) -> Config:
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()
    if getattr(args, "provider", None):
        cfg.provider = args.provider
    if getattr(args, "account", None):
        cfg.account.account_size = float(args.account)
    if getattr(args, "risk", None):
        cfg.account.risk_per_trade_pct = float(args.risk)
    if getattr(args, "min_score", None) is not None:
        cfg.output.min_score = float(args.min_score)
    if getattr(args, "max_ideas", None):
        cfg.output.max_ideas = int(args.max_ideas)
    if getattr(args, "workers", None):
        cfg.max_workers = int(args.workers)
    if getattr(args, "out", None):
        cfg.output.directory = args.out
    if getattr(args, "allow_earnings", False):
        cfg.earnings.allow_through_earnings = True
    if getattr(args, "no_cache", False):
        cfg.use_cache = False
    if getattr(args, "tickers", None):
        cfg.tickers = resolve(args.tickers)
    elif getattr(args, "universe", None):
        cfg.tickers = resolve(args.universe)
    return cfg


def cmd_scan(args) -> int:
    from .core.scanner import Scanner
    from .report import exports
    from .report.console import format_report

    cfg = _build_cfg(args)
    tickers = cfg.tickers or resolve("default")
    prov = _provider(cfg.provider, cfg)
    store = IVHistoryStore()

    quiet = args.quiet
    if not quiet:
        print(f"Scanning {len(tickers)} symbols via {prov.name} ... ", flush=True)

    interactive = sys.stdout.isatty()

    def progress(done, total, tk, status):
        if not quiet and interactive:
            bar = int(done / total * 28)
            sys.stdout.write(f"\r  [{'#' * bar}{'.' * (28 - bar)}] {done}/{total} "
                             f"{tk:<7}{status:<10}")
            sys.stdout.flush()

    res = Scanner(prov, cfg, store).scan(tickers, progress=progress)
    if not quiet and interactive:
        sys.stdout.write("\r" + " " * 72 + "\r")

    print(format_report(res, cfg, show_diagnostics=not args.brief,
                        show_detail=not args.brief))

    out = Path(cfg.output.directory)
    stamp = res.as_of.strftime("%Y-%m-%d")
    written = []
    if cfg.output.html:
        written.append(exports.write_html(res, cfg, out / f"scan_{stamp}.html"))
    if cfg.output.csv:
        written.append(exports.write_csv(res, out / f"scan_{stamp}.csv"))
    if cfg.output.json:
        written.append(exports.write_json(res, out / f"scan_{stamp}.json"))
    written.append(exports.write_text(res, cfg, out / f"scan_{stamp}.txt"))
    n = exports.log_suggestions(res, PositionLog())

    print(f"\nWritten:")
    for w in written:
        print(f"  {w}")
    print(f"  {n} suggestion(s) appended to the audit log "
          f"(~/.optionscan/suggestions.jsonl)")
    if res.provider == "synthetic":
        print("\n  NOTE: simulated data. Use --provider yahoo for live quotes.")
    return 0


def cmd_init_config(args) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} already exists. Use --force to overwrite.")
        return 1
    Config().save(path)
    print(f"Wrote starter config to {path}\nEdit it, then run:\n"
          f"  python -m optionscan scan --config {path}")
    return 0


def cmd_backtest(args) -> int:
    from .backtest import run_backtest, format_backtest
    cfg = _build_cfg(args)
    prov = _provider(cfg.provider, cfg)
    tickers = cfg.tickers or resolve("small")
    res = run_backtest(prov, tickers, horizons=(5, 10, 21, 42),
                       lookback=int(args.lookback))
    print(format_backtest(res))
    return 0


def cmd_history(args) -> int:
    store = IVHistoryStore()
    df = store.load()
    if df.empty:
        print("No IV history recorded yet. Each scan appends one row per symbol;\n"
              "after about 40 trading days IV rank switches from the estimated\n"
              "proxy to a true IV rank.")
        return 0
    g = df.groupby("ticker").agg(obs=("atm_iv", "size"), first=("date", "min"),
                                 last=("date", "max"), mean_iv=("atm_iv", "mean"))
    g = g.sort_values("obs", ascending=False)
    print(f"IV history: {len(df):,} rows across {df['ticker'].nunique()} symbols\n")
    print(f"{'SYMBOL':<8}{'OBS':>5}{'MEAN IV':>9}  {'FROM':<12}{'TO':<12}STATUS")
    for tk, row in g.iterrows():
        status = "true IV rank" if row["obs"] >= 40 else f"proxy ({40 - int(row['obs'])} more needed)"
        print(f"{tk:<8}{int(row['obs']):>5}{row['mean_iv']:>8.1%}  "
              f"{row['first']:%Y-%m-%d}  {row['last']:%Y-%m-%d}  {status}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="optionscan",
        description="Daily options scanner: finds statistically favourable setups, "
                    "picks the contracts, and writes a complete trade plan with "
                    "position sizing. A screening tool, not financial advice.")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--config", help="YAML or JSON config file")
        p.add_argument("--provider", choices=["yahoo", "synthetic"],
                       help="data source (default yahoo; synthetic runs offline)")
        p.add_argument("--tickers", help="comma-separated symbols, or a preset name: "
                                         + ", ".join(PRESETS))
        p.add_argument("--universe", help="preset universe name")
        p.add_argument("--account", type=float, help="account size in dollars")
        p.add_argument("--risk", type=float, help="percent of account risked per trade")
        p.add_argument("--workers", type=int, help="parallel data fetches")
        p.add_argument("--no-cache", action="store_true")

    s = sub.add_parser("scan", help="run today's scan")
    common(s)
    s.add_argument("--out", help="output directory (default ./output)")
    s.add_argument("--min-score", type=float, help="minimum score to report")
    s.add_argument("--max-ideas", type=int, help="maximum ideas to report")
    s.add_argument("--allow-earnings", action="store_true",
                   help="permit short premium through an earnings report (off by default)")
    s.add_argument("--brief", action="store_true", help="summary table only")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("init-config", help="write a starter config file")
    s.add_argument("path", nargs="?", default="optionscan.yaml")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_init_config)

    s = sub.add_parser("backtest", help="validate the regime signals on history")
    common(s)
    s.add_argument("--lookback", type=int, default=750, help="bars of history to test")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("history", help="show recorded IV history / IV-rank readiness")
    s.set_defaults(func=cmd_history)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
