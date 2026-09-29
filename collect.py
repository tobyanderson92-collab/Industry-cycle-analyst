"""
Industry cycle data collector.

Pulls free public data for each industry in industries.yaml and writes:
  data/latest.md               human-readable report Claude reads each session
  data/latest.json             full structured output
  data/history/<date>.json     dated snapshot for quarter-on-quarter comparison
  data/history/index.txt       list of snapshot dates (newest last)

Sources: Yahoo Finance (prices, EPS estimate trends), FRED (prices,
capacity utilisation), SEC EDGAR XBRL (capex, D&A, margins).
Every failure is recorded in the data health section instead of stopping
the run. Nothing is estimated or filled in: missing means missing.
"""

import datetime as dt
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf

ROOT = Path(__file__).parent
DATA = ROOT / "data"
HIST = DATA / "history"
FRED_KEY = os.environ.get("FRED_API_KEY", "")
SEC_UA = os.environ.get("SEC_USER_AGENT", "Industry cycle research tool admin@example.com")

TODAY = dt.date.today()
health = {"prices_missing": [], "fred_failed": [], "edgar_failed": [],
          "revisions_failed": [], "notes": []}


def r(x, n=2):
    """Round for output; keep None as None."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), n)


def pct_rank(series, value):
    s = pd.Series(series).dropna()
    if len(s) < 5 or value is None:
        return None
    return float((s <= value).mean() * 100)


# ---------------------------------------------------------------- prices / RS

def download_prices(tickers):
    tickers = sorted(set(tickers))
    closes = {}
    # batch download in chunks to reduce rate-limit risk
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        try:
            df = yf.download(chunk, period="3y", interval="1d", auto_adjust=True,
                             progress=False, threads=True, group_by="column")
            close = df["Close"] if "Close" in df else df
            if isinstance(close, pd.Series):
                close = close.to_frame(chunk[0])
            for t in chunk:
                if t in close and close[t].dropna().shape[0] > 60:
                    closes[t] = close[t].dropna()
        except Exception as e:
            health["notes"].append(f"Yahoo batch download failed for {chunk}: {e}")
        time.sleep(2)
    missing = [t for t in tickers if t not in closes]
    health["prices_missing"].extend(missing)
    return closes


def basket_index(closes, tickers):
    avail = [t for t in tickers if t in closes]
    if not avail:
        return None, []
    rets = pd.concat([closes[t].pct_change() for t in avail], axis=1)
    idx = (1 + rets.mean(axis=1, skipna=True).fillna(0)).cumprod()
    return idx, avail


def rs_metrics(closes, tickers, bench):
    if not tickers or bench not in closes:
        return None
    idx, used = basket_index(closes, tickers)
    if idx is None:
        return None
    b = closes[bench]
    df = pd.concat([idx, b], axis=1, join="inner").dropna()
    df.columns = ["x", "b"]
    if len(df) < 260:
        return None
    ratio = df["x"] / df["b"]
    out = {"tickers_used": used}
    for label, n in [("rel_3m_pct", 63), ("rel_6m_pct", 126), ("rel_12m_pct", 252)]:
        out[label] = r((ratio.iloc[-1] / ratio.iloc[-n - 1] - 1) * 100, 1)
    ma200 = ratio.rolling(200).mean().iloc[-1]
    out["ratio_vs_200d_pct"] = r((ratio.iloc[-1] / ma200 - 1) * 100, 1)
    last252 = df["x"].iloc[-252:]
    out["off_52w_high_pct"] = r((df["x"].iloc[-1] / last252.max() - 1) * 100, 1)
    out["above_52w_low_pct"] = r((df["x"].iloc[-1] / last252.min() - 1) * 100, 1)
    # has the relative line made a higher low? compare 6m min vs prior 6m min
    out["rel_higher_low"] = bool(ratio.iloc[-126:].min() > ratio.iloc[-252:-126].min())
    return out


# ---------------------------------------------------------------- FRED

def fred_series(sid):
    if not FRED_KEY:
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": sid, "api_key": FRED_KEY, "file_type": "json",
              "observation_start": "2008-01-01"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            health["fred_failed"].append(f"{sid} (HTTP {resp.status_code})")
            return None
        obs = resp.json().get("observations", [])
        s = pd.Series({pd.Timestamp(o["date"]): float(o["value"])
                       for o in obs if o["value"] not in (".", "")})
        if s.empty:
            health["fred_failed"].append(f"{sid} (no data)")
            return None
        return s.resample("MS").mean().dropna()
    except Exception as e:
        health["fred_failed"].append(f"{sid} ({e})")
        return None


def fred_metrics(sid, kind):
    s = fred_series(sid)
    if s is None or len(s) < 30:
        return None
    last = s.iloc[-1]
    out = {"series": sid, "latest_date": s.index[-1].strftime("%Y-%m"),
           "latest": r(last, 2)}
    yoy = s.pct_change(12) * 100
    out["yoy_pct"] = r(yoy.iloc[-1], 1)
    out["yoy_6m_ago_pct"] = r(yoy.iloc[-7], 1) if len(yoy.dropna()) > 7 else None
    out["pct_rank_15y"] = r(pct_rank(s.iloc[-180:], last), 0)
    if kind == "capu":
        out["chg_12m_pts"] = r(last - s.iloc[-13], 1) if len(s) > 13 else None
    return out


# ---------------------------------------------------------------- EDGAR

CAPEX_TAGS = ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets",
              "PaymentsToAcquireOilAndGasPropertyAndEquipment",
              "PaymentsToAcquireOtherPropertyPlantAndEquipment",
              "PaymentsForCapitalImprovements"]
DA_TAGS = ["Depreciation", "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
           "DepreciationAmortizationAndAccretionNet", "DepreciationAndAmortizationExcludingNonrecurringCharges",
           "DepreciationDepletionAndAmortizationExcludingNonrecurringCharges"]
REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense"]
# operating income; pretax income as fallback for companies with no operating line
OPINC_TAGS = ["OperatingIncomeLoss",
              "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
              "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"]

_cik_map = None


def cik_for(ticker):
    global _cik_map
    if _cik_map is None:
        try:
            resp = requests.get("https://www.sec.gov/files/company_tickers.json",
                                headers={"User-Agent": SEC_UA}, timeout=30)
            _cik_map = {v["ticker"].upper(): int(v["cik_str"])
                        for v in resp.json().values()}
        except Exception as e:
            health["notes"].append(f"SEC ticker map failed: {e}")
            _cik_map = {}
    return _cik_map.get(ticker.upper().replace("-", "."), _cik_map.get(ticker.upper()))


def fiscal_label(end):
    """Label a fiscal year by the calendar year it mostly covers, so a year
    ending March 2026 counts as 2025 alongside December-2025 year ends."""
    return end.year if end.month >= 7 else end.year - 1


def tag_values(gaap, tag):
    units = gaap.get(tag, {}).get("units", {}).get("USD", [])
    best = {}
    for u in units:
        if u.get("form") not in ("10-K", "10-K/A") or "start" not in u:
            continue
        d0, d1 = pd.Timestamp(u["start"]), pd.Timestamp(u["end"])
        if not 340 <= (d1 - d0).days <= 380:
            continue
        yr = fiscal_label(d1)
        # keep the most recently filed figure for each period
        if yr not in best or u.get("filed", "") > best[yr][1]:
            best[yr] = (u["val"], u.get("filed", ""))
    return {y: v for y, (v, _) in best.items()}


def annual_values(facts, tags):
    """Return {fiscal_year: value}. Prefer a single tag (highest priority
    with 5+ years including the latest year) so the series doesn't jump
    between definitions; otherwise fall back to merging tags by priority."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    per_tag = [(t, tag_values(gaap, t)) for t in tags]
    per_tag = [(t, v) for t, v in per_tag if v]
    if not per_tag:
        return {}
    latest = max(max(v) for _, v in per_tag)
    for _, v in per_tag:
        if len(v) >= 5 and max(v) == latest:
            return v
    merged = {}
    for _, v in per_tag:
        for y, val in v.items():
            merged.setdefault(y, val)
    return merged


def pair_series(comps, num_key, den_key):
    """Aggregate ratio sum(num)/sum(den) across companies, per fiscal year.

    Active set: companies whose paired data reaches the industry's typical
    latest year (the most common latest year across companies). Companies
    whose data stops earlier (tag change, delisting, merger) are dropped
    and reported, rather than silently dragging the latest year back.
    The latest year must include every active company; history years need
    at least 75% of them."""
    spans = {}
    for t, c in comps.items():
        yrs = set(c[num_key]) & set(c[den_key])
        yrs = {y for y in yrs if y <= TODAY.year}
        if len(yrs) >= 3:
            spans[t] = yrs
    if not spans:
        return None, [], list(comps)
    latests = [max(v) for v in spans.values()]
    typical = max(set(latests), key=lambda y: (latests.count(y), y))
    active = [t for t, v in spans.items() if max(v) >= typical]
    dropped = [t for t in comps if t not in active]
    need = max(1, int(np.ceil(0.75 * len(active))))
    out = {}
    for y in range(min(min(spans[t]) for t in active), typical + 1):
        have = [t for t in active if y in spans[t]]
        if len(have) < need or (y == typical and len(have) < len(active)):
            continue
        den = sum(comps[t][den_key][y] for t in have)
        if den > 0:
            out[y] = sum(comps[t][num_key][y] for t in have) / den
    return (pd.Series(out).sort_index() if out else None), active, dropped


def edgar_industry(tickers, capex_relevant, cache):
    comps = {}
    for t in tickers:
        if t not in cache:
            cache[t] = edgar_company(t)
        if cache[t]:
            comps[t] = cache[t]
    if not comps:
        return None
    out = {}

    if capex_relevant:
        s, active, dropped = pair_series(comps, "capex", "da")
        if s is not None and len(s) >= 4:
            out["capex_companies"] = active
            if dropped:
                out["capex_dropped"] = dropped
            out["capex_da_latest"] = r(s.iloc[-1])
            out["capex_da_latest_year"] = int(s.index[-1])
            out["capex_da_3y_avg"] = r(s.iloc[-3:].mean())
            out["capex_da_longrun_avg"] = r(s.mean())
            out["capex_da_years_below_1_last5"] = int((s.iloc[-5:] < 1).sum())
            out["capex_da_history"] = {int(k): r(v) for k, v in s.items()}

    s, active, dropped = pair_series(comps, "opinc", "rev")
    if s is not None and len(s) >= 4:
        s = s * 100
        out["margin_companies"] = active
        if dropped:
            out["margin_dropped"] = dropped
        if s.iloc[-4:].abs().max() > 100:
            out["margin_note"] = "not meaningful (revenue too small relative to costs, e.g. pre-production)"
        else:
            out["op_margin_latest_pct"] = r(s.iloc[-1], 1)
            out["op_margin_latest_year"] = int(s.index[-1])
            out["op_margin_prior_pct"] = r(s.iloc[-2], 1)
            out["op_margin_pct_rank"] = r(pct_rank(s, s.iloc[-1]), 0) if len(s) >= 6 else None
            out["op_margin_history"] = {int(k): r(v, 1) for k, v in s.items()}
    return out or None


def edgar_diagnostics(cache):
    """Per company: year range found for each field, so gaps can be fixed."""
    diag = {}
    for t, c in cache.items():
        if not c:
            continue
        diag[t] = {k: (f"{min(v)}-{max(v)}" if v else "none") for k, v in c.items()
                   if k in ("capex", "da", "rev", "opinc")}
    return diag


# ---------------------------------------------------------------- revisions

def eps_revision(ticker):
    """% change in next-fiscal-year consensus EPS vs 90 days ago."""
    try:
        tr = yf.Ticker(ticker).eps_trend
        if tr is None or tr.empty or "+1y" not in tr.index:
            return None
        row = tr.loc["+1y"]
        now, then = row.get("current"), row.get("90daysAgo")
        if now is None or then is None or pd.isna(now) or pd.isna(then) or then <= 0:
            return None
        return (now / then - 1) * 100
    except Exception:
        return None


def revisions_industry(tickers, cache):
    vals = {}
    for t in tickers:
        if t not in cache:
            cache[t] = eps_revision(t)
            time.sleep(0.5)
        if cache[t] is not None:
            vals[t] = r(cache[t], 1)
        else:
            health["revisions_failed"].append(t)
    if not vals:
        return None
    return {"median_fy2_eps_rev_90d_pct": r(np.median(list(vals.values())), 1),
            "by_ticker": vals}


# ---------------------------------------------------------------- screen

def screen(ind):
    """Transparent signal counts. Each signal is True/False/None (no data).
    None never counts for or against."""
    e, pr, cu = ind.get("edgar") or {}, ind.get("price") or {}, ind.get("capu") or {}
    rv, rs_us, rs_asx = ind.get("revisions") or {}, ind.get("rs_us") or {}, ind.get("rs_asx") or {}

    def lt(a, b):
        return None if a is None or b is None else a < b

    def gt(a, b):
        return None if a is None or b is None else a > b

    supply = {
        "capex_below_DA_latest": lt(e.get("capex_da_latest"), 1.0),
        "capex_3y_below_longrun": lt(e.get("capex_da_3y_avg"),
                                     None if e.get("capex_da_longrun_avg") is None
                                     else 0.9 * e["capex_da_longrun_avg"]),
    }
    trough = {
        "margin_bottom_40pct": lt(e.get("op_margin_pct_rank"), 40),
        "price_bottom_40pct_15y": lt(pr.get("pct_rank_15y"), 40),
        "capu_bottom_40pct_15y": lt(cu.get("pct_rank_15y"), 40),
    }
    rs = rs_us or rs_asx
    turn = {
        "margin_up_yoy": gt(e.get("op_margin_latest_pct"), e.get("op_margin_prior_pct")),
        "price_yoy_improving": gt(pr.get("yoy_pct"), pr.get("yoy_6m_ago_pct")),
        "capu_up_12m": gt(cu.get("chg_12m_pts"), 0),
        "eps_revisions_positive": gt(rv.get("median_fy2_eps_rev_90d_pct"), 0),
        "rel_strength_above_200d": gt(rs.get("ratio_vs_200d_pct"), 0),
        "rel_higher_low": rs.get("rel_higher_low") if rs else None,
    }

    def tally(d):
        vals = [v for v in d.values() if v is not None]
        return {"hits": sum(vals), "available": len(vals), "of": len(d)}

    return {"supply": supply, "trough": trough, "turn": turn,
            "supply_tally": tally(supply), "trough_tally": tally(trough),
            "turn_tally": tally(turn)}


def share(t):
    return t["hits"] / t["available"] if t["available"] else 0


# ---------------------------------------------------------------- report

def fmt(v, suffix=""):
    return "n/a" if v is None else f"{v}{suffix}"


def write_markdown(results, meta):
    L = []
    L.append(f"# Industry cycle data — {meta['run_date']}\n")
    L.append("Generated automatically. Signal counts are a screen, not a verdict. "
             "`n/a` means the data was unavailable; it is never estimated.\n")

    L.append("## Screen summary\n")
    L.append("Sorted by (supply+trough share) + turn share. Format hits/available (of total).\n")
    L.append("| Industry | Supply | Trough | Turn | Rel 6m US | Rel 6m ASX | Capex/D&A | Margin rank | FY2 EPS rev 90d |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ind in results:
        s = ind["screen"]
        t = lambda k: f"{s[k]['hits']}/{s[k]['available']} ({s[k]['of']})"
        e = ind.get("edgar") or {}
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            ind["name"], t("supply_tally"), t("trough_tally"), t("turn_tally"),
            fmt((ind.get("rs_us") or {}).get("rel_6m_pct"), "%"),
            fmt((ind.get("rs_asx") or {}).get("rel_6m_pct"), "%"),
            fmt(e.get("capex_da_latest")),
            fmt(e.get("op_margin_pct_rank"), "th"),
            fmt((ind.get("revisions") or {}).get("median_fy2_eps_rev_90d_pct"), "%")))

    L.append("\n## Industry detail\n")
    for ind in results:
        L.append(f"### {ind['name']}")
        e = ind.get("edgar")
        if e:
            if "capex_da_latest" in e:
                L.append(f"- Capex/D&A ({', '.join(e['capex_companies'])}): {e['capex_da_latest']} in "
                         f"{e['capex_da_latest_year']}, 3y avg {e['capex_da_3y_avg']} vs long-run "
                         f"{e['capex_da_longrun_avg']}, years <1.0 in last 5: {e['capex_da_years_below_1_last5']}"
                         + (f" [dropped, data ends early: {', '.join(e['capex_dropped'])}]" if e.get('capex_dropped') else ""))
                L.append(f"  - History: {e['capex_da_history']}")
            if "op_margin_latest_pct" in e:
                L.append(f"- Op margin ({', '.join(e['margin_companies'])}): {e['op_margin_latest_pct']}% in "
                         f"{e['op_margin_latest_year']} (prior {e['op_margin_prior_pct']}%, "
                         f"{fmt(e.get('op_margin_pct_rank'), 'th')} pct of own history)"
                         + (f" [dropped, data ends early: {', '.join(e['margin_dropped'])}]" if e.get('margin_dropped') else ""))
                L.append(f"  - History: {e['op_margin_history']}")
            if e.get("margin_note"):
                L.append(f"- Op margin: {e['margin_note']}")
        else:
            L.append("- Fundamentals: n/a")
        for key, label in [("price", "Pricing"), ("capu", "Capacity utilisation")]:
            p = ind.get(key)
            if p:
                extra = f", 12m chg {fmt(p.get('chg_12m_pts'))} pts" if key == "capu" else ""
                L.append(f"- {label} [{p['series']}, {p['latest_date']}]: {p['latest']}, "
                         f"YoY {fmt(p.get('yoy_pct'), '%')} (6m ago {fmt(p.get('yoy_6m_ago_pct'), '%')}), "
                         f"{fmt(p.get('pct_rank_15y'), 'th')} pct of 15y{extra}")
        for key, label in [("rs_us", "Rel strength US vs SPY"), ("rs_asx", "Rel strength ASX vs STW")]:
            x = ind.get(key)
            if x:
                L.append(f"- {label} ({', '.join(x['tickers_used'])}): 3m {fmt(x['rel_3m_pct'], '%')}, "
                         f"6m {fmt(x['rel_6m_pct'], '%')}, 12m {fmt(x['rel_12m_pct'], '%')}, "
                         f"vs 200d {fmt(x['ratio_vs_200d_pct'], '%')}, higher low: {x['rel_higher_low']}, "
                         f"off 52w high {fmt(x['off_52w_high_pct'], '%')}")
        rv = ind.get("revisions")
        if rv:
            L.append(f"- FY2 EPS revisions 90d: median {rv['median_fy2_eps_rev_90d_pct']}% {rv['by_ticker']}")
        sig = ind["screen"]
        for grp in ("supply", "trough", "turn"):
            L.append(f"- {grp.title()} signals: " + ", ".join(
                f"{k}={'n/a' if v is None else ('Y' if v else 'N')}" for k, v in sig[grp].items()))
        L.append("")

    L.append("## Data health\n")
    if meta.get("edgar_diag"):
        L.append("EDGAR year ranges per company (capex / D&A / revenue / operating income):")
        L.append("; ".join(f"{t}: {d['capex']} / {d['da']} / {d['rev']} / {d['opinc']}"
                           for t, d in sorted(meta["edgar_diag"].items())) + "\n")
    for k, v in health.items():
        if v:
            L.append(f"- **{k}**: {', '.join(sorted(set(map(str, v))))}")
    if not any(health.values()):
        L.append("- All sources returned data.")
    (DATA / "latest.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    cfg = yaml.safe_load((ROOT / "industries.yaml").read_text())
    inds = cfg["industries"]
    # guard: YAML turns unquoted tickers like ON / YES / NO into booleans
    for i in inds:
        for k in ("rs_us", "rs_asx", "bellwethers"):
            bad = [t for t in i.get(k, []) if not isinstance(t, str)]
            if bad:
                health["notes"].append(f"{i['name']} {k}: non-text ticker {bad} - put it in quotes in industries.yaml")
                i[k] = [t for t in i[k] if isinstance(t, str)]
    bus, basx = cfg["benchmarks"]["us"], cfg["benchmarks"]["asx"]
    if not FRED_KEY:
        health["notes"].append("FRED_API_KEY not set; pricing and utilisation skipped")

    all_tickers = {bus, basx}
    for i in inds:
        all_tickers.update(i.get("rs_us", []), i.get("rs_asx", []))
    closes = download_prices(list(all_tickers))

    edgar_cache, rev_cache, fred_cache = {}, {}, {}
    results = []
    for i in inds:
        print("Processing", i["name"], flush=True)
        out = {"name": i["name"]}
        out["rs_us"] = rs_metrics(closes, i.get("rs_us", []), bus)
        out["rs_asx"] = rs_metrics(closes, i.get("rs_asx", []), basx)
        for key, kind in [("fred_price", "price"), ("fred_capu", "capu")]:
            sid = i.get(key)
            if sid:
                if sid not in fred_cache:
                    fred_cache[sid] = fred_metrics(sid, kind)
                out[kind] = fred_cache[sid]
        bw = i.get("bellwethers", [])
        out["edgar"] = edgar_industry(bw, i.get("capex_relevant", True), edgar_cache)
        out["revisions"] = revisions_industry(bw, rev_cache)
        out["screen"] = screen(out)
        results.append(out)

    results.sort(key=lambda x: -(
        (share(x["screen"]["supply_tally"]) + share(x["screen"]["trough_tally"])) / 2
        + share(x["screen"]["turn_tally"])))

    meta = {"run_date": TODAY.isoformat(), "benchmarks": cfg["benchmarks"],
            "edgar_diag": edgar_diagnostics(edgar_cache)}
    DATA.mkdir(exist_ok=True)
    HIST.mkdir(exist_ok=True)
    payload = {"meta": meta, "health": health, "industries": results}
    (DATA / "latest.json").write_text(json.dumps(payload, indent=1, default=str))
    (HIST / f"{TODAY.isoformat()}.json").write_text(json.dumps(payload, default=str))
    dates = sorted(p.stem for p in HIST.glob("*.json"))
    (HIST / "index.txt").write_text("\n".join(dates) + "\n")
    write_markdown(results, meta)
    print("Done.", {k: len(v) for k, v in health.items()})


if __name__ == "__main__":
    main()
