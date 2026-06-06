"""Helius API client for Solana token transfer history."""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from night_shift.core.logging import log

HELIUS_BASE = "https://api.helius.xyz/v0"
DEFAULT_HISTORY_DAYS = 180
DEFAULT_REQUEST_DELAY_S = 0.25
MAX_RETRIES = 5


def get_api_key() -> Optional[str]:
    return os.environ.get("HELIUS_API_KEY") or os.environ.get("HELIUS_API_KEY_ID")


def fetch_address_transactions(
    address: str,
    api_key: Optional[str] = None,
    limit: int = 100,
    before: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch parsed transactions for a Solana address (mint or wallet)."""
    key = api_key or get_api_key()
    if not key:
        return []

    params = {"api-key": key, "limit": str(limit)}
    if before:
        params["before"] = before
    url = f"{HELIUS_BASE}/addresses/{address}/transactions?{urllib.parse.urlencode(params)}"

    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
                return data if isinstance(data, list) else []
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                sleep_s = DEFAULT_REQUEST_DELAY_S * (2**attempt)
                log(f"  Helius HTTP {exc.code} for {address[:12]}... retry in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue
            log(f"  Helius HTTP {exc.code} for {address[:12]}...")
            return []
        except urllib.error.URLError as exc:
            if attempt < MAX_RETRIES - 1:
                time.sleep(DEFAULT_REQUEST_DELAY_S * (2**attempt))
                continue
            log(f"  Helius network error: {exc.reason}")
            return []
    return []


def extract_transfer_events(transactions: List[Dict[str, Any]], mint: str) -> List[Dict[str, Any]]:
    """Parse Helius enhanced transactions into transfer/mint/burn events."""
    events: List[Dict[str, Any]] = []
    for tx in transactions:
        ts = tx.get("timestamp")
        if ts is None:
            continue

        for transfer in tx.get("tokenTransfers", []):
            if transfer.get("mint") != mint:
                continue
            amount = float(transfer.get("tokenAmount", 0) or 0)
            if amount <= 0:
                continue
            events.append({"timestamp": ts, "amount": amount, "type": "transfer"})

        for change in tx.get("accountData", []):
            token_changes = change.get("tokenBalanceChanges", [])
            for tc in token_changes:
                if tc.get("mint") != mint:
                    continue
                raw = tc.get("rawTokenAmount", {})
                delta = float(raw.get("tokenAmount", 0) or 0)
                if delta > 0:
                    events.append({"timestamp": ts, "amount": delta, "type": "mint"})
                elif delta < 0:
                    events.append({"timestamp": ts, "amount": abs(delta), "type": "burn"})

    return events


def fetch_token_daily_history(
    mint: str,
    days: int = DEFAULT_HISTORY_DAYS,
    api_key: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Paginate Helius transactions until history cutoff or pagination ends.

    max_pages=None paginates until cutoff (recommended for batch backfill).
    """
    key = api_key or get_api_key()
    if not key:
        return []

    all_events: List[Dict[str, Any]] = []
    before: Optional[str] = None
    cutoff = time.time() - days * 86400
    page = 0

    while True:
        if max_pages is not None and page >= max_pages:
            break

        txs = fetch_address_transactions(mint, api_key=key, limit=100, before=before)
        page += 1
        if not txs:
            break

        all_events.extend(extract_transfer_events(txs, mint))
        oldest = min((t.get("timestamp", time.time()) for t in txs), default=time.time())
        if oldest < cutoff:
            break

        before = txs[-1].get("signature")
        if not before:
            break
        time.sleep(DEFAULT_REQUEST_DELAY_S)

    return [e for e in all_events if e.get("timestamp", 0) >= cutoff]