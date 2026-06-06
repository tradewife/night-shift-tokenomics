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

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data if isinstance(data, list) else []
    except urllib.error.HTTPError as exc:
        log(f"  Helius HTTP {exc.code} for {address[:12]}...")
        return []
    except urllib.error.URLError as exc:
        log(f"  Helius network error: {exc.reason}")
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
    max_pages: int = 5,
) -> List[Dict[str, Any]]:
    """
    Paginate Helius transactions and return raw transfer events.

    Limited pagination keeps fetch cost reasonable for overnight runs.
    """
    key = api_key or get_api_key()
    if not key:
        return []

    all_events: List[Dict[str, Any]] = []
    before: Optional[str] = None
    cutoff = time.time() - days * 86400

    for _ in range(max_pages):
        txs = fetch_address_transactions(mint, api_key=key, limit=100, before=before)
        if not txs:
            break

        all_events.extend(extract_transfer_events(txs, mint))
        oldest = min((t.get("timestamp", time.time()) for t in txs), default=time.time())
        if oldest < cutoff:
            break

        before = txs[-1].get("signature")
        if not before:
            break
        time.sleep(0.15)

    return [e for e in all_events if e.get("timestamp", 0) >= cutoff]