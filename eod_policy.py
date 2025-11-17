from typing import Optional, Callable

from models import SymbolState, TargetPosition
from config import EOD_POLICIES, DEFAULT_EOD_POLICY


def _pnl_percent(state: SymbolState) -> Optional[float]:
    return state.pnl_percent()


# ---------------------------------------------------------------------------
# Policy-type handlers
# ---------------------------------------------------------------------------

def _eod_no_eod_action(
    state: SymbolState,
    target: TargetPosition,
    pnl_pct: Optional[float],
    cfg: dict,
) -> TargetPosition:
    """Do nothing at EOD; whatever the normal strategy decided stands."""
    return target


def _eod_always_flatten(
    state: SymbolState,
    target: TargetPosition,
    pnl_pct: Optional[float],
    cfg: dict,
) -> TargetPosition:
    """Always flatten at EOD, regardless of PnL."""
    return TargetPosition(
        symbol=state.symbol,
        target_qty=0.0,
        reason="EOD policy: always_flatten; forcing flat at EOD.",
    )


def _eod_band_hold(
    state: SymbolState,
    target: TargetPosition,
    pnl_pct: Optional[float],
    cfg: dict,
) -> TargetPosition:
    """
    Allow overnight only if min_pnl_pct <= PnL% <= max_pnl_pct.
    If outside the band, flatten.
    """
    if pnl_pct is None:
        # Can't compute PnL% → fall back to original target
        return target

    min_pnl = cfg.get("min_pnl_pct", -5.0)
    max_pnl = cfg.get("max_pnl_pct", 5.0)

    if pnl_pct < min_pnl or pnl_pct > max_pnl:
        return TargetPosition(
            symbol=state.symbol,
            target_qty=0.0,
            reason=(
                "EOD policy: band_hold; "
                f"PnL {pnl_pct:.2f}% outside [{min_pnl}%, {max_pnl}%]; "
                "flattening before overnight."
            ),
        )

    # Within allowed band → hold current position overnight
    return TargetPosition(
        symbol=state.symbol,
        target_qty=state.position_qty,
        reason=(
            "EOD policy: band_hold; "
            f"PnL {pnl_pct:.2f}% within [{min_pnl}%, {max_pnl}%]; "
            "allowing overnight hold."
        ),
    )


def _eod_min_profit_flatten(
    state: SymbolState,
    target: TargetPosition,
    pnl_pct: Optional[float],
    cfg: dict,
) -> TargetPosition:
    """
    Flatten at EOD if PnL% >= min_pnl_pct; otherwise hold overnight.
    """
    if pnl_pct is None:
        return target

    min_pnl = cfg.get("min_pnl_pct", 1.0)

    if pnl_pct >= min_pnl:
        return TargetPosition(
            symbol=state.symbol,
            target_qty=0.0,
            reason=(
                "EOD policy: min_profit_flatten; "
                f"PnL {pnl_pct:.2f}% >= {min_pnl}%; flattening before overnight."
            ),
        )

    return TargetPosition(
        symbol=state.symbol,
        target_qty=state.position_qty,
        reason=(
            "EOD policy: min_profit_flatten; "
            f"PnL {pnl_pct:.2f}% < {min_pnl}%; allowing overnight hold."
        ),
    )


# Map policy type strings to handler functions
_POLICY_HANDLERS: dict[str, Callable[[SymbolState, TargetPosition, Optional[float], dict], TargetPosition]] = {
    "no_eod_action": _eod_no_eod_action,
    "always_flatten": _eod_always_flatten,
    "band_hold": _eod_band_hold,
    "min_profit_flatten": _eod_min_profit_flatten,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_eod_policy(state: SymbolState, target: TargetPosition) -> TargetPosition:
    """
    Apply the configured EOD policy for this symbol (if any) on top of the
    normal strategy decision.

    - If there's no open position, return target unchanged.
    - Look up EOD_POLICIES[symbol]; fall back to DEFAULT_EOD_POLICY if missing.
    - Dispatch by cfg["type"] to one of the policy handlers.
    """
    # No position → no EOD action.
    if state.position_qty == 0.0:
        return target

    cfg = EOD_POLICIES.get(state.symbol, DEFAULT_EOD_POLICY)
    policy_type = cfg.get("type", "no_eod_action")

    handler = _POLICY_HANDLERS.get(policy_type)
    if handler is None:
        # Unknown policy type → safest is to do nothing special at EOD.
        return target

    pnl_pct = _pnl_percent(state)
    return handler(state, target, pnl_pct, cfg)
