"""
Snapshot Triangle Builder

Converts the SynthETIC transaction dataset into a snapshot-date-triangle
dataset for individual claim reserving, per the spec in CLAUDE.md (Phase 2)
and the reference paper (Llaguno et al., "Reserving with Machine Learning:
Applications for Loyalty Programs and Individual Insurance Claims", CAS
E-Forum Summer 2017).

build_snapshot_triangle(df) is core mechanism only (8 columns) -- matches
notebooks/02_snapshot_triangle_build.ipynb exactly, no feature engineering,
just the claim/snapshot/age grid and the future_paid target. Feature
engineering for modeling (Phase 3+) lives in feature_engineering.py instead,
built on top of this module's output.

Scope: dataset construction only. No modeling, no clustering, no chain ladder.
"""
import math
import pandas as pd

CUTOFF = 40

# Grid spacing (periods) between both snapshots AND observation ages -- the two
# must match, or the triangle degenerates into a parallelogram and the paper's
# cell-count formula N(N-1)/2 no longer holds (see CLAUDE.md). Default 1 = the
# finest grid the data supports (payment_period is the atomic unit); pass
# STEP=3 for a coarser, more classically "quarterly-report-cadence" triangle.
STEP = 1


def ceil_period(t):
    """Round a continuous time value up to its period bucket (period=quarter)."""
    return math.ceil(t) if t != int(t) else max(1, int(t))


def load_visible(df, cutoff=CUTOFF):
    """Everything the build is allowed to see. Enforced by dropping the rows at
    load time, not by discipline downstream."""
    visible = df[df["payment_period"] <= cutoff].copy()
    # real payments are whole cents; round here once so every downstream sum is
    # cent-exact and stays consistent regardless of summation order
    visible["payment_size"] = visible["payment_size"].round(2)
    return visible


def load_truth(df):
    """Full uncensored data. Scoring only -- never features, never training."""
    return df.copy()


def snapshot_grid(cutoff=CUTOFF, step=STEP):
    """The shared grid every claim's snapshots (and observation ages) are drawn
    from: step, 2*step, ... <= cutoff. One grid for the whole portfolio, not
    per-claim offsets, so per-cluster aggregate triangles built later don't
    have misaligned rows.

    Example:
        >>> snapshot_grid(cutoff=40, step=3)
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]

        >>> snapshot_grid(cutoff=40, step=1)
        [1, 2, 3, 4, ..., 40]   # every single period

    A claim doesn't get its own personal grid -- it just joins this one
    shared list at whichever stop comes first at or after its notification
    period, like catching the next scheduled bus rather than calling one
    just for you. See build_snapshot_triangle()'s `entry` line for exactly
    where that "which stop do I join at" decision happens.
    """
    last = cutoff - (cutoff % step)
    return list(range(step, last + 1, step))


def build_snapshot_triangle(df, cutoff=CUTOFF, step=STEP):
    """Core mechanism only -- claim_no, snapshot_period, observation_age,
    observation_period, row_status, future_paid, paid_to_date, is_settled_at_obs.
    No feature columns -- see feature_engineering.add_features() for that."""
    visible = load_visible(df, cutoff)
    grid = snapshot_grid(cutoff, step)

    # claim identity (occurrence, notidel, setldel) comes from the FULL data, not
    # `visible` -- a claim notified before the cutoff but with zero payments yet
    # (nothing due until after the cutoff) has zero rows in `visible` and would
    # silently vanish otherwise, even though it's a valid "known, nothing paid
    # yet" claim that belongs in the triangle
    claims = df.drop_duplicates("claim_no").copy()
    claims["notification_period"] = (claims["occurrence_time"] + claims["notidel"]).apply(ceil_period)
    claims["settlement_period"] = (
        claims["occurrence_time"] + claims["notidel"] + claims["setldel"]
    ).apply(ceil_period)

    n_before = len(claims)
    # notified AT the wall (not just after it) is also excluded: with nothing but the
    # wall itself left, there's no room for even one observed row to show development.
    # (The ~2 such claims that also happen to settle exactly at the wall would still
    # get a single settled marker row otherwise -- not worth keeping for that alone,
    # since a settled-only row has no future_paid target and isn't usable for training.)
    claims = claims[claims["notification_period"] < grid[-1]]
    excluded = n_before - len(claims)
    print(f"portfolio: {n_before} claims")
    print(f"excluded (notified at or after the last grid point, {grid[-1]}): "
          f"{excluded} ({excluded/n_before*100:.1f}%) -- no room left to show any development")

    rows = []
    for claim in claims.itertuples():
        payments = visible[visible["claim_no"] == claim.claim_no][["payment_period", "payment_size"]].values

        def cum_paid(p, _payments=payments):
            return _payments[_payments[:, 0] <= p][:, 1].sum()

        entry = next((g for g in grid if g >= claim.notification_period), None)
        if entry is None:
            continue

        # exit is the claim's TRUE settlement_period, where that's at or before the
        # cutoff -- NOT "outstanding <= 0 derived from cum_paid(cutoff)". The latter
        # reads as 0 for any claim with nothing paid *yet* inside the visible window,
        # which would wrongly flag every such immature claim as settled.
        exit_period = claim.settlement_period if claim.settlement_period <= cutoff else None

        for s in grid:
            if s < entry:
                continue

            paid_to_date = cum_paid(s)

            if exit_period is not None and s >= exit_period:
                rows.append(dict(
                    claim_no=claim.claim_no, snapshot_period=s, observation_age=None,
                    observation_period=None, row_status="settled", future_paid=None,
                    paid_to_date=round(paid_to_date, 2),
                ))
                break  # one exit row, then stop -- no further snapshots for this claim

            for g in grid:
                if g <= s:
                    continue
                if g > cutoff:
                    break
                k = g - s
                future_paid = round(cum_paid(g) - paid_to_date, 2)
                rows.append(dict(
                    claim_no=claim.claim_no, snapshot_period=s, observation_age=k,
                    observation_period=g, row_status="observed", future_paid=future_paid,
                    paid_to_date=round(paid_to_date, 2),
                    is_settled_at_obs=(g >= claim.settlement_period),
                ))

    result = pd.DataFrame(rows)
    n_observed = (result["row_status"] == "observed").sum()
    n_settled = (result["row_status"] == "settled").sum()
    print(f"snapshot triangle: {len(result)} rows ({n_observed} observed, {n_settled} settled-exit), "
          f"{result['claim_no'].nunique()} claims represented")
    return result
