"""
Snapshot Triangle Builder

Converts the SynthETIC transaction dataset into a snapshot-date-triangle
dataset for individual claim reserving and the reference paper 
(Llaguno et al., "Reserving with Machine Learning:
Applications for Loyalty Programs and Individual Insurance Claims", CAS
E-Forum Summer 2017).

build_snapshot_triangle(df) is core mechanism only (8 columns) -- no feature engineering,
just the claim/snapshot/age grid and the future_paid target. Feature
engineering for modeling (Phase 3+) lives in feature_engineering.py instead,
built on top of this module's output.

Scope: dataset construction only. No modeling, no clustering, no chain ladder.
"""
import math
import pandas as pd

CUTOFF = 40

# Only two grid spacings are created -- matching how insurers actually
# review claims (quarterly or annually). Periods are already quarters, so
# "quarterly" is every native period and "yearly" is every 4th (4 quarters =
# 12 months). build_snapshot_triangle() takes granularity="quarterly"/"yearly"
# directly; bad input raises KeyError.
GRANULARITY_TO_STEP = {"quarterly": 1, "yearly": 4}


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


def snapshot_grid(cutoff=CUTOFF, step=GRANULARITY_TO_STEP["quarterly"]):
    """The shared list of periods every claim's snapshots (and observation
    ages) are drawn from: step, 2*step, ... up to cutoff. Every claim uses
    this same fixed set of periods -- no claim gets its own personal
    schedule -- so that triangles built later by combining many claims
    (e.g. per-cluster in Phase 5) line up correctly instead of each claim's
    rows landing on different periods.

    Example:
        >>> snapshot_grid(cutoff=40, step=4)
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

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


def build_snapshot_triangle(df, cutoff=CUTOFF, granularity="quarterly"):
    """Core mechanism only -- claim_no, snapshot_period, observation_age,
    observation_period, row_status, future_paid, paid_to_date, is_settled_at_obs.
    No feature columns -- see feature_engineering.add_features() for that."""
    step = GRANULARITY_TO_STEP[granularity]  
    visible = load_visible(df, cutoff)
    grid = snapshot_grid(cutoff, step)

    # Claim identity (occurrence, notidel, setldel) comes from the FULL data,
    # not `visible`. Reason: a claim can be notified before the cutoff and
    # still have nothing due yet -- zero payments, so zero rows in `visible`.
    # If identity were built from `visible` alone, that claim would silently
    # vanish, even though it's a real, already-known claim that belongs in
    # the triangle.
    claims = df.drop_duplicates("claim_no").copy()
    claims["notification_period"] = (claims["occurrence_time"] + claims["notidel"]).apply(ceil_period)
    claims["settlement_period"] = (
        claims["occurrence_time"] + claims["notidel"] + claims["setldel"]
    ).apply(ceil_period)

    n_before = len(claims)
    # Claims notified exactly AT the last grid point (not just after it) are also
    # excluded: with nothing left but that single point, there's no room for even
    # one observed row to show development.
    # (The few claims that also happen to settle exactly at that same point
    # would still get a single settled marker row otherwise -- not worth keeping
    # for that alone, since a settled-only row has no future_paid target and
    # isn't usable for training.)
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

        entry = next((grid_point for grid_point in grid if grid_point >= claim.notification_period), None)
        if entry is None:
            continue

        # exit is the claim's TRUE settlement_period, where that's at or before the
        # cutoff -- NOT "outstanding <= 0 derived from cum_paid(cutoff)". The latter
        # reads as 0 for any claim with nothing paid *yet* inside the visible window,
        # which would wrongly flag every such immature claim as settled.
        exit_period = claim.settlement_period if claim.settlement_period <= cutoff else None

        for snapshot in grid:
            if snapshot < entry:
                continue

            paid_to_date = cum_paid(snapshot)

            if exit_period is not None and snapshot >= exit_period:
                rows.append(dict(
                    claim_no=claim.claim_no, snapshot_period=snapshot, observation_age=None,
                    observation_period=None, row_status="settled", future_paid=None,
                    paid_to_date=round(paid_to_date, 2),
                ))
                break  # one exit row, then stop -- no further snapshots for this claim

            for observation_period in grid:
                if observation_period <= snapshot:
                    continue
                if observation_period > cutoff:
                    break
                k = observation_period - snapshot
                future_paid = round(cum_paid(observation_period) - paid_to_date, 2)
                rows.append(dict(
                    claim_no=claim.claim_no, snapshot_period=snapshot, observation_age=k,
                    observation_period=observation_period, row_status="observed", future_paid=future_paid,
                    paid_to_date=round(paid_to_date, 2),
                    is_settled_at_obs=(observation_period >= claim.settlement_period),
                ))

    result = pd.DataFrame(rows)
    n_observed = (result["row_status"] == "observed").sum()
    n_settled = (result["row_status"] == "settled").sum()
    print(f"snapshot triangle: {len(result)} rows ({n_observed} observed, {n_settled} settled-exit), "
        f"{result['claim_no'].nunique()} claims represented")
    return result
