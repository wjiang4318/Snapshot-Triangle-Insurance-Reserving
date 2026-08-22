"""
Feature Engineering

Adds the modeling features on top of a core snapshot triangle from
snapshot_builder.build_snapshot_triangle(). Kept in its own module rather
than folded into snapshot_builder.py -- that module is core mechanism only
(see its own docstring), and feature engineering is Phase-3-specific, not
part of the triangle's own definition.

Promoted here from notebooks/03_model_training_revised.ipynb and
notebooks/04_clustering_revised.ipynb once the logic was stable and
identical in both -- the two notebooks had already drifted out of sync once
(adding age_of_claimant required editing both by hand), which is exactly the
duplication risk a shared module avoids going forward.
"""
import numpy as np
import pandas as pd

from snapshot_builder import ceil_period, load_visible

FEATURE_COLUMNS = [
    "injury_severity", "legal_representation", "notidel", "age_of_claimant",
    "paid_to_date", "periods_since_notification", "periods_since_last_payment",
    "has_any_payment", "payment_acceleration",
]


def add_features(df, snapshot_triangle):
    """Merge static (injury_severity, legal_representation, notidel,
    age_of_claimant) and dynamic (periods_since_notification,
    periods_since_last_payment, payment_acceleration, has_any_payment)
    features onto a snapshot triangle built by build_snapshot_triangle().

    Returns (snapshot_triangle_with_features, claims). `claims` is the full
    one-row-per-claim frame, returned alongside since some notebooks (e.g.
    the oracle/ceiling test in 03) need claim-level fields -- like the true
    settlement date, from `setldel` -- that aren't part of the feature set
    itself and must never feed into a real model (used only to build a
    diagnostic ceiling test).
    """
    visible = load_visible(df)
    claims = df.drop_duplicates("claim_no").copy()
    claims["notification_period"] = (claims["occurrence_time"] + claims["notidel"]).apply(ceil_period)

    static_cols = claims[["claim_no", "injury_severity", "legal_representation", "notidel", "age_of_claimant"]].copy()
    static_cols["notidel"] = static_cols["notidel"].round(3)

    payments_by_claim = {
        claim_no: claim_payments[["payment_period", "payment_size"]].values
        for claim_no, claim_payments in visible.groupby("claim_no")
    }
    notif_by_claim = claims.set_index("claim_no")["notification_period"]

    # one dynamic-feature row per (claim_no, snapshot_period) -- shared across every
    # observation_age at that snapshot, since these only depend on the claim's state
    # as of the snapshot, not on how far ahead a given row is predicting
    combos = snapshot_triangle[["claim_no", "snapshot_period"]].drop_duplicates()

    dyn_rows = []
    for claim_no, snapshot in combos.itertuples(index=False):
        payments = payments_by_claim.get(claim_no, np.empty((0, 2)))

        def cum_paid(p, _payments=payments):
            return _payments[_payments[:, 0] <= p][:, 1].sum()

        paid_so_far = payments[payments[:, 0] <= snapshot]
        n_payments_to_date = int(len(paid_so_far))  # only needed internally -- not kept as a feature
        periods_since_last_payment = (
            snapshot - int(paid_so_far[:, 0].max()) if n_payments_to_date else None
        )
        # payment_acceleration: this period's payment vs. the one before it (a fixed
        # 1-period lookback, unrelated to the triangle's own grid spacing -- no
        # variable needed since it's never anything but 1)
        paid_last = cum_paid(snapshot) - cum_paid(snapshot - 1)
        paid_prior = cum_paid(snapshot - 1) - cum_paid(snapshot - 2)
        payment_acceleration = round(paid_last / paid_prior, 3) if paid_prior > 1e-6 else None

        dyn_rows.append(dict(
            claim_no=claim_no, snapshot_period=snapshot,
            periods_since_notification=snapshot - notif_by_claim[claim_no],
            periods_since_last_payment=periods_since_last_payment,
            payment_acceleration=payment_acceleration,
            has_any_payment=n_payments_to_date > 0,
        ))
    dynamic_cols = pd.DataFrame(dyn_rows)

    snapshot_triangle = snapshot_triangle.merge(static_cols, on="claim_no", how="left")
    snapshot_triangle = snapshot_triangle.merge(dynamic_cols, on=["claim_no", "snapshot_period"], how="left")

    return snapshot_triangle, claims
