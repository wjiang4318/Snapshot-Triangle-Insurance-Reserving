"""
Model Training & Scoring Helpers

Shared by notebooks/03_model_training_revised.ipynb and
notebooks/04_clustering_revised.ipynb -- created this file here since both notebooks
need the exact same time-based train/test split, chain-ladder baseline,
and scoring functions.

Scope: model training/scoring helpers only.
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from snapshot_builder import CUTOFF


def is_train(frame, cutoff=CUTOFF):
    """Train on snapshots up to 1 year before cutoff, test on everything
    after. Split by TIME, not randomly -- a random split would put the same
    claim in both halves, letting its own future payments leak into its
    training features, and would let the model learn from calendar periods
    that hadn't happened yet."""
    return frame["snapshot_period"] <= cutoff - 12


def chain_ladder(frame, by=None):
    """Predict future payments as: paid_to_date x (growth factor - 1)

    Main idea: look at how much claims grew in the past, and assume new
    claims grow the same way.

    by=None  -> one growth factor for every claim
    by='col' -> a separate growth factor for each value of that column
    """
    train = frame[is_train(frame)]  # create training data

    before = train["paid_to_date"].sum()  # money paid as of the snapshot
    after = before + train["future_paid"].sum()  # ...and one observation window later
    overall_growth = after / before  # e.g. 1.5 -> claims grow 50%

    if by is None:
        growth = pd.Series(overall_growth, index=frame.index)
    else:
        # same formula, but computed separately inside each group
        sums = train.groupby(by)[["paid_to_date", "future_paid"]].sum()
        table = (sums["paid_to_date"] + sums["future_paid"]) / sums["paid_to_date"]
        # look up each row's own factor; fall back to overall if the group is unseen
        growth = frame[by].map(table).fillna(overall_growth)

    # "- 1" because we predict the NEW money, not the new total:
    # $10,000 paid x 1.5 growth = $15,000 total, so $5,000 is still to come
    return (frame["paid_to_date"] * (growth - 1)).clip(lower=0)


def score(name, actual_train, pred_train, actual_test, pred_test):
    """Every model is measured the same way, so the numbers are directly
    comparable.

    bias_pct = total predicted dollars vs total actual dollars. R2 and MAE
    both ignore whether the money adds up, which is what reserving cares about.
    """
    return dict(model=name,
                train_r2=r2_score(actual_train, pred_train),
                test_r2=r2_score(actual_test, pred_test),
                test_mae=mean_absolute_error(actual_test, pred_test),
                bias_pct=(pred_test.sum() / actual_test.sum() - 1) * 100)


def show(rows):
    """Print a list of score() results as one table."""
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "train_r2": "{:.3f}".format, "test_r2": "{:.3f}".format,
        "test_mae": "${:,.0f}".format, "bias_pct": "{:+.1f}%".format}))
