import pandas as pd

# ============================================================
#  RULE ENGINE
# ============================================================

def rule_engine(transaction_df, probs, threshold):

    decisions = []

    for idx, (_, row) in enumerate(transaction_df.iterrows()):

        p = probs[idx]

        if row["Amount_original"] > 5000:
            decisions.append("BLOCK_RULE")

        elif p >= threshold:
            decisions.append("BLOCK_MODEL")

        elif p >= threshold*0.6:
            decisions.append("REVIEW")

        else:
            decisions.append("APPROVE")

    return decisions