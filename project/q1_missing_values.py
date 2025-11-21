import numpy as np

def generate_matrix_with_nans(seed=42):
    """
    Q1: Generate 1000×5 numeric matrix and insert 200 NaNs.
    """
    rng = np.random.default_rng(seed)
    # create the matrix with 1000 rows and 5 columns
    X_full = rng.normal(0, 1, size=(1000, 5))

    X_missing = X_full.copy()

    # choose 100 differnet rows and select 2 different columns in each row to set to NaN
    rows = rng.choice(1000, size=100, replace=False)
    nan_positions = []  # list of row and column where we will put NaN
    true_values = []    # corresponding true values before we put NaN


    for r in rows:
        # 2 different columns in that row
        cols = rng.choice(5, size=2, replace=False)
        for c in cols:
            nan_positions.append((r, c))
            true_values.append(X_full[r, c])    # store teh value before overwriting it
            X_missing[r, c] = np.nan            # insert the missing value

    print("Q1 → Shape:", X_missing.shape, "| Total NaNs:", np.isnan(X_missing).sum())
    return X_full, X_missing, nan_positions, np.array(true_values)

if __name__ == "__main__":
    generate_matrix_with_nans()
