
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.adult_income.data import load_raw_data, FEATURE_COLUMNS
from src.adult_income.preprocessing import build_preprocessor


def main():
    # Load the dataset
    train_df, test_df = load_raw_data()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["income"]

    # Create preprocessing + model pipeline
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LGBMClassifier(
            random_state=42,
            verbosity=-1
        ))
    ])

    # Parameters to test
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__num_leaves": [15, 31, 50],
        "classifier__max_depth": [-1, 10, 20],
    }

    # Search for the best settings using training data only
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10,
        scoring="f1",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nBest cross-validation F1-score:")
    print(round(search.best_score_, 4))


if __name__ == "__main__":
    main()
