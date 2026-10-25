import joblib
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.adult_income.data import load_raw_data, FEATURE_COLUMNS
from src.adult_income.preprocessing import build_preprocessor


def main():
    model_dir = Path(__file__).resolve().parents[2] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the training and official test datasets
    train_df, test_df = load_raw_data()

    # 2. Separate features and target
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["income"].str.strip()

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["income"].str.strip()

    # 3. Define the baseline and three actual classifiers
    classifiers = {
        "Dummy Classifier (Baseline)": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=-1,
        ),
    }

    results = []

    # 4. Train and evaluate each classifier
    for name, classifier in classifiers.items():
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        print(f"\n{'=' * 55}")
        print(f"Training: {name}")
        print("=" * 55)

        # Train the model
        model.fit(X_train, y_train)


                # Save the fitted pipeline for the API
        model_filename = {
            "Dummy Classifier (Baseline)": "dummy_baseline.joblib",
            "Logistic Regression": "logistic_regression.joblib",
            "Random Forest": "random_forest.joblib",
            "LightGBM": "lightgbm.joblib",
        }[name]

        model_path = model_dir / model_filename
        joblib.dump(model, model_path)

        print(f"Saved fitted pipeline: {model_path}")

        # Predict income classes
        predictions = model.predict(X_test)

        # Calculate overall accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Evaluate the higher-income class
        precision = precision_score(
            y_test,
            predictions,
            pos_label=">50K",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label=">50K",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label=">50K",
            zero_division=0,
        )

        # Display metrics
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision (>50K): {precision:.4f}")
        print(f"Recall (>50K):    {recall:.4f}")
        print(f"F1-score (>50K):  {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                labels=["<=50K", ">50K"],
                zero_division=0,
            )
        )

        # Store results for comparison
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        })

        # Create and save the confusion matrix
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=["<=50K", ">50K"],
            cmap="Blues",
            values_format="d",
        )

        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()

        # Save the plot instead of opening a GUI window
        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        plt.savefig(f"{filename}_confusion_matrix.png", dpi=150)
        plt.close()

        print(f"Confusion matrix saved: {filename}_confusion_matrix.png")

    # 5. Compare all four classifiers
    print("\n\nMODEL COMPARISON")
    print("=" * 85)

    print(
        f"{'Model':<32}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-score':>12}"
    )
    print("-" * 80)

    # Sort by F1-score for the >50K class
    for result in sorted(
        results,
        key=lambda item: item["F1-score"],
        reverse=True,
    ):
        print(
            f"{result['Model']:<32}"
            f"{result['Accuracy']:>12.4f}"
            f"{result['Precision']:>12.4f}"
            f"{result['Recall']:>12.4f}"
            f"{result['F1-score']:>12.4f}"
        )


if __name__ == "__main__":
    main()
        # Directory for saving trained model pipelines
    model_dir = Path(__file__).resolve().parents[2] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
