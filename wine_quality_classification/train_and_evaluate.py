from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from wine_quality_classification.data_loader import (
    get_train_test_split,
    load_data,
    prepare_target,
    select_features,
)
from wine_quality_classification.decision_tree import DecisionTree
from wine_quality_classification.random_forest import RandomForest


def main():
    # Load & tiền xử lý dữ liệu
    df = load_data()
    df = select_features(df)
    df = prepare_target(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}\n")

    # --- Assignment 1: Custom Decision Tree ---
    dt_custom = DecisionTree(max_depth=10, min_samples_split=2)
    dt_custom.fit(X_train, y_train)

    y_pred_dt = dt_custom.predict(X_test)
    f1_dt = f1_score(y_test, y_pred_dt, average="micro")
    print(f"[Custom Decision Tree]  F1 Score: {f1_dt:.4f}")

    # --- Assignment 2: Custom Random Forest ---
    rf_custom = RandomForest(n_trees=10, max_depth=10, min_samples_split=2)
    rf_custom.fit(X_train, y_train)

    y_pred_rf = rf_custom.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average="micro")
    print(f"[Custom Random Forest]  F1 Score: {f1_rf:.4f}")

    # --- Assignment 3: Sklearn Decision Tree ---
    dt_sklearn = DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=42)
    dt_sklearn.fit(X_train, y_train)
    y_pred_dt_sklearn = dt_sklearn.predict(X_test)
    f1_dt_sklearn = f1_score(y_test, y_pred_dt_sklearn, average="micro")
    print(f"[Sklearn Decision Tree] F1 Score: {f1_dt_sklearn:.4f}")

    # --- Assignment 3: Sklearn Random Forest ---
    rf_sklearn = RandomForestClassifier(
        n_estimators=10, max_depth=10, min_samples_split=2, random_state=42
    )
    rf_sklearn.fit(X_train, y_train)
    y_pred_rf_sklearn = rf_sklearn.predict(X_test)
    f1_rf_sklearn = f1_score(y_test, y_pred_rf_sklearn, average="micro")
    print(f"[Sklearn Random Forest] F1 Score: {f1_rf_sklearn:.4f}")


if __name__ == "__main__":
    main()
