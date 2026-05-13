import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

DATA_PATH = "winequality-white.csv"
TARGET_COL = "quality"
QUALITY_THRESHOLD = 6       # >= threshold → label 1 (good), else 0 (bad)
LOW_CORR_THRESHOLD = 0.05   # Drop features whose |corr| with target < this value


def load_data(path=DATA_PATH):
    """Đọc file CSV và trả về DataFrame thô."""
    return pd.read_csv(path, sep=";")


def explore_data(df):
    """In thông tin tổng quan và vẽ biểu đồ EDA."""
    print(df.info())
    print(df.describe())

    plt.figure(figsize=(10, 6))
    sns.countplot(x=TARGET_COL, data=df)
    plt.title("Distribution of Wine Quality")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
    plt.title("Correlation Matrix")
    plt.show()


def select_features(df, target_col=TARGET_COL, threshold=LOW_CORR_THRESHOLD):
    """Loại bỏ các feature có tương quan thấp với target."""
    corr = df.corr()[target_col].abs().sort_values(ascending=False)
    print(corr)

    low_corr_cols = corr[corr < threshold].index.tolist()
    if target_col in low_corr_cols:
        low_corr_cols.remove(target_col)

    return df.drop(columns=low_corr_cols)


def prepare_target(df, target_col=TARGET_COL, threshold=QUALITY_THRESHOLD):
    """Chuyển target thành bài toán phân loại nhị phân (0/1)."""
    df = df.copy()
    df[target_col] = (df[target_col] >= threshold).astype(int)
    return df


def get_train_test_split(df, target_col=TARGET_COL, test_size=0.2, random_state=42):
    """Tách features và labels, trả về X_train, X_test, y_train, y_test."""
    X = df.drop(columns=target_col).to_numpy()
    y = df[target_col].to_numpy()

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


if __name__ == "__main__":
    df = load_data()
    explore_data(df)
    df = select_features(df)
    df = prepare_target(df)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
