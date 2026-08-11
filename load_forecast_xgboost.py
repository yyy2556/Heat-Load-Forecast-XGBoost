"""建筑负荷预测：基于 XGBoost 的小时级时间序列回归。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
PREFERRED_DATA_FILE = BASE_DIR / "heat_exchange_data_with_iforest.csv"
FALLBACK_DATA_FILE = BASE_DIR / "heat_exchange_data.csv"
OUTPUT_DIR = BASE_DIR / "docs"


def configure_matplotlib_font() -> None:
    """按字体名称查找系统中文字体，避免中文字符触发字体缺失警告。"""
    font_candidates = ["Microsoft YaHei", "SimHei", "SimSun", "DengXian"]
    for font_name in font_candidates:
        try:
            font_path = font_manager.findfont(
                font_manager.FontProperties(family=font_name),
                fallback_to_default=False,
            )
        except (OSError, ValueError):
            continue

        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            matplotlib.rcParams["font.family"] = font_name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return

    # 没有 Windows 中文字体时，至少避免坐标轴负号显示异常。
    matplotlib.rcParams["axes.unicode_minus"] = False


configure_matplotlib_font()


def load_data() -> pd.DataFrame:
    """读取候选数据文件，并只保留建模需要的三列。"""
    if PREFERRED_DATA_FILE.exists():
        data_file = PREFERRED_DATA_FILE
    elif FALLBACK_DATA_FILE.exists():
        data_file = FALLBACK_DATA_FILE
    else:
        raise FileNotFoundError(
            "未找到 heat_exchange_data_with_iforest.csv 或 heat_exchange_data.csv。"
        )

    data = pd.read_csv(
        data_file,
        usecols=["timestamp", "outside_temp", "heat_power"],
        parse_dates=["timestamp"],
    )
    data = data.dropna(subset=["timestamp", "outside_temp", "heat_power"])
    data = data.sort_values("timestamp").set_index("timestamp")
    return data


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """构造时间、滞后和气象特征。"""
    features = pd.DataFrame(index=data.index)
    features["hour"] = data.index.hour
    features["day_of_week"] = data.index.dayofweek
    features["is_weekend"] = (data.index.dayofweek >= 5).astype(int)
    features["lag_1"] = data["heat_power"].shift(1)
    features["lag_2"] = data["heat_power"].shift(2)
    features["lag_24"] = data["heat_power"].shift(24)
    features["outside_temp"] = data["outside_temp"]

    # 仅对序列开头没有历史值的滞后特征补 0，避免丢失前 24 个样本。
    features[["lag_1", "lag_2", "lag_24"]] = features[
        ["lag_1", "lag_2", "lag_24"]
    ].fillna(0)
    return features, data["heat_power"]


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """统一计算 MAE、MAPE 和 sMAPE；MAPE 会忽略真实值为 0 的样本。"""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if len(y_true_array) == 0:
        return float("nan"), float("nan"), float("nan")

    mae = float(mean_absolute_error(y_true_array, y_pred_array))

    non_zero = y_true_array != 0
    if np.any(non_zero):
        mape = float(
            np.mean(
                np.abs(
                    (y_true_array[non_zero] - y_pred_array[non_zero])
                    / y_true_array[non_zero]
                )
            )
            * 100
        )
    else:
        mape = float("nan")

    smape_denominator = np.abs(y_true_array) + np.abs(y_pred_array)
    valid_smape = smape_denominator > 1e-8
    if np.any(valid_smape):
        smape = float(
            np.mean(
                2
                * np.abs(y_true_array[valid_smape] - y_pred_array[valid_smape])
                / smape_denominator[valid_smape]
            )
            * 100
        )
    else:
        smape = float("nan")

    return mae, mape, smape


def save_plots(
    model: XGBRegressor,
    feature_names: list[str],
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    """保存特征重要性图和测试集预测对比图。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True)

    plt.figure(figsize=(9, 6))
    importance.plot(kind="barh", color="#2f75b5")
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()

    preview_size = min(100, len(y_test))
    plt.figure(figsize=(12, 5))
    plt.plot(
        y_test.index[:preview_size],
        y_test.iloc[:preview_size],
        label="Actual heat_power",
        linewidth=1.8,
        color="#222222",
    )
    plt.plot(
        y_test.index[:preview_size],
        predictions[:preview_size],
        label="Predicted heat_power",
        linewidth=1.5,
        color="#d9534f",
    )
    plt.title("Test Set: Actual vs Predicted Heat Power (First 100 Points)")
    plt.xlabel("Timestamp")
    plt.ylabel("Heat Power")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_prediction_vs_actual.png", dpi=150)
    plt.close()


def main() -> None:
    data = load_data()
    features, target = build_features(data)

    split_index = int(len(features) * 0.8)
    if split_index == 0 or split_index == len(features):
        raise ValueError("数据量不足以按 80%/20% 划分训练集和测试集。")

    X_train = features.iloc[:split_index]
    X_test = features.iloc[split_index:]
    y_train = target.iloc[:split_index]
    y_test = target.iloc[split_index:]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 仅在评估阶段排除真实热功率低于 1.0 的低负荷样本。
    evaluation_mask = y_test >= 1.0
    y_true_filtered = y_test.loc[evaluation_mask]
    y_pred_filtered = predictions[evaluation_mask.to_numpy()]
    mae, mape, smape = calculate_metrics(y_true_filtered, y_pred_filtered)
    _, _, full_test_smape = calculate_metrics(y_test, predictions)
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    print("低负荷时段已排除，当前计算基于正常运行工况。")
    print(f"正常工况评估样本数: {len(y_true_filtered)}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MAE: {mae:.4f}")
    print(f"全量测试集 sMAPE: {full_test_smape:.2f}%")

    thresholds = [10.0, 20.0, 30.0]
    print("\n按热负荷阈值分档评估:")
    print(f"{'阈值':>8}{'样本数':>10}{'MAE':>14}{'MAPE':>14}{'sMAPE':>14}")
    for threshold in thresholds:
        threshold_mask = y_test >= threshold
        threshold_y_true = y_test.loc[threshold_mask]
        threshold_y_pred = predictions[threshold_mask.to_numpy()]
        threshold_mae, threshold_mape, threshold_smape = calculate_metrics(
            threshold_y_true, threshold_y_pred
        )
        print(
            f"{threshold:>8.1f}{len(threshold_y_true):>10}"
            f"{threshold_mae:>14.4f}{threshold_mape:>13.2f}%"
            f"{threshold_smape:>13.2f}%"
        )

    # 按业务核心时段切片，观察模型在真实供热场景中的可用性。
    # 这些结果用于业务场景分析，不代表模型整体泛化能力提升。
    test_hours = y_test.index.hour
    test_months = y_test.index.month
    heating_season = test_months.isin([11, 12, 1, 2, 3])
    daytime = (test_hours >= 8) & (test_hours <= 18)
    stable_heating = y_test >= 30.0

    business_conditions = {
        "工况A：稳定供热（y_test >= 30 kW）": stable_heating,
        "工况B：采暖季 + 稳定供热": heating_season & stable_heating,
        "工况C：白天 + 稳定供热": daytime & stable_heating,
        "工况D：采暖季 + 白天 + 稳定供热": (
            heating_season & daytime & stable_heating
        ),
    }

    def print_business_metrics(label: str, condition: pd.Series) -> None:
        condition_y_true = y_test.loc[condition]
        condition_y_pred = predictions[condition.to_numpy()]
        condition_mae, condition_mape, condition_smape = calculate_metrics(
            condition_y_true, condition_y_pred
        )
        print(
            f"{label} | 样本数: {len(condition_y_true)} | "
            f"MAE: {condition_mae:.4f} kW | "
            f"MAPE: {condition_mape:.2f}% | "
            f"sMAPE: {condition_smape:.2f}%"
        )

    print("\n" + "-" * 78)
    print("按时间维度的业务核心时段切片分析")
    print("-" * 78)
    for condition_name, condition in business_conditions.items():
        if condition_name.startswith("工况D"):
            print("\n" + "=" * 78)
            print("工况D：采暖季白天稳定供热工况预测效果（重点展示）")
            print("=" * 78)
        print_business_metrics(condition_name, condition)

    condition_d = business_conditions["工况D：采暖季 + 白天 + 稳定供热"]
    d_y_true = y_test.loc[condition_d]
    d_y_pred = predictions[condition_d.to_numpy()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.scatter(
        d_y_true,
        d_y_pred,
        alpha=0.6,
        s=28,
        color="#2f75b5",
        edgecolors="none",
        label="样本点",
    )
    if len(d_y_true) > 0:
        axis_min = min(float(d_y_true.min()), float(d_y_pred.min()))
        axis_max = max(float(d_y_true.max()), float(d_y_pred.max()))
        plt.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            linewidth=1.5,
            color="#d9534f",
            label="理想预测线 y=x",
        )
    plt.title("采暖季白天稳定供热工况预测效果")
    plt.xlabel("真实热功率 (kW)")
    plt.ylabel("预测热功率 (kW)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "winter_day_stable_heating_scatter.png", dpi=150)
    plt.close()

    save_plots(model, list(features.columns), y_test, predictions)
    print(f"图表已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
