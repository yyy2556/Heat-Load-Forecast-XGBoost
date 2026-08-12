"""建筑负荷预测：基于 XGBoost 的小时级时间序列回归。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
PREFERRED_DATA_FILE = BASE_DIR / "heat_exchange_data_with_iforest.csv"
FALLBACK_DATA_FILE = BASE_DIR / "heat_exchange_data.csv"
OUTPUT_DIR = BASE_DIR / "docs"

# Serbia non-working public and religious holidays covered by this dataset.
# The list includes statutory carry-over days when a public holiday falls on Sunday.
SERBIA_PUBLIC_HOLIDAYS = pd.to_datetime(
    [
        # 2017
        "2017-01-01",
        "2017-01-02",
        "2017-01-03",
        "2017-01-07",
        "2017-02-15",
        "2017-02-16",
        "2017-04-14",
        "2017-04-15",
        "2017-04-16",
        "2017-04-17",
        "2017-05-01",
        "2017-05-02",
        "2017-11-11",
        # 2018
        "2018-01-01",
        "2018-01-02",
        "2018-01-07",
        "2018-02-15",
        "2018-02-16",
        "2018-04-06",
        "2018-04-07",
        "2018-04-08",
        "2018-04-09",
        "2018-05-01",
        "2018-05-02",
        "2018-11-11",
        "2018-11-12",
        # 2019
        "2019-01-01",
        "2019-01-02",
        "2019-01-07",
        "2019-02-15",
        "2019-02-16",
        "2019-04-26",
        "2019-04-27",
        "2019-04-28",
        "2019-04-29",
        "2019-05-01",
        "2019-05-02",
        "2019-11-11",
        # 2020
        "2020-01-01",
        "2020-01-02",
        "2020-01-07",
        "2020-02-15",
        "2020-02-16",
        "2020-02-17",
        "2020-04-17",
        "2020-04-18",
        "2020-04-19",
        "2020-04-20",
        "2020-05-01",
        "2020-05-02",
        "2020-11-11",
    ]
).normalize()


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
    original_rows = len(data)
    data = data.dropna(subset=["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")

    # 保留 heat_power == 0 的真实停机/低负荷状态，只处理负值和越界室外温度。
    negative_heat_power_mask = data["heat_power"] < 0
    outside_temp_out_of_range_mask = ~data["outside_temp"].between(-40, 50)
    negative_heat_power_count = int(negative_heat_power_mask.sum())
    outside_temp_out_of_range_count = int(outside_temp_out_of_range_mask.sum())
    data.loc[negative_heat_power_mask, "heat_power"] = np.nan
    data.loc[outside_temp_out_of_range_mask, "outside_temp"] = np.nan

    value_columns = ["outside_temp", "heat_power"]
    missing_before_imputation = int(data[value_columns].isna().sum().sum())
    interpolated_data = data[value_columns].interpolate(
        method="linear", limit=2
    )
    missing_after_interpolation = int(interpolated_data.isna().sum().sum())
    linear_interpolated_count = (
        missing_before_imputation - missing_after_interpolation
    )

    filled_data = interpolated_data.ffill()
    missing_after_ffill = int(filled_data.isna().sum().sum())
    forward_filled_count = missing_after_interpolation - missing_after_ffill
    data[value_columns] = filled_data

    remaining_missing_mask = data[value_columns].isna().any(axis=1)
    remaining_missing_count = int(remaining_missing_mask.sum())
    if remaining_missing_count > 0:
        print(
            f"警告：插补后仍有 {remaining_missing_count} 行存在缺失值，"
            "这些行将被删除。"
        )
        data = data.loc[~remaining_missing_mask]

    total_imputed_count = linear_interpolated_count + forward_filled_count
    removed_rows = original_rows - len(data)
    print(f"原始行数: {original_rows}")
    print(f"负值替换为 NaN 的数量（heat_power）: {negative_heat_power_count}")
    print(
        "越界替换为 NaN 的数量（outside_temp）: "
        f"{outside_temp_out_of_range_count}"
    )
    print(f"线性插补数量: {linear_interpolated_count}")
    print(f"前向填充数量: {forward_filled_count}")
    print(f"成功插补的数量: {total_imputed_count}")
    print(f"最终剩余缺失值数量: {missing_after_ffill}")
    print(f"因缺失值删除的行数: {removed_rows}")

    return data


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """构造时间、滞后和气象特征。"""
    features = pd.DataFrame(index=data.index)
    hour = data.index.hour
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["day_of_week"] = data.index.dayofweek
    features["is_weekend"] = (data.index.dayofweek >= 5).astype(int)
    features["lag_1"] = data["heat_power"].shift(1)
    features["lag_2"] = data["heat_power"].shift(2)
    features["lag_24"] = data["heat_power"].shift(24)
    shifted_heat_power = data["heat_power"].shift(1)
    features["rolling_mean_6h"] = shifted_heat_power.rolling(
        window=6, min_periods=1
    ).mean()
    features["rolling_std_6h"] = shifted_heat_power.rolling(
        window=6, min_periods=1
    ).std()
    features["outside_temp"] = data["outside_temp"]
    features["is_holiday"] = data.index.normalize().isin(
        SERBIA_PUBLIC_HOLIDAYS
    ).astype(int)

    # 仅使用历史数据构造滞后和滚动特征，避免将未来信息泄露到训练样本。
    history_features = [
        "lag_1",
        "lag_2",
        "lag_24",
        "rolling_mean_6h",
        "rolling_std_6h",
    ]
    features[history_features] = features[history_features].fillna(0)
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

    # 使用完全相同的训练/测试特征建立线性回归基线，便于比较非线性模型的收益。
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_mae, _, baseline_smape = calculate_metrics(
        y_test, baseline_predictions
    )
    print(f"线性回归 MAE: {baseline_mae:.4f}")
    print(f"线性回归 sMAPE: {baseline_smape:.2f}%")

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

    # 分层评估只改变指标切片，不改变模型训练和预测结果。
    # 使用 y_true >= 1.0 与主 MAPE 口径保持一致，避免低负荷样本放大百分比误差。
    normal_load_mask = y_test >= 1.0
    test_day_of_week = pd.Series(y_test.index.dayofweek, index=y_test.index)
    test_hour = pd.Series(y_test.index.hour, index=y_test.index)
    temporal_slices = {
        "工作日": test_day_of_week < 5,
        "周末": test_day_of_week >= 5,
        "白天（08-18时）": (test_hour >= 8) & (test_hour <= 18),
        "夜间（19-07时）": (test_hour >= 19) | (test_hour <= 7),
    }
    print("\n按时间维度分层评估（y_true >= 1.0）:")
    print(f"{'时段':<18}{'样本数':>10}{'MAPE':>14}")
    for slice_name, slice_mask in temporal_slices.items():
        evaluation_mask = normal_load_mask & slice_mask
        slice_y_true = y_test.loc[evaluation_mask]
        slice_y_pred = predictions[evaluation_mask.to_numpy()]
        _, slice_mape, _ = calculate_metrics(slice_y_true, slice_y_pred)
        print(f"{slice_name:<18}{len(slice_y_true):>10}{slice_mape:>13.2f}%")

    save_plots(model, list(features.columns), y_test, predictions)
    print(f"图表已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
