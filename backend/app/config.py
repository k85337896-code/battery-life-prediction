from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "battery_system.sqlite"

MODEL_OPTIONS = {
    "xgboost": "XGBoost 回归模型",
    "lstm": "LSTM 循环神经网络",
    "tcn": "TCN 时序卷积网络",
    "cnn": "CNN 卷积神经网络",
    "gpr": "GPR 高斯过程回归",
}

BATTERY_TYPES = {
    "LCO": "钴酸锂",
    "LFP": "磷酸铁锂",
    "LS": "锂硫电池",
    "G1": "真实数据 G1 组",
    "G2": "真实数据 G2 组",
    "G3": "真实数据 G3 组",
    "G4": "真实数据 G4 组",
}

EOL_SOH = 80.0

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def model_path(model_key: str):
    return MODEL_DIR / f"{model_key}_life_model.joblib"
