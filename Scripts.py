"""
Xây dựng dataset dự báo giá vàng (GC=F) giai đoạn 2015-01-01 -> 2024-12-31
và chia Train / Validation / Test THEO THỨ TỰ THỜI GIAN (Chronological Split).

Hỗ trợ 3 nguồn dữ liệu (đổi biến SOURCE bên dưới):
  "yfinance" : Yahoo Finance qua thư viện yfinance (mặc định)
  "stooq"    : Stooq.com (cần apikey lấy miễn phí qua captcha trên website)
  "local"    : đọc file CSV bạn tự tải về (đặt trong thư mục data_raw/)
               -> dùng được khi máy bạn không gọi được API nào cả.

Cài đặt:
    pip install --upgrade pandas yfinance requests
"""

import io
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. CẤU HÌNH
# ---------------------------------------------------------------------------
SOURCE = "yfinance"            # "yfinance" | "stooq" | "local"

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"        # đã BAO GỒM ngày này

RAW_DIR = Path("data_raw")     # thư mục chứa CSV khi SOURCE = "local"
STOOQ_APIKEY = ""              # dán apikey Stooq vào đây khi SOURCE = "stooq"

# tên tài sản -> (mã Yahoo, mã Stooq, tên file CSV khi dùng "local")
# Lưu ý: mã Stooq bên dưới chưa được tôi kiểm chứng, hãy tra lại trên stooq.com.
ASSETS = {
    "Gold":   ("GC=F",     "gc.f",    "GC_F.csv"),
    "DXY":    ("DX-Y.NYB", "dx.f",    "DXY.csv"),
    "US10Y":  ("^TNX",     "10usy.b", "US10Y.csv"),
    "SP500":  ("^GSPC",    "^spx",    "SP500.csv"),
    "WTI":    ("CL=F",     "cl.f",    "WTI.csv"),
    "Silver": ("SI=F",     "si.f",    "SI_F.csv"),   # bạc COMEX = SI=F (không phải SL=F)
}
MACRO_COLUMNS = {
    "DXY": "DXY_Close",
    "US10Y": "US10Y_Close",
    "SP500": "SP500_Close",
    "WTI": "WTI_Close",
    "Silver": "Silver_Close",
}

# --- Chia tập dữ liệu ------------------------------------------------------
SPLIT_MODE = "date"            # "date" (theo mốc năm) | "ratio" (70/15/15)
TRAIN_END = "2021-12-31"
VAL_END = "2023-12-31"         # test = từ 2024-01-01 trở đi
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15

# Tên file có hậu tố theo nguồn (vd gold_forecast_dataset_stooq.csv) để chạy nhiều
# nguồn mà không ghi đè lên nhau, và có thể đối chiếu bằng compare_sources.py
OUTPUT_FULL = f"gold_forecast_dataset_{SOURCE}.csv"
OUTPUT_TRAIN = f"gold_train_{SOURCE}.csv"
OUTPUT_VAL = f"gold_val_{SOURCE}.csv"
OUTPUT_TEST = f"gold_test_{SOURCE}.csv"

PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# 2. CHUẨN HÓA DỮ LIỆU TỪ NHIỀU NGUỒN
# ---------------------------------------------------------------------------
def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đưa tên cột của Yahoo / Stooq / Investing.com về Date, Open, High, Low, Close, Volume."""
    lowered = {c: str(c).strip().lower().rstrip(".") for c in df.columns}
    has_close = "close" in lowered.values()
    mapping = {}
    for col, k in lowered.items():
        if k in ("date", "time", "datetime"):
            mapping[col] = "Date"
        elif k in ("open", "high", "low", "close"):
            mapping[col] = k.capitalize()
        elif k in ("price", "last") and not has_close:  # Investing.com dùng "Price"
            mapping[col] = "Close"
        elif k in ("volume", "vol"):
            mapping[col] = "Volume"
    return df.rename(columns=mapping)


def _to_number(s: pd.Series) -> pd.Series:
    """Chuyển cột sang số; xử lý dấu phẩy ngăn cách nghìn và hậu tố K/M/B (vd '152.3K')."""
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
        mult = (s.str.extract(r"([KMBkmb])$", expand=False)
                .str.upper().map({"K": 1e3, "M": 1e6, "B": 1e9}).fillna(1.0))
        s = pd.to_numeric(s.str.replace(r"[KMBkmb%]$", "", regex=True), errors="coerce") * mult
    return pd.to_numeric(s, errors="coerce")


def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = _rename_columns(df)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"[{label}] Không tìm thấy cột Date/Close. Các cột hiện có: {list(df.columns)}")

    # Lưu ý: định dạng ngày kiểu 01/02/2020 được hiểu là tháng/ngày (chuẩn Mỹ).
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")]

    for c in PRICE_COLS:
        if c in df.columns:
            df[c] = _to_number(df[c])
    df = df.loc[START_DATE:END_DATE]
    if df.empty:
        raise ValueError(f"[{label}] Không có dữ liệu trong khoảng {START_DATE} -> {END_DATE}")
    return df


# ---------------------------------------------------------------------------
# 3. CÁC HÀM TẢI DỮ LIỆU
# ---------------------------------------------------------------------------
def load_yfinance(ticker: str, max_retries: int = 5) -> pd.DataFrame:
    import yfinance as yf

    end_exclusive = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, start=START_DATE, end=end_exclusive, interval="1d",
                             auto_adjust=False, progress=False)
        except Exception as e:
            print(f"[{ticker}] Lỗi lần {attempt}: {e}")
            df = None
        if df is not None and not df.empty:
            break
        wait = 5 * attempt
        print(f"[{ticker}] Chưa có dữ liệu (lần {attempt}/{max_retries}), chờ {wait}s...")
        time.sleep(wait)

    if df is None or df.empty:
        raise ValueError(f"Không tải được {ticker}. Kiểm tra lại mã hoặc kết nối mạng.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return normalize(df.reset_index(), ticker)


def load_stooq(symbol: str) -> pd.DataFrame:
    import requests

    if not STOOQ_APIKEY:
        raise ValueError("Chưa có STOOQ_APIKEY. Lấy key tại https://stooq.com/q/d/?s=spy.us&get_apikey")
    r = requests.get(
        "https://stooq.com/q/d/l/",
        params={"s": symbol, "i": "d",
                "d1": START_DATE.replace("-", ""), "d2": END_DATE.replace("-", ""),
                "apikey": STOOQ_APIKEY},
        timeout=30,
    )
    r.raise_for_status()
    text = r.text.strip()
    if not text.lower().startswith("date"):
        raise RuntimeError(f"Stooq không trả về CSV cho '{symbol}' (sai mã, sai key hoặc hết hạn mức): {text[:120]!r}")
    return normalize(pd.read_csv(io.StringIO(text)), symbol)


def load_local(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Thiếu file {path}. Hãy tải CSV và đặt đúng tên này.")
    return normalize(pd.read_csv(path), filename)


def fetch(name: str) -> pd.DataFrame:
    yahoo, stooq, local = ASSETS[name]
    if SOURCE == "yfinance":
        df = load_yfinance(yahoo)
    elif SOURCE == "stooq":
        df = load_stooq(stooq)
    elif SOURCE == "local":
        df = load_local(local)
    else:
        raise ValueError("SOURCE phải là 'yfinance', 'stooq' hoặc 'local'")
    print(f"[OK] {name:<7} ({SOURCE}) {len(df):>5} dòng | {df.index.min().date()} -> {df.index.max().date()}")
    if SOURCE != "local":
        time.sleep(2)  # tránh bị giới hạn truy cập
    return df


# ---------------------------------------------------------------------------
# 4. GỘP, XỬ LÝ KHUYẾT THIẾU
# ---------------------------------------------------------------------------
def build_dataset() -> pd.DataFrame:
    gold_raw = fetch("Gold")
    cols = [c for c in PRICE_COLS if c in gold_raw.columns]
    if "Volume" not in cols:
        print("[CẢNH BÁO] File vàng không có cột Volume, dataset sẽ thiếu Gold_Volume.")
    gold = gold_raw[cols].add_prefix("Gold_")

    macro = [fetch(name)["Close"].rename(col) for name, col in MACRO_COLUMNS.items()]

    df = pd.concat([gold] + macro, axis=1, join="outer", sort=True)

    # Forward-fill CHỈ các cột vĩ mô; không dùng nội suy tuyến tính.
    macro_cols = list(MACRO_COLUMNS.values())
    df[macro_cols] = df[macro_cols].ffill()

    df = df.dropna(subset=["Gold_Close"])   # giữ nguyên lịch giao dịch của GC=F
    df = df.dropna(subset=macro_cols)       # bỏ dòng đầu chưa có giá vĩ mô

    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    return df


# ---------------------------------------------------------------------------
# 5. CHIA TẬP THEO THỨ TỰ THỜI GIAN
# ---------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, mode: str = SPLIT_MODE):
    if mode == "date":
        train = df.loc[:TRAIN_END]
        val = df.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):VAL_END]
        test = df.loc[pd.Timestamp(VAL_END) + pd.Timedelta(days=1):]
    elif mode == "ratio":
        n = len(df)
        n_train, n_val = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
        train = df.iloc[:n_train]
        val = df.iloc[n_train:n_train + n_val]
        test = df.iloc[n_train + n_val:]
    else:
        raise ValueError("SPLIT_MODE phải là 'date' hoặc 'ratio'")
    assert train.index.max() < val.index.min() < val.index.max() < test.index.min()
    return train, val, test


def describe_split(name: str, part: pd.DataFrame, total: int) -> None:
    print(f"{name:<11}: {len(part):>5} phiên ({len(part) / total:>5.1%}) | "
          f"{part.index.min().date()} -> {part.index.max().date()}")


# ---------------------------------------------------------------------------
# 6. CHẠY
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_dataset()
    n = len(df)

    print("\n=== TỔNG QUAN DATASET ===")
    print(f"Khung thời gian : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Tổng số phiên   : {n} dòng x {df.shape[1]} cột")
    print(f"NaN còn lại     : {int(df.isna().sum().sum())}")

    train, val, test = chronological_split(df, SPLIT_MODE)
    print(f"\n=== CHIA TẬP (mode = '{SPLIT_MODE}') ===")
    describe_split("Train", train, n)
    describe_split("Validation", val, n)
    describe_split("Test", test, n)


    df.to_csv(OUTPUT_FULL, date_format="%Y-%m-%d")
    train.to_csv(OUTPUT_TRAIN, date_format="%Y-%m-%d")
    val.to_csv(OUTPUT_VAL, date_format="%Y-%m-%d")
    test.to_csv(OUTPUT_TEST, date_format="%Y-%m-%d")
    print(f"\nĐã lưu: {OUTPUT_FULL}, {OUTPUT_TRAIN}, {OUTPUT_VAL}, {OUTPUT_TEST}")
    print("Nhắc lại: chỉ fit scaler trên Train rồi áp dụng cho Validation/Test.")