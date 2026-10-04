import pandas as pd
import numpy as np

# 1. 讀取原始 CSV 檔案
df = pd.read_csv('PT_RC_QPSK_200symbol.csv')

# 單位轉換函數
def parse_val_with_unit(s):
    if pd.isna(s): return np.nan
    parts = str(s).strip().split()
    if not parts: return np.nan
    val = float(parts[0])
    if len(parts) > 1:
        unit = parts[1].lower()
        if unit.startswith('f'): val *= 1e-15
        elif unit.startswith('p'): val *= 1e-12
        elif unit.startswith('n'): val *= 1e-9
        elif unit.startswith('u') or unit.startswith('μ'): val *= 1e-6
        elif unit.startswith('m'): val *= 1e-3
    return val

# 解析原始資料
t_orig = df[df.columns[0]].apply(parse_val_with_unit).values
cc1_orig = df[df.columns[1]].apply(parse_val_with_unit).values
cc2_orig = df[df.columns[2]].apply(parse_val_with_unit).values

# 2. 設定固定區間與固定步階 (5 ns 到 11.75 ns, dt = 7.8125 ps)
t_start = 5e-9
t_stop = 11.75e-9
dt = 7.8125e-12

t_fixed = np.arange(t_start, t_stop + dt/2, dt)

# 3. 線性內插 (Interpolate)
cc1_fixed = np.interp(t_fixed, t_orig, cc1_orig)
cc2_fixed = np.interp(t_fixed, t_orig, cc2_orig)

# 4. 存成乾淨的固定時間區間 CSV
df_fixed = pd.DataFrame({
    'Time_s': t_fixed,
    'Time_ns': t_fixed * 1e9,
    'Cc1_V': cc1_fixed,
    'Cc2_V': cc2_fixed
})