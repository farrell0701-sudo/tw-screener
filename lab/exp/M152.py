# M-152：進入端確認——A1＆B1 同時下彎才擋、A1 轉好即恢復（描述性篩選）。定義見 registry_M.md M-152。
import sys
from pathlib import Path
import numpy as np, pandas as pd
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / 'data')); sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'exp'))
import M151


def machine_entry(enter, a):
    out = np.zeros(len(a), bool); st = False
    for i in range(len(a)):
        if not st and enter[i]: st = True
        elif st and not a[i]: st = False
        out[i] = st
    return out


def main():
    orig = M151.machine
    def patched_D(A1, B1n, we):
        return pd.Series(machine_entry(A1.to_numpy() & B1n.to_numpy(), A1.to_numpy()), index=we)
    # 以 M151 的資料流程為底，替換辨識器組合
    src = open(Path(M151.__file__)).read()
    src = src.replace("    for n in (21, 16, 26):\n        B1 = wfall(SP, n); D[f\"A1→A1＋B1({n}){'（主）' if n == 21 else ''}\"] = pd.Series(machine(A1.to_numpy(), B1.to_numpy()), index=we)",
        "    for n in (21, 16, 26):\n        B1 = wfall(SP, n); D[f\"A1＆B1({n})進場擋{'（主）' if n == 21 else ''}\"] = pd.Series(machine_entry(A1.to_numpy() & B1.to_numpy(), A1.to_numpy()), index=we)\n"
        "    run = A1.astype(int).groupby((~A1).cumsum()).cumsum(); D['A1 連2週才擋（對照）'] = pd.Series(machine_entry((run >= 2).to_numpy(), A1.to_numpy()), index=we)")
    src = src.replace("# M-151 結果：A1 觸發擋、A1＋B1 同時轉好才恢復（狀態機）", "# M-152 結果：進入端確認——A1＆B1 同時下彎才擋、A1 轉好即恢復")
    src = src.replace('headline=f"A1→A1＋B1 狀態機篩選', 'headline=f"進入端確認篩選')
    ns = {'machine_entry': machine_entry, '__name__': 'M152_inner', '__file__': str(Path(M151.__file__))}
    exec(compile(src, 'M152_inner', 'exec'), ns)
    return ns['main']()
