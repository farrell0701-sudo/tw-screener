# 實驗執行器：檢查登記 → 執行 lab/exp/<編號>.py → 存結果摘要 → 累計試驗次數 → 提交並推送
# 用法：python lab/core/run.py L-113 [--no-push] [--rerun]
#   - 登記簿（registry_L.md／registry_M.md）沒有「## <編號>」標題 → 拒絕執行。
#   - 實驗腳本 lab/exp/<L113>.py 必須定義 main() → dict(summary=Markdown 字串, headline=一句話結果,
#     series={名稱: 日報酬 Series}（選填）, n_variants=本次新變體數（計入 DSR 試驗次數）)。
#   - 結果：lab/results/<編號>/summary.md、series.npz（小檔，入庫；只有策略報酬，沒有原始資料）。
#   - --rerun：同一編號重跑（例如修正程式後），不再累加試驗次數。
import sys, json, re, time, subprocess, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; LAB = ROOT / 'lab'
TRIALS = LAB / 'registry' / 'trials.json'


def registered(eid):
    for f in ('registry_L.md', 'registry_M.md'):
        txt = (LAB / 'registry' / f).read_text(encoding='utf-8')
        if re.search(rf'^##\s+{re.escape(eid)}[（(\s：:]', txt, re.M): return f
    return None


def trials():
    if TRIALS.exists(): return json.loads(TRIALS.read_text(encoding='utf-8'))
    return {'N0': 520, 'note': 'L-113 決定 4：L-096～L-112 已執行 14 編號共 65 變體，平均 4.64 × 112 編號', 'log': []}


def n_trials():
    t = trials(); return t['N0'] + sum(x['variants'] for x in t['log'])


def main(argv):
    if not argv: sys.exit('用法：python lab/core/run.py <編號>')
    eid = argv[0].upper(); push = '--no-push' not in argv; rerun = '--rerun' in argv
    reg = registered(eid)
    if not reg: sys.exit(f'拒絕執行：登記簿中找不到「## {eid}」。先登記（假說、定義、判準、事前預期），再執行。')
    key = eid.replace('-', '')
    script = LAB / 'exp' / f'{key}.py'
    if not script.exists(): sys.exit(f'找不到實驗腳本 {script.relative_to(ROOT)}')
    head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'lab/core', 'lab/data', str(script)],
                           capture_output=True, text=True).stdout.strip()
    if dirty: print('注意：引擎或實驗腳本有未提交的變更，結果會連同這些變更一起提交。', flush=True)
    spec = importlib.util.spec_from_file_location(key, script); mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(LAB / 'core')); sys.path.insert(0, str(LAB / 'data'))
    t0 = time.time(); spec.loader.exec_module(mod); out = mod.main(); sec = round(time.time() - t0)

    t = trials(); nv = int(out.get('n_variants', 0))
    if any(x['id'] == eid for x in t['log']) and not rerun:
        sys.exit(f'{eid} 已執行過；若是修正後重跑請加 --rerun（不再累加試驗次數）')
    if not rerun:
        t['log'].append({'id': eid, 'variants': nv, 'date': time.strftime('%Y-%m-%d')})
        TRIALS.write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding='utf-8')
    d = LAB / 'results' / key; d.mkdir(parents=True, exist_ok=True)
    try:
        info = json.loads((LAB / 'cache' / 'panel' / 'info.json').read_text(encoding='utf-8'))
    except Exception:
        info = {}
    meta = (f"\n\n---\n編號 {eid}（{reg}）｜程式版本 {head}{'（含未提交變更）' if dirty else ''}｜資料 tw-data {info.get('tag', '?')}"
            f"｜面板 {info.get('built', '?')}｜執行 {time.strftime('%Y-%m-%d %H:%M')}，{sec} 秒｜本次新變體 {nv}｜累計試驗 {n_trials()}"
            f"{'｜重跑（不累加）' if rerun else ''}\n")
    (d / 'summary.md').write_text(out['summary'] + meta, encoding='utf-8')
    ser = out.get('series') or {}
    if ser:
        df = pd.concat(ser, axis=1).sort_index()
        np.savez_compressed(d / 'series.npz', dates=df.index.values.astype('datetime64[D]').astype('int64'),
                            names=np.array(df.columns, dtype=str), values=df.values.astype(np.float32))
    print(out['summary'] + meta, flush=True)
    if push:
        r = subprocess.run(['bash', str(LAB / 'save.sh'), f"{eid} {out.get('headline', '')}",
                            'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'], cwd=ROOT)
        if r.returncode: sys.exit('提交失敗，請手動執行 ./lab/save.sh')


if __name__ == '__main__':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    main(sys.argv[1:])
