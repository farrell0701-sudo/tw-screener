# 沙盒專用：這個工作環境沒有 pyarrow 也不能安裝，用 libzstd（ctypes）＋自寫解碼讀 parquet。
# 只支援 pyarrow 寫出的平面表格（無巢狀）：PLAIN / 字典 / RLE、資料頁 v1/v2、ZSTD/無壓縮。
# install() 之後：pd.read_parquet、DataFrame.to_parquet（改存 pickle）、pyarrow.parquet.read_table 可用。
import ctypes, struct, sys, types, pickle
import numpy as np, pandas as pd

_z = ctypes.CDLL('libzstd.so.1')
_z.ZSTD_decompress.restype = ctypes.c_size_t
_z.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
_z.ZSTD_isError.restype = ctypes.c_uint
_z.ZSTD_isError.argtypes = [ctypes.c_size_t]


def _unzstd(src, n):
    dst = ctypes.create_string_buffer(n)
    r = _z.ZSTD_decompress(dst, n, src, len(src))
    if _z.ZSTD_isError(r) or r != n:
        raise IOError(f'zstd 解壓失敗（{r} vs {n}）')
    return dst.raw


# ---------------- Thrift compact protocol ----------------
class _R:
    __slots__ = ('b', 'p')
    def __init__(self, b, p=0): self.b, self.p = b, p
    def byte(self):
        v = self.b[self.p]; self.p += 1; return v
    def varint(self):
        s = r = 0
        while True:
            x = self.b[self.p]; self.p += 1; r |= (x & 0x7f) << s
            if not x & 0x80: return r
            s += 7
    def zz(self):
        v = self.varint(); return (v >> 1) ^ -(v & 1)
    def val(self, t):
        if t == 1: return True
        if t == 2: return False
        if t == 3: v = self.b[self.p]; self.p += 1; return v
        if t in (4, 5, 6): return self.zz()
        if t == 7: v = struct.unpack_from('<d', self.b, self.p)[0]; self.p += 8; return v
        if t == 8:
            n = self.varint(); v = bytes(self.b[self.p:self.p + n]); self.p += n; return v
        if t in (9, 10):
            h = self.byte(); n = h >> 4; et = h & 15
            if n == 15: n = self.varint()
            if et in (1, 2): out = [self.byte() == 1 for _ in range(n)]
            else: out = [self.val(et) for _ in range(n)]
            return out
        if t == 11:
            n = self.varint()
            if n == 0: return {}
            h = self.byte(); kt, vt = h >> 4, h & 15
            return {self.val(kt): self.val(vt) for _ in range(n)}
        if t == 12: return self.struct()
        raise ValueError(f'thrift 型別 {t}')
    def struct(self):
        d = {}; last = 0
        while True:
            h = self.byte()
            if h == 0: return d
            t = h & 15; dl = h >> 4
            fid = last + dl if dl else self.zz()
            d[fid] = self.val(t); last = fid


# ---------------- 編碼 ----------------
def _bitunpack(buf, width, count):
    """little-endian 位元打包 → uint32 陣列"""
    if width == 0: return np.zeros(count, np.uint32)
    bits = np.unpackbits(np.frombuffer(buf, np.uint8), bitorder='little')
    need = count * width
    bits = bits[:need].reshape(count, width).astype(np.uint32)
    return (bits << np.arange(width, dtype=np.uint32)).sum(1).astype(np.uint32)


def _rle(buf, width, count):
    """RLE / bit-packed 混合編碼（不含長度前綴）"""
    out = np.empty(count, np.uint32); i = 0; r = _R(buf); bw = (width + 7) // 8
    while i < count and r.p < len(buf):
        h = r.varint()
        if h & 1:
            g = h >> 1; n = g * 8; nb = g * width
            v = _bitunpack(buf[r.p:r.p + nb], width, n); r.p += nb
            k = min(n, count - i); out[i:i + k] = v[:k]; i += k
        else:
            n = h >> 1; v = int.from_bytes(buf[r.p:r.p + bw], 'little') if bw else 0; r.p += bw
            k = min(n, count - i); out[i:i + k] = v; i += k
    if i < count: raise IOError('RLE 資料不足')
    return out


def _plain(buf, ptype, n, tlen=None):
    if ptype == 1: return np.frombuffer(buf, '<i4', n), len(buf[:4 * n]) and 4 * n
    if ptype == 2: return np.frombuffer(buf, '<i8', n), 8 * n
    if ptype == 4: return np.frombuffer(buf, '<f4', n), 4 * n
    if ptype == 5: return np.frombuffer(buf, '<f8', n), 8 * n
    if ptype == 0: return _bitunpack(buf, 1, n).astype(bool), (n + 7) // 8
    if ptype == 6:
        out = [None] * n; p = 0; mv = memoryview(buf)
        for k in range(n):
            L = int.from_bytes(mv[p:p + 4], 'little'); p += 4; out[k] = bytes(mv[p:p + L]); p += L
        return np.array(out, dtype=object), p
    if ptype == 7:
        out = [bytes(buf[k * tlen:(k + 1) * tlen]) for k in range(n)]
        return np.array(out, dtype=object), n * tlen
    raise ValueError(f'不支援的型別 {ptype}')


# ---------------- 讀檔 ----------------
def _meta(raw):
    if raw[-4:] != b'PAR1': raise IOError('不是 parquet')
    n = struct.unpack('<I', raw[-8:-4])[0]
    return _R(raw[-8 - n:-8]).struct()


def _schema(fm):
    cols = []
    for se in fm[2][1:]:
        cols.append(dict(name=se[4].decode(), ptype=se.get(1), rep=se.get(3, 0), tlen=se.get(2),
                         conv=se.get(6), logical=se.get(10)))
    return cols


def _convert(vals, c):
    lg = c['logical'] or {}
    if c['ptype'] == 6 and (c['conv'] == 0 or 1 in lg or c['conv'] is None):
        return np.array([None if v is None else v.decode('utf-8') for v in vals], dtype=object)
    if c['conv'] == 6 or 6 in lg:           # DATE
        return vals.astype('datetime64[D]').astype('datetime64[ns]')
    if c['conv'] in (9, 10) or 8 in lg:      # TIMESTAMP
        unit = 'ms' if c['conv'] == 9 else 'us'
        if 8 in lg:
            u = lg[8].get(2, {}); unit = 'ms' if 1 in u else ('us' if 2 in u else 'ns')
        return vals.astype(f'datetime64[{unit}]').astype('datetime64[ns]')
    return vals


def _read_chunk(raw, cm, c, nrows):
    ptype = c['ptype']; codec = cm[4]
    start = cm.get(11) if cm.get(11) is not None else cm[9]
    start = min(start, cm[9]); end = start + cm[7]
    r = _R(raw, start); dic = None; parts = []; total = 0; nv_all = cm[5]
    while total < nv_all and r.p < end:
        ph = r.struct(); ptyp = ph[1]; usz = ph[2]; csz = ph[3]
        body = raw[r.p:r.p + csz]; r.p += csz
        if ptyp == 2:                                             # 字典頁
            b = _unzstd(body, usz) if codec == 6 else body
            dic, _ = _plain(b, ptype, ph[7][1], c['tlen']); continue
        if ptyp == 0:                                             # 資料頁 v1
            b = _unzstd(body, usz) if codec == 6 else body
            h = ph[5]; nv = h[1]; enc = h[2]; p = 0
            if c['rep'] == 1:
                L = struct.unpack_from('<I', b, 0)[0]; dl = _rle(b[4:4 + L], 1, nv); p = 4 + L
            else:
                dl = None
            vb = b[p:]
        elif ptyp == 3:                                           # 資料頁 v2
            h = ph[8]; nv = h[1]; enc = h[4]; dlen = h[5]; rlen = h[6]
            lv = body[:rlen + dlen]; vb = body[rlen + dlen:]
            if h.get(7, True) and codec == 6: vb = _unzstd(vb, usz - rlen - dlen)
            dl = _rle(lv[rlen:rlen + dlen], 1, nv) if c['rep'] == 1 else None
        else:
            continue
        nn = int(dl.sum()) if dl is not None else nv
        if enc in (2, 8):
            bw = vb[0]; idx = _rle(vb[1:], bw, nn); vals = dic[idx]
        elif enc == 0:
            vals, _ = _plain(vb, ptype, nn, c['tlen'])
        elif enc == 3 and ptype == 0:
            L = struct.unpack_from('<I', vb, 0)[0]; vals = _rle(vb[4:4 + L], 1, nn).astype(bool)
        else:
            raise ValueError(f'不支援的編碼 {enc}（欄位 {c["name"]}）')
        if dl is not None and nn < nv:
            if ptype in (1, 2, 4, 5):
                full = np.full(nv, np.nan); full[dl == 1] = vals
            else:
                full = np.empty(nv, dtype=object); full[:] = None; full[dl == 1] = vals
            vals = full
        parts.append(vals); total += nv
    v = np.concatenate(parts) if parts else np.array([])
    has_null = (c['rep'] == 1) and v.dtype == object or (v.dtype.kind == 'f' and c['ptype'] in (1, 2))
    if v.dtype == object and c['ptype'] == 0:
        pass
    return v


def read_parquet(path, columns=None, **kw):
    with open(path, 'rb') as f: raw = f.read()
    if raw[:4] != b'PAR1':
        df = pd.read_pickle(path)                              # 本沙盒 to_parquet 的替代存檔
        return df[columns] if columns else df
    fm = _meta(raw); cols = _schema(fm)
    names = [c['name'] for c in cols]
    want = columns or [n for n in names if not n.startswith('__index_level_')]
    data = {}
    for name in want:
        if name not in names: raise KeyError(name)
        k = names.index(name); c = cols[k]; segs = []
        for rg in fm[4]:
            segs.append(_read_chunk(raw, rg[1][k][3], c, rg[3]))
        v = np.concatenate(segs) if segs else np.array([])
        if c['ptype'] in (1, 2) and v.dtype.kind == 'f' and not (c['conv'] == 6 or (c['logical'] or {}).get(6) is not None):
            pass
        if c['ptype'] in (1, 2) and v.dtype.kind == 'f' and (c['conv'] in (6, 9, 10) or any(x in (c['logical'] or {}) for x in (6, 8))):
            ok = ~np.isnan(v); out = np.full(len(v), np.datetime64('NaT'), 'datetime64[ns]')
            out[ok] = _convert(v[ok].astype(np.int64), c); v = out
        else:
            v = _convert(v, c) if v.dtype != object or c['ptype'] == 6 else v
        data[name] = v
    return pd.DataFrame(data)


class _Col:
    def __init__(self, s): self.s = s
    def to_pandas(self): return self.s


class _Table:
    def __init__(self, df): self.df = df
    def column(self, n): return _Col(self.df[n])
    def to_pandas(self): return self.df
    @property
    def num_rows(self): return len(self.df)


def install():
    pd.read_parquet = read_parquet
    def _to(self, path, index=True, **kw):
        (self if index else self.reset_index(drop=True)).to_pickle(path)
    pd.DataFrame.to_parquet = _to
    pa = types.ModuleType('pyarrow'); pqm = types.ModuleType('pyarrow.parquet')
    pqm.read_table = lambda path, columns=None, **kw: _Table(read_parquet(path, columns))
    pqm.read_schema = lambda path: [c['name'] for c in _schema(_meta(open(path, 'rb').read()))]
    pa.parquet = pqm; pa.__version__ = 'shim'
    sys.modules['pyarrow'] = pa; sys.modules['pyarrow.parquet'] = pqm
