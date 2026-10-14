// Cloudflare Worker: relays quote requests to Fugle MarketData API.
// Secret required in Cloudflare: FUGLE_KEY  (optional variable: ALLOW_ORIGIN)
export default {
  async fetch(req, env, ctx) {
    const cors = {
      "Access-Control-Allow-Origin": env.ALLOW_ORIGIN || "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Content-Type": "application/json; charset=utf-8",
    };
    if (req.method === "OPTIONS") {
      return new Response(null, { headers: cors });
    }

    const url = new URL(req.url);
    if (url.pathname !== "/quote") {
      return new Response(JSON.stringify({ error: "usage: /quote?codes=2330,2317" }), { status: 404, headers: cors });
    }
    if (!env.FUGLE_KEY) {
      return new Response(JSON.stringify({ error: "FUGLE_KEY not set" }), { status: 500, headers: cors });
    }

    const raw = (url.searchParams.get("codes") || "").split(",");
    const codes = [];
    for (const s of raw) {
      const c = s.trim();
      if (/^[0-9]{4}$/.test(c) && codes.indexOf(c) < 0 && codes.length < 50) {
        codes.push(c);
      }
    }

    const cache = caches.default;
    const out = {};

    await Promise.all(codes.map(async (code) => {
      const key = new Request("https://cache.local/quote/" + code);
      const hit = await cache.match(key);
      if (hit) {
        out[code] = await hit.json();
        return;
      }

      const r = await fetch(
        "https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/" + code,
        { headers: { "X-API-KEY": env.FUGLE_KEY } }
      );
      if (!r.ok) {
        out[code] = { err: r.status };
        return;
      }

      const d = await r.json();
      const q = {
        price: d.lastPrice ?? d.closePrice ?? null,
        chg: d.changePercent ?? null,
        open: d.openPrice ?? null,
        high: d.highPrice ?? null,
        low: d.lowPrice ?? null,
        vol: d.total ? (d.total.tradeVolume ?? null) : null,
        time: d.lastUpdated ?? null,
      };
      out[code] = q;
      ctx.waitUntil(cache.put(key, new Response(JSON.stringify(q), {
        headers: { "Cache-Control": "max-age=45" },
      })));
    }));

    return new Response(JSON.stringify(out), { headers: cors });
  },
};
