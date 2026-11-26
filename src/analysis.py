"""Online Retail II: cleaning + aggregate analysis.

Source: UCI Machine Learning Repository, "Online Retail II" (CC BY 4.0)
https://archive.ics.uci.edu/dataset/502/online+retail+ii
Run: python src/analysis.py   (downloads the zip if needed, writes data/*.csv and data/summary.json)
Only aggregates are written. Row-level data is never saved to the repo.
"""
import io, json, os, sys, urllib.request, zipfile
import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT, exist_ok=True)

def load():
    cache = "/tmp/online_retail_II.xlsx"
    if not os.path.exists(cache):
        raw = urllib.request.urlopen(URL).read()
        zipfile.ZipFile(io.BytesIO(raw)).extract("online_retail_II.xlsx", "/tmp")
    pk = "/tmp/online_retail_II.pkl"  # speed cache (optional)
    if os.path.exists(pk):
        return pd.read_pickle(pk)
    sheets = pd.read_excel(cache, sheet_name=None)
    out = pd.concat(sheets.values(), ignore_index=True)
    out.to_pickle(pk)
    return out

df = load()
df.columns = ["invoice", "stock", "desc", "qty", "invoice_date", "price", "customer", "country"]
log = {"raw_rows": len(df)}

# 1. The two yearly sheets overlap (Dec 2010). Drop exact duplicate rows.
n = len(df); df = df.drop_duplicates(); log["exact_duplicates_removed"] = n - len(df)
df["invoice"] = df["invoice"].astype(str)
df["is_cancel"] = df["invoice"].str.startswith("C")

# 2. Non-product stock codes (postage, fees, manual adjustments, samples...)
nonprod = df["stock"].astype(str).str.upper().str.match(r"^(POST|D|DOT|M|C2|BANK CHARGES|AMAZONFEE|S|B|CRUK|ADJUST.*|TEST.*|GIFT.*|PADS)$")
log["non_product_rows_removed"] = int(nonprod.sum()); df = df[~nonprod]

# 3. Zero/negative prices are stock adjustments or bad debt, not sales
bad = df["price"] <= 0
log["zero_or_negative_price_rows_removed"] = int(bad.sum()); df = df[~bad]

df["revenue"] = df["qty"] * df["price"]
sales = df[~df["is_cancel"] & (df["qty"] > 0)].copy()
rets = df[df["is_cancel"] | (df["qty"] < 0)].copy()
log["sales_rows"] = len(sales); log["return_rows"] = len(rets)
log["rows_without_customer_id"] = int(sales["customer"].isna().sum())
log["share_rows_without_customer_id"] = round(float(sales["customer"].isna().mean()) * 100, 1)
sales["month"] = sales["invoice_date"].dt.to_period("M").astype(str)
rets["month"] = rets["invoice_date"].dt.to_period("M").astype(str)
sales["desc"] = sales["desc"].astype(str).str.strip()

gross = sales["revenue"].sum(); returned = -rets["revenue"].sum()
inv = sales.groupby("invoice").agg(rev=("revenue", "sum"), items=("qty", "sum"), lines=("stock", "count"),
                                   date=("invoice_date", "min"), country=("country", "first"), cust=("customer", "first"))
kpi = dict(
    gross_revenue=round(gross), returns_value=round(returned), net_revenue=round(gross - returned),
    return_rate_pct=round(returned / gross * 100, 1),
    orders=int(inv.shape[0]), customers=int(sales["customer"].nunique()),
    aov=round(float(inv["rev"].mean()), 1), median_order=round(float(inv["rev"].median()), 1),
    units=int(sales["qty"].sum()), countries=int(sales["country"].nunique()),
    products=int(sales["stock"].nunique()),
    first_date=str(sales["invoice_date"].min().date()), last_date=str(sales["invoice_date"].max().date()),
)

# Monthly (Dec 2009 - Dec 2011; Dec 2011 is a partial month ending Dec 9)
m = sales.groupby("month").agg(revenue=("revenue", "sum"), orders=("invoice", "nunique"), customers=("customer", "nunique"))
m["returns"] = -rets.groupby("month")["revenue"].sum()
m = m.fillna(0).round(0).reset_index(); m.to_csv(f"{OUT}/monthly.csv", index=False)

# Countries
c = sales.groupby("country").agg(revenue=("revenue", "sum"), orders=("invoice", "nunique"), customers=("customer", "nunique")).sort_values("revenue", ascending=False)
c["share_pct"] = (c["revenue"] / c["revenue"].sum() * 100).round(2)
c["aov"] = (c["revenue"] / c["orders"]).round(1)
c[["revenue","orders","customers"]] = c[["revenue","orders","customers"]].round(0); c = c.reset_index(); c.to_csv(f"{OUT}/countries.csv", index=False)

# Products
p = sales.groupby("stock").agg(desc=("desc", lambda s: s.mode().iat[0]), gross=("revenue", "sum"), units=("qty", "sum"), orders=("invoice", "nunique"))
# Rank by NET revenue (sales minus returns). Two products (23843, 23166) each had one huge order that was fully cancelled.
p["returned"] = -rets.groupby("stock")["revenue"].sum()
p["returned"] = p["returned"].fillna(0)
p["revenue"] = p["gross"] - p["returned"]
p = p.sort_values("revenue", ascending=False)
p["cum_share_pct"] = (p["revenue"].cumsum() / p["revenue"].sum() * 100).round(1)
p.head(50).round(0).reset_index().to_csv(f"{OUT}/top_products.csv", index=False)
top_prod = p.head(10).round(0).reset_index()
pareto_prod_20 = float(p["revenue"].head(int(len(p) * .2)).sum() / p["revenue"].sum() * 100)
rp = rets.groupby("stock").agg(desc=("desc", lambda s: s.dropna().astype(str).mode().iat[0] if s.notna().any() else ""), value=("revenue", lambda s: -s.sum()), units=("qty", lambda s: -s.sum()))
top_ret = rp.sort_values("value", ascending=False).head(6).round(0).reset_index()

# Day x hour heatmap (revenue)
sales["dow"] = sales["invoice_date"].dt.dayofweek; sales["hour"] = sales["invoice_date"].dt.hour
hh = sales.groupby(["dow", "hour"])["revenue"].sum().round(0).reset_index()

# RFM (customers with an ID, sales only), snapshot = last date + 1 day
cs = sales.dropna(subset=["customer"]).copy()
snap = cs["invoice_date"].max().normalize() + pd.Timedelta(days=1)
rfm = cs.groupby("customer").agg(recency=("invoice_date", lambda s: (snap - s.max()).days),
                                 frequency=("invoice", "nunique"), monetary=("revenue", "sum"))
rfm["R"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
def seg(r):
    if r.R >= 4 and r.F >= 4 and r.M >= 4: return "Champions"
    if r.R >= 3 and r.F >= 4: return "Loyal"
    if r.R >= 4 and r.F <= 2: return "New / Promising"
    if r.R >= 3 and r.F == 3: return "Potential Loyalists"
    if r.R <= 2 and r.F >= 4: return "At Risk (was loyal)"
    if r.R <= 2 and r.F >= 2: return "Hibernating"
    return "Lost / One-off"
rfm["segment"] = rfm.apply(seg, axis=1)
sg = rfm.groupby("segment").agg(customers=("monetary", "size"), revenue=("monetary", "sum"), avg_recency_days=("recency", "mean"),
                                avg_orders=("frequency", "mean"), avg_value=("monetary", "mean")).round(1)
sg["revenue_share_pct"] = (sg["revenue"] / sg["revenue"].sum() * 100).round(1)
sg = sg.sort_values("revenue", ascending=False).reset_index(); sg.to_csv(f"{OUT}/rfm_segments.csv", index=False)
cust_sorted = rfm["monetary"].sort_values(ascending=False)
top20_share = float(cust_sorted.head(int(len(cust_sorted) * .2)).sum() / cust_sorted.sum() * 100)
repeat_rate = float((rfm["frequency"] > 1).mean() * 100)
# scatter sample for chart: aggregated bins instead of customer rows
fb = rfm.groupby("F").agg(customers=("monetary", "size"), revenue=("monetary", "sum")).reset_index()

# Cohorts: first purchase month -> retention by months since
cs["cohort"] = cs.groupby("customer")["invoice_date"].transform("min").dt.to_period("M")
cs["period"] = cs["invoice_date"].dt.to_period("M")
cs["offset"] = (cs["period"] - cs["cohort"]).apply(lambda x: x.n)
coh = cs.groupby(["cohort", "offset"])["customer"].nunique().unstack(fill_value=0)
ret = (coh.div(coh[0], axis=0) * 100).round(1)
ret.index = ret.index.astype(str); coh.index = coh.index.astype(str)
cohort_sizes = coh[0].to_dict()
ret.to_csv(f"{OUT}/cohort_retention.csv")
avg_ret = {int(k): round(float(ret[k].iloc[: len(ret) - k].mean()), 1) for k in [1, 3, 6, 12] if k in ret.columns}

# Basket
basket = inv["lines"].clip(upper=60)
bins = pd.cut(inv["rev"], [0, 100, 250, 500, 1000, 2500, 1e9], labels=["<100", "100-250", "250-500", "500-1k", "1k-2.5k", "2.5k+"])
ob = inv.groupby(bins, observed=True).agg(orders=("rev", "size"), revenue=("rev", "sum")).round(0).reset_index()
ob.columns = ["bucket", "orders", "revenue"]

# Returns by month rate
mm = m.copy(); mm["return_rate"] = (mm["returns"] / mm["revenue"] * 100).round(1)


# ---------------------------------------------------------------------------
# DEEPER ANALYSIS
# ---------------------------------------------------------------------------
last = sales["invoice_date"].max()
cs_all = sales.dropna(subset=["customer"]).copy()

def rfm_at(snap, data):
    d = data[data["invoice_date"] < snap]
    r = d.groupby("customer").agg(recency=("invoice_date", lambda s: (snap - s.max()).days),
                                  frequency=("invoice", "nunique"), monetary=("revenue", "sum"))
    r["R"] = pd.qcut(r["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    r["F"] = pd.qcut(r["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    r["M"] = pd.qcut(r["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    r["segment"] = r.apply(seg, axis=1)
    return r

# 1. Back-test: score customers on year 1 (to 9 Dec 2010), measure what they really spent in year 2.
SNAP = pd.Timestamp("2010-12-10")
r1 = rfm_at(SNAP, cs_all)
y2 = cs_all[cs_all["invoice_date"] >= SNAP].groupby("customer")["revenue"].sum()
r1["y2"] = y2.reindex(r1.index).fillna(0.0)
r1["back"] = r1["y2"] > 0
bt = r1.groupby("segment").agg(customers=("monetary", "size"), y1_revenue=("monetary", "sum"), y2_revenue=("y2", "sum"),
                              retained_pct=("back", lambda s: s.mean() * 100)).reset_index()
bt["avg_y2_per_customer"] = bt["y2_revenue"] / bt["customers"]
bt["avg_y2_per_returner"] = r1[r1["back"]].groupby("segment")["y2"].mean().reindex(bt["segment"]).values
bt["nrr_pct"] = bt["y2_revenue"] / bt["y1_revenue"] * 100
bt = bt.sort_values("y1_revenue", ascending=False).round(1)
bt.to_csv(f"{OUT}/backtest_segments.csv", index=False)
tot_y1 = float(r1["monetary"].sum()); tot_y2 = float(r1["y2"].sum())
lost_y1 = float(r1.loc[~r1["back"], "monetary"].sum())
backtest = dict(rows=bt.to_dict("records"), y1_customers=int(len(r1)), retained_pct=round(float(r1["back"].mean() * 100), 1),
                y1_revenue=round(tot_y1), y2_revenue_from_y1_customers=round(tot_y2), nrr_pct=round(tot_y2 / tot_y1 * 100, 1),
                lost_y1_revenue=round(lost_y1), lost_share_pct=round(lost_y1 / tot_y1 * 100, 1))

# 2. Customer lifetime value (observed): 12-month revenue after first order, by 90-day reorder behaviour
fo = cs_all.groupby("customer")["invoice_date"].min().rename("first")
cs_all = cs_all.join(fo, on="customer")
cs_all["days"] = (cs_all["invoice_date"] - cs_all["first"]).dt.days
elig = fo[fo <= last - pd.Timedelta(days=365)].index          # customers with a full 12 months of history
e = cs_all[cs_all["customer"].isin(elig)]
v12 = e[e["days"] < 365].groupby("customer")["revenue"].sum()
first_inv_rev = e[e["days"] == 0].groupby("customer")["revenue"].sum()
orders90 = e[e["days"] <= 90].groupby("customer")["invoice"].nunique()
clv = pd.DataFrame({"v12": v12, "first": first_inv_rev, "reorder90": orders90 > 1}).dropna()
clv_stats = dict(customers=int(len(clv)), avg_12m=round(float(clv["v12"].mean())), median_12m=round(float(clv["v12"].median())),
                 reorder90_pct=round(float(clv["reorder90"].mean() * 100), 1),
                 avg_12m_reorderers=round(float(clv.loc[clv["reorder90"], "v12"].mean())),
                 avg_12m_nonreorderers=round(float(clv.loc[~clv["reorder90"], "v12"].mean())),
                 avg_first_order=round(float(clv["first"].mean())),
                 top10pct_share=round(float(clv["v12"].sort_values(ascending=False).head(int(len(clv)*.1)).sum() / clv["v12"].sum() * 100), 1))
new_per_year = int(fo[(fo >= "2010-12-10") & (fo <= last)].shape[0])
clv_stats["new_customers_last_12m"] = new_per_year

# 3. Seasonality + simple forecast. Seasonal naive x trend; back-tested on Jan-Nov 2011 (no peeking).
rev_m = m.set_index("month")["revenue"]
trend_2010_to_2011 = float(sum(rev_m[f"2011-{i:02d}"] for i in range(1, 11)) / sum(rev_m[f"2010-{i:02d}"] for i in range(1, 11)))
bt_pred = {f"2011-{i:02d}": float(rev_m[f"2010-{i:02d}"]) for i in range(1, 12)}      # naive: same month last year
mape = float(np.mean([abs(bt_pred[k] - rev_m[k]) / rev_m[k] for k in bt_pred]) * 100)
fc = {}
for k, src in [("2011-12", "2010-12"), ("2012-01", "2011-01"), ("2012-02", "2011-02"), ("2012-03", "2011-03")]:
    fc[k] = float(rev_m[src]) * trend_2010_to_2011
season_idx = {int(mm): float(np.mean([rev_m[f"2010-{mm:02d}"] / np.mean([rev_m[f"2010-{i:02d}"] for i in range(1, 13)]),
                                       rev_m[f"2011-{mm:02d}"] / np.mean([rev_m[f"2011-{i:02d}"] for i in range(1, 11)] + [rev_m["2010-11"], rev_m["2010-12"]])])) for mm in range(1, 12)}
season_idx[12] = float(rev_m["2010-12"] / np.mean([rev_m[f"2010-{i:02d}"] for i in range(1, 13)]))
q4_2011 = float(sum(rev_m[f"2011-{i:02d}"] for i in (9, 10, 11)))
forecast = dict(trend=round(trend_2010_to_2011, 3), mape_pct=round(mape, 1), fc={k: round(v) for k, v in fc.items()},
                actual_2011_jan_nov=[round(float(rev_m[f"2011-{i:02d}"])) for i in range(1, 12)],
                naive_2011_jan_nov=[round(bt_pred[f"2011-{i:02d}"]) for i in range(1, 12)],
                q4_2011=round(q4_2011), q4_share_2010=round(float(sum(rev_m[f"2010-{i:02d}"] for i in (9, 10, 11)) / sum(rev_m[f"2010-{i:02d}"] for i in range(1, 13)) * 100), 1))

# 4. Market basket: product pairs bought together (top 250 products by orders), lift + confidence
top_codes = sales.groupby("stock")["invoice"].nunique().sort_values(ascending=False).head(250).index
desc_map = sales.groupby("stock")["desc"].agg(lambda s: s.mode().iat[0])
bk = sales[sales["stock"].isin(top_codes)].drop_duplicates(["invoice", "stock"])
mat = pd.crosstab(bk["invoice"], bk["stock"]).astype(np.float32)
n_inv = len(inv); co = mat.T.values @ mat.values; cnt = np.diag(co).copy()
pairs = []
for i in range(len(cnt)):
    for j in range(i + 1, len(cnt)):
        if co[i, j] >= 200:
            lift = co[i, j] * n_inv / (cnt[i] * cnt[j])
            pairs.append((mat.columns[i], mat.columns[j], int(co[i, j]), float(lift), float(co[i, j] / min(cnt[i], cnt[j]))))
pairs = sorted(pairs, key=lambda x: -x[3])[:12]
basket_pairs = [dict(a=str(a), b=str(b), a_desc=desc_map[a], b_desc=desc_map[b], orders=o, lift=round(l, 1), conf_pct=round(c * 100, 0)) for a, b, o, l, c in pairs]
basket_stats = dict(avg_lines=round(float(inv["lines"].mean()), 1), median_lines=float(inv["lines"].median()),
                    single_line_pct=round(float((inv["lines"] == 1).mean() * 100), 1), products_tested=250)

# 5. Returns / product risk
mega = ["23843", "23166"]
rets_u = rets.copy(); rets_u["stock"] = rets_u["stock"].astype(str)
sales_u = sales.copy(); sales_u["stock"] = sales_u["stock"].astype(str)
ret_prod = rets_u.groupby("stock").agg(r_val=("revenue", lambda s: -s.sum()), r_units=("qty", lambda s: -s.sum()))
sal_prod = sales_u.groupby("stock").agg(s_val=("revenue", "sum"), s_units=("qty", "sum"), desc=("desc", lambda s: s.mode().iat[0]))
rp2 = sal_prod.join(ret_prod, how="inner")
rp2["rate"] = rp2["r_val"] / rp2["s_val"] * 100
risk = rp2[(rp2["s_val"] >= 10000) & (~rp2.index.isin(mega))].sort_values("rate", ascending=False).head(8)
mega_val = float(rets_u[rets_u["stock"].isin(mega)]["revenue"].sum() * -1)
ret_inv = rets_u.groupby("invoice").size().shape[0]
returns = dict(total=round(returned), mega_cancelled=round(mega_val), underlying=round(returned - mega_val),
               underlying_rate_pct=round((returned - mega_val) / (gross - 0) * 100, 1), return_invoices=int(ret_inv),
               risk=[dict(stock=i, desc=r.desc, sales=round(r.s_val), returned=round(r.r_val), rate=round(r.rate, 1)) for i, r in risk.iterrows()])

# 6. Concentration risk
cust_rev = cs_all.groupby("customer")["revenue"].sum().sort_values(ascending=False)
tot_c = cust_rev.sum()
conc = dict(top1=round(float(cust_rev.iloc[0] / tot_c * 100), 1), top10=round(float(cust_rev.head(10).sum() / tot_c * 100), 1),
            top50=round(float(cust_rev.head(50).sum() / tot_c * 100), 1), top100=round(float(cust_rev.head(100).sum() / tot_c * 100), 1),
            top10_gbp=round(float(cust_rev.head(10).sum())), top10_countries=cs_all[cs_all["customer"].isin(cust_rev.head(10).index)].groupby("customer")["country"].first().value_counts().to_dict(),
            unidentified_share_pct=round(float(sales.loc[sales["customer"].isna(), "revenue"].sum() / gross * 100), 1))
x = np.arange(1, len(cust_rev) + 1) / len(cust_rev) * 100; yv = cust_rev.cumsum().values / tot_c * 100
idx = np.linspace(0, len(x) - 1, 40).astype(int)
conc["curve"] = [[round(float(x[i]), 1), round(float(yv[i]), 1)] for i in idx]

# 7. Fix-tab sizing (scenarios, each with its assumption; revenue not profit)
now_seg = rfm.groupby("segment").agg(n=("monetary", "size"), rev=("monetary", "sum"))
bsr = bt.set_index("segment")
sizing = []
lapsed_now = int(now_seg.loc[["At Risk (was loyal)", "Hibernating"], "n"].sum())
lapsed_ret = float((r1[r1["segment"].isin(["At Risk (was loyal)", "Hibernating"]) & r1["back"]]["y2"]).mean())
for s_ in ["At Risk (was loyal)", "Hibernating"]:
    pass
ar = bsr.loc["At Risk (was loyal)"]; hb = bsr.loc["Hibernating"]
sizing.append(dict(key="winback", title="Win back lapsed buyers (At Risk + Hibernating)",
    base=f"{lapsed_now:,} customers lapsed at Dec 2011",
    assumption="10% of them reorder in the next 12 months, each spending what a returning At Risk/Hibernating customer spent in the back-test",
    gbp=round(lapsed_now * 0.10 * lapsed_ret)))
ch = bsr.loc["Champions"]
n_champ = int(now_seg.loc["Champions", "n"])
sizing.append(dict(key="champions", title="Keep Champions from lapsing",
    base=f"{n_champ:,} Champions; {ch.retained_pct:.0f}% bought again in the back-test year",
    assumption="Cut Champion churn by 5 percentage points; each saved Champion spends the back-test average for a returning Champion (year-2 spend of returning Champions)",
    gbp=round(n_champ * 0.05 * float(ch.avg_y2_per_returner))))
gap = clv_stats["avg_12m_reorderers"] - clv_stats["avg_12m_nonreorderers"]
sizing.append(dict(key="reorder", title="Win the second order within 90 days",
    base=f"{new_per_year:,} new customers in the last 12 months; {clv_stats['reorder90_pct']}% reorder within 90 days",
    assumption=f"Lift 90-day reorder rate by 5 points; each extra reorderer is worth the 12-month gap vs non-reorderers (GBP {gap:,}). This is a correlation: reorderers may simply be better customers",
    gbp=round(new_per_year * 0.05 * gap)))
sizing.append(dict(key="q4", title="Avoid Q4 stock-outs",
    base=f"Sep to Nov 2011 revenue was GBP {q4_2011/1e6:.2f}M",
    assumption="Every 1% of Q4 sales lost to stock-outs or late stock; sensitivity, not a measured loss",
    gbp=round(q4_2011 * 0.01)))
sizing.append(dict(key="returns", title="Cut avoidable returns",
    base=f"Underlying returns GBP {(returned-mega_val)/1e3:,.0f}k after removing two fully cancelled mega-orders",
    assumption="Reduce returned value by 20%. Returned value is revenue, not margin; profit effect is unknown without cost data",
    gbp=round((returned - mega_val) * 0.20)))

summary = dict(
    log=log, kpi=kpi,
    monthly=dict(x=m["month"].tolist(), revenue=m["revenue"].tolist(), orders=m["orders"].tolist(), customers=m["customers"].tolist(), rr=mm["return_rate"].tolist(), returns=m["returns"].tolist()),
    countries=c.head(12)[["country", "revenue", "orders", "customers", "share_pct", "aov"]].values.tolist(),
    uk_share=float(c.loc[c.country == "United Kingdom", "share_pct"].iat[0]),
    top_products=top_prod[["stock", "desc", "revenue", "units", "orders", "gross"]].values.tolist(),
    top_returns=top_ret[["stock", "desc", "value", "units"]].values.tolist(),
    pareto_products_top20_pct=round(pareto_prod_20, 1),
    heat=hh.values.tolist(),
    rfm=sg.to_dict("records"), top20_customers_share=round(top20_share, 1), repeat_rate=round(repeat_rate, 1),
    freq_bins=fb.values.tolist(),
    cohort=dict(x=list(range(ret.shape[1])), y=list(ret.index), v=[[int(j), i, (None if i + j >= len(ret) else float(ret.iloc[i, j]))] for i in range(len(ret)) for j in range(ret.shape[1])], sizes=cohort_sizes),
    avg_retention=avg_ret,
    order_buckets=ob.values.tolist(),
    backtest=backtest, clv=clv_stats, forecast=forecast, basket_pairs=basket_pairs, basket_stats=basket_stats, returns=returns, conc=conc, sizing=sizing,
    items_per_order_median=float(inv["lines"].median()),
)
json.dump(summary, open(f"{OUT}/summary.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: summary[k] for k in ["log", "kpi", "uk_share", "pareto_products_top20_pct", "top20_customers_share", "repeat_rate", "avg_retention", "items_per_order_median"]}, indent=1))
print(json.dumps({k:summary[k] for k in ["backtest","clv","forecast","basket_stats","returns","conc","sizing"]},indent=1,default=str)); print(basket_pairs); print(sg.to_string()); print(c.head(8).to_string()); print(top_prod.to_string()); print(top_ret.to_string()); print(m.to_string())
