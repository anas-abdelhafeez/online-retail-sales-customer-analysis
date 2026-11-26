# Is flat revenue hiding a retention problem?

Retention, back-test, forecast and basket analysis of a UK gift wholesaler (Online Retail II), with an interactive dark dashboard. Python (pandas) pipeline, aggregates only.

**Live dashboard:** https://anas-abdelhafeez.github.io/online-retail-sales-customer-analysis/

## The question and the answer

Revenue grew only 3.3% (Jan to Oct 2011 vs 2010). Is the business healthy, or replacing lost customers just to stand still?

**It is standing still.** Of the 4,285 customers who bought in year 1 (to 9 Dec 2010), 37% never bought again, and the rest spent less. Those customers produced £8.64M in year 1 and £6.74M in year 2: **net revenue retention of 78%**. £1.23M of year-1 revenue left with customers who never returned. New customers filled the gap.

## What each tab proves

| Tab | Question | Main result |
|---|---|---|
| Answer | Is growth real? | NRR 78%, 62.9% of year-1 customers bought again |
| Who | Who holds the revenue? | 1,280 Champions = 68% of identified revenue; top 10 customers = 16%; UK = 86% |
| Retention | Who stayed, and what did they spend? | Back-test by RFM segment; second order within 90 days is worth £3.8k vs £0.9k over 12 months |
| Seasonality | When does demand peak, and can we forecast it? | Sep to Nov = 36% of revenue; seasonal-naive forecast back-tested at 9.3% average error |
| Products | What sells, what is returned? | Variant pairs have 20 to 30x lift; return rates reach 36% on a few lines |
| Fix | What is each lever worth? | Five levers sized in GBP with their assumptions |

## Methods

- **Cleaning:** 1,067,371 raw rows. Removed 34,335 exact duplicates (the two sheets overlap in Dec 2010), 5,819 non-product lines (postage, fees, manual adjustments) and 5,963 rows with zero or negative price. Cancellations are kept as returns. All counts are in `data/summary.json`.
- **Back-test:** customers are scored with RFM on year 1 only (to 9 Dec 2010), then their actual spend in the next 12 months is measured. Net revenue retention = year-2 revenue / year-1 revenue of the same customers.
- **Forecast:** seasonal naive (same month last year) with a growth trend from Jan to Oct 2011 vs 2010. Tested on Jan to Nov 2011 using 2010 only.
- **Basket analysis:** pairs among the 250 most-ordered products with at least 200 shared orders, ranked by lift.
- **Product risk:** return value as % of sales for products with at least £10k sales. Two products (23843, 23166) each had one huge order that was fully cancelled (£246k combined); they are excluded from product risk and ranked by net revenue.
- **Fix sizing:** scenarios built from the back-test. Each states its assumption in the dashboard and in `data/summary.json`. They overlap and are not additive.

## Limitations

- Revenue, not profit: there is no cost or margin data.
- One retailer, two years, no price or marketing data. The forecast cannot see promotions or a lost large account.
- 13% of gross revenue has no Customer ID and is excluded from customer analysis.
- RFM segments are rule-based, not a fitted model. The "second order" lever is a correlation, not proven cause.
- Dec 2011 is a partial month (data ends 9 Dec 2011).

## Run it

```bash
pip install -r requirements.txt
python src/analysis.py          # downloads the UCI zip, writes data/*.csv and data/summary.json
python src/build_dashboard.py   # writes index.html
```

## Layout

```
index.html               dashboard (ECharts, data embedded)
src/analysis.py          cleaning, back-test, forecast, baskets, sizing (pandas)
src/build_dashboard.py   builds index.html from the aggregates
src/template.html        dashboard template
data/                    aggregate outputs only
```

## Source and licence

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii. Licensed CC BY 4.0. Only aggregates derived from the dataset are published here. Code by Anas Abdelhafeez.
