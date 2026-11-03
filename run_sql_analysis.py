"""Execute all seven saved SQL queries and calculate the required pandas cross-checks."""
from pathlib import Path
import json, sqlite3, re
import pandas as pd
ROOT=Path(__file__).resolve().parent; DB=ROOT/"data/instamart.db"; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
text=(ROOT/"sql/queries.sql").read_text(encoding="utf-8")
queries=[]
for part in text.split(';'):
    lines=[line for line in part.splitlines() if not line.lstrip().startswith('--')]
    sql='\n'.join(lines).strip()
    if sql: queries.append(sql)
results=[]
with sqlite3.connect(DB) as con:
    for i,q in enumerate(queries,1):
        df=pd.read_sql_query(q,con)
        row={"query":f"Q{i}","rows":len(df),"columns":list(df.columns)}
        if i==2 and "revenue" in df.columns:
            row["revenue_total"]=round(float(df.revenue.sum()),2)
            row["largest_cell"]=df.sort_values("revenue",ascending=False).iloc[0].to_dict()
        elif i==5 and "total_spend" in df.columns:
            row["average_spend_returned"]=round(float(df.total_spend.mean()),2)
        elif i==6:
            row["rank_diff_row_number"]=int((df.spend_rank!=df.row_num).sum())
            row["city_top_customers"]={r.city:{"customer_id":r.customer_id,"total_spend":round(float(r.total_spend),2)} for _,r in df.sort_values("row_num").groupby("city",sort=True).head(1).iterrows()}
        elif i==7 and "cumulative_revenue" in df.columns:
            row["daily_rows"]=len(df); row["final_cumulative_revenue"]=round(float(df.cumulative_revenue.iloc[-1]),2)
        results.append(row)
        print(f"Q{i}: rows={len(df)}")
assert [item["rows"] for item in results] == [30, 30, 11, 25, 550, 1175, 183]
orders=pd.read_csv(ROOT/"data/clean/orders.csv",keep_default_na=False)
orders=orders[orders.status.eq("completed")].copy()
for c in ["order_amount","discount_pct","quantity","delivery_minutes"]: orders[c]=pd.to_numeric(orders[c],errors="coerce")
customers=pd.read_csv(ROOT/"data/clean/customers.csv")
products=pd.read_csv(ROOT/"data/clean/products.csv")
orders=orders.merge(customers[["customer_id","city"]],on="customer_id",how="left").merge(products[["product_id","category"]],on="product_id",how="left")
pivot=pd.pivot_table(orders,index="city",columns="category",values="order_amount",aggfunc="sum",fill_value=0,margins=True,margins_name="All")
city=orders.groupby("city").order_amount.agg(total="sum",mean="mean",count="count").reset_index()
x=orders.order_amount; q1=float(x.quantile(.25)); q3=float(x.quantile(.75)); iqr=q3-q1; lower=q1-1.5*iqr; upper=q3+1.5*iqr
corr=orders[["order_amount","discount_pct","quantity","delivery_minutes"]].corr()
eda={"pivot_grand_total":round(float(pivot.loc["All","All"]),2),"city_stats":city.round(2).to_dict(orient="records"),
 "amount_mean":float(x.mean()),"amount_median":float(x.median()),"amount_skewness":float(x.skew()),"q1":q1,"q3":q3,"lower_fence":lower,"upper_fence":upper,
 "outliers_below":int((x<lower).sum()),"outliers_above":int((x>upper).sum()),"correlation_matrix":corr.round(4).to_dict(),
 "largest_abs_correlation_pair":max(((abs(float(corr.loc[a,b])),a,b,float(corr.loc[a,b])) for i,a in enumerate(corr.columns) for b in corr.columns[i+1:]),key=lambda t:t[0])[1:]}
assert abs(eda["pivot_grand_total"] - 2998668.80) < 0.01
assert eda["outliers_above"] == 897 and eda["outliers_below"] == 0
summary={"sql":results,"eda":eda}
(OUT/"part_a_analysis.json").write_text(json.dumps(summary,indent=2,default=float),encoding="utf-8")
print("PANDAS_EDA",json.dumps(eda,default=float))
