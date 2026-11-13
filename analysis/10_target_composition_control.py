from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; D=ROOT/"analysis"/"data"; EPS=1e-12
DATASETS=["dunnhumby","instacart","sam","tafeng","valuedshopper"]
METHODS=["recency","frequency","tifuknn","fpmc","triple2vec","dnntsp","cbp","diffrec"]
FEATURES=["n_baskets","avg_basket_size","repeat_rate","adjacent_basket_jaccard","recent_basket_novelty","popularity_profile","neighbor_similarity","transition_strength","relational_strength"]
TARGETS=["target_last_repeat_fraction","target_older_repeat_fraction"]; Z=["z_"+x for x in FEATURES]; ZT=["z_"+x for x in TARGETS]

p=pd.read_csv(D/"performance.csv",dtype={"user":str}); f=pd.read_csv(D/"regime_features.csv",dtype={"user":str}); t=pd.read_csv(D/"target_mechanism_features.csv",dtype={"user":str})
p=p[p.dataset.isin(DATASETS)].copy(); f=f[f.dataset.isin(DATASETS)].copy(); t=t[t.dataset.isin(DATASETS)].copy()
r=p[["dataset","user"]].drop_duplicates().merge(f[["dataset","user"]+FEATURES],on=["dataset","user"],validate="one_to_one").merge(t[["dataset","user"]+TARGETS+["target_novel_fraction"]],on=["dataset","user"],validate="one_to_one")

for ds in DATASETS:
    m=r.dataset.eq(ds)
    for x in FEATURES+TARGETS:
        s=r.loc[m,x].std(ddof=0); r.loc[m,"z_"+x]=(r.loc[m,x]-r.loc[m,x].mean())/s if pd.notna(s) and s>EPS else np.nan

a=p.merge(r[["dataset","user"]+Z+ZT],on=["dataset","user"],validate="many_to_one"); a["user_mean"]=a[METHODS].mean(axis=1)
for m in METHODS: a[m+"_centered"]=a[m]-a["user_mean"]

def fit(d,y,x):
    X=np.c_[np.ones(len(d)),d[x].to_numpy(float)]; yy=d[y].to_numpy(float); b=np.linalg.lstsq(X,yy,rcond=None)[0]; pred=X@b; ss=np.sum((yy-yy.mean())**2)
    return 1-np.sum((yy-pred)**2)/ss if ss>0 else np.nan,dict(zip(x,b[1:]))

rows=[]; controls=[]
for (ds,k),g in a.groupby(["dataset","keyset"]):
    g=g[Z+ZT+[m+"_centered" for m in METHODS]].dropna()
    for m in METHODS:
        r2b,bb=fit(g,m+"_centered",Z); r2c,bc=fit(g,m+"_centered",Z+ZT)
        rows += [{"dataset":ds,"keyset":k,"method":m,"feature":x,"n_users":len(g),"coef_base":bb["z_"+x],"coef_controlled":bc["z_"+x],"r2_base":r2b,"r2_controlled":r2c} for x in FEATURES]
        controls += [{"dataset":ds,"keyset":k,"method":m,"control":x.removeprefix("z_"),"coef":bc[x],"n_users":len(g)} for x in ZT]

e=pd.DataFrame(rows); e["delta_coef"]=e.coef_controlled-e.coef_base; e["same_sign"]=np.sign(e.coef_base)==np.sign(e.coef_controlled)
keys=["dataset","method","feature"]
s=e.groupby(keys,as_index=False).agg(n_users=("n_users","mean"),coef_base=("coef_base","mean"),coef_controlled=("coef_controlled","mean"),delta_coef=("delta_coef","mean"),same_sign=("same_sign","all"),r2_base=("r2_base","mean"),r2_controlled=("r2_controlled","mean"))
s["abs_change"]=s.coef_controlled.abs()-s.coef_base.abs(); s["pct_abs_retained"]=np.where(s.coef_base.abs()>EPS,s.coef_controlled.abs()/s.coef_base.abs(),np.nan)

e.to_csv(D/"target_controlled_effects.csv",index=False); s.to_csv(D/"target_controlled_summary.csv",index=False); pd.DataFrame(controls).to_csv(D/"target_control_coefficients.csv",index=False)

print("\nHistory-feature coefficients before vs after target-composition control:")
show=s[s.feature.isin(["repeat_rate","adjacent_basket_jaccard","recent_basket_novelty"])].copy()
print(show[["dataset","method","feature","coef_base","coef_controlled","pct_abs_retained","same_sign"]].round(4).to_string(index=False))
print("\nOverall sign retention:",s.same_sign.mean().round(4))
print("\nMean R² gain from target composition:",(s.r2_controlled-s.r2_base).mean().round(4))