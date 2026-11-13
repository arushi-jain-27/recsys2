from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; D=ROOT/"analysis"/"data"; EPS=1e-12
DATASETS=["dunnhumby","instacart","sam","tafeng","valuedshopper"]
METHODS=["recency","frequency","tifuknn","fpmc","triple2vec","dnntsp","cbp","diffrec"]
FEATURES=["n_baskets","avg_basket_size","repeat_rate","adjacent_basket_jaccard","recent_basket_novelty","popularity_profile","neighbor_similarity","transition_strength","relational_strength"]
Z=["z_"+x for x in FEATURES]

p=pd.read_csv(D/"performance.csv",dtype={"user":str}); f=pd.read_csv(D/"regime_features.csv",dtype={"user":str})
p=p[p.dataset.isin(DATASETS)].copy(); f=f[f.dataset.isin(DATASETS)].copy()
r=p[["dataset","user"]].drop_duplicates().merge(f[["dataset","user"]+FEATURES],on=["dataset","user"],validate="one_to_one")

for ds in DATASETS:
    m=r.dataset.eq(ds)
    for x in FEATURES:
        s=r.loc[m,x].std(ddof=0); r.loc[m,"z_"+x]=(r.loc[m,x]-r.loc[m,x].mean())/s if pd.notna(s) and s>EPS else np.nan

a=p.merge(r[["dataset","user"]+Z],on=["dataset","user"],validate="many_to_one"); a["user_mean"]=a[METHODS].mean(axis=1)
for m in METHODS: a[m+"_centered"]=a[m]-a["user_mean"]

def fit(d,y):
    d=d[[y]+Z].dropna(); X=np.c_[np.ones(len(d)),d[Z].to_numpy(float)]; yy=d[y].to_numpy(float); b=np.linalg.lstsq(X,yy,rcond=None)[0]
    pred=X@b; ss=np.sum((yy-yy.mean())**2); return len(d),1-np.sum((yy-pred)**2)/ss if ss>0 else np.nan,b[1:]

rows=[]
for (ds,k),g in a.groupby(["dataset","keyset"]):
    g=g[g[METHODS].max(axis=1)>EPS]
    for m in METHODS:
        n,r2,b=fit(g,m+"_centered")
        rows += [{"dataset":ds,"keyset":k,"method":m,"feature":x,"n_users":n,"r2":r2,"coef":c} for x,c in zip(FEATURES,b)]

e=pd.DataFrame(rows); keys=["dataset","method","feature"]
s=e.groupby(keys,as_index=False).agg(n_users=("n_users","mean"),coef_mean=("coef","mean"),coef_sd=("coef",lambda x:x.std(ddof=0)),r2_mean=("r2","mean"))
sg=e.groupby(keys).coef.apply(lambda x:"positive" if (x>0).all() else "negative" if (x<0).all() else "mixed").reset_index(name="sign_nonzero"); s=s.merge(sg,on=keys)

base=pd.read_csv(D/"joint_regime_effects.csv")
base=base[(base.outcome=="centered")&(base.population=="all")][["dataset","keyset","method","feature","coef"]].rename(columns={"coef":"coef_all"})
c=base.merge(e[["dataset","keyset","method","feature","coef"]].rename(columns={"coef":"coef_nonzero"}),on=["dataset","keyset","method","feature"])
c["same_sign"]=(np.sign(c.coef_all)==np.sign(c.coef_nonzero)); c["delta"]=c.coef_nonzero-c.coef_all

e.to_csv(D/"joint_regime_nonzero_effects.csv",index=False); s.to_csv(D/"joint_regime_nonzero_summary.csv",index=False); c.to_csv(D/"joint_regime_nonzero_comparison.csv",index=False)

print("\nCross-keyset nonzero-user effects:")
print(s.sort_values(["dataset","coef_mean"],key=lambda x:x.abs() if x.name=="coef_mean" else x).to_string(index=False))
print("\nSign agreement with all-user model:",c.same_sign.mean().round(4))
print(c.groupby("dataset").same_sign.mean().round(3))