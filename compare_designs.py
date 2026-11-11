"""Compare DHBO-v1, DHBO-v2 (BiLSTM-TCN) and DHBO-v2-CNN (BiLSTM-CNN) on one country set.

    python compare_designs.py <tcn_dir> <cnn_dir> <countries comma list> <v1_dir>
Each dir may hold both targets, or be split into <dir>_daily and <dir>_cumulative.
e.g.  python compare_designs.py results_v2 results_cnn India,Brazil,Russia,USA results
      python compare_designs.py val_tcn val_cnn "France,Germany,Japan,South Korea" val_v1
Set TARGETS=daily to restrict the targets.
"""
import json, os, sys
import numpy as np
import dhbo as D

tcn_p, cnn_p, C = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
v1_p = sys.argv[4] if len(sys.argv) > 4 else "results"
R = lambda p, t, c, ext: f"{p}_{t}/runs/{t}_{c}.{ext}" if os.path.isdir(f"{p}_{t}") else f"{p}/runs/{t}_{c}.{ext}"
T = os.environ.get("TARGETS", "daily,cumulative").split(",")
BASE = ["Naive", "ARIMA", "SARIMA", "ETS (damped)", "LSTM", "GRU", "ARIMA-LSTM hybrid"]
rows, out = [], {}
for t in T:
    for c in C:
        rt = json.load(open(R(tcn_p, t, c, "json")))
        rc = json.load(open(R(cnn_p, t, c, "json")))
        zt = np.load(R(tcn_p, t, c, "npz")); zc = np.load(R(cnn_p, t, c, "npz"))
        assert np.allclose(zt["y_true"], zc["y_true"])
        y = zt["y_true"]
        P = {"DHBO-v1": zt["pred__DHBO (proposed)"], "BiLSTM-TCN": zt["pred__DHBO-v2 (proposed)"], "BiLSTM-CNN": zc["pred__DHBO-v2-CNN"]}
        for b in BASE:
            P[b] = zt[f"pred__{b}"]
        mase = {k: rt["metrics"][k]["MASE"] for k in BASE}
        mase["DHBO-v1"] = rt["metrics"]["DHBO (proposed)"]["MASE"]
        mase["BiLSTM-TCN"] = rt["metrics"]["DHBO-v2 (proposed)"]["MASE"]
        mase["BiLSTM-CNN"] = rc["metrics"]["DHBO-v2-CNN"]["MASE"]
        e = {k: np.mean(np.abs(y - v), axis=1) for k, v in P.items()}
        dm = {}
        for a, b in [("BiLSTM-CNN", "BiLSTM-TCN"), ("BiLSTM-CNN", "DHBO-v1"), ("BiLSTM-TCN", "DHBO-v1")]:
            dm[f"{a} vs {b}"] = D.dm_test(e[a], e[b], 7)
        sig = {}
        for d in ["DHBO-v1", "BiLSTM-TCN", "BiLSTM-CNN"]:
            better = worse = 0
            for b in BASE:
                s, p = D.dm_test(e[d], e[b], 7)
                better += (p < 0.05 and s < 0); worse += (p < 0.05 and s > 0)
            sig[d] = (better, worse)
        order = sorted(mase, key=mase.get)
        rank = {k: order.index(k) + 1 for k in mase}       # rank among 10 models
        iv = {"BiLSTM-TCN": rt["intervals"]["ACI 95%"]["coverage"], "BiLSTM-CNN": rc["intervals"]["ACI 95%"]["coverage"]}
        out[(t, c)] = dict(mase=mase, dm=dm, sig=sig, rank=rank, aci95=iv, hp_cnn=rc["hp_tcn"], w_cnn=rc["mean_weights_test"])
        print(f"{t:10s} {c:12s} MASE v1 {mase['DHBO-v1']:.3f}  TCN {mase['BiLSTM-TCN']:.3f}  CNN {mase['BiLSTM-CNN']:.3f}  best base {min(BASE,key=mase.get)} {min(mase[b] for b in BASE):.3f}"
              f" | rank v1/TCN/CNN {rank['DHBO-v1']}/{rank['BiLSTM-TCN']}/{rank['BiLSTM-CNN']} (of 10)"
              f" | DM CNN-TCN {dm['BiLSTM-CNN vs BiLSTM-TCN'][0]:+.2f} p={dm['BiLSTM-CNN vs BiLSTM-TCN'][1]:.3f}"
              f" | ACI95 TCN {iv['BiLSTM-TCN']:.3f} CNN {iv['BiLSTM-CNN']:.3f}")
print()
for d in ["DHBO-v1", "BiLSTM-TCN", "BiLSTM-CNN"]:
    for t in T:
        m = np.mean([out[(t, c)]["mase"][d] for c in C])
        print(f"{d:11s} mean MASE {t:10s} {m:.3f}")
    allm = np.mean([out[k]["mase"][d] for k in out])
    print(f"{d:11s} mean MASE all        {allm:.3f} | mean rank {np.mean([out[k]['rank'][d] for k in out]):.2f} | firsts {sum(out[k]['rank'][d]==1 for k in out)}"
          f" | sig better/worse vs 7 baselines {sum(out[k]['sig'][d][0] for k in out)}/{sum(out[k]['sig'][d][1] for k in out)}")
base_best = np.mean([min(out[k]['mase'][b] for b in BASE) for k in out])
print("best-baseline-per-series mean MASE", round(base_best, 3))
for b in BASE:
    print(f"   {b:18s} mean MASE all {np.mean([out[k]['mase'][b] for k in out]):.3f}")
for pair in ["BiLSTM-CNN vs BiLSTM-TCN", "BiLSTM-CNN vs DHBO-v1", "BiLSTM-TCN vs DHBO-v1"]:
    w = sum(out[k]["dm"][pair][1] < 0.05 and out[k]["dm"][pair][0] < 0 for k in out)
    l = sum(out[k]["dm"][pair][1] < 0.05 and out[k]["dm"][pair][0] > 0 for k in out)
    print(f"{pair}: first significantly better in {w}, worse in {l}, of {len(out)}")
json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()}, open(f"compare_{os.path.basename(tcn_p.rstrip('/'))}_{os.path.basename(cnn_p.rstrip('/'))}.json", "w"), indent=1, default=float)

# ---- scale-free summary: geometric mean of MASE relative to SARIMA (and to v1) ----
D3 = ["DHBO-v1", "BiLSTM-TCN", "BiLSTM-CNN"]
print("\n## geometric-mean relative MASE (model / SARIMA), per target and overall")
for t in T + ["all"]:
    ks = [k for k in out if t == "all" or k[0] == t]
    line = []
    for m in D3 + BASE:
        g = np.exp(np.mean([np.log(out[k]["mase"][m] / out[k]["mase"]["SARIMA"]) for k in ks]))
        line.append(f"{m} {g:.3f}")
    print(t, " | ".join(line))
print("\n## mean rank (of 10) per target")
for t in T:
    ks = [k for k in out if k[0] == t]
    print(t, " | ".join(f"{m} {np.mean([out[k]['rank'][m] for k in ks]):.2f}" for m in D3 + BASE))
print("\n## pairwise DM detail per series (stat, p)")
for k in out:
    print(k, {p: (round(v[0], 2), round(v[1], 3)) for p, v in out[k]["dm"].items()})
print("\n## ACI 95% coverage incl. v1")
for k in out:
    t, c = k
    v1 = json.load(open(R(v1_p, t, c, "json")))
    print(k, "v1", round(v1["intervals"]["ACI 95%"]["coverage"], 3), "TCN", round(out[k]["aci95"]["BiLSTM-TCN"], 3), "CNN", round(out[k]["aci95"]["BiLSTM-CNN"], 3))
