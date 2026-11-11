"""Summaries for the v2 paper update, printed as Markdown, straight from results_v2_* JSON."""
import glob, json, sys
import numpy as np

R = {}
for f in glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/results_v2_*/runs/*.json"):
    r = json.load(open(f)); R[(r["target"], r["country"])] = r
C = sys.argv[2].split(",") if len(sys.argv) > 2 else ["India", "Brazil", "Russia", "USA"]; T = ["daily", "cumulative"]
SER = [(t, c) for t in T for c in C]
M = ["Naive", "ARIMA", "SARIMA", "ETS (damped)", "LSTM", "GRU", "ARIMA-LSTM hybrid", "DHBO (proposed)", "DHBO-v2 (proposed)"]
lab = lambda m: m.replace("ARIMA-LSTM", "ARIMA–LSTM").replace("DHBO (proposed)", "DHBO-v1").replace("DHBO-v2 (proposed)", "DHBO-v2 (proposed)")
assert len(R) == 8, f"only {len(R)} series"

ranks = {m: [] for m in M}
for k in SER:
    for i, m in enumerate(sorted(M, key=lambda m: R[k]["metrics"][m]["MASE"])):
        ranks[m].append(i + 1)
mase = lambda m, t: np.mean([R[(t, c)]["metrics"][m]["MASE"] for c in C])
print("## overall (9 models)")
for m in sorted(M, key=lambda m: np.mean(ranks[m])):
    print(f"| {lab(m)} | {np.mean(ranks[m]):.2f} | {ranks[m].count(1)} | {mase(m,'daily'):.3f} | {mase(m,'cumulative'):.3f} | {(mase(m,'daily')+mase(m,'cumulative'))/2:.3f} |  ranks {ranks[m]}")

print("\n## MASE per series (v2, v1, hybrid, best baseline)")
for k in SER:
    mt = R[k]["metrics"]; base = [m for m in M if not m.startswith("DHBO")]
    best = min(base, key=lambda m: mt[m]["MASE"])
    v2, v1 = mt["DHBO-v2 (proposed)"]["MASE"], mt["DHBO (proposed)"]["MASE"]
    print(k, f"v2 {v2:.3f} v1 {v1:.3f} change {100*(v2/v1-1):+.1f}% | hybrid {mt['ARIMA-LSTM hybrid']['MASE']:.3f} | best baseline {best} {mt[best]['MASE']:.3f} | rank v2 {ranks['DHBO-v2 (proposed)'][SER.index(k)]}")

print("\n## MAPE per series")
for t in T:
    for m in M:
        vals = [R[(t, c)]["metrics"][m]["MAPE"] for c in C]
        print(t, f"{lab(m):22s}", " ".join(f"{v:.3f}" for v in vals), f"mean {np.mean(vals):.4f}")

print("\n## DM v2 vs each (p<0.05)")
b = w = n = 0
for k in SER:
    for m, v in R[k]["dm"].items():
        if v["p"] < 0.05: b += v["DM"] < 0; w += v["DM"] > 0
        else: n += 1
print("all comparisons incl. v1:", b, w, n)
b = w = n = 0
for k in SER:
    for m, v in R[k]["dm"].items():
        if m == "DHBO (proposed)": continue
        if v["p"] < 0.05: b += v["DM"] < 0; w += v["DM"] > 0
        else: n += 1
print("vs 7 baselines:", b, w, n)
for k in SER:
    d = R[k]["dm"]
    print(k, "vs v1", f"{d['DHBO (proposed)']['DM']:.2f} p={d['DHBO (proposed)']['p']:.3f}", "| vs hybrid", f"{d['ARIMA-LSTM hybrid']['DM']:.2f} p={d['ARIMA-LSTM hybrid']['p']:.3f}")

print("\n## variants MASE (stacking-level ablation)")
V = list(R[SER[0]]["metrics_variants"])
for t in T:
    full = np.mean([R[(t, c)]["metrics_variants"]["DHBO-v2 (full)"]["MASE"] for c in C])
    for v in V:
        vals = [R[(t, c)]["metrics_variants"][v]["MASE"] for c in C]
        print(t, f"{v:30s}", " ".join(f"{x:.3f}" for x in vals), f"mean {np.mean(vals):.3f} {100*(np.mean(vals)/full-1):+.1f}%")

print("\n## intervals")
for meth in ["Split CP 95%", "ACI 95%", "Split CP 80%", "ACI 80%"]:
    nom = 0.95 if "95" in meth else 0.8
    cov = [R[k]["intervals"][meth]["coverage"] for k in SER]
    print(meth, "range", round(min(cov), 3), round(max(cov), 3), "mean abs dev", round(np.mean([abs(c - nom) for c in cov]), 3))
for k in SER:
    iv = R[k]["intervals"]
    print(k, *(f"{iv[m]['coverage']:.3f}" for m in ["Split CP 80%", "Split CP 95%", "ACI 80%", "ACI 95%"]), f"{iv['ACI 95%']['width']:.0f} {iv['ACI 95%']['winkler']:.0f}")

print("\n## hyper-parameters and weights")
for k in SER:
    r = R[k]; hb, ht = r["hp_bilstm"], r["hp_tcn"]
    print(k, "K", hb["K"], "L", hb["L"], "bilstm", hb["hidden"], hb["layers"], "| tcn ch", ht["channels"], "lev", ht["levels"], "ker", ht["kernel"],
          "| weights", np.round(r["mean_weights_test"], 2), "| times", {a: b for a, b in r["log"].items() if a.startswith("time")})
