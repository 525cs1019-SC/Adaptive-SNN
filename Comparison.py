from sklearn.ensemble import HistGradientBoostingClassifier

def probs(L): return torch.softmax(torch.from_numpy(L), -1).numpy()

def feats(L):
    """Per-timestep signals used by all exit rules."""
    P = probs(L); s = np.sort(P, -1)
    conf, margin, pred = s[..., -1], s[..., -1] - s[..., -2], P.argmax(-1)
    ent = -(P * np.log(P + 1e-9)).sum(-1)
    N, Tn = pred.shape
    runlen = np.ones((N, Tn), int)
    for t in range(1, Tn):
        runlen[:, t] = np.where(pred[:, t] == pred[:, t - 1], runlen[:, t - 1] + 1, 1)
    return conf, margin, pred, runlen, ent

def exit_conf_stable(Fe, p, k):
    """Exit at first t where confidence >= p AND prediction unchanged for k steps."""
    conf, _, _, runlen, _ = Fe; N, Tn = conf.shape
    ok = (conf >= p) & (runlen >= k)
    return np.where(ok.any(1), ok.argmax(1), Tn - 1) + 1

def run(Fe, Y, E):
    """avg T, accuracy at exit points E (1-indexed)"""
    _, _, pred, _, _ = Fe
    return float(E.mean()), float((pred[np.arange(len(Y)), E - 1] == Y).mean())

# ---------------- fit the learned gate on validation, cross-fitted ----------------
valF = [feats(L) for L, Y in val_data]
def gate_X(Fe):
    conf, margin, _, runlen, ent = Fe; N, Tn = conf.shape
    prev_conf = np.concatenate([conf[:, :1], conf[:, :-1]], 1)
    pos = np.broadcast_to(np.arange(1, Tn + 1), (N, Tn))
    return np.stack([conf, margin, ent, runlen, pos, prev_conf], -1)

def fit_gate():
    n = len(val_data[0][1]); fold = np.arange(n) % 2
    mk = lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_depth=4)
    XB = [gate_X(Fe) for Fe in valF]
    CB = [(Fe[2] == Y[:, None]) for Fe, (_, Y) in zip(valF, val_data)]
    def xy(mask):
        X = np.concatenate([x[mask].reshape(-1, x.shape[-1]) for x in XB])
        y = np.concatenate([c[mask].reshape(-1) for c in CB])
        return X, y
    val_cf = [np.zeros((n, cfg.MAX_T)) for _ in XB]
    for f in (0, 1):
        m = mk().fit(*xy(fold == f))
        for s, x in enumerate(XB):
            o = x[fold != f]
            val_cf[s][fold != f] = m.predict_proba(o.reshape(-1, o.shape[-1]))[:, 1].reshape(-1, cfg.MAX_T)
    full_model = mk().fit(*xy(np.ones(n, bool)))
    return val_cf, full_model

val_gate_probs, gate_model = fit_gate()
test_gate_probs = [gate_model.predict_proba(gate_X(feats(L)).reshape(-1, 6))[:, 1].reshape(-1, cfg.MAX_T)
                   for L, Y in test_data]

def exit_gate(gate_p, thresh):
    N, Tn = gate_p.shape
    ok = gate_p >= thresh
    return np.where(ok.any(1), ok.argmax(1), Tn - 1) + 1

# ---------------- define method families ----------------
PGRID = list(np.round(np.linspace(0.3, 0.99, 24), 3))
TGRID = list(np.round(np.linspace(0.4, 0.98, 25), 3))

METHODS = {
    "Fixed-T":            lambda Fe, s, c: np.full(len(Fe[2]), c[0]),
    "conf (SEENN-style)": lambda Fe, s, c: exit_conf_stable(Fe, c[0], 1),
    "stable k=2":         lambda Fe, s, c: exit_conf_stable(Fe, c[0], 2),
    "gate (learned)":     lambda Fe, s, c: exit_gate(c[1][s], c[0]),
}
GRIDS = {
    "Fixed-T":            [(t,) for t in range(1, cfg.MAX_T + 1)],
    "conf (SEENN-style)": [(p,) for p in PGRID],
    "stable k=2":         [(p,) for p in PGRID],
    "gate (learned)":     [(t, val_gate_probs) for t in TGRID],
}
TEST_GRIDS = {**GRIDS, "gate (learned)": [(t, test_gate_probs) for t in TGRID]}

def evaluate_cfg(F_list, Y_list, fam, c, grids_source):
    f = METHODS[fam]
    r = [run(Fe, Y, f(Fe, s, c)) for s, (Fe, Y) in enumerate(zip(F_list, Y_list))]
    return tuple(np.mean(r, 0))

def pareto(pts):
    out, best = [], -1
    for t, a, c in sorted(pts, key=lambda x: (x[0], -x[1])):
        if a > best + 1e-9: out.append((t, a, c)); best = a
    return out

testF  = [feats(L) for L, Y in test_data]
testY  = [Y for L, Y in test_data]
valY   = [Y for L, Y in val_data]

val_front, test_curve = {}, {}
for fam in METHODS:
    val_front[fam] = pareto([(*evaluate_cfg(valF, valY, fam, c, GRIDS), c) for c in GRIDS[fam]])
    test_curve[fam] = pareto([(*evaluate_cfg(testF, testY, fam, c, TEST_GRIDS), c)
                               for _, _, c in val_front[fam]])

def T_needed(curve, acc):
    y = np.array([p[1] for p in curve]); x = np.array([p[0] for p in curve])
    return np.nan if (acc < y[0] or acc > y[-1]) else float(np.interp(acc, y, x))

# ---------------- Report 1: iso-accuracy table ----------------
print("="*90, "\nAverage timesteps needed to reach a given TEST accuracy\n", "="*90, sep="")
targets = [0.68, 0.70, 0.71, 0.715, 0.72]
print(f"{'Target':<8}" + "".join(f"{f:<20}" for f in METHODS))
for a in targets:
    print(f"{100*a:<8.1f}" + "".join(f"{T_needed(test_curve[f], a):<20.2f}" for f in METHODS))

ref = "conf (SEENN-style)"
print("\nSaving vs SEENN-style at same accuracy:")
for a in targets:
    r0 = T_needed(test_curve[ref], a)
    row = "".join(f"{100*(1-T_needed(test_curve[f], a)/r0):<20.1f}" for f in METHODS if f != ref)
    print(f"{100*a:<8.1f}" + row)

# ---------------- Report 2: operating points at allowed accuracy drop ----------------
val_t6 = np.mean([(Fe[2][:, -1] == Y).mean() for Fe, Y in zip(valF, valY)])
for drop in [0.5, 1.0, 2.0]:
    tgt = val_t6 - drop / 100
    print("\n" + "="*90 + f"\nAllowed drop <= {drop}pt vs T=6 (val T=6 acc = {100*val_t6:.2f}%)\n" + "="*90)
    print(f"{'Method':<22}{'test avgT':>12}{'test acc':>11}{'saved vs T=6':>15}{'saved vs SEENN':>16}")
    base = None
    for fam in METHODS:
        ok = [x for x in val_front[fam] if x[1] >= tgt]
        if not ok: print(f"{fam:<22}  target not reachable"); continue
        c = min(ok, key=lambda x: x[0])[2]
        at, ac = evaluate_cfg(testF, testY, fam, c, TEST_GRIDS)
        if fam == "conf (SEENN-style)": base = at
        sv = f"{100*(1-at/base):>15.1f}%" if base else f"{'-':>16}"
        print(f"{fam:<22}{at:>12.2f}{100*ac:>10.2f}%{100*(1-at/cfg.MAX_T):>14.1f}%{sv}")

print("\nDone.")
