import time, datetime as dt
import numpy as np, pandas as pd, streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ndtr

BD_YEAR = 252
DT = 1.0/BD_YEAR
BLK = 20          # business days per month block
SETTLE_LAG = 5    # settlement = 5th business day after the block's last fixing

# ====================================================================== engine
def b76(kind, F, K, T, vol):
    sd = vol*np.sqrt(T)
    d1 = (np.log(F/K) + 0.5*sd**2)/sd; d2 = d1 - sd
    return F*ndtr(d1) - K*ndtr(d2) if kind == "C" else K*ndtr(-d2) - F*ndtr(-d1)

def eff_vol_arr(sig, tau_first, m):
    tau = tau_first + DT*np.arange(m)
    w = 2*(m - np.arange(m)) - 1
    m2 = (np.exp(np.outer(sig**2, tau)) @ w)/m**2
    T = tau[-1]
    return np.sqrt(np.maximum(np.log(m2), 1e-12)/T), T

def leg_values(S_t, sig, Ks, fixed_sum, tau_first, m):
    """Unit values of [long call, short call, short put, long put] monthly-Asian options (N=BLK fixings,
    m remaining, flat forward = S_t, partially fixed average handled via adjusted strike)."""
    N = BLK
    if m == 0:
        A = fixed_sum/N
        return [np.maximum(A-Ks[0],0), np.maximum(A-Ks[1],0), np.maximum(Ks[2]-A,0), np.maximum(Ks[3]-A,0)]
    ve, T = eff_vol_arr(sig, tau_first, m)
    out = []
    for kind, K in (("C",Ks[0]),("C",Ks[1]),("P",Ks[2]),("P",Ks[3])):
        Kp = (N*K - fixed_sum)/m
        pos = Kp > 1e-9
        Kc = np.where(pos, Kp, 1.0)
        v = b76(kind, S_t, Kc, T, ve)
        v = np.where(pos, v, (S_t - Kp) if kind == "C" else 0.0)
        out.append(v*m/N)
    return out

def net_unit(v):
    return v[0] - v[1] - v[2] + v[3]

def simulate(P, n_days, s0, vol0, vlo, vhi, volvol, rho, drift, lam, mu_j, sig_j, use_jumps, seed):
    rng = np.random.default_rng(seed)
    S = np.empty((n_days+1, P)); V = np.empty((n_days+1, P))
    S[0] = S[1] = s0; V[0] = V[1] = vol0
    if use_jumps and lam > 0 and sig_j > 0:
        adj = drift - lam*(np.exp(mu_j + 0.5*sig_j**2) - 1.0)
    else:
        adj = drift
    for t in range(1, n_days):
        z1 = rng.standard_normal(P); z2 = rng.standard_normal(P)
        zv = rho*z1 + np.sqrt(1-rho**2)*z2
        sig = V[t]
        jump = np.zeros(P)
        if use_jumps and lam > 0 and sig_j > 0:
            nj = rng.poisson(lam*DT, size=P)
            jump = nj*mu_j + np.sqrt(nj)*sig_j*rng.standard_normal(P)
        S[t+1] = S[t]*np.exp((adj - 0.5*sig**2)*DT + sig*np.sqrt(DT)*z1 + jump)
        V[t+1] = np.clip(sig*np.exp(-0.5*volvol**2*DT + volvol*np.sqrt(DT)*zv), vlo, vhi)
    return S, V

def solve_trade(S0, sig0, t0, k, vols, oc1, oc2, k3_lo, k3_hi, k4_lo, X, Y, step=1.0, chunk=1000):
    """Per-path solve for one trade date. Long call (1+oc1)*S0 and short call (1+oc2)*S0 fixed; short put k3 and
    long put k4 (fractions of S0) minimise net premium subject to premium<=X and worst-case expiry loss>=Y."""
    P = len(S0)
    grid = np.arange(k4_lo, k3_hi + 1e-9, step)/100.0
    nk = len(grid)
    c1, c2 = 1+oc1, 1+oc2
    Vtot = float(sum(vols))
    Cs = np.zeros(P); A = np.zeros((nk, P))
    for mth, vm in enumerate(vols, start=1):
        if vm <= 0: continue
        tau_first = (BLK*(k+mth) + 1 - t0)*DT
        ve, T = eff_vol_arr(sig0, tau_first, BLK)
        Cs += vm*(b76("C", 1.0, c1, T, ve) - b76("C", 1.0, c2, T, ve))
        for i, g in enumerate(grid):
            A[i] += vm*b76("P", 1.0, g, T, ve)
    x = np.linspace(0.0, 3.0, 601)
    base = np.maximum(x-c1,0) - np.maximum(x-c2,0)
    minpay = np.full((nk, nk), np.nan)
    for i3, g3 in enumerate(grid):
        for i4, g4 in enumerate(grid):
            if g4 < g3 - 1e-12:
                minpay[i3, i4] = (base - np.maximum(g3-x,0) + np.maximum(g4-x,0)).min()
    valid = ~np.isnan(minpay)
    out = {n: np.full(P, np.nan) for n in ("k3","k4","cost","loss","need_spend","best_loss")}
    feas = np.zeros(P, bool)
    for a in range(0, P, chunk):
        sl = slice(a, min(P, a+chunk)); s0 = S0[sl]; n_ = sl.stop - sl.start
        cost = s0[None,None,:]*(Cs[sl][None,None,:] - A[:, None, sl] + A[None, :, sl])
        loss = s0[None,None,:]*Vtot*np.nan_to_num(minpay, nan=-1e9)[:,:,None] - cost
        vm_ = valid[:,:,None]
        c_ok = np.where(vm_ & (cost <= X) & (loss >= Y), cost, np.inf).reshape(-1, n_)
        idx = c_ok.argmin(axis=0); mincost = c_ok[idx, np.arange(n_)]
        f = np.isfinite(mincost)
        i3s, i4s = np.divmod(idx, nk)
        pp = np.arange(sl.start, sl.stop); wf = np.where(f)[0]
        feas[sl] = f
        out["k3"][pp[f]] = grid[i3s[f]]; out["k4"][pp[f]] = grid[i4s[f]]
        out["cost"][pp[f]] = mincost[f]; out["loss"][pp[f]] = loss[i3s[f], i4s[f], wf]
        bad = ~f
        if bad.any():
            lm = np.where(vm_ & (loss >= Y), cost, np.inf).reshape(-1, n_).min(axis=0)
            bl = np.where(vm_ & (cost <= X), loss, -np.inf).reshape(-1, n_).max(axis=0)
            out["need_spend"][pp[bad]] = lm[bad]; out["best_loss"][pp[bad]] = bl[bad]
    out["feas"] = feas
    return out

def build_book(S, V, n_trades, vols, oc1, oc2, k3_lo, k3_hi, k4_lo, X, Y, halt=True):
    positions, log = [], []
    Vtot = float(sum(vols))
    for k in range(n_trades):
        t0 = BLK*k + 1
        S0, sig0 = S[t0], V[t0]
        sol = solve_trade(S0, sig0, t0, k, vols, oc1, oc2, k3_lo, k3_hi, k4_lo, X, Y)
        nfail = int((~sol["feas"]).sum())
        entry = dict(trade=k+1, day=t0, n_fail=nfail, paths=len(S0), sol=sol, S0=S0, sig0=sig0)
        if nfail > 0 and halt:
            return dict(ok=False, failed=entry, log=log, positions=positions)
        live = sol["feas"]
        Ks = [S0*(1+oc1), S0*(1+oc2), np.where(live, S0*sol["k3"], S0), np.where(live, S0*sol["k4"], S0)]
        pl = np.zeros_like(S0); ps = np.zeros_like(S0)
        for mth, vm in enumerate(vols, start=1):
            if vm <= 0: continue
            j = k + mth
            vals = leg_values(S0, sig0, Ks, np.zeros_like(S0), (BLK*j + 1 - t0)*DT, BLK)
            long_, short_ = (vals[0] + vals[3])*vm, (vals[1] + vals[2])*vm
            long_, short_ = np.where(live, long_, 0.0), np.where(live, short_, 0.0)
            pl += long_; ps += short_
            positions.append(dict(trade=k, month=mth, t0=t0, block=j, vol=vm, Ks=Ks, live=live,
                                  prem_long=long_, prem_short=short_, prem=long_-short_, settle=BLK*j + BLK + SETTLE_LAG))
        width = np.where(live, S0*Vtot*(sol["k3"]-sol["k4"]), 0.0)          # max liability on the short put spread
        net = np.where(live, pl - ps, 0.0)
        entry.update(prem_long=pl, prem_short=ps, prem_net=net, liab_max=width, theo=net + width, Vtot=Vtot)
        log.append(entry)
    return dict(ok=True, log=log, positions=positions)

def mark_book(S, V, positions, n_days):
    P = S.shape[1]
    csum = np.vstack([np.zeros((1,P)), np.cumsum(S[1:], axis=0)])
    mtm = np.zeros((n_days+1, P)); cash = np.zeros((n_days+1, P)); prem_cum = np.zeros((n_days+1, P))
    for pos in positions:
        j, vm, t0, st_, live, Ks = pos["block"], pos["vol"], pos["t0"], pos["settle"], pos["live"], pos["Ks"]
        prem_cum[t0:] += pos["prem"]
        for t in range(t0, min(st_, n_days) + 1):
            nf = int(np.clip(t - BLK*j, 0, BLK)); m = BLK - nf
            fs = (csum[min(t, BLK*j+BLK)] - csum[BLK*j]) if nf > 0 else np.zeros(P)
            if t == st_:
                vals = leg_values(S[t], V[t], Ks, fs, DT, 0)
                cash[t:] += np.where(live, net_unit(vals)*vm, 0.0)[None, :]
            else:
                vals = leg_values(S[t], V[t], Ks, fs, (BLK*j + nf + 1 - t)*DT, m)
                mtm[t] += np.where(live, net_unit(vals)*vm, 0.0)
    return mtm, cash, prem_cum

def bd_date(day, today):
    return np.busday_offset(np.datetime64(today), day-1, roll="forward").astype("datetime64[D]").astype(object)

# ====================================================================== UI
st.set_page_config(page_title="Option Strip PFE Simulator", layout="wide", initial_sidebar_state="expanded")
st.title("🛢️ Oil 4-Way Option Book – PFE Exposure Simulator")
st.markdown("**Jump-diffusion + stochastic implied vol Monte Carlo for rolling monthly-Asian 4-way (call spread + put spread) hedges**")
st.markdown("---")

if "results" not in st.session_state: st.session_state.results = None
if "failure" not in st.session_state: st.session_state.failure = None

sb = st.sidebar
sb.header("📊 Simulation Parameters")
tenor = sb.slider("Option tenor (months per trade)", 1, 6, 3)
n_trades = sb.slider("Number of monthly trade dates", 1, 6, 4)
quick = sb.checkbox("⚡ Quick Mode (500 paths)", value=False)
num_paths = 500 if quick else sb.slider("MC Paths", 500, 10000, 2000, 500)
seed = int(sb.number_input("Random seed", min_value=0, max_value=1000000, value=123, step=1))

sb.subheader("📈 Volumes added at each trade date (bbl)")
sb.caption("M1 = next whole month, M2 = month after, etc. The same schedule is added on every trade date.")
default_vols = [65000.0, 25000.0, 14000.0, 0.0, 0.0, 0.0]
vols = [float(sb.number_input(f"M{i+1}", min_value=0.0, max_value=5000000.0, value=default_vols[i], step=1000.0, key=f"v{i}"))
        for i in range(tenor)]

sb.subheader("💲 Market & implied vol")
spot0 = float(sb.number_input("Spot (USD/bbl)", min_value=1.0, max_value=1000.0, value=196.0, step=1.0))
vol0 = float(sb.number_input("Start implied vol (%)", min_value=1.0, max_value=300.0, value=55.0, step=1.0))/100
vlo = float(sb.number_input("Vol lower bound (%)", min_value=1.0, max_value=300.0, value=25.0, step=1.0))/100
vhi = float(sb.number_input("Vol upper bound (%)", min_value=1.0, max_value=300.0, value=70.0, step=1.0))/100
volvol = float(sb.number_input("Vol-of-vol (annual, relative changes)", min_value=0.0, max_value=5.0, value=0.80, step=0.05, format="%.3f"))
rho = float(sb.number_input("Spot / vol correlation", min_value=-0.99, max_value=0.99, value=-0.30, step=0.05, format="%.2f"))
drift = float(sb.number_input("Drift rate μ", min_value=-0.2, max_value=0.2, value=0.0, step=0.01, format="%.3f"))

sb.subheader("⚡ Jump-Diffusion Parameters")
use_jumps = sb.checkbox("Enable Jump-Diffusion", value=True)
lam = float(sb.number_input("Lambda (Intensity)", min_value=0.0, max_value=5.0, value=0.8, step=0.1, format="%.3f"))
mu_j = float(sb.number_input("Mean Jump Size", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, format="%.3f"))
sig_j = float(sb.number_input("Jump Std Dev", min_value=0.0, max_value=2.0, value=0.2, step=0.01, format="%.3f"))

sb.subheader("🧩 4-way solver criteria (per trade)")
oc1 = float(sb.number_input("Long call strike (% OTM)", min_value=0.0, max_value=100.0, value=0.0, step=1.0))/100
oc2 = float(sb.number_input("Short call strike (% OTM)", min_value=0.0, max_value=200.0, value=10.0, step=1.0))/100
k3_lo = float(sb.number_input("Short put strike % of spot – lower bound", min_value=1.0, max_value=200.0, value=70.0, step=1.0))
k3_hi = float(sb.number_input("Short put strike % of spot – upper bound", min_value=1.0, max_value=200.0, value=100.0, step=1.0))
k4_lo = float(sb.number_input("Long put strike % of spot – lower bound", min_value=1.0, max_value=200.0, value=50.0, step=1.0))
X = float(sb.number_input("Max spend X per trade (USD, net premium)", min_value=0.0, max_value=1e9, value=750000.0, step=10000.0))
Y = float(sb.number_input("Max loss Y per trade (USD, negative) – solver only", min_value=-1e9, max_value=0.0, value=-2000000.0, step=100000.0))
halt = sb.checkbox("Halt simulation if solver cannot solve", value=True)
sb.caption("Calendar: 20 business-day months; trades on business day 1, 21, 41, ...; each month's APO settles on the 5th business day after its last fixing. "
           "Strikes are solved for every path from that path's spot and implied vol. X and Y apply per trade only; the cumulative book has no limit "
           "(theoretical max loss is reported).")

if oc2 <= oc1: st.warning("Short call OTM % should exceed long call OTM %.")
if k4_lo >= k3_hi: st.warning("Long put lower bound must be below short put upper bound.")

today = dt.date.today()
n_days = BLK*(n_trades + tenor - 1) + BLK + SETTLE_LAG + 1
day_axis = np.arange(n_days+1)
trade_days = [BLK*k+1 for k in range(n_trades)]
settle_days = sorted({BLK*(k+m)+BLK+SETTLE_LAG for k in range(n_trades) for m in range(1, tenor+1)})

if sb.button("🚀 Run Simulation", type="primary", use_container_width=True, key="run_btn"):
    st.session_state.results = None; st.session_state.failure = None
    bar = st.progress(0); msg = st.empty()
    msg.text("🎲 Simulating price and implied-vol paths..."); bar.progress(10)
    S, V = simulate(num_paths, n_days, spot0, vol0, vlo, vhi, volvol, rho, drift, lam, mu_j, sig_j, use_jumps, seed)
    msg.text("🧩 Solving the 4-way on each trade date..."); bar.progress(35)
    book = build_book(S, V, n_trades, vols, oc1, oc2, k3_lo, k3_hi, k4_lo, X, Y, halt=halt)
    if not book["ok"]:
        st.session_state.failure = dict(entry=book["failed"], solved=len(book["log"]), X=X, Y=Y)
        bar.progress(100); msg.text("⛔ Solver could not solve – simulation halted"); time.sleep(0.3); st.rerun()
    msg.text("📐 Marking the cumulative 4-way book to market every day..."); bar.progress(60)
    mtm, cash, prem_cum = mark_book(S, V, book["positions"], n_days)
    cum = mtm + cash - prem_cum
    exposure = np.maximum(0.0, mtm)
    log = book["log"]
    st.session_state.results = dict(
        S=S, V=V, mtm=mtm, cum=cum, log=log, positions=book["positions"],
        pfe99=np.percentile(exposure, 99, axis=1), pfe95=np.percentile(exposure, 95, axis=1), ee=exposure.mean(axis=1),
        neg99=np.percentile(mtm, 1, axis=1), neg95=np.percentile(mtm, 5, axis=1),
        cneg99=np.percentile(cum, 1, axis=1), cneg95=np.percentile(cum, 5, axis=1), cmed=np.median(cum, axis=1),
        prem_net=sum(e["prem_net"] for e in log), prem_long=sum(e["prem_long"] for e in log),
        prem_short=sum(e["prem_short"] for e in log), theo=sum(e["theo"] for e in log),
        theo_steps=np.array([np.median(sum(e["theo"] for e in log[:i+1])) for i in range(len(log))]),
        n_days=n_days, trade_days=trade_days, settle_days=settle_days, Y=Y, X=X, vols=vols)
    bar.progress(100); msg.text("✅ Simulation complete"); time.sleep(0.3); st.rerun()

def mark_lines(ax):
    for d in trade_days: ax.axvline(d, color="tab:blue", ls=":", alpha=0.5)
    for d in settle_days: ax.axvline(d, color="tab:green", ls=":", alpha=0.35)

# ---------------------------------------------------------------- failure report
fail = st.session_state.failure
if fail is not None:
    e = fail["entry"]; s = e["sol"]; bad = ~s["feas"]
    st.error(f"⛔ The solver could not find a 4-way meeting max spend {fail['X']:,.0f} USD and max loss {fail['Y']:,.0f} USD "
             f"on trade {e['trade']} (business day {e['day']}) for {e['n_fail']:,} of {e['paths']:,} paths "
             f"({e['n_fail']/e['paths']:.1%}). {fail['solved']} earlier trade date(s) solved on every path. Simulation halted.")
    ns_ = s["need_spend"][bad]; bl_ = s["best_loss"][bad]
    spend_alone = bool(np.all(np.isfinite(ns_)))
    c = st.columns(4)
    c[0].metric("Failing spot (median)", f"{np.median(e['S0'][bad]):,.1f}")
    c[1].metric("Failing implied vol (median)", f"{np.median(e['sig0'][bad]):.1%}")
    c[2].metric("Spend needed to fix all failing paths",
                f"{np.max(ns_[np.isfinite(ns_)]):,.0f} USD" if spend_alone else "Not fixable by spend alone")
    c[3].metric("Loss limit needed to fix all failing paths", f"{np.min(bl_[np.isfinite(bl_)]):,.0f} USD")
    st.markdown("**What must change** (any one of these, then re-run):")
    if spend_alone:
        st.markdown(f"- Raise max spend X to at least **{np.max(ns_):,.0f} USD** (currently {fail['X']:,.0f}); 95% of failing paths need {np.percentile(ns_,95):,.0f}.")
    else:
        st.markdown("- Raising max spend alone cannot fix every failing path: even the cheapest structures breach the loss limit on some paths.")
    st.markdown(f"- Relax max loss Y to at least **{np.min(bl_[np.isfinite(bl_)]):,.0f} USD** (currently {fail['Y']:,.0f}).")
    st.markdown("- Or widen the put strike bounds, move the short call further out-of-the-money, reduce volumes, or untick 'Halt' to skip the failing paths.")
    st.stop()

# ---------------------------------------------------------------- results
R = st.session_state.results
if R is not None:
    td = day_axis[:R["n_days"]+1]
    c = st.columns(5)
    c[0].metric("Max PFE 99% (MTM)", f"${np.max(R['pfe99'])/1e6:.3f}M")
    c[1].metric("Max PFE 95% (MTM)", f"${np.max(R['pfe95'])/1e6:.3f}M")
    c[2].metric("Max EE", f"${np.max(R['ee'])/1e6:.3f}M")
    c[3].metric("Max MTM Liability (99%)", f"${abs(np.min(R['neg99']))/1e6:.3f}M")
    c[4].metric("Worst Cum. P&L (99%)", f"${abs(np.min(R['cneg99']))/1e6:.3f}M")

    st.subheader("Premium spend and theoretical maximum loss (whole book)")
    nl = abs(R["Y"]) if R["Y"] else np.nan
    pn, pl_, ps_, th = R["prem_net"], R["prem_long"], R["prem_short"], R["theo"]
    d = st.columns(5)
    d[0].metric("Premium paid – long legs (median)", f"${np.median(pl_)/1e6:.3f}M", help="Long call + long put premium across all trades")
    d[1].metric("Premium received – short legs (median)", f"${np.median(ps_)/1e6:.3f}M", help="Short call + short put premium across all trades")
    d[2].metric("Net premium (median | P95)", f"${np.median(pn)/1e6:.3f}M", delta=f"P95 {np.percentile(pn,95)/1e6:.3f}M", delta_color="off",
                help="Negative = net credit received")
    d[3].metric("Theoretical max loss (median)", f"${np.median(th)/1e6:.3f}M",
                delta=f"{np.median(th)/nl:.1f}x per-trade limit" if nl == nl and nl > 0 else None, delta_color="off",
                help="Net premium + max liability on the put spreads, summed over all trades")
    d[4].metric("Theoretical max loss (P95 | max)", f"${np.percentile(th,95)/1e6:.3f}M", delta=f"max {np.max(th)/1e6:.3f}M", delta_color="off")
    st.caption("Theoretical max loss per trade = net premium paid (lost premium) + maximum MTM liability = volume × spot at trade × (short put strike − long put strike). "
               "The call spread can only gain for the airline. Summed over all trades – no cap is applied to the cumulative book.")
    st.markdown("---")

    fig1, ax1 = plt.subplots(figsize=(16, 6))
    ax1.plot(td, R["pfe99"]/1e6, "r-", lw=3, label="PFE 99%")
    ax1.plot(td, R["pfe95"]/1e6, "orange", ls="--", lw=2.5, label="PFE 95%")
    ax1.plot(td, R["ee"]/1e6, "b-", lw=2.5, label="Expected Exposure")
    ax1.plot(td, R["neg99"]/1e6, "darkgreen", lw=3, label="Liability 99%")
    ax1.plot(td, R["neg95"]/1e6, "lightgreen", ls="--", lw=2.5, label="Liability 95%")
    ax1.axhline(0, color="k", alpha=0.5); mark_lines(ax1)
    ax1.set_title("4-Way Option Book Two-Sided MTM Exposure Profile (Daily) – blue: trade days, green: settlement days",
                  fontsize=15, fontweight="bold", pad=20)
    ax1.set_xlabel("Business Days"); ax1.set_ylabel("Exposure / Liability ($ Millions)")
    ax1.ticklabel_format(style="plain", axis="y"); ax1.grid(True, alpha=0.3); ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(16, 6))
    for i in range(min(10, R["cum"].shape[1])):
        ax2.plot(td, R["cum"][:, i]/1e6, color="gray", alpha=0.4, lw=1)
    ax2.plot(td, R["cneg99"]/1e6, "darkred", lw=4, label="1% Centile (99% loss)")
    ax2.plot(td, R["cneg95"]/1e6, "red", ls="--", lw=3, label="5% Centile (95% loss)")
    ax2.plot(td, R["cmed"]/1e6, "b-", lw=2, label="Median")
    tds = list(R["trade_days"]) + [R["n_days"]]
    for i, tv in enumerate(R["theo_steps"]):
        ax2.hlines(-tv/1e6, tds[i], tds[i+1], colors="purple", linestyles="-.", lw=2.5,
                   label="Theoretical max loss (median, builds with trades)" if i == 0 else None)
    ax2.axhline(0, color="k", alpha=0.5); mark_lines(ax2)
    ax2.set_title("Cumulative Net P&L of 4-Way Book (MTM + settled cash − premium) + Loss Centiles",
                  fontsize=15, fontweight="bold")
    ax2.set_xlabel("Business Days"); ax2.set_ylabel("Cumulative P&L ($ Millions)")
    ax2.ticklabel_format(style="plain", axis="y"); ax2.grid(True, alpha=0.3); ax2.legend()
    st.pyplot(fig2)

    S_ = R["S"]; Np = S_.shape[1]
    rng = np.random.default_rng(123); idx = rng.choice(Np, size=min(100, Np), replace=False)
    fig3, ax3 = plt.subplots(figsize=(16, 6))
    for i in idx: ax3.plot(td, S_[:, i], color="gray", alpha=0.15, lw=0.6)
    p1, p5, p95, p99 = [np.percentile(S_, q, axis=1) for q in (1, 5, 95, 99)]
    ax3.fill_between(td, p1, p99, color="red", alpha=0.15, label="1st-99th centile")
    ax3.fill_between(td, p5, p95, color="orange", alpha=0.25, label="5th-95th centile")
    ax3.plot(td, S_.min(axis=1), "k--", lw=1, alpha=0.6, label="Min path"); ax3.plot(td, S_.max(axis=1), "k--", lw=1, alpha=0.6, label="Max path")
    mark_lines(ax3)
    ax3.set_title("Monte Carlo Price Paths (USD/bbl)", fontsize=15, fontweight="bold")
    ax3.set_xlabel("Business Days"); ax3.set_ylabel("Price (USD/bbl)"); ax3.grid(True, alpha=0.3); ax3.legend()
    st.pyplot(fig3)

    V_ = R["V"]
    fig4, ax4 = plt.subplots(figsize=(16, 5))
    for i in idx[:50]: ax4.plot(td, V_[:, i]*100, color="gray", alpha=0.2, lw=0.6)
    for q, col, ls in ((5, "orange", "--"), (50, "b", "-"), (95, "orange", "--")):
        ax4.plot(td, np.percentile(V_, q, axis=1)*100, color=col, ls=ls, lw=2, label=f"{q}th centile")
    ax4.axhline(vlo*100, color="red", ls=":", label="Vol bounds"); ax4.axhline(vhi*100, color="red", ls=":")
    mark_lines(ax4)
    ax4.set_title("Simulated Implied Volatility Paths (relative changes, bounded)", fontsize=15, fontweight="bold")
    ax4.set_xlabel("Business Days"); ax4.set_ylabel("Implied vol (%)"); ax4.grid(True, alpha=0.3); ax4.legend()
    st.pyplot(fig4)

    worst = R["cum"].min(axis=0)/1e6
    fig5, ax5 = plt.subplots(figsize=(16, 5))
    ax5.hist(worst, bins=60, color="tab:red", alpha=0.7)
    for q, col in ((50, "b"), (95, "orange"), (99, "darkred")):
        ax5.axvline(np.percentile(worst, 100-q), color=col, ls="--", label=f"{q}% worst-loss level: {np.percentile(worst, 100-q):.2f}M")
    ax5.axvline(-np.median(th)/1e6, color="purple", ls="-.", lw=2, label=f"Theoretical max loss (median): {-np.median(th)/1e6:.2f}M")
    ax5.set_title("Distribution of Each Path's Worst Cumulative P&L ($M)", fontsize=15, fontweight="bold")
    ax5.set_xlabel("Worst cumulative P&L ($ Millions)"); ax5.set_ylabel("Paths"); ax5.legend(); ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)

    # ------------------------------------------------------------ trade details (single path)
    st.subheader("Trade details – selected path")
    sel = st.radio("Show trades for", ["Median path (final cum. P&L)", "Worst path (lowest cum. P&L)", "Best path", "Path number"], horizontal=True)
    fin = R["cum"][-1]; wrs = R["cum"].min(axis=0)
    if sel.startswith("Median"): p = int(np.argsort(fin)[len(fin)//2])
    elif sel.startswith("Worst"): p = int(np.argmin(wrs))
    elif sel.startswith("Best"): p = int(np.argmax(fin))
    else: p = int(st.number_input("Path number", min_value=0, max_value=int(R["S"].shape[1]-1), value=0, step=1))
    st.caption(f"Path {p}: final cumulative P&L {fin[p]:,.0f} USD, worst cumulative P&L {wrs[p]:,.0f} USD.")

    trows = []
    for e in R["log"]:
        s = e["sol"]
        if not s["feas"][p]:
            trows.append({"Trade": e["trade"], "Business day": e["day"], "Trade date": bd_date(e["day"], today).strftime("%d %b %Y"),
                          "Status": "NOT SOLVED – no trade"}); continue
        S0p = e["S0"][p]
        trows.append({"Trade": e["trade"], "Business day": e["day"], "Trade date": bd_date(e["day"], today).strftime("%d %b %Y"),
                      "Spot": S0p, "Implied vol": e["sig0"][p], "Total volume (bbl)": e["Vtot"],
                      "Long call K": S0p*(1+oc1), "Short call K": S0p*(1+oc2),
                      "Short put K": S0p*s["k3"][p], "Long put K": S0p*s["k4"][p],
                      "Premium paid (long legs)": e["prem_long"][p], "Premium received (short legs)": e["prem_short"][p],
                      "Net premium": e["prem_net"][p], "Solver worst-case loss": s["loss"][p],
                      "Max put-spread liability": e["liab_max"][p], "Theoretical max loss": e["theo"][p], "Status": "Solved"})
    tdf = pd.DataFrame(trows)
    fm = {c: "{:,.0f}" for c in ["Total volume (bbl)", "Premium paid (long legs)", "Premium received (short legs)", "Net premium",
                                 "Solver worst-case loss", "Max put-spread liability", "Theoretical max loss"] if c in tdf}
    fm.update({c: "{:,.2f}" for c in ["Spot", "Long call K", "Short call K", "Short put K", "Long put K"] if c in tdf})
    if "Implied vol" in tdf: fm["Implied vol"] = "{:.1%}"
    st.dataframe(tdf.style.format(fm, na_rep=""), use_container_width=True)

    mrows = []
    for pos in R["positions"]:
        if not pos["live"][p]: continue
        j = pos["block"]
        mrows.append({"Trade": pos["trade"]+1, "Trade date": bd_date(pos["t0"], today).strftime("%d %b %Y"),
                      "Month": f"M{pos['month']}",
                      "Averaging from": bd_date(BLK*j+1, today).strftime("%d %b %Y"),
                      "Averaging to": bd_date(BLK*j+BLK, today).strftime("%d %b %Y"),
                      "Settlement": bd_date(pos["settle"], today).strftime("%d %b %Y"),
                      "Volume (bbl)": pos["vol"], "Implied vol at trade": R["V"][pos["t0"], p],
                      "Long call K": pos["Ks"][0][p], "Short call K": pos["Ks"][1][p],
                      "Short put K": pos["Ks"][2][p], "Long put K": pos["Ks"][3][p],
                      "Premium USD/bbl": pos["prem"][p]/pos["vol"], "Premium USD (net)": pos["prem"][p]})
    mdf = pd.DataFrame(mrows)
    st.markdown("**Month-by-month legs of each trade**")
    if len(mdf):
        st.dataframe(mdf.style.format({"Volume (bbl)": "{:,.0f}", "Implied vol at trade": "{:.1%}", "Long call K": "{:,.2f}",
                     "Short call K": "{:,.2f}", "Short put K": "{:,.2f}", "Long put K": "{:,.2f}",
                     "Premium USD/bbl": "{:,.3f}", "Premium USD (net)": "{:,.0f}"}), use_container_width=True)
        st.download_button("📥 Download trade blotter (this path)", mdf.to_csv(index=False), f"trade_blotter_path_{p}.csv")

    st.subheader("Solved 4-way per trade date (distribution across all paths)")
    rows = []
    for e in R["log"]:
        s = e["sol"]; f = s["feas"]
        rows.append({"Trade": e["trade"], "Business day": e["day"], "Trade date": bd_date(e["day"], today).strftime("%d %b %Y"),
                     "Spot P5/P50/P95": f"{np.percentile(e['S0'],5):,.0f} / {np.median(e['S0']):,.0f} / {np.percentile(e['S0'],95):,.0f}",
                     "Implied vol median": f"{np.median(e['sig0']):.1%}",
                     "Short put % spot (P5-P95)": f"{np.nanpercentile(s['k3'],5)*100:.0f}-{np.nanpercentile(s['k3'],95)*100:.0f}" if f.any() else "n/a",
                     "Long put % spot (P5-P95)": f"{np.nanpercentile(s['k4'],5)*100:.0f}-{np.nanpercentile(s['k4'],95)*100:.0f}" if f.any() else "n/a",
                     "Premium paid median": float(np.median(e["prem_long"])), "Premium received median": float(np.median(e["prem_short"])),
                     "Net premium median": float(np.median(e["prem_net"])), "Net premium P95": float(np.percentile(e["prem_net"], 95)),
                     "Theoretical max loss median": float(np.median(e["theo"])), "Unsolved paths": e["n_fail"]})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:,.0f}" for c in ["Premium paid median", "Premium received median",
                 "Net premium median", "Net premium P95", "Theoretical max loss median"]}), use_container_width=True)
    df = pd.DataFrame({"Business_Day": td, "PFE_99_M": R["pfe99"]/1e6, "PFE_95_M": R["pfe95"]/1e6, "EE_M": R["ee"]/1e6,
                       "MTM_P1_M": R["neg99"]/1e6, "MTM_P5_M": R["neg95"]/1e6,
                       "CumPnL_P1_M": R["cneg99"]/1e6, "CumPnL_P5_M": R["cneg95"]/1e6, "CumPnL_Median_M": R["cmed"]/1e6})
    st.download_button("📥 Download Exposure CSV", df.to_csv(index=False), "option_pfe_results.csv")
    st.caption("Monthly-Asian options priced with a moment-matched lognormal approximation on a flat forward curve (forward = spot, r = 0). "
               "Each position is revalued daily with the path's spot, realised average so far and implied vol. Trade dates assume business day 1 = today.")
else:
    st.info("🚀 Set parameters and click **Run Simulation** to solve each trade date's 4-way and generate the option-book PFE, "
            "cumulative P&L loss distribution, price paths and implied-vol paths.")
