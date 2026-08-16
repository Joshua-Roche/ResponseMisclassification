import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from joblib import Parallel, delayed
from scipy.stats import norm

QUICK = False          # set True for a fast sanity check
NCORES = 110


# =============================================================================
# Step 0 - the numbers we're working with
# =============================================================================

lam_os   = np.log(2) / 14.5 # death before progression, control arm (RIOMeso)
lam_fast = np.log(2) / 3.4 # death after a progression nobody noticed
lam_pfs  = np.log(2) / 5.6 # progression, control arm (MS01)

visits = [3, 6, 9, 12, 15, 18, 21, 24] # scans every 3 months
cutoff = 24 # everyone censored here

true_hr = 0.70 # applied to both progression and pre-progression death
alpha, power = 0.05, 0.80

if QUICK:
    m_grid, n_reps, n_big, reps_big = np.arange(0, 1.01, 0.25), 200, 4000, 2
else:
    m_grid, n_reps, n_big, reps_big = np.arange(0, 1.01, 0.01), 10000, 20000, 5


# =============================================================================
# Step 1 - what "misclassification rate m" actually means
# =============================================================================
# Three mRECIST categories but only one boundary matters, because the only thing
# that changes anything is whether the scan says PD. So we only need the two
# error rates on that boundary:
#
#     fn = P(read non-PD | truly PD)     = 1 - sensitivity
#     fp = P(read PD | truly non-PD)     = 1 - specificity
#
# In the baseline case a patient is misread with probability m, and a misread
# non-PD patient only crosses the boundary half the time (PR can go to SD, which
# is wrong but changes nothing). So fn = m and fp = m/2, and the errors are
# already 2:1 in favour of false negatives before we do anything.
#
# The total, fn + fp = 3m/2, is one minus the Youden index, and that is what
# decides how much information the reader carries. So when we ask what happens
# if the errors are mostly one type, we hold the total fixed and change only the
# split. Otherwise we would be changing the direction of the error and the
# quality of the reader at the same time and we could not tell them apart.

ratio = {"balanced": 2.0, "mostly_fn": 7.5, "mostly_fp": 1 / 7.5}   # fn : fp


def error_rates(m, scenario):
    total = 1.5 * m # = 1 - J
    r = ratio[scenario]
    fn = total * r / (r + 1)
    fp = total * 1 / (r + 1)
    return min(fn, 1.0), min(fp, 1.0) # can't have a probability above 1


# Whichever of fn and fp takes the bigger share of the budget hits 1 first, so
# solve 1.5 * m * share = 1. 
def clip_point(scenario):
    r = ratio[scenario]
    share = max(r, 1) / (r + 1)
    m = 1 / (1.5 * share)
    return m if m <= 1 else None


# =============================================================================
# Step 2 - simulate one trial
# =============================================================================
# Interval by interval. People die, people progress, then
# they get scanned. The PFS event is whichever comes first: a scan that says PD,
# or death. A true progression nobody catches doesn't count as an event - the
# patient stays in the risk set - but it does put them on the fast death rate,
# so a missed progression tends to show up later as a death instead since they were
# on an ineffective, potentially toxic treatment.

def one_trial(seed, m, n, scenario, hr):
    rng = np.random.default_rng(42 + seed)
    fn, fp = error_rates(m, scenario)

    arm = np.array([1] * (n // 2) + [2] * (n - n // 2)) # 1 control, 2 experimental
    at_risk = np.ones(n, dtype=bool) # no PFS event yet
    progressed = np.zeros(n, dtype=bool) # true state, which nobody observes

    pfs_time = np.full(n, np.nan)
    pfs_event = np.zeros(n, dtype=int)

    n_pd_scans = n_nonpd_scans = n_fn = n_fp = 0
    last_t = 0.0

    for t in visits:
        dt = t - last_t
        if not at_risk.any():
            break

        # --- who dies in this interval (death is a PFS event) ---
        here = np.where(at_risk)[0]
        rate = np.where(progressed[here], lam_fast,
                        np.where(arm[here] == 1, lam_os, lam_os * hr))
        wait = rng.exponential(1 / rate)
        dead = wait <= dt
        if dead.any():
            pfs_time[here[dead]] = last_t + wait[dead]
            pfs_event[here[dead]] = 1
            at_risk[here[dead]] = False
        if not at_risk.any():
            break

        # --- who progresses in this interval (not an event until it's seen) ---
        here = np.where(at_risk & ~progressed)[0]
        if len(here):
            prate = np.where(arm[here] == 1, lam_pfs, lam_pfs * hr)
            progressed[here[rng.exponential(1 / prate) <= dt]] = True

        # --- the scan --- #
        here = np.where(at_risk)[0]
        truth = progressed[here]
        u = rng.random(len(here))
        says_pd = np.where(truth, u > fn, u < fp) # miss a real PD, or invent one

        n_pd_scans += int(truth.sum())
        n_nonpd_scans += int((~truth).sum())
        n_fn += int((truth & ~says_pd).sum())
        n_fp += int((~truth & says_pd).sum())

        # --- a scan that says PD is the event, whether or not it's real --- #
        called = here[says_pd]
        if len(called):
            pfs_time[called] = t
            pfs_event[called] = 1
            at_risk[called] = False

        last_t = t

    # anyone with no recorded PD and no death by 24 months is censored there
    left = np.where(at_risk)[0]
    pfs_time[left], pfs_event[left] = cutoff, 0

    df = pd.DataFrame({"arm": arm, "time": pfs_time, "event": pfs_event})
    try:
        fit = CoxPHFitter().fit(df, duration_col="time", event_col="event")
        row = fit.summary.loc["arm"]
        hr_hat = np.exp(fit.params_["arm"])
        lo, hi = np.exp(row["coef lower 95%"]), np.exp(row["coef upper 95%"])
        p = row["p"]
    except Exception:
        hr_hat = lo = hi = p = np.nan

    return {"hr": hr_hat, "lo": lo, "hi": hi, "p": p,
            "events": int(pfs_event.sum()),
            "pd_scans": n_pd_scans, "nonpd_scans": n_nonpd_scans,
            "fn": n_fn, "fp": n_fp}


# =============================================================================
# Step 3 - how many patients we need
# =============================================================================

print("Measuring the PFS event rate with no misclassification")
big = Parallel(n_jobs=NCORES)(
    delayed(one_trial)(100000 + i, 0.0, n_big, "balanced", true_hr)
    for i in range(reps_big))

ev_rate = np.mean([b["events"] for b in big]) / n_big
hr_at_zero = np.nanmean([b["hr"] for b in big])

print(f"  event rate            {ev_rate:.3f}")
print(f"  Cox HR at m = 0       {hr_at_zero:.4f}   (target {true_hr})")

n_events = int(np.ceil(4 * (norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2
                       / np.log(true_hr) ** 2))
n_patients = int(np.ceil(n_events / ev_rate))
n_patients += n_patients % 2      # keep it even for 1:1

print(f"  events needed         {n_events}")
print(f"  sample size           {n_patients}\n")

for s in ratio:
    c = clip_point(s)
    if c is not None:
        print(f"note: '{s}' error rates hit 1.0 at m = {c:.3f}")


# =============================================================================
# Step 4 - run over all misclassification rates
# =============================================================================

rows = []
for scenario in ratio:
    print(f"\n--- {scenario} ---")
    for m in m_grid:
        out = Parallel(n_jobs=NCORES)(
            delayed(one_trial)(i, m, n_patients, scenario, true_hr)
            for i in range(n_reps))
        d = pd.DataFrame(out)

        sig = (d["p"] < alpha).astype(float)
        cov = ((d["lo"] <= true_hr) & (true_hr <= d["hi"])).astype(float)
        fn_hat = d["fn"].sum() / max(d["pd_scans"].sum(), 1)
        fp_hat = d["fp"].sum() / max(d["nonpd_scans"].sum(), 1)

        rows.append({"scenario": scenario, "m": m,
                     "true_hr": true_hr, "n": n_patients, "reps": n_reps,
                     "power": sig.mean(), "power_mcse": sig.std(ddof=1) / np.sqrt(n_reps),
                     "coverage": cov.mean(), "coverage_mcse": cov.std(ddof=1) / np.sqrt(n_reps),
                     "mean_hr": d["hr"].mean(),
                     "mean_events": d["events"].mean(),
                     "fn_rate": fn_hat, "fp_rate": fp_hat,
                     "fn_fp_ratio": fn_hat / fp_hat if fp_hat > 0 else np.nan})

        print(f"  m={m:4.2f}  power={sig.mean():.3f}  coverage={cov.mean():.3f}  "
              f"HR={d['hr'].mean():.3f}  fn:fp={rows[-1]['fn_fp_ratio']:.2f}")


# =============================================================================
# Step 5 - save
# =============================================================================

results = pd.DataFrame(rows)
results.to_csv("pfs_results.csv", index=False)
print("\nsaved pfs_results.csv")