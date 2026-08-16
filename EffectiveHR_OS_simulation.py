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
# All medians in months, converted to exponential rates.

lam_os   = np.log(2) / 14.5 # death before progression, control arm (RIOMeso)
lam_fast = np.log(2) / 3.4 # death after a progression nobody noticed
lam_2L   = np.log(2) / 8 # death after switching to second line
lam_pfs  = np.log(2) / 5.6 # progression, control arm (MS01)

visits = [3, 6, 9, 12, 15, 18, 21, 24] # scans every 3 months
cutoff = 24 # everyone censored here

target_hr = 0.70 # the EFFECTIVE HR we're aiming for
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
# Interval by interval. In each interval people
# can die, they can progress, and then they get scanned. A scan that says PD
# stops the trial drug and puts them on the second-line death rate. A true
# progression nobody catches puts them on the fast death rate instead.

def one_trial(seed, m, n, scenario, hr1):
    rng = np.random.default_rng(42 + seed)
    fn, fp = error_rates(m, scenario)

    arm = np.array([1] * (n // 2) + [2] * (n - n // 2)) # 1 control, 2 experimental
    on_trial = np.ones(n, dtype=bool) # still on first line, no PD recorded
    progressed = np.zeros(n, dtype=bool) # true state, which nobody observes

    os_time = np.full(n, np.nan)
    os_event = np.ones(n, dtype=int)

    n_pd_scans = n_nonpd_scans = n_fn = n_fp = 0
    last_t = 0.0

    for t in visits:
        dt = t - last_t
        if not on_trial.any():
            break

        # --- who dies in this interval ---
        here = np.where(on_trial)[0]
        rate = np.where(progressed[here], lam_fast,
                        np.where(arm[here] == 1, lam_os, lam_os * hr1))
        wait = rng.exponential(1 / rate)
        dead = wait <= dt
        if dead.any():
            os_time[here[dead]] = last_t + wait[dead]
            on_trial[here[dead]] = False
        if not on_trial.any():
            break

        # --- who progresses in this interval ---
        here = np.where(on_trial & ~progressed)[0]
        if len(here):
            prate = np.where(arm[here] == 1, lam_pfs, lam_pfs * hr1)
            progressed[here[rng.exponential(1 / prate) <= dt]] = True

        # --- the scan ---
        here = np.where(on_trial)[0]
        truth = progressed[here]
        u = rng.random(len(here))
        says_pd = np.where(truth, u > fn, u < fp) # miss a real PD, or invent one

        n_pd_scans += int(truth.sum())
        n_nonpd_scans += int((~truth).sum())
        n_fn += int((truth & ~says_pd).sum())
        n_fp += int((~truth & says_pd).sum())

        # --- anyone the scan calls PD comes off treatment ---
        switch = here[says_pd]
        if len(switch):
            os_time[switch] = t + rng.exponential(1 / lam_2L, size=len(switch))
            on_trial[switch] = False

        last_t = t

    # Everyone still going at 24 months is censored. 
    os_time[np.isnan(os_time)] = cutoff
    os_event[np.where(on_trial)[0]] = 0
    late = os_time > cutoff
    os_time[late], os_event[late] = cutoff, 0

    df = pd.DataFrame({"arm": arm, "time": os_time, "event": os_event})
    try:
        fit = CoxPHFitter().fit(df, duration_col="time", event_col="event")
        row = fit.summary.loc["arm"]
        hr = np.exp(fit.params_["arm"])
        lo, hi = np.exp(row["coef lower 95%"]), np.exp(row["coef upper 95%"])
        p = row["p"]
    except Exception:
        hr = lo = hi = p = np.nan

    return {"hr": hr, "lo": lo, "hi": hi, "p": p, "events": int(os_event.sum()),
            "pd_scans": n_pd_scans, "nonpd_scans": n_nonpd_scans,
            "fn": n_fn, "fp": n_fp}


# =============================================================================
# Step 3 - find the first-line HR that gives us an effective HR of 0.7
# =============================================================================
# Run a very big trial with no misclassification at all and see what the Cox
# model comes back with. That's the effective HR for that first-line HR. Then
# bisect until it lands on 0.7.

def effective_hr(hr1):
    out = Parallel(n_jobs=NCORES)(
        delayed(one_trial)(100000 + i, 0.0, n_big, "balanced", hr1)
        for i in range(reps_big))
    return (np.nanmean([o["hr"] for o in out]),
            np.nanmean([o["events"] for o in out]) / n_big)


print(f"Looking for the first-line HR that gives an effective HR of {target_hr}")
lo_hr, hi_hr = 0.30, 0.95
for step in range(12):
    hr1 = (lo_hr + hi_hr) / 2
    eff, ev_rate = effective_hr(hr1)
    print(f"  first-line {hr1:.4f}  ->  effective {eff:.4f}   (events {ev_rate:.3f})")
    if abs(eff - target_hr) < 0.002:
        break
    if eff > target_hr:
        hi_hr = hr1      # treatment too weak, push it down
    else:
        lo_hr = hr1

print(f"\nfirst-line HR  {hr1:.4f}")
print(f"effective HR   {eff:.4f}")
print(f"event rate     {ev_rate:.3f}")


# =============================================================================
# Step 4 - how many patients we need
# =============================================================================
# Schoenfeld for the number of events, then divide by the event rate the
# simulation actually produces rather than one worked out from the
# pre-progression hazard alone, which ignores the fact that progression speeds
# everyone up.

n_events = int(np.ceil(4 * (norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2
                       / np.log(target_hr) ** 2))
n_patients = int(np.ceil(n_events / ev_rate))
n_patients += n_patients % 2      # keep it even for 1:1

print(f"events needed  {n_events}")
print(f"sample size    {n_patients}\n")

for s in ratio:
    c = clip_point(s)
    if c is not None:
        print(f"note: '{s}' error rates hit 1.0 at m = {c:.3f}")


# =============================================================================
# Step 5 - run over all misclassification rates
# =============================================================================

rows = []
for scenario in ratio:
    print(f"\n--- {scenario} ---")
    for m in m_grid:
        out = Parallel(n_jobs=NCORES)(
            delayed(one_trial)(i, m, n_patients, scenario, hr1)
            for i in range(n_reps))
        d = pd.DataFrame(out)

        sig = (d["p"] < alpha).astype(float)
        cov = ((d["lo"] <= target_hr) & (target_hr <= d["hi"])).astype(float)
        fn_hat = d["fn"].sum() / max(d["pd_scans"].sum(), 1)
        fp_hat = d["fp"].sum() / max(d["nonpd_scans"].sum(), 1)

        rows.append({"scenario": scenario, "m": m,
                     "first_line_hr": hr1, "effective_hr": target_hr,
                     "n": n_patients, "reps": n_reps,
                     "power": sig.mean(), "power_mcse": sig.std(ddof=1) / np.sqrt(n_reps),
                     "coverage": cov.mean(), "coverage_mcse": cov.std(ddof=1) / np.sqrt(n_reps),
                     "mean_hr": d["hr"].mean(),
                     "fn_rate": fn_hat, "fp_rate": fp_hat,
                     "fn_fp_ratio": fn_hat / fp_hat if fp_hat > 0 else np.nan})

        print(f"  m={m:4.2f}  power={sig.mean():.3f}  coverage={cov.mean():.3f}  "
              f"fn:fp={rows[-1]['fn_fp_ratio']:.2f}")


# =============================================================================
# Step 6 - save
# =============================================================================

results = pd.DataFrame(rows)
results.to_csv("effective_hr_results.csv", index=False)
print("\nsaved effective_hr_results.csv")