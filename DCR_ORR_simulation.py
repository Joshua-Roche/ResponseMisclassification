import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm
from joblib import Parallel, delayed

QUICK = False          # set True for a fast sanity check
NCORES = 110


# =============================================================================
# Step 0 - the numbers we're working with
# =============================================================================

lam_os   = np.log(2) / 14.5 # death before progression (RIOMeso)
lam_fast = np.log(2) / 3.4 # death after a progression nobody noticed
lam_pfs  = np.log(2) / 5.6 # progression, control arm (MS01)

visits = [3, 6, 9, 12, 15, 18, 21, 24]
cutoff = 24

orr_control, orr_effect = 0.30, 0.20 # 30% control, 50% experimental

# We solve below for the HR that moves the landmark rate by 20 points.
dcr_landmark_scan = 2 # 2nd scan, i.e. 6 months
dcr_effect = 0.20

alpha, power = 0.05, 0.80

if QUICK:
    m_grid, n_reps, n_big, reps_big = np.arange(0, 1.01, 0.25), 400, 20000, 2
else:
    m_grid, n_reps, n_big, reps_big = np.arange(0, 1.01, 0.01), 10000, 100000, 5

t_landmark = visits[dcr_landmark_scan - 1]


# =============================================================================
# Step 1 - what "misclassification rate m" actually means
# =============================================================================
# Three categories, so three things can happen to a scan. Writing the true
# category as PR, SD or PD:
#
#     fn = P(read non-PD | truly PD)     split evenly between PR and SD
#     fp = P(read PD | truly non-PD)
#     sw = P(read the other non-PD label | truly non-PD)   i.e. PR <-> SD
#
# In the baseline case a patient is misread with probability m and the wrong
# label is picked uniformly from the other two, so fn = m, fp = m/2, sw = m/2.
# That is the same thing the survival scripts used, just with the PR/SD half of
# the error now visible instead of ignored.
#
# For the mostly-FN and mostly-FP scenarios we skew the PD boundary, because
# that is the boundary that changes management, and we hold fn + fp = 3m/2
# fixed so that the reader carries the same amount of information in all three
# scenarios and only the direction of the error changes. The PR <-> SD swap is a
# different boundary and we leave it at m/2 throughout, so reader quality on the
# response boundary is held constant while the PD split moves.

ratio = {"balanced": 2.0, "mostly_fn": 7.5, "mostly_fp": 1 / 7.5} # fn : fp


def error_rates(m, scenario):
    total = 1.5 * m
    r = ratio[scenario]
    fn = min(total * r / (r + 1), 1.0)
    fp = min(total * 1 / (r + 1), 1.0)
    sw = m / 2
    return fn, fp, min(sw, 1.0 - fp) # the two non-PD errors can't exceed 1


def clip_point(scenario):
    r = ratio[scenario]
    share = max(r, 1) / (r + 1)
    m = 1 / (1.5 * share)
    return m if m <= 1 else None


# =============================================================================
# Step 2 - simulate one trial
# =============================================================================
# is_orr = True runs the ORR trial (treatment acts on response probability).
# is_orr = False runs the DCR trial (treatment acts on the progression hazard).

def one_trial(seed, m, n, scenario, is_orr, p_resp, prog_hr):
    rng = np.random.default_rng(42 + seed)
    fn, fp, sw = error_rates(m, scenario)

    arm = np.array([1] * (n // 2) + [2] * (n - n // 2))
    responder = rng.random(n) < np.where(arm == 1, p_resp[0], p_resp[1])
    prog_rate = np.where(arm == 1, lam_pfs, lam_pfs * prog_hr)

    # Two separate things: Whether a patient is alive, and whether we are still
    # scanning them. A false PD takes them off study but their disease carries on
    # doing whatever it was going to do, and the truth we score against has to
    # follow that rather than stopping when we stopped looking.
    alive = np.ones(n, dtype=bool)
    on_study = np.ones(n, dtype=bool)
    progressed = np.zeros(n, dtype=bool)

    recorded_pr = np.zeros(n, dtype=bool) # has any scan said PR
    true_pr_ever = np.zeros(n, dtype=bool) # would a perfect reader have seen PR

    # DCR only looks inside the landmark window, so these three are snapshots
    # taken at the landmark rather than running totals over the whole follow-up
    pd_called_by_lm = np.zeros(n, dtype=bool)
    progressed_at_lm = np.zeros(n, dtype=bool)
    dead_by_lm = np.zeros(n, dtype=bool)

    last_t = 0.0
    for t in visits:
        dt = t - last_t
        if not alive.any():
            break

        # --- deaths, for everyone still alive whether or not we're scanning them ---
        here = np.where(alive)[0]
        rate = np.where(progressed[here], lam_fast, lam_os)
        died = rng.exponential(1 / rate) <= dt
        if died.any():
            d = here[died]
            alive[d] = False
            on_study[d] = False
            if t <= t_landmark:
                dead_by_lm[d] = True
        if not alive.any():
            break

        # --- true progressions, again for everyone alive ---
        here = np.where(alive & ~progressed)[0]
        if len(here):
            progressed[here[rng.exponential(1 / prog_rate[here]) <= dt]] = True

        # --- the scan, only for those still on study ---
        here = np.where(on_study)[0]
        if len(here) == 0:
            if t <= t_landmark:
                progressed_at_lm = progressed.copy()
            last_t = t
            continue
        is_pd = progressed[here]
        is_pr = responder[here] & ~is_pd # responders show PR until they progress
        true_pr_ever[here[is_pr]] = True

        u = rng.random(len(here))
        says = np.empty(len(here), dtype="<U2")

        # truly PD: read as PD unless missed, in which case PR or SD evenly
        says[is_pd] = np.where(u[is_pd] > fn, "PD",
                               np.where(u[is_pd] > fn / 2, "PR", "SD"))
        # truly non-PD: can be called PD, or swapped to the other non-PD label
        non = ~is_pd
        own = np.where(is_pr[non], "PR", "SD")
        other = np.where(is_pr[non], "SD", "PR")
        says[non] = np.where(u[non] < fp, "PD",
                             np.where(u[non] < fp + sw, other, own))

        recorded_pr[here[says == "PR"]] = True
        pd_called = here[says == "PD"]
        on_study[pd_called] = False # off study, no more scans

        if t <= t_landmark:
            pd_called_by_lm[pd_called] = True
            progressed_at_lm = progressed.copy() # state as at the landmark

        last_t = t

    # --- the two endpoints ---
    # ORR: a response was recorded at some point. Because a recorded PD stops
    # assessment, a PR can only be recorded before one.
    obs_orr = recorded_pr
    true_orr = true_pr_ever

    # DCR: no PD recorded by the landmark and still alive at it. Both are taken
    # as at the landmark, so a patient who is controlled at month 6 and
    # progresses at month 12 still counts as controlled.
    obs_dcr = ~pd_called_by_lm & ~dead_by_lm
    true_dcr = ~progressed_at_lm & ~dead_by_lm

    y = obs_orr if is_orr else obs_dcr
    truth = true_orr if is_orr else true_dcr

    a, b = int(y[arm == 2].sum()), int((~y[arm == 2]).sum())
    c, d = int(y[arm == 1].sum()), int((~y[arm == 1]).sum())
    p_val = fisher_exact([[a, b], [c, d]])[1]

    p2, p1 = a / (a + b), c / (c + d)
    diff = p2 - p1
    se = np.sqrt(p1 * (1 - p1) / (c + d) + p2 * (1 - p2) / (a + b))
    z = norm.ppf(1 - alpha / 2)

    return {"p": p_val, "diff": diff, "lo": diff - z * se, "hi": diff + z * se,
            "obs_rate_ctrl": p1, "obs_rate_exp": p2,
            "true_rate_ctrl": truth[arm == 1].mean(),
            "true_rate_exp": truth[arm == 2].mean(),
            # endpoint-level operating characteristics, to check against 2.3.2
            "n_true_pos": int((truth & y).sum()), "n_true": int(truth.sum()),
            "n_false_pos": int((~truth & y).sum()), "n_nontrue": int((~truth).sum())}


# =============================================================================
# Step 3 - pin down the latent parameters, then the sample sizes
# =============================================================================
# For ORR a responder shows PR from their first scan onwards, so a perfect
# reader records a response for everyone who responds and is still there to be
# scanned at month 3 - which means they have to have avoided both progression
# and death. So true ORR = p_resp * exp(-(lam_pfs + lam_os) * 3), and we invert
# that. Note this makes p_resp quite a bit larger than the target ORR, because a
# fair number of patients are lost before they ever get the chance to respond.

intact_at_first_scan = np.exp(-(lam_pfs + lam_os) * visits[0])
p_resp_orr = (orr_control / intact_at_first_scan,
              (orr_control + orr_effect) / intact_at_first_scan)

# Same for DCR: being controlled at the landmark means neither progressing nor
# dying before it, so both hazards go into the control rate, and we solve for
# the progression HR that lifts it by the target amount.
dcr_control = np.exp(-(lam_pfs + lam_os) * t_landmark)
dcr_target = dcr_control + dcr_effect
hr_dcr = (-np.log(dcr_target) / t_landmark - lam_os) / lam_pfs

print(f"ORR   responder probability   {p_resp_orr[0]:.3f} control, "
      f"{p_resp_orr[1]:.3f} experimental")
print(f"DCR   landmark                scan {dcr_landmark_scan} (month {t_landmark})")
print(f"DCR   control rate at the landmark     {dcr_control:.3f}")
print(f"DCR   progression HR needed for +{dcr_effect:.2f}   {hr_dcr:.3f}")


def sample_size(p1, p2):
    pbar = (p1 + p2) / 2
    num = (norm.ppf(1 - alpha / 2) * np.sqrt(2 * pbar * (1 - pbar))
           + norm.ppf(power) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    per_arm = int(np.ceil(num / (p2 - p1) ** 2))
    return 2 * per_arm


# measure what the mechanism actually delivers before sizing anything
settings = {"ORR": (True, p_resp_orr, 1.0),
            "DCR": (False, (0.0, 0.0), hr_dcr)}

n_by_endpoint = {}
for name, (is_orr, pr, hr) in settings.items():
    big = Parallel(n_jobs=NCORES)(
        delayed(one_trial)(100000 + i, 0.0, n_big, "balanced", is_orr, pr, hr)
        for i in range(reps_big))
    tc = np.mean([b["true_rate_ctrl"] for b in big])
    te = np.mean([b["true_rate_exp"] for b in big])
    n_by_endpoint[name] = sample_size(tc, te)
    print(f"{name}   true rates {tc:.3f} vs {te:.3f}  (difference {te - tc:.3f})"
          f"   ->  n = {n_by_endpoint[name]}")

print()
for s in ratio:
    c = clip_point(s)
    if c is not None:
        print(f"note: '{s}' error rates hit 1.0 at m = {c:.3f}")


# =============================================================================
# Step 4 -  run over all misclassification rates
# =============================================================================

rows = []
for name, (is_orr, pr, hr) in settings.items():
    n = n_by_endpoint[name]
    for scenario in ratio:
        print(f"\n--- {name}, {scenario} (n = {n}) ---")
        for m in m_grid:
            out = Parallel(n_jobs=NCORES)(
                delayed(one_trial)(i, m, n, scenario, is_orr, pr, hr)
                for i in range(n_reps))
            d = pd.DataFrame(out)

            true_diff = d["true_rate_exp"].mean() - d["true_rate_ctrl"].mean()
            sig = ((d["p"] < alpha) & (d["diff"] > 0)).astype(float)
            cov = ((d["lo"] <= true_diff) & (true_diff <= d["hi"])).astype(float)

            # endpoint-level sensitivity and specificity
            Phi = d["n_true_pos"].sum() / max(d["n_true"].sum(), 1)
            Psi = 1 - d["n_false_pos"].sum() / max(d["n_nontrue"].sum(), 1)

            rows.append({"endpoint": name, "scenario": scenario, "m": m, "n": n,
                         "reps": n_reps, "true_diff": true_diff,
                         "power": sig.mean(),
                         "power_mcse": sig.std(ddof=1) / np.sqrt(n_reps),
                         "coverage": cov.mean(),
                         "coverage_mcse": cov.std(ddof=1) / np.sqrt(n_reps),
                         "mean_diff": d["diff"].mean(),
                         "endpoint_sens": Phi, "endpoint_spec": Psi})

            print(f"  m={m:4.2f}  power={sig.mean():.3f}  coverage={cov.mean():.3f}  "
                  f"diff={d['diff'].mean():+.3f}  Phi={Phi:.3f}  Psi={Psi:.3f}")


# =============================================================================
# Step 5 - save
# =============================================================================

results = pd.DataFrame(rows)
results.to_csv("orr_dcr_results.csv", index=False)
print("\nsaved orr_dcr_results.csv")
