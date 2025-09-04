# -*- coding: utf-8 -*-
# 서울보증 SAA — Equity VaR(연 300억) + Shortfall(5Y) 제약
# 속도개선: 연도별 VaR 1차 필터 → 샤프 상위 K개만 Shortfall MC → 공통난수·벡터화
# - 상관행렬: Risk-Free 제외 9×9 사용자 입력만 사용 → 10×10 임베딩
# - μ, σ: 연도별(Year 1~5) 직접 입력 가능(미입력 시 base 그대로 5년 적용)
# - Equity VaR: 경로의 매년 P(연간 손실≥300억원) < 1%
# - Shortfall: 5년 누적수익률<0 확률 < 1% (경로 & 타깃 5년 고정 모두)
# - 타깃 목표수익률(≥3.3%)은 Year1 μ 기준

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import norm

# =========================================================
# ===============  CONFIG — 사용자 입력(독립변수)  ===============
# ---------------------------------------------------------
ASSETS = [
    "Risk-Free","Fixed-Income","Global FI","Domestic Eq","Global Eq",
    "Private Eq","Private Credit","Hedgefund","Infrastructure","Real Estate"
]

# (A) 기준(한 해) μ, σ — 연도별을 따로 주지 않으면 이 값을 5년 동일 적용
RF_ANN   = 0.025
MU_ANN   = np.array([RF_ANN, 0.0251, 0.0323, 0.056, 0.064, 0.074, 0.0568, 0.046, 0.057, 0.052])
VOL_ANN  = np.array([0.0,   0.028,  0.029,  0.146, 0.135, 0.163, 0.085,  0.015, 0.059, 0.053])

# (B) 연도별 μ/σ (선택) — None이면 위 값 5년 동일 적용
MU_ANN_BY_YEAR  = None
VOL_ANN_BY_YEAR = None
"""
# 예시:
MU_ANN_BY_YEAR = [
    [RF_ANN, 0.024, 0.031, 0.055, 0.063, 0.073, 0.056, 0.046, 0.057, 0.052],
    [RF_ANN, 0.0245,0.0315,0.0555,0.0635,0.0735,0.0562,0.046, 0.057, 0.052],
    [RF_ANN, 0.025, 0.032, 0.056, 0.064, 0.074, 0.0564,0.046, 0.057, 0.052],
    [RF_ANN, 0.0255,0.0325,0.0565,0.0645,0.0745,0.0566,0.046, 0.057, 0.052],
    [RF_ANN, 0.026, 0.033, 0.057, 0.065, 0.075, 0.0568,0.046, 0.057, 0.052],
]
VOL_ANN_BY_YEAR = [
    [0.0, 0.028, 0.029, 0.146, 0.135, 0.163, 0.085, 0.015, 0.059, 0.053],
    [0.0, 0.028, 0.029, 0.145, 0.134, 0.162, 0.086, 0.015, 0.059, 0.053],
    [0.0, 0.028, 0.029, 0.144, 0.133, 0.161, 0.086, 0.015, 0.059, 0.053],
    [0.0, 0.028, 0.029, 0.143, 0.132, 0.160, 0.087, 0.015, 0.059, 0.053],
    [0.0, 0.028, 0.029, 0.142, 0.131, 0.159, 0.087, 0.015, 0.059, 0.053],
]
"""

# 단위 일치(억원/원)
AUM_KRW            = 71693.1564   # 억원 운용규모
LOSS_THRESHOLD_KRW = 300.0        # 억원 주식 VaR 한도
PROB_LIMIT_VAR     = 0.01         # 1%  Shortfall 한도

# Shortfall 설정 (MC=Monte Carlo 샘플 수) — 샘플수는 유지
PROB_LIMIT_SF = 0.01
SF_YEARS      = 5
SF_N_MC       = 50_000
SF_SEED       = 1234

# 속도 손잡이 (샘플수는 그대로, 나머지 최적화)
MC_TOPK = 500           # VaR 통과 + Sharpe 상위 K개만 Shortfall MC
USE_ANTITHETIC = True   # 안티시메트릭 난수로 분산 절감(샘플수는 동일하게 유지)

# 타깃 목표수익률 (Year1 μ 기준)
TARGET_RETURN = 0.033

# 현재비중/밴드/시나리오
W_CURR   = np.array([0.02, 0.7246, 0.1046, 0.0052, 0.0075, 0.0174, 0.0681, 0.0017, 0.0249, 0.0260])
MIN_BAND = np.array([-10, -20, -10,  10,  10, -50, -10, -10, -10, -10], float)
MAX_BAND = np.array([ 10,  10,  50, 100,  50,  10, 400,3000, 400, 400], float)
N_SCENARIOS = 10
SEED_SCEN   = 2029

# ===== Risk-Free 제외 9×9 상관행렬 (사용자 입력) =====
# 순서: [Fixed-Income, Global FI, Domestic Eq, Global Eq, Private Eq, Private Credit, Hedgefund, Infrastructure, Real Estate]
C_INPUT_RISKY = [
    [1.0, 0.701636697, -0.53797132,  -0.581272379, -0.563229065, -0.552949825, -0.473362391, 0.120633713, -0.007199747],
    [0.701636697, 1.0, 0.002671012,   0.008953527,  -0.217461037, -0.249235507, -0.223532661, 0.438921681,  0.042296936],
    [-0.53797132, 0.002671012, 1.0,   0.838644061,   0.532711651,  0.079347047,  0.399396062, 0.058919553, -0.329014317],
    [-0.581272379,0.008953527,0.838644061, 1.0,       0.861985025,  0.511832146,  0.758504663, 0.473232601,  0.124729145],
    [-0.563229065,-0.217461037,0.532711651,0.861985025,1.0,         0.643440814,  0.934438051, 0.553815197,  0.297972219],
    [-0.552949825,-0.249235507,0.079347047,0.511832146,0.643440814, 1.0,          0.519205453, 0.44725613,   0.560183029],
    [-0.473362391,-0.223532661,0.399396062,0.758504663,0.934438051, 0.519205453,  1.0,         0.606403149,  0.433543825],
    [0.120633713, 0.438921681,0.058919553,0.473232601,0.553815197,  0.44725613,   0.606403149, 1.0,          0.526516441],
    [-0.007199747,0.042296936,-0.329014317,0.124729145,0.297972219, 0.560183029,  0.433543825, 0.526516441,  1.0]
]

# --- apply overrides AFTER defaults are set ---
try:
    import overrides as _ov
    print("[overrides] using", _ov.__file__)
    # 대문자 변수만 골라서 현재 네임스페이스에 덮어쓰기
    for _k in dir(_ov):
        if _k.isupper():
            globals()[_k] = getattr(_ov, _k)
    print("[overrides] applied keys:", [k for k in dir(_ov) if k.isupper()])
except Exception as e:
    print("[overrides] not applied:", e)
# ---------------------------------------------

# from time import perf_counter
# print("[PARAMS] N_SCENARIOS=", N_SCENARIOS)
# print("[PARAMS] MC_TOPK    =", MC_TOPK)
# print("[PARAMS] SF_N_MC    =", SF_N_MC)
# print("[PARAMS] USE_ANTITHETIC=", USE_ANTITHETIC)
# print("[PARAMS] TARGET_RETURN=", TARGET_RETURN)
# print("[PARAMS] BANDS sum-min/max =", MIN_BAND.sum(), MAX_BAND.sum())
# T0 = perf_counter()

# === 타입 보정 (엑셀에서 들어온 float/str을 int로 정리) ===
def _as_int(name):
    if name in globals():
        v = globals()[name]
        try:
            globals()[name] = int(v) if isinstance(v, int) else int(float(v))
        except Exception:
            pass

for _name in ("SF_YEARS", "N_SCENARIOS", "MC_TOPK", "SF_N_MC", "SEED_SCEN", "SF_SEED"):
    _as_int(_name)
# ==========================================================

# -------------------- Helper --------------------
def show_df(title, df, max_rows=40):
    print(f"\n=== {title} ===")
    if len(df) > max_rows:
        print(df.head(max_rows).to_string()); print(f"... ({len(df)-max_rows} more rows)")
    else:
        print(df.to_string())

def build_cov_from_vol_corr(vol, corr):
    vol = np.asarray(vol).reshape(-1); C = np.asarray(corr)
    return np.outer(vol, vol) * C

def nearest_psd_corr(C):
    C = np.array(C, dtype=float); C = 0.5*(C + C.T); np.fill_diagonal(C, 1.0)
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-12, None)
    C_psd = V @ np.diag(w) @ V.T; C_psd = 0.5*(C_psd + C_psd.T); np.fill_diagonal(C_psd, 1.0); return C_psd

def bands_from_relative(current_w, min_pct, max_pct):
    current_w = np.asarray(current_w, float)
    w_min = np.clip(current_w * (1.0 + np.asarray(min_pct)/100.0), 0.0, 1.0)
    w_max = np.clip(current_w * (1.0 + np.asarray(max_pct)/100.0), 0.0, 1.0)
    return w_min, w_max

def feasible_projection(w, w_min, w_max, max_iter=1000):
    w = np.clip(w, w_min, w_max).astype(float)
    for _ in range(max_iter):
        s = w.sum()
        if abs(s-1.0) < 1e-12: return w, True
        if s < 1.0:
            free = (w < (w_max-1e-15))
            if not np.any(free): return w, False
            add = 1.0 - s; head = (w_max - w)*free; k = head.sum()
            if k <= 0: return w, False
            w = np.minimum(w + head/k*add, w_max)
        else:
            free = (w > (w_min+1e-15))
            if not np.any(free): return w, False
            sub = s - 1.0; room = (w - w_min)*free; k = room.sum()
            if k <= 0: return w, False
            w = np.maximum(w - room/k*sub, w_min)
    return w, False

def sample_weights_scenarios(current_w, min_pct, max_pct, n_scenarios=100000, seed=2029, enforce_strict=True):
    rng = default_rng(seed); current_w = np.asarray(current_w, float)
    w_min, w_max = bands_from_relative(current_w, min_pct, max_pct)
    if w_min.sum() - 1.0 > 1e-9 or w_max.sum() + 1e-9 < 1.0:
        raise ValueError("Infeasible bands: sum(w_min)>1 or sum(w_max)<1.")
    W = np.zeros((n_scenarios, len(current_w))); acc=0; trials=0; max_trials=n_scenarios*50
    while acc < n_scenarios and trials < max_trials:
        trials += 1
        u = rng.uniform(np.asarray(min_pct)/100.0, np.asarray(max_pct)/100.0, size=len(current_w))
        w0 = current_w * (1.0 + u); w0 = w0 / w0.sum()
        w, ok = feasible_projection(w0, w_min, w_max)
        if (not enforce_strict) or ok:
            W[acc,:] = w; acc += 1
    if acc < n_scenarios: W = W[:acc,:]
    return W, w_min, w_max

def port_metrics(W, mu_ann, Sigma, rf_ann=0.0):
    W = np.atleast_2d(W); mu = np.asarray(mu_ann).reshape(-1)
    rets = W @ mu
    vols = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', W, Sigma, W), 1e-18))
    sr   = (rets - rf_ann) / np.maximum(vols, 1e-18)
    return rets, vols, sr

def linear_transition(w_start, w_target, n_years):
    ts = np.linspace(1/n_years, 1.0, n_years)
    P = np.stack([w_start + t*(w_target - w_start) for t in ts], axis=0)
    P = np.clip(P, 0, 1); P = P / P.sum(axis=1, keepdims=True); return P

# -------- 연도별 파라미터 준비 --------
def make_yearly_params(mu_base, vol_base, mu_by_year, vol_by_year, n_years, n_assets):
    if mu_by_year is None:
        MUy = np.tile(np.asarray(mu_base, float), (n_years, 1))
    else:
        MUy = np.asarray(mu_by_year, float)
    if vol_by_year is None:
        VOLy = np.tile(np.asarray(vol_base, float), (n_years, 1))
    else:
        VOLy = np.asarray(vol_by_year, float)
    assert MUy.shape == (n_years, n_assets) and VOLy.shape == (n_years, n_assets), \
        f"연도별 μ/σ의 shape는 {(n_years, n_assets)} 이어야 합니다."
    return MUy, VOLy

def make_sigma_list_by_year(VOLy, C_full):
    n_years = VOLy.shape[0]; n_assets = VOLy.shape[1]
    sigmas = []
    I = np.eye(n_assets)
    for t in range(n_years):
        Sigma_t = build_cov_from_vol_corr(VOLy[t], C_full) + 1e-12 * I
        sigmas.append(Sigma_t)
    return sigmas

# --- Equity VaR(연 300억) — 연도별 μ/σ 사용 ---
def equity_mu_sigma_year(w, mu_y_t, vol_y_t, rho_dg, idx_dom, idx_glb):
    w_d, w_g = float(w[idx_dom]), float(w[idx_glb])
    mu_d, sd_d = float(mu_y_t[idx_dom]), float(vol_y_t[idx_dom])
    mu_g, sd_g = float(mu_y_t[idx_glb]), float(vol_y_t[idx_glb])
    mu_s  = w_d*mu_d + w_g*mu_g
    var_s = (w_d*sd_d)**2 + (w_g*sd_g)**2 + 2*w_d*w_g*sd_d*sd_g*float(rho_dg)
    sig_s = np.sqrt(max(var_s, 1e-18))
    return mu_s, sig_s

def prob_equity_loss_ge_threshold_year(w, mu_y_t, vol_y_t, rho_dg, idx_dom, idx_glb, aum_same_unit, loss_threshold_same_unit):
    mu_s, sig_s = equity_mu_sigma_year(w, mu_y_t, vol_y_t, rho_dg, idx_dom, idx_glb)
    thr_ret = -float(loss_threshold_same_unit) / float(aum_same_unit)
    z = (thr_ret - mu_s) / max(sig_s, 1e-18)
    return float(norm.cdf(z)), mu_s, sig_s, thr_ret

def path_prob_table_yearly(pathW, MUy, VOLy, rho_dg, idx_dom, idx_glb, aum_same_unit, loss_threshold_same_unit, prob_limit):
    rows=[]
    for t, w in enumerate(pathW, start=1):
        p, mu_s, sig_s, thr = prob_equity_loss_ge_threshold_year(w, MUy[t-1], VOLy[t-1], rho_dg, idx_dom, idx_glb, aum_same_unit, loss_threshold_same_unit)
        rows.append({"Year":t,"μ_stock(연)":mu_s,"σ_stock(연)":sig_s,"임계수익률(= -L/AUM)":thr,"P(loss≥L)":p,"제약충족":(p<prob_limit)})
    return pd.DataFrame(rows)

def path_constraint_ok_equityVaR_yearly(pathW, MUy, VOLy, rho_dg, idx_dom, idx_glb, aum_same_unit, loss_threshold_same_unit, prob_limit):
    for t, w in enumerate(pathW):
        p, *_ = prob_equity_loss_ge_threshold_year(w, MUy[t], VOLy[t], rho_dg, idx_dom, idx_glb, aum_same_unit, loss_threshold_same_unit)
        if not (p < prob_limit): return False
    return True

# ---------- Shortfall MC: 공통난수 생성 + 벡터화 ----------
def precompute_returns_yearly(MUy, Sigma_list, n_mc=50000, seed=1234, use_antithetic=False):
    rng = default_rng(seed)
    T = len(Sigma_list); n_assets = MUy.shape[1]
    R_by_year = []
    for t in range(T):
        S = Sigma_list[t]
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(S)
            L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-18, None)))
        if use_antithetic:
            m = n_mc // 2
            Z = rng.standard_normal(size=(m, n_assets))
            Z = np.vstack([Z, -Z])  # 합쳐서 n_mc
        else:
            Z = rng.standard_normal(size=(n_mc, n_assets))
        Rt = MUy[t] + Z @ L.T        # (n_mc, n_assets)
        if Rt.shape[0] != n_mc:
            # n_mc가 홀수인데 antithetic를 켰을 때 등, 크기 보정
            Rt = Rt[:n_mc, :]
        R_by_year.append(Rt)
    return R_by_year  # 길이 T 리스트

def shortfall_prob_bulk(paths, R_by_year):
    """
    paths: (M, T, n_assets) 여러 후보의 이행경로
    R_by_year[t]: (n_mc, n_assets) 공통난수로 생성된 연도별 자산수익률
    반환: (M,) 각 후보의 P(5Y 누적<0)
    """
    paths = np.asarray(paths)
    M, T, n_assets = paths.shape
    n_mc = R_by_year[0].shape[0]
    wealth = np.ones((n_mc, M))
    for t in range(T):
        Rt = R_by_year[t]          # (n_mc, n_assets)
        Wt = paths[:, t, :]        # (M, n_assets)
        r_t = Rt @ Wt.T            # (n_mc, M)
        wealth *= (1.0 + r_t)
    return np.mean(wealth - 1.0 < 0.0, axis=0)

# ===================== 메인 =====================
if __name__ == "__main__":
    assets = ASSETS; n_full = len(assets)

    # (1) Risk-Free 제외 9×9 → 10×10 임베딩
    C_risky = nearest_psd_corr(C_INPUT_RISKY)
    C = np.eye(n_full)
    risky_labels   = [a for a in ASSETS if a != "Risk-Free"]
    risky_idx_full = [i for i,a in enumerate(ASSETS) if a != "Risk-Free"]
    for i_r, i_full in enumerate(risky_idx_full):
        for j_r, j_full in enumerate(risky_idx_full):
            C[i_full, j_full] = C_risky[i_r, j_r]
    # VaR용 Domestic–Global 상관 자동 추출
    rho_dg = float(C_risky[risky_labels.index("Domestic Eq"), risky_labels.index("Global Eq")])

    # (2) 연도별 μ/σ + 연도별 Σ 리스트
    MUy, VOLy = make_yearly_params(MU_ANN, VOL_ANN, MU_ANN_BY_YEAR, VOL_ANN_BY_YEAR, SF_YEARS, n_full)
    Sigma_list = make_sigma_list_by_year(VOLy, C)

    # (3) 시나리오 생성 & Year1 기준 스코어(Sharpe)
    w_curr = W_CURR / W_CURR.sum()
    W, _, _ = sample_weights_scenarios(w_curr, MIN_BAND, MAX_BAND, n_scenarios=N_SCENARIOS, seed=SEED_SCEN)
    ret_all, vol_all, sr_all = port_metrics(W, MUy[0], Sigma_list[0], RF_ANN)

    # (4) 타깃 후보: 목표수익률(Year1 μ) 필터
    mask_tr = (W @ MUy[0] >= TARGET_RETURN)
    W2 = W[mask_tr]; ret2, vol2, sr2 = ret_all[mask_tr], vol_all[mask_tr], sr_all[mask_tr]
    print(f"[목표수익률-타깃(Year1 μ)] 충족: {W2.shape[0]} / {W.shape[0]} (≥ {TARGET_RETURN:.2%})")

    # (5) 경로 제약 1차: 연도별 Equity VaR 통과하는 후보만 남김
    i_dom = assets.index("Domestic Eq"); i_glb = assets.index("Global Eq")
    paths = []; keep_idx = []
    for i, w_tgt in enumerate(W2):
        path = linear_transition(w_curr, w_tgt, n_years=SF_YEARS)
        ok_var = path_constraint_ok_equityVaR_yearly(path, MUy, VOLy, rho_dg, i_dom, i_glb,
                                                     AUM_KRW, LOSS_THRESHOLD_KRW, PROB_LIMIT_VAR)
        if ok_var:
            paths.append(path); keep_idx.append(i)
    if len(paths) == 0:
        print("VaR(연도별) 통과 후보가 없습니다.")
        exit(0)

    paths = np.array(paths)                 # (M, T, n_assets)
    sr_kept = sr2[keep_idx]
    W2_kept = W2[keep_idx]

    # (6) 샤프 상위 K개만 Shortfall MC
    K = min(MC_TOPK, len(W2_kept))
    top = np.argsort(-sr_kept)[:K]
    paths_top = paths[top]                  # (K, T, n_assets)
    W2_top    = W2_kept[top]                # (K, n_assets)
    sr_top    = sr_kept[top]

    # (7) 공통난수로 연도별 자산수익률 한 번만 생성
    R_by_year = precompute_returns_yearly(MUy, Sigma_list, n_mc=SF_N_MC,
                                          seed=SF_SEED, use_antithetic=USE_ANTITHETIC)

    # (8) Shortfall 확률 벡터화 계산
    p_sf_path_all = shortfall_prob_bulk(paths_top, R_by_year)
    target_paths_top = np.repeat(W2_top[:, None, :], SF_YEARS, axis=1)  # 타깃 5년 고정
    p_sf_tgt_all  = shortfall_prob_bulk(target_paths_top, R_by_year)

    # (9) 제약 통과 중 샤프 최대 선택
    ok = (p_sf_path_all < PROB_LIMIT_SF) & (p_sf_tgt_all < PROB_LIMIT_SF)
    if not np.any(ok):
        print("⚠️ '경로VaR+Shortfall' 통과 포트 없음 — K를 늘리거나 파라미터 재점검 필요.")
        exit(0)

    cand = np.where(ok)[0]
    best_loc = cand[np.argmax(sr_top[ok])]
    best_path   = paths_top[best_loc]
    best_target = W2_top[best_loc]
    best_p_sf_path   = float(p_sf_path_all[best_loc])
    best_p_sf_target = float(p_sf_tgt_all[best_loc])

    # (10) 결과표 생성/저장
    years = [f"Year {i}" for i in range(1, SF_YEARS+1)]
    # 연도별 성과(연 μ/Σ 사용)
    retsY, volsY, srY = [], [], []
    for t in range(SF_YEARS):
        r, v, s = port_metrics(best_path[t], MUy[t], Sigma_list[t], RF_ANN)
        retsY.append(r[0]); volsY.append(v[0]); srY.append(s[0])

    weights_df = pd.DataFrame(best_path, columns=assets, index=years).round(6)
    metrics_df = pd.DataFrame({"연 기대수익률": retsY, "연 변동성": volsY, "샤프비율": srY}, index=years).round(6)
    prob_tbl = path_prob_table_yearly(best_path, MUy, VOLy, rho_dg, i_dom, i_glb,
                                      AUM_KRW, LOSS_THRESHOLD_KRW, PROB_LIMIT_VAR)\
                .set_index(pd.Index(years)).round(8)

    # 타깃 자체의 연간 Equity VaR(타깃 5년 고정 시, 해마다 μ/σ가 달라지므로 연도별 확률 다를 수 있음)
    p_tgt_eq_y = []
    for t in range(SF_YEARS):
        p, _, _, _ = prob_equity_loss_ge_threshold_year(best_target, MUy[t], VOLy[t], rho_dg, i_dom, i_glb,
                                                        AUM_KRW, LOSS_THRESHOLD_KRW)
        p_tgt_eq_y.append(p)
    target_row_eq = pd.DataFrame({"P(loss≥300억)_Target": p_tgt_eq_y}, index=years).round(8)

    sf_row = pd.DataFrame({
        "P(5Y 누적<0)_경로": [best_p_sf_path],
        "P(5Y 누적<0)_타깃고정": [best_p_sf_target],
        "허용한도": [PROB_LIMIT_SF],
        "MC_N": [SF_N_MC]
    }, index=["Shortfall"]).round(8)

    # 출력
    show_df("5개년 이행포트폴리오 - 연도별 가중치(현재→타깃)", weights_df)
    show_df("5개년 이행포트폴리오 - 연도별 성과지표(연 μ/σ/Sharpe)", metrics_df)
    show_df("5개년 주식VaR 확률표 (연간 300억원 손실 확률 < 1%)", prob_tbl)
    show_df("타깃 포트폴리오 연간 주식VaR 확률(5년 표시)", target_row_eq)
    show_df("Shortfall Risk 결과 (5년 누적수익률<0)", sf_row)

    # 저장
    weights_df.to_csv("transition_5y_weights.csv", encoding="utf-8-sig")
    metrics_df.to_csv("transition_5y_metrics.csv", encoding="utf-8-sig")
    prob_tbl.to_csv("transition_5y_equityVaR_prob.csv", encoding="utf-8-sig")
    target_row_eq.to_csv("target_equityVaR_prob.csv", encoding="utf-8-sig")
    sf_row.to_csv("shortfall_results.csv", encoding="utf-8-sig")
    print("\nSaved: transition_5y_weights.csv, transition_5y_metrics.csv, "
          "transition_5y_equityVaR_prob.csv, target_equityVaR_prob.csv, shortfall_results.csv")
