# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json, io, sys, subprocess, zipfile
from pathlib import Path

st.set_page_config(page_title="서울보증 SAA모델", layout="wide")
st.title("📄 서울보증 SAA모델")

ASSETS = ["Fixed-Income","Global FI","Domestic Eq","Global Eq",
          "Private Credit","Private Eq","Real Estate","Infrastructure","Hedgefund"]
RISKY9 = ["Fixed-Income","Global FI","Domestic Eq","Global Eq",
          "Private Credit","Private Eq","Real Estate","Infrastructure","Hedgefund"]

# ---------------- 템플릿 엑셀 생성 ----------------
def make_template_bytes():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        # MU (기준연 예시 1×10)
        mu_df = pd.DataFrame([ [0.025, 0.0251, 0.0323, 0.056, 0.064, 0.074, 0.0568, 0.046, 0.057, 0.052] ],
                             columns=ASSETS)
        mu_df.to_excel(xw, sheet_name="MU", index=False)

        # VOL (기준연 예시 1×10)
        vol_df = pd.DataFrame([ [0.0, 0.028, 0.029, 0.146, 0.135, 0.163, 0.085, 0.015, 0.059, 0.053] ],
                              columns=ASSETS)
        vol_df.to_excel(xw, sheet_name="VOL", index=False)

        # CORR (9×9, Risk-Free 제외) — 대각=1 예시
        corr_df = pd.DataFrame(np.eye(9), columns=RISKY9, index=RISKY9)
        corr_df.to_excel(xw, sheet_name="CORR")

        # BANDS (최대 3×10) — 1행 W_CURR, 2행 MIN_BAND, 3행 MAX_BAND
        bands = pd.DataFrame([
            [0.02, 0.7246, 0.1046, 0.0052, 0.0075, 0.0174, 0.0681, 0.0017, 0.0249, 0.0260],  # W_CURR
            [-10, -20, -10, 10, 10, -50, -10, -10, -10, -10],                               # MIN_BAND (%)
            [ 10,  10,  50,100, 50,  10, 400,3000, 400, 400],                                # MAX_BAND (%)
        ], columns=ASSETS)
        bands.to_excel(xw, sheet_name="BANDS", index=False)

        # PARAMS (Key, Value)
        params_kv = pd.DataFrame({
            "Key": ["TARGET_RETURN","PROB_LIMIT_VAR","PROB_LIMIT_SF","SF_YEARS","SF_N_MC","USE_ANTITHETIC",
                    "MC_TOPK","N_SCENARIOS","SEED_SCEN","AUM_KRW","LOSS_THRESHOLD_KRW","RF_ANN","SF_SEED"],
            "Value": [0.033, 0.01, 0.01, 5, 50000, True, 500, 10, 2029, 71693.1564, 300.0, 0.025, 1234]
        })
        params_kv.to_excel(xw, sheet_name="PARAMS", index=False)
    buf.seek(0)
    return buf

with st.sidebar:
    st.markdown("### 1) 템플릿")
    st.download_button(
        "⬇️ 템플릿 엑셀 다운로드",
        data=make_template_bytes().read(),
        file_name="template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ---------------- 단일 엑셀 업로드 ----------------
    st.markdown("### 2) 파일 업로드")
    template_xlsx = st.file_uploader("template.xlsx 업로드", type=["xlsx"])

    model_path = st.text_input("model.py 경로", str(Path("model.py").resolve()))
    workdir = str(Path(model_path).parent.resolve())

def kv_to_dict(df):
    out = {}
    for _, row in df.iterrows():
        k = str(row[0]).strip()
        v = row[1]
        if isinstance(v, str):
            v_strip = v.strip()
            # True/False/None/숫자 파싱 시도
            if v_strip.lower() in ("true","false"):
                v = (v_strip.lower()=="true")
            else:
                try:
                    v = json.loads(v_strip)
                except Exception:
                    try:
                        v = float(v_strip)
                    except Exception:
                        pass
        out[k] = v
    return out

def build_overrides_from_excel(xls: pd.ExcelFile) -> str:
    lines = ["import numpy as np"]
    # MU
    if "MU" in xls.sheet_names:
        MU = pd.read_excel(xls, sheet_name="MU")
        if list(MU.columns) == ASSETS:
            arr = MU.to_numpy(dtype=float)
            if arr.shape == (1, 10):
                lines.append(f"MU_ANN = np.array({arr[0].tolist()}, float)")
                lines.append("MU_ANN_BY_YEAR = None")
            elif arr.shape == (5, 10):
                lines.append("MU_ANN_BY_YEAR = " + json.dumps(arr.tolist()))
            else:
                lines.append("# [warn] MU는 (1×10) 또는 (5×10) 이어야 합니다.")
        else:
            lines.append("# [warn] MU 열 순서가 ASSETS와 다릅니다.")
    # VOL
    if "VOL" in xls.sheet_names:
        VOL = pd.read_excel(xls, sheet_name="VOL")
        if list(VOL.columns) == ASSETS:
            arr = VOL.to_numpy(dtype=float)
            if arr.shape == (1, 10):
                lines.append(f"VOL_ANN = np.array({arr[0].tolist()}, float)")
                lines.append("VOL_ANN_BY_YEAR = None")
            elif arr.shape == (5, 10):
                lines.append("VOL_ANN_BY_YEAR = " + json.dumps(arr.tolist()))
            else:
                lines.append("# [warn] VOL은 (1×10) 또는 (5×10) 이어야 합니다.")
        else:
            lines.append("# [warn] VOL 열 순서가 ASSETS와 다릅니다.")
    # CORR (9×9)
    if "CORR" in xls.sheet_names:
        CORR = pd.read_excel(xls, sheet_name="CORR", index_col=0)
        C = CORR.to_numpy(dtype=float)
        if C.shape == (9, 9) and list(CORR.columns)==RISKY9 and list(CORR.index)==RISKY9:
            lines.append("C_INPUT_RISKY = " + json.dumps(C.tolist()))
        else:
            lines.append("# [warn] CORR은 (9×9)이고 행/열 라벨이 RISKY9 순서여야 합니다.")
    # BANDS (최대 3×10)
    if "BANDS" in xls.sheet_names:
        B = pd.read_excel(xls, sheet_name="BANDS")
        if list(B.columns) == ASSETS and B.shape[1] == 10:
            arr = B.to_numpy(dtype=float)
            if arr.shape[0] >= 1:
                lines.append(f"W_CURR = np.array({arr[0].tolist()}, float)")
            if arr.shape[0] >= 2:
                lines.append(f"MIN_BAND = np.array({arr[1].tolist()}, float)")
            if arr.shape[0] >= 3:
                lines.append(f"MAX_BAND = np.array({arr[2].tolist()}, float)")
        else:
            lines.append("# [warn] BANDS는 열 10개가 필요합니다 (ASSETS 순서).")
    # PARAMS (Key, Value)
    if "PARAMS" in xls.sheet_names:
        P = pd.read_excel(xls, sheet_name="PARAMS")
        if P.shape[1] >= 2:
            params = kv_to_dict(P.iloc[:, :2])
            for k, v in params.items():
                if isinstance(v, bool):
                    lines.append(f"{k} = {str(v)}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{k} = {v}")
                else:
                    lines.append(f"{k} = {json.dumps(v)}")
    return "\n".join(lines) + "\n"

if st.button("▶ 모델 실행"):
    if not Path(model_path).exists():
        st.error(f"model.py를 찾을 수 없습니다: {model_path}")
    elif template_xlsx is None:
        st.error("템플릿 엑셀 파일을 업로드해 주세요.")
    else:
        xls = pd.ExcelFile(template_xlsx, engine="openpyxl")
        ov_txt = build_overrides_from_excel(xls)
        overrides_path = Path(workdir) / "overrides.py"
        with open(overrides_path, "w", encoding="utf-8") as f:
            f.write(ov_txt)

        with st.expander("📜 입력 데이터", expanded=False):
            st.code(ov_txt, language="python")


        # 기존 결과 제거
        expected = [
            "transition_5y_weights.csv",
            "transition_5y_metrics.csv",
            "transition_5y_equityVaR_prob.csv",
            "target_equityVaR_prob.csv",
            "shortfall_results.csv",
        ]
        for fn in expected:
            p = Path(workdir) / fn
            if p.exists():
                try: p.unlink()
                except: pass

        # model.py 실행
        with st.spinner("model.py 실행 중..."):
            proc = subprocess.run(
                [sys.executable, Path(model_path).name],
                cwd=workdir,
                capture_output=True,
                text=True
            )

        with st.expander("📜 실행 로그 보기", expanded=False):
            if proc.stdout:
                st.subheader("stdout")
                st.code(proc.stdout)
            if proc.stderr:
                st.subheader("stderr")
                st.code(proc.stderr)

        # 결과 표시/다운로드
        existing = []
        for fn in expected:
            p = Path(workdir) / fn
            if p.exists():
                existing.append(p)

        if not existing:
            st.warning("결과 CSV가 생성되지 않았습니다. 파라미터/밴드/목표수익률 등을 점검해 보세요.")
        else:
            st.success(f"생성된 결과 파일 {len(existing)}개")
            for p in existing:
                st.markdown(f"**📄 {p.name}**")
                try:
                    # 1) CSV를 문자열로 읽고 → 공백/빈문자 → NaN 정규화
                    df = pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False)
                    df = df.applymap(lambda x: np.nan if (x is None or (isinstance(x, str) and x.strip() == "")) else x)

                    # 2) 완전히 빈 행/열 제거
                    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

                    if df.empty:
                        st.info("모든 셀이 빈 값이라 미리보기를 생략합니다.")
                    else:
                        # 3) 값이 있는 최소 바운딩 박스만 표시
                        nonempty_rows = df.index[df.notna().any(axis=1)]
                        nonempty_cols = df.columns[df.notna().any(axis=0)]

                        r0, r1 = nonempty_rows.min(), nonempty_rows.max()
                        c0, c1 = nonempty_cols[0], nonempty_cols[-1]

                        df_trim = df.loc[r0:r1, c0:c1].fillna("").reset_index(drop=True)

                        # (선택) 실제로 항상 5행까지만 보고 싶으면 아래 한 줄 활성화
                        # df_trim = df_trim.head(5)

                        # 4) 행 수에 맞춰 높이 자동 조정(빈 공간 최소화)
                        base, row_h, cap = 44, 28, 520   # 헤더/행높이/최대높이
                        height = min(base + row_h * max(1, len(df_trim)), cap)

                        st.dataframe(df_trim, use_container_width=True, height=height)

                        # # 필요 시 원본 전체는 접어서 제공
                        # with st.expander("원본 전체 보기 (여백 포함)", expanded=False):
                        #     st.dataframe(df.fillna(""), use_container_width=True, height=360)

                except Exception as e:
                    st.info(f"(미리보기 불가) {e}")

                # 다운로드 버튼은 그대로 유지
                with open(p, "rb") as fh:
                    st.download_button(
                        f"⬇️ 다운로드: {p.name}",
                        fh.read(),
                        file_name=p.name,
                        mime="text/csv"
                    )


            # ZIP 묶음
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in existing:
                    zf.write(p, arcname=p.name)
            buf.seek(0)
            st.download_button("📦 결과 ZIP 다운로드", buf, file_name="model_outputs.zip", mime="application/zip")


