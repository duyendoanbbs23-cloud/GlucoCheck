"""Diabetes Risk Check: a multilingual Streamlit screening app (CDC BRFSS 2015).

User-decision statement: an adult answering questions at home uses the app to decide
whether to get a blood sugar test. Educational project, not a medical device.
"""
import re
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

import core
import viz
from i18n import LANGS, tr

# --------------------------------------------------------------------------- language & page config
_qp = {}
try:
    _qp = dict(st.query_params)
except Exception:  # very old Streamlit
    pass
_lang0 = _qp.get("lang", "vi")
if _lang0 not in LANGS:
    _lang0 = "vi"

st.set_page_config(page_title=tr(_lang0, "app_title"), page_icon="🩺", layout="centered",
                   initial_sidebar_state="expanded")

# Widget state is dropped when a widget is not rendered (e.g. on another page). Seeding defaults and
# re-assigning every run keeps the user's answers and costs when they move between pages.
DEFAULTS = {"lang": _lang0, "page": 0, "show": False,
            "w_age": 45, "w_h": 165, "w_w": 65, "w_gen": 3,
            "w_HighBP": 0, "w_HighChol": 0, "w_Heart": 0, "w_Diff": 0,
            "c_fp": 1.0, "c_fn": 5.7, "fair_attr": "Income", "fair_measure": "auc"}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
    st.session_state[_k] = st.session_state[_k]

lang = st.session_state["lang"]
L = lambda key, **kw: tr(lang, key, **kw)

# --------------------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stRadio, .stButton, .stNumberInput { font-family: 'Inter', system-ui, sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 760px; }
h1, h2, h3 { letter-spacing: -0.02em; color: #1F2A44; }
.hero { background: linear-gradient(135deg, #4F5BD5 0%, #1F9D8B 100%); color: #fff; border-radius: 22px;
        padding: 26px 28px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(79,91,213,.22); }
.hero h1 { color: #fff; font-size: 1.9rem; margin: 0 0 6px 0; font-weight: 800; }
.hero p { margin: 0; font-size: 1.04rem; opacity: .95; line-height: 1.5; }
.card { background: #fff; border: 1px solid #E3E8F2; border-radius: 18px; padding: 18px 20px; margin: 12px 0;
        box-shadow: 0 2px 10px rgba(31,42,68,.04); }
.card h4 { margin: 0 0 6px 0; color: #1F2A44; font-size: 1.02rem; font-weight: 700; }
.card p { margin: 0; color: #3B4663; line-height: 1.6; font-size: .97rem; }
.tier { border-radius: 20px; padding: 20px 22px; margin: 6px 0 12px 0; border: 1.5px solid; }
.tier .t { font-size: 1.35rem; font-weight: 800; margin-bottom: 6px; color: #1F2A44; }
.tier .x { color: #2C3652; line-height: 1.6; font-size: 1rem; }
.tier.low  { background: #E7F6F3; border-color: #1F9D8B; }
.tier.mid  { background: #FFF6E0; border-color: #F2B84B; }
.tier.high { background: #ECEEFE; border-color: #4F5BD5; }
.kicker { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #6B7690; font-weight: 700; margin-bottom: 4px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 10px 0 6px 0; }
.tile { background: #fff; border: 1px solid #E3E8F2; border-radius: 16px; padding: 14px 16px; }
.tile .v { font-size: 1.55rem; font-weight: 800; color: #4F5BD5; line-height: 1.15; }
.tile .l { font-size: .8rem; color: #6B7690; margin-top: 4px; line-height: 1.35; }
.note { background: #F0F3FA; border-radius: 14px; padding: 12px 16px; color: #4A5573; font-size: .9rem; line-height: 1.55; margin: 10px 0; }
.disc { background: #FFF9EC; border: 1px solid #F2D9A0; border-radius: 14px; padding: 14px 18px; color: #5B4A1F; line-height: 1.6; margin: 10px 0; }
.big { font-size: 2.6rem; font-weight: 800; color: #4F5BD5; line-height: 1; }
.foot { text-align: center; color: #8A93A8; font-size: .8rem; margin-top: 36px; }
div[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #E3E8F2; }
.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; padding: .55rem 1.1rem; }
div[role="radiogroup"] label { font-size: .96rem; }
</style>
""", unsafe_allow_html=True)


def bold(s: str) -> str:
    """Escape text and turn **x** into <b>x</b> (for strings placed inside our own HTML blocks)."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(s)).replace("\n", "<br>")


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def card(title: str, text: str):
    html(f'<div class="card"><h4>{escape(title)}</h4><p>{bold(text)}</p></div>')


def tiles(items):
    html('<div class="tiles">' + "".join(
        f'<div class="tile"><div class="v">{escape(str(v))}</div><div class="l">{escape(l)}</div></div>'
        for v, l in items) + "</div>")


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


# --------------------------------------------------------------------------- cached data
@st.cache_resource
def get_model():
    return core.load_model()


@st.cache_data
def get_metrics():
    return core.load_metrics()


@st.cache_data
def get_scores():
    return core.load_scores()


model, metrics, scores = get_model(), get_metrics(), get_scores()
OV = metrics["overall"]
FEATS = model["features"]
AVG = OV["prevalence"]


def threshold() -> float:
    return core.threshold_from_costs(st.session_state["c_fp"], st.session_state["c_fn"])


def advice(tier_name: str) -> str:
    return L("adv_" + tier_name)


# --------------------------------------------------------------------------- sidebar
NAV = ["nav_check", "nav_sim", "nav_batch", "nav_fair", "nav_about", "nav_card"]


def _sync_lang():
    try:
        st.query_params["lang"] = st.session_state["lang"]
    except Exception:
        pass


with st.sidebar:
    st.selectbox(L("lang_label"), list(LANGS), format_func=LANGS.get, key="lang", on_change=_sync_lang)
    st.radio(L("nav_menu"), list(range(len(NAV))), format_func=lambda i: L(NAV[i]), key="page")
    html(f'<div class="note">{escape(L("side_disc"))}</div>')

page = st.session_state["page"]
hero_sub = L("tagline")


def hero(title, sub):
    html(f'<div class="hero"><h1>{escape(title)}</h1><p>{escape(sub)}</p></div>')


# =========================================================================== 0. My check
def page_check():
    hero("🩺 " + L("app_title"), hero_sub)
    st.subheader(L("check_title"))
    st.caption(L("check_intro"))

    c1, c2, c3 = st.columns(3)
    age = c1.number_input(L("q_age"), min_value=1, max_value=110, step=1, key="w_age")
    height = c2.number_input(L("q_height"), min_value=100, max_value=230, step=1, key="w_h")
    weight = c3.number_input(L("q_weight"), min_value=25, max_value=250, step=1, key="w_w")
    bmi = core.bmi_from(height, weight)
    st.markdown(L("bmi_is", bmi=f"{bmi:.1f}"))

    st.markdown(f"**{L('q_gen')}**")
    gen_opts = L("gen_opts")
    gen = st.radio(L("q_gen"), [1, 2, 3, 4, 5], format_func=lambda i: gen_opts[i - 1], horizontal=True,
                   label_visibility="collapsed", key="w_gen")

    answers = {}
    for key, qkey in [("HighBP", "q_HighBP"), ("HighChol", "q_HighChol"),
                      ("HeartDiseaseorAttack", "q_Heart"), ("DiffWalk", "q_Diff")]:
        wkey = {"HighBP": "w_HighBP", "HighChol": "w_HighChol",
                "HeartDiseaseorAttack": "w_Heart", "DiffWalk": "w_Diff"}[key]
        st.markdown(f"**{L(qkey)}**")
        answers[key] = st.radio(L(qkey), [0, 1], format_func=lambda v: L("yes") if v else L("no"),
                                horizontal=True, label_visibility="collapsed", key=wkey)
    st.caption(L("unsure_hint"))

    if age < 18:
        st.warning(L("minor_warn"))
        return
    if st.button("👉 " + L("btn_show"), type="primary"):
        st.session_state["show"] = True
    if not st.session_state["show"]:
        return

    case = {"GenHlth": gen, "Age": core.age_to_code(age), "BMI": float(np.clip(bmi, 12, 98)), **answers}
    p = float(core.predict_proba(pd.DataFrame([case]), model)[0])
    st.session_state["case"] = {"p": p, **case}
    t = threshold()
    tier = core.tier(p, t)

    st.divider()
    st.subheader(L("res_title"))
    icon = {"low": "🌿", "mid": "🔎", "high": "🩺"}[tier]
    html(f'<div class="tier {tier}"><div class="kicker">{escape(L("action"))}</div>'
         f'<div class="t">{icon} {escape(L("tier_" + tier + "_t"))}</div>'
         f'<div class="x">{escape(L("tier_" + tier + "_x"))}</div></div>')

    n_in_100 = min(100, max(1, round(p * 100))) if p > 0 else 0
    left, right = st.columns([3, 2])
    with left:
        html(f'<div class="card"><p>{bold(L("res_you", n=n_in_100 if p >= 0.005 else "<1"))}</p></div>')
        st.caption(L("res_avg"))
    with right:
        html(viz.icon_array(n_in_100, L("res_you", n=n_in_100)))

    html(viz.meter(p, AVG, t, {"you": L("meter_you"), "avg": L("meter_avg"), "thr": L("meter_thr")}))
    st.caption(L("thr_used", t=f"{t * 100:.0f}"))

    # --- explanation: top factors for THIS case
    st.subheader(L("exp_title"))
    st.write(L("exp_intro"))
    contrib = core.contributions(case, model)
    top = [(k, v) for k, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:5] if abs(v) >= 0.04]
    if top:
        def answer_text(k):
            v = case[k]
            if k == "GenHlth":
                return gen_opts[int(v) - 1]
            if k == "Age":
                return core.age_code_label(int(v))
            if k == "BMI":
                return f"{v:.1f}"
            return L("yes") if v else L("no")
        html(viz.contrib_bars([(f"{L('f_' + k)} · {answer_text(k)}", v) for k, v in top],
                              L("exp_up"), L("exp_down"), L("f_axis")))
    else:
        st.info(L("exp_none"))
    st.caption(L("exp_note"))
    html(f'<div class="note">{escape(L("not_dx"))}</div>')


# =========================================================================== 1. Threshold & cost
def _set_costs(fp, fn):
    st.session_state["c_fp"], st.session_state["c_fn"] = float(fp), float(fn)


def page_sim():
    hero("⚖️ " + L("sim_title"), "")
    st.markdown(L("sim_intro"))

    st.markdown(f"**{L('preset_title')}**")
    b1, b2, b3 = st.columns(3)
    b1.button(L("preset_cheap"), on_click=_set_costs, args=(1, 10))
    b2.button(L("preset_bal"), on_click=_set_costs, args=(1, 5))
    b3.button(L("preset_costly"), on_click=_set_costs, args=(1, 2))

    c1, c2 = st.columns(2)
    fp = c1.number_input(L("c_fp"), min_value=0.1, max_value=1000.0, step=0.1, key="c_fp")
    fn = c2.number_input(L("c_fn"), min_value=0.1, max_value=1000.0, step=0.1, key="c_fn")
    st.caption(L("c_help"))

    t = core.threshold_from_costs(fp, fn)
    html(f'<div class="card"><div class="kicker">{escape(L("thr_res"))}</div>'
         f'<div class="big">{t * 100:.1f}%</div>'
         f'<p style="margin-top:8px">{bold(L("thr_formula", t=f"{t * 100:.1f}"))}</p></div>')

    c = core.confusion(scores, t)
    tiles([(pct(c["flagged"]), L("sim_flagged")), (pct(c["sensitivity"]), L("sim_sens")),
           (pct(c["specificity"]), L("sim_spec")), (pct(c["ppv"]), L("sim_ppv")),
           (f"{core.expected_cost_per_1000(scores, t, fp, fn):,.0f}", L("sim_cost"))])
    st.caption(L("sim_note", n=len(scores)))

    st.subheader(L("curve_title"))
    cc = core.cost_curve(scores, fp, fn)
    best = float(cc.loc[cc["cost"].idxmin(), "threshold"])
    html(viz.line_chart(cc["threshold"].tolist(), cc["cost"].tolist(),
                        [(t, f"{L('curve_you')} {t * 100:.0f}%", viz.INDIGO),
                         (best, f"{L('curve_best')} {best * 100:.0f}%", viz.TEAL)],
                        L("curve_x"), L("curve_y")))

    st.subheader(L("case_title"))
    case = st.session_state.get("case")
    if case:
        a = advice(core.tier(case["p"], t))
        p_txt = f"{case['p'] * 100:.0f}"
        html(f'<div class="card"><p>{bold(L("case_line", p=p_txt, a=a))}</p></div>')
    else:
        st.info(L("case_none"))


# =========================================================================== 2. Batch
def page_batch():
    hero("📄 " + L("batch_title"), "")
    st.write(L("batch_intro"))
    with st.expander(L("batch_cols")):
        st.markdown(L("batch_cols_help"))
    st.download_button("⬇️ " + L("batch_example"),
                       (core.ROOT / "data" / "sample_batch.csv").read_bytes(),
                       file_name="example_batch.csv", mime="text/csv")

    up = st.file_uploader(L("batch_upload"), type=["csv"])
    if up is None:
        return
    try:
        raw = pd.read_csv(up)
    except Exception:
        st.error(L("batch_read_err"))
        return
    work, ok, missing = core.normalize_batch(raw, FEATS)
    if missing:
        st.error(L("batch_missing", c=", ".join(missing)))
        return
    skipped = int((~ok).sum())
    if skipped:
        st.warning(L("batch_skipped", n=skipped))
    if not ok.any():
        return

    t = threshold()
    probs = core.predict_proba(work.loc[ok, FEATS], model)
    res = raw.loc[ok].copy()
    # the user's own file may already contain columns with these names
    res = res.drop(columns=[c for c in (L("col_rank"), L("col_risk"), L("col_advice")) if c in res.columns])
    res.insert(0, L("col_risk"), np.round(probs * 100, 1))
    res.insert(1, L("col_advice"), [advice(core.tier(p, t)) for p in probs])
    res = res.sort_values(L("col_risk"), ascending=False).reset_index(drop=True)
    res.insert(0, L("col_rank"), np.arange(1, len(res) + 1))

    st.success(L("batch_ok", n=len(res), k=int((probs >= t).sum()), t=f"{t * 100:.0f}"))
    tiles([(f"{int((np.array([core.tier(p, t) for p in probs]) == k).sum()):,}", advice(k))
           for k in ["high", "mid", "low"]])
    st.dataframe(res.head(1000), hide_index=True)
    st.download_button("⬇️ " + L("batch_dl"), res.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ranked_results.csv", mime="text/csv", type="primary")
    st.caption(L("thr_used", t=f"{t * 100:.0f}"))


# =========================================================================== 3. Fairness
ATTRS = ["Income", "Education", "Sex", "Age"]


def page_fair():
    hero("🤝 " + L("fair_title"), "")
    st.write(L("fair_intro"))
    attr = st.radio(L("fair_attr"), ATTRS, format_func=lambda a: L("attr_" + a), horizontal=True, key="fair_attr")
    t = threshold()

    static = {(r["attribute"], r["group"]): r for r in metrics["fairness"]}
    rows = []
    for g in [r["group"] for r in metrics["fairness"] if r["attribute"] == attr]:
        sub = scores[scores[attr] == g]
        c = core.confusion(sub, t)
        rows.append({"key": g, "label": L("grp_" + g), "n": len(sub), "prev": float(sub["y"].mean()),
                     "auc": static[(attr, g)]["auc"], "flag": c["flagged"], "sens": c["sensitivity"],
                     "spec": c["specificity"]})

    df = pd.DataFrame({
        L("attr_" + attr): [r["label"] for r in rows],
        L("m_n"): [f"{r['n']:,}" for r in rows],
        L("m_prev"): [pct(r["prev"], 1) for r in rows],
        L("m_auc"): [f"{r['auc']:.2f}" for r in rows],
        L("m_flag"): [pct(r["flag"]) for r in rows],
        L("m_sens"): [pct(r["sens"]) for r in rows],
        L("m_spec"): [pct(r["spec"]) for r in rows],
    })
    st.dataframe(df, hide_index=True)
    st.caption(L("thr_here", t=f"{t * 100:.0f}"))

    measures = {"auc": "m_auc", "sens": "m_sens", "spec": "m_spec", "flag": "m_flag"}
    m = st.radio(L("fair_chart"), list(measures), format_func=lambda k: L(measures[k]), horizontal=True,
                 key="fair_measure")
    is_auc = m == "auc"
    vmax = 1.0 if is_auc else min(1.0, max(r[m] for r in rows) * 1.2 + 0.02)
    ref = {"auc": OV["auc"], "sens": core.confusion(scores, t)["sensitivity"],
           "spec": core.confusion(scores, t)["specificity"], "flag": core.confusion(scores, t)["flagged"]}[m]
    html(viz.hbars([(r["label"], r[m], f"n = {r['n']:,}") for r in rows], vmax,
                   (lambda v: f"{v:.2f}") if is_auc else (lambda v: pct(v)), ref=ref))

    best, worst = max(rows, key=lambda r: r["auc"]), min(rows, key=lambda r: r["auc"])
    st.markdown(L("fair_gap_auc", g=best["auc"] - worst["auc"], b=best["label"], w=worst["label"]))
    hi, lo = max(rows, key=lambda r: r["sens"]), min(rows, key=lambda r: r["sens"])
    st.markdown(L("fair_gap_sens", lo=f"{lo['sens'] * 100:.0f}", lw=lo["label"],
                  hi=f"{hi['sens'] * 100:.0f}", hw=hi["label"]))
    with st.expander(L("fair_read")):
        st.markdown(L("fair_read_x").replace("\n", "\n\n"))


# =========================================================================== 4. About
def page_about():
    hero("ℹ️ " + L("about_title"), "")
    st.subheader(L("about_data_h"))
    st.write(L("about_data", n=OV["n_train"] + OV["n_test"]))
    st.subheader(L("about_model_h"))
    st.write(L("about_model", a7=f"{OV['auc']:.3f}", a16=f"{OV['auc_all_16_lr']:.3f}"))

    st.subheader(L("about_metrics_h"))
    tiles([(pct(OV["prevalence"], 1), L("ref_prev")), (f"{OV['auc']:.3f}", L("ref_auc")),
           (f"{OV['pr_auc']:.3f}", L("ref_pr")), (f"{OV['brier']:.3f}", L("ref_brier"))])
    st.markdown(f"**{L('ref_thr')}: {OV['default_threshold'] * 100:.0f}%**")
    tiles([(pct(OV["sensitivity"]), L("ref_sens")), (pct(OV["specificity"]), L("ref_spec")),
           (pct(OV["ppv"]), L("ref_ppv"))])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{L('cal_title')}**")
        cal = metrics["calibration"]
        html(viz.scatter_cal([r["pred"] for r in cal], [r["obs"] for r in cal], L("cal_x"), L("cal_y")))
        st.caption(L("cal_note"))
    with c2:
        st.markdown(f"**{L('sel_title')}**")
        sel = metrics["selection_curve"]
        html(viz.step_line([r["n"] for r in sel], [r["auc"] for r in sel], L("sel_x"), L("sel_y"), mark_n=len(FEATS)))
        st.caption(L("sel_note"))

    st.subheader(L("about_lim_h"))
    for i, lim in enumerate(L("limits"), 1):
        html(f'<div class="card"><p>{i}. {bold(lim)}</p></div>')
    st.subheader("⚕️ " + L("disc_h"))
    html(f'<div class="disc">{escape(L("disc_x"))}</div>')


# =========================================================================== 5. Model card
def page_card():
    hero("📋 " + L("card_title"), "")
    for h, text in L("card"):
        card(h, text)
    html(f'<div class="disc">{escape(L("disc_x"))}</div>')


PAGES = [page_check, page_sim, page_batch, page_fair, page_about, page_card]
PAGES[page]()
html(f'<div class="foot">{escape(L("footer"))}</div>')
