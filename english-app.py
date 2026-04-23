import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Vietnam English Score Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 16px; }

    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #5b6ef5;
        border-radius: 0 8px 8px 0;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0 0.75rem 0;
        color: #374151;
        font-size: 0.97rem;
        line-height: 1.7;
    }
    .insight-box strong { color: #1e293b; }

    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #64748b;
        margin-bottom: 0.15rem;
    }
    .stat-card .stat-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .stat-card .stat-desc {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.5;
    }

    [data-testid="stMetricValue"] { font-size: 1.7rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem !important; }
    h1 { font-size: 2.1rem !important; font-weight: 700 !important; }
    p, li { font-size: 1rem; line-height: 1.7; }
    /* Add spacing between tabs */
div[data-baseweb="tab-list"] {
    gap: 12px;
    margin-bottom: 10px;
}

/* Individual tab styling */
button[data-baseweb="tab"] {
    background: #f1f5f9;
    border-radius: 999px; /* pill shape */
    padding: 8px 18px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    font-weight: 500;
}

/* Hover effect */
button[data-baseweb="tab"]:hover {
    background: #e2e8f0;
}

/* Active tab */
button[aria-selected="true"] {
    background: #5b6ef5;
    color: white;
    border: 1px solid #5b6ef5;
}

/* Remove ugly bottom border line and red highlight */
div[data-baseweb="tab-border"] {
    display: none !important;
}
div[data-baseweb="tab-highlight"] {
    display: none !important;
}
div[data-baseweb="tab-list"] {
    border-bottom: none !important;
}

</style>
""", unsafe_allow_html=True)

PALETTE = ['#5b6ef5', '#0ea5e9', '#10b981', '#f97316', '#ef4444', '#8b5cf6', '#ec4899']
C_GREEN = '#10b981'
C_RED   = '#ef4444'
C_BLUE  = '#5b6ef5'

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def stat_card(label, value, description):
    st.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'<div class="stat-desc">{description}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def ols(X_series, y_series):
    X = X_series.values.astype(float)
    y = y_series.values.astype(float)
    mask = ~(np.isnan(X) | np.isnan(y))
    X, y = X[mask], y[mask]
    mean_X, mean_y = X.mean(), y.mean()
    slope     = ((X - mean_X) * (y - mean_y)).sum() / ((X - mean_X) ** 2).sum()
    intercept = mean_y - slope * mean_X
    corr      = np.corrcoef(X, y)[0, 1]
    return slope, intercept, corr, X, y


# ── Province / region maps ────────────────────────────────────
PROVINCES = {
    "01":"Hà Nội","02":"TP. Hồ Chí Minh","03":"Hải Phòng","04":"Đà Nẵng",
    "05":"Hà Giang","06":"Cao Bằng","07":"Lai Châu","08":"Lào Cai",
    "09":"Tuyên Quang","10":"Lạng Sơn","11":"Bắc Kạn","12":"Thái Nguyên",
    "13":"Yên Bái","14":"Sơn La","15":"Phú Thọ","16":"Vĩnh Phúc",
    "17":"Quảng Ninh","18":"Bắc Giang","19":"Bắc Ninh","21":"Hải Dương",
    "22":"Hưng Yên","23":"Hòa Bình","24":"Hà Nam","25":"Nam Định",
    "26":"Thái Bình","27":"Ninh Bình","28":"Thanh Hóa","29":"Nghệ An",
    "30":"Hà Tĩnh","31":"Quảng Bình","32":"Quảng Trị","33":"Thừa Thiên-Huế",
    "34":"Quảng Nam","35":"Quảng Ngãi","36":"Kon Tum","37":"Bình Định",
    "38":"Gia Lai","39":"Phú Yên","40":"Đắk Lắk","41":"Khánh Hòa",
    "42":"Lâm Đồng","43":"Bình Phước","44":"Bình Dương","45":"Ninh Thuận",
    "46":"Tây Ninh","47":"Bình Thuận","48":"Đồng Nai","49":"Long An",
    "50":"Đồng Tháp","51":"An Giang","52":"Bà Rịa - Vũng Tàu",
    "53":"Tiền Giang","54":"Kiên Giang","55":"Cần Thơ","56":"Bến Tre",
    "57":"Vĩnh Long","58":"Trà Vinh","59":"Sóc Trăng","60":"Bạc Liêu",
    "61":"Cà Mau","62":"Điện Biên","63":"Đắk Nông","64":"Hậu Giang",
}

REGIONS = {
    "North": [
        "Hà Nội","Vĩnh Phúc","Bắc Ninh","Quảng Ninh","Hải Dương","Hải Phòng",
        "Hưng Yên","Thái Bình","Hà Nam","Nam Định","Ninh Bình","Hà Giang",
        "Cao Bằng","Bắc Kạn","Tuyên Quang","Lào Cai","Yên Bái","Thái Nguyên",
        "Lạng Sơn","Bắc Giang","Phú Thọ","Điện Biên","Lai Châu","Sơn La","Hòa Bình",
    ],
    "Central": [
        "Thanh Hóa","Nghệ An","Hà Tĩnh","Quảng Bình","Quảng Trị","Thừa Thiên-Huế",
        "Đà Nẵng","Quảng Nam","Quảng Ngãi","Bình Định","Phú Yên","Khánh Hòa",
        "Ninh Thuận","Bình Thuận","Kon Tum","Gia Lai","Đắk Lắk","Đắk Nông","Lâm Đồng",
    ],
    "South": [
        "Bình Phước","Tây Ninh","Bình Dương","Đồng Nai","Bà Rịa - Vũng Tàu",
        "TP. Hồ Chí Minh","Long An","Tiền Giang","Bến Tre","Trà Vinh","Vĩnh Long",
        "Đồng Tháp","An Giang","Kiên Giang","Cần Thơ","Hậu Giang","Sóc Trăng",
        "Bạc Liêu","Cà Mau",
    ],
}

def get_region(province):
    for region, provs in REGIONS.items():
        if province in provs:
            return region
    return None


# ── Load data ─────────────────────────────────────────────────
GITHUB_RAW = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/data"

@st.cache_data
def load_data():
    def resolve(filename):
        local = os.path.join("data", filename)
        if os.path.isfile(local):
            return local
        return f"{GITHUB_RAW}/{filename}"

    scores = pd.read_csv(resolve("diem_thi_thpt_2024.csv"), dtype={'sbd': str})
    scores['hometown'] = scores['sbd'].str[:2].map(PROVINCES)
    scores['region']   = scores['hometown'].apply(get_region)
    eng = scores[scores['ma_ngoai_ngu'] == 'N1'].copy()

    computer = pd.read_excel(resolve("computer23.xlsx"), skiprows=2, names=['Province','computer_pct'])
    poverty  = pd.read_excel(resolve("poverty23.xlsx"),  skiprows=2, names=['Province','poverty_pct'])
    gdp      = pd.read_excel(resolve("gdp23.xlsx"), skiprows=3,
                             names=['Province','avg_income','group1','group2','group3','group4','group5'])

    socio = computer.merge(poverty, on='Province', how='left') \
                    .merge(gdp[['Province','avg_income']], on='Province', how='left')

    eng_avg = eng.groupby('hometown', as_index=False)['ngoai_ngu'].mean()
    eng_avg.columns = ['Province', 'avg_score']

    merged = socio.merge(eng_avg, on='Province', how='inner')
    merged = merged.dropna(subset=['avg_score','computer_pct','poverty_pct','avg_income'])

    return eng, eng_avg, merged

eng, eng_avg, merged = load_data()


# ── KPIs ─────────────────────────────────────────────────────
n_students   = len(eng)
mean_score   = eng['ngoai_ngu'].mean()
median_score = eng['ngoai_ngu'].median()
pass_rate    = round(100 * (eng['ngoai_ngu'] >= 5).sum() / n_students, 1)
top_province = eng_avg.sort_values('avg_score', ascending=False).iloc[0]
bot_province = eng_avg.sort_values('avg_score').iloc[0]


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
with st.container(horizontal_alignment="center"):
    st.title("🎓 Vietnam 2024 English Exam Analysis")
    st.markdown("""
    This study presents a comprehensive analysis of the 2024 ***Vietnamese National High School Exam English*** scores across all 63 provinces, 
    with the aim of uncovering regional disparities in academic performance and examining how these differences may be associated with underlying socioeconomic conditions. 
    
    The score dataset, courtesy of J2Team, provides province-level insights into student outcomes, enabling a nationwide comparative perspective. 
    The analysis incorporates socioeconomic data from 2023 published by  ***Tổng cục Thống kê (General Statistics Office of Vietnam)***. 
    The selected indicators include computer access rate (%), poverty rate (%), and GDP per capita (VND), which together serve as proxies for technological accessibility, economic well-being, and overall development.
    
    By exploring the relationships between these variables and English exam performance, this study seeks to identify meaningful correlations and highlight potential structural inequalities that may influence educational outcomes across different regions of Vietnam."""
    )

st.space()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Students (English)",  f"{n_students:,}")
k2.metric("National Mean Score", f"{mean_score:.2f} / 10")
k3.metric("Median Score",        f"{median_score:.1f} / 10")
k4.metric("Pass Rate (≥ 5)",     f"{pass_rate}%")
k5.metric("Score Range",         f"{eng['ngoai_ngu'].min():.1f} – {eng['ngoai_ngu'].max():.1f}")

st.space("large")


tab_dist, tab_prov, tab_region, tab_reg = st.tabs([
    "📊 Score Distribution",
    "🗺️ By Province",
    "🧭 By Region",
    "📈 Socioeconomic Factors"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — SCORE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════
with tab_dist:
    st.subheader("National English Score Distribution")
    st.caption("Full distribution of all students who sat the English exam in 2024.")

    score_counts = (
        eng['ngoai_ngu'].dropna()
        .value_counts().sort_index().reset_index()
    )
    score_counts.columns = ['score', 'count']

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Score histogram")
        hist = (
            alt.Chart(score_counts)
            .mark_bar(color=C_BLUE, opacity=0.85, binSpacing=0)
            .encode(
                alt.X("score:Q", title="Score (0–10)",
                      axis=alt.Axis(labelFontSize=13, tickCount=10),
                      scale=alt.Scale(domain=[0, 10])),
                alt.Y("count:Q", title="Number of Students",
                      axis=alt.Axis(format=",d", labelFontSize=13)),
                tooltip=[alt.Tooltip("score:Q", title="Score", format=".2f"),
                         alt.Tooltip("count:Q", title="Students", format=",")]
            )
            .properties(height=320)
        )
        mean_rule = (
            alt.Chart(pd.DataFrame({'mean': [mean_score]}))
            .mark_rule(color=C_RED, strokeDash=[6,3], strokeWidth=2.5)
            .encode(x="mean:Q", tooltip=[alt.Tooltip("mean:Q", title="National Mean", format=".2f")])
        )
        st.altair_chart(hist + mean_rule, use_container_width=True)
        st.caption(f"Red dashed line = national mean ({mean_score:.2f})")
        insight(
            f"The distribution is <strong>bimodal</strong> — two distinct peaks reveal a "
            f"<strong>clear performance divide</strong> between <strong>low achievers</strong> "
            f"(clustered around <strong>3–4</strong>) and <strong>high achievers</strong> "
            f"(clustered around <strong>7–8</strong>). "
            f"The national mean is <strong>{mean_score:.2f}</strong> while the median is "
            f"<strong>{median_score:.2f}</strong>, suggesting a <strong>slight right-skew</strong> "
            f"driven by higher-performing students."
        )
    with cols[1]:
        st.subheader("Score bracket breakdown")
        brackets = pd.DataFrame({
            'Bracket': ['Excellent (8–10)','Good (6.5–8)','Average (5–6.5)','Below average (<5)'],
            'Min':     [8, 6.5, 5, 0],
            'Max':     [10.01, 8, 6.5, 5],
        })
        brackets['Count'] = brackets.apply(
            lambda r: int(((eng['ngoai_ngu'] >= r['Min']) & (eng['ngoai_ngu'] < r['Max'])).sum()), axis=1
        )
        brackets['%'] = (brackets['Count'] / n_students * 100).round(1)

        bracket_chart = (
            alt.Chart(brackets)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                alt.X("Bracket:N",
                      sort=['Excellent (8–10)','Good (6.5–8)','Average (5–6.5)','Below average (<5)'],
                      axis=alt.Axis(labelAngle=-20, labelFontSize=12), title=None),
                alt.Y("Count:Q", title="Students", axis=alt.Axis(format=",d", labelFontSize=13)),
                alt.Color("Bracket:N",
                          scale=alt.Scale(
                              domain=['Excellent (8–10)','Good (6.5–8)','Average (5–6.5)','Below average (<5)'],
                              range=[C_GREEN, C_BLUE, '#f97316', C_RED]
                          ), legend=None),
                tooltip=[alt.Tooltip("Bracket:N"), alt.Tooltip("Count:Q", format=","),
                         alt.Tooltip("%:Q", title="% of Students", format=".1f")]
            )
            .properties(height=240)
        )
        st.altair_chart(bracket_chart, use_container_width=True)
        st.dataframe(
            brackets[['Bracket','Count','%']].rename(columns={'%':'% of Students'}),
            hide_index=True, use_container_width=True,
            column_config={
                'Count':         st.column_config.NumberColumn(format="localized"),
                '% of Students': st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            }
        )




# ═══════════════════════════════════════════════════════════════
# TAB 2 — BY PROVINCE
# ═══════════════════════════════════════════════════════════════
with tab_prov:
    st.subheader("English Scores by Province")
    st.caption("Average English score for each of Vietnam's 63 provinces, sorted by performance.")

    eng_avg_sorted = eng_avg.sort_values('avg_score', ascending=False).copy()
    eng_avg_sorted['rank'] = range(1, len(eng_avg_sorted) + 1)

    col_ctrl, _ = st.columns([2, 3])
    with col_ctrl:
        score_thresh = st.slider("Highlight provinces with avg score above", 3.0, 8.0, 5.0, 0.1)

    eng_avg_sorted['color'] = eng_avg_sorted['avg_score'].apply(
        lambda s: C_GREEN if s >= score_thresh else C_RED
    )

    bar_chart = (
        alt.Chart(eng_avg_sorted)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            alt.X("Province:N",
                  sort=alt.EncodingSortField(field="avg_score", order="descending"),
                  axis=alt.Axis(labelAngle=-70, labelFontSize=11), title=None),
            alt.Y("avg_score:Q", title="Avg English Score",
                  scale=alt.Scale(domain=[0, 10]),
                  axis=alt.Axis(labelFontSize=13)),
            alt.Color("color:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("Province:N"),
                     alt.Tooltip("avg_score:Q", title="Avg Score", format=".2f"),
                     alt.Tooltip("rank:Q", title="Rank")]
        )
        .properties(height=380)
    )
    threshold_rule = (
        alt.Chart(pd.DataFrame({'y': [score_thresh]}))
        .mark_rule(color='#64748b', strokeDash=[5,3], strokeWidth=1.5)
        .encode(y="y:Q")
    )
    st.altair_chart(bar_chart + threshold_rule, use_container_width=True)

    insight(
        f"<strong>{top_province['Province']}</strong> leads nationally with an average of "
        f"<strong>{top_province['avg_score']:.2f}</strong>, while "
        f"<strong>{bot_province['Province']}</strong> scores lowest at "
        f"<strong>{bot_province['avg_score']:.2f}</strong> — a gap of "
        f"<strong>{top_province['avg_score'] - bot_province['avg_score']:.2f} points</strong>."
        f"It's also worth noting that high-performing provinces are predominantly <strong>major metropolitan areas</strong> "
    f"such as <strong>HCMC</strong>, <strong>Bình Dương</strong>, <strong>Hà Nội</strong>, and "
    f"<strong>Đà Nẵng</strong>, whereas lower scores are concentrated in "
    f"<strong>northern mountainous regions</strong>, where <strong>access to quality English education</strong> "
    f"remains more limited."
    )

    st.subheader("Pass / Fail Rate by Province")
    pf_col, _ = st.columns([2, 3])
    with pf_col:
        pf_thresh_prov = st.slider(
            "Pass threshold (province)", min_value=1.0, max_value=9.0, value=5.0, step=0.25,
            help="Drag to change what counts as a passing score"
        )
    st.caption(f"Proportion of students scoring ≥ {pf_thresh_prov:.2f} (pass) vs below (fail) per province.")

    pass_fail = eng.groupby('hometown').apply(
        lambda g: pd.Series({
            f'Pass (≥{pf_thresh_prov:.2f})': (g['ngoai_ngu'] >= pf_thresh_prov).mean() * 100,
            f'Fail (<{pf_thresh_prov:.2f})': (g['ngoai_ngu'] < pf_thresh_prov).mean() * 100,
        })
    ).reset_index().rename(columns={'hometown': 'Province'})

    pass_col = f'Pass (≥{pf_thresh_prov:.2f})'
    fail_col = f'Fail (<{pf_thresh_prov:.2f})'
    pass_fail = pass_fail.sort_values(pass_col, ascending=False)
    pass_fail_long = pass_fail.melt(id_vars='Province', var_name='Category', value_name='Pct')

    stacked = (
        alt.Chart(pass_fail_long)
        .mark_bar()
        .encode(
            alt.X("Province:N", sort=list(pass_fail['Province']),
                  axis=alt.Axis(labelAngle=-70, labelFontSize=11), title=None),
            alt.Y("Pct:Q", title="% of Students", stack="normalize",
                  axis=alt.Axis(format=".0%", labelFontSize=13)),
            alt.Color("Category:N",
                      scale=alt.Scale(domain=[pass_col, fail_col], range=[C_GREEN, C_RED]), title=""),
            tooltip=[alt.Tooltip("Province:N"), alt.Tooltip("Category:N"),
                     alt.Tooltip("Pct:Q", title="%", format=".1f")]
        )
        .properties(height=340)
    )
    st.altair_chart(stacked, use_container_width=True)

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Top 10 provinces", help="Ranked by average English score")
        st.dataframe(
            eng_avg_sorted.head(10)[['rank','Province','avg_score']].rename(
                columns={'rank':'Rank','avg_score':'Avg Score'}),
            hide_index=True, use_container_width=True,
            column_config={'Avg Score': st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.2f")}
        )
    with cols[1]:
        st.subheader("Bottom 10 provinces", help="Ranked by average English score")
        st.dataframe(
            eng_avg_sorted.tail(10)[['rank','Province','avg_score']].rename(
                columns={'rank':'Rank','avg_score':'Avg Score'}),
            hide_index=True, use_container_width=True,
            column_config={'Avg Score': st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.2f")}
        )


# ═══════════════════════════════════════════════════════════════
# TAB 3 — BY REGION
# ═══════════════════════════════════════════════════════════════
with tab_region:
    st.subheader("English Scores by Region")
    st.caption("Vietnam's three regions: North (Miền Bắc), Central (Miền Trung), South (Miền Nam).")

    region_stats = eng.groupby('region')['ngoai_ngu'].agg(
        mean='mean', median='median', std='std', count='count'
    ).round(3).reset_index().dropna()
    region_stats.columns = ['Region','Mean','Median','Std Dev','Students']

    r_cols = st.columns(3)
    for i, row in region_stats.iterrows():
        r_cols[i % 3].metric(
            f"{row['Region']} — Mean Score",
            f"{row['Mean']:.2f}",
            delta=f"{row['Mean'] - mean_score:+.2f} vs national avg"
        )

    cols = st.columns(2, border=True)
    with cols[0]:
        st.subheader("Mean score by region")
        region_bar = (
            alt.Chart(region_stats)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                alt.X("Region:N", axis=alt.Axis(labelAngle=0, labelFontSize=14), title=None),
                alt.Y("Mean:Q", title="Avg English Score",
                      scale=alt.Scale(domain=[0, 8]),
                      axis=alt.Axis(labelFontSize=13)),
                alt.Color("Region:N",
                          scale=alt.Scale(domain=['North','Central','South'],
                                          range=[C_BLUE, '#f97316', C_GREEN]), legend=None),
                tooltip=[alt.Tooltip("Region:N"), alt.Tooltip("Mean:Q", format=".2f"),
                         alt.Tooltip("Students:Q", format=",")]
            )
            .properties(height=300)
        )
        labels = region_bar.mark_text(dy=-12, fontSize=16, fontWeight=700).encode(
            text=alt.Text("Mean:Q", format=".2f"), color=alt.value("#1e293b")
        )
        nat_rule = (
            alt.Chart(pd.DataFrame({'y': [mean_score]}))
            .mark_rule(color='#64748b', strokeDash=[5,3], strokeWidth=1.5)
            .encode(y="y:Q", tooltip=[alt.Tooltip("y:Q", title="National mean", format=".2f")])
        )
        st.altair_chart(region_bar + labels + nat_rule, use_container_width=True)
        st.caption("Dashed line = national mean")

    with cols[1]:
        st.subheader("Pass / fail rate by region")
        pf_thresh_reg = st.slider(
            "Pass threshold (region)", min_value=1.0, max_value=9.0, value=5.0, step=0.25,
            help="Drag to change what counts as a passing score"
        )
        region_pf = eng.dropna(subset=['region']).groupby('region').apply(
            lambda g: pd.Series({
                f'Pass (≥{pf_thresh_reg:.2f})': (g['ngoai_ngu'] >= pf_thresh_reg).mean() * 100,
                f'Fail (<{pf_thresh_reg:.2f})': (g['ngoai_ngu'] < pf_thresh_reg).mean() * 100,
            })
        ).reset_index().rename(columns={'region': 'Region'})
        rpf_pass_col = f'Pass (≥{pf_thresh_reg:.2f})'
        rpf_fail_col = f'Fail (<{pf_thresh_reg:.2f})'
        region_pf_long = region_pf.melt(id_vars='Region', var_name='Category', value_name='Pct')

        region_stacked = (
            alt.Chart(region_pf_long)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                alt.X("Region:N", axis=alt.Axis(labelAngle=0, labelFontSize=14), title=None),
                alt.Y("Pct:Q", title="% of Students", stack="normalize",
                      axis=alt.Axis(format=".0%", labelFontSize=13)),
                alt.Color("Category:N",
                          scale=alt.Scale(domain=[rpf_pass_col, rpf_fail_col],
                                          range=[C_GREEN, C_RED]), title=""),
                tooltip=[alt.Tooltip("Region:N"), alt.Tooltip("Category:N"),
                         alt.Tooltip("Pct:Q", title="%", format=".1f")]
            )
            .properties(height=300)
        )
        st.altair_chart(region_stacked, use_container_width=True)

    insight(
        "There is a clear <strong>North–South divide</strong> in English performance. "
        "Urban centres like Hà Nội and TP. Hồ Chí Minh anchor their respective regions, "
        "while mountainous Northern provinces and parts of the Central Highlands "
        "consistently show the lowest scores."
    )

    st.subheader("Score distribution within each region")
    st.caption(
        "Overlapping distributions show how each region's students are spread across score ranges. "
        "Scores are normalised within each region so regions of different sizes are comparable."
    )

    eng_region_scores = eng.dropna(subset=['region','ngoai_ngu'])[['region','ngoai_ngu']].copy()
    region_score_counts = (
        eng_region_scores
        .groupby(['region','ngoai_ngu']).size().reset_index(name='count')
    )
    region_totals = region_score_counts.groupby('region')['count'].transform('sum')
    region_score_counts['pct'] = region_score_counts['count'] / region_totals * 100

    region_hist = (
        alt.Chart(region_score_counts)
        .mark_area(opacity=0.45, interpolate='monotone', line=True)
        .encode(
            alt.X("ngoai_ngu:Q", title="Score (0–10)",
                  scale=alt.Scale(domain=[0, 10]),
                  axis=alt.Axis(labelFontSize=13, tickCount=10)),
            alt.Y("pct:Q", title="% of Students in Region",
                  axis=alt.Axis(labelFontSize=13, format=".1f")),
            alt.Color("region:N", title="Region",
                      scale=alt.Scale(domain=['North','Central','South'],
                                      range=[C_BLUE, '#f97316', C_GREEN])),
            tooltip=[alt.Tooltip("region:N", title="Region"),
                     alt.Tooltip("ngoai_ngu:Q", title="Score", format=".2f"),
                     alt.Tooltip("pct:Q", title="% of Region", format=".2f"),
                     alt.Tooltip("count:Q", title="Students", format=",")]
        )
        .properties(height=360)
    )
    mean_rule_region = (
        alt.Chart(pd.DataFrame({'mean': [mean_score]}))
        .mark_rule(color=C_RED, strokeDash=[6,3], strokeWidth=2)
        .encode(x="mean:Q", tooltip=[alt.Tooltip("mean:Q", title="National Mean", format=".2f")])
    )
    st.altair_chart(region_hist + mean_rule_region, use_container_width=True)
    st.caption("Red dashed line = national mean.")


# ═══════════════════════════════════════════════════════════════
# TAB 4 — SOCIOECONOMIC REGRESSIONS
# ═══════════════════════════════════════════════════════════════
with tab_reg:
    st.subheader("Socioeconomic Factors & English Score")
    st.caption("Each point is a province. OLS regression shows the linear relationship between each factor and average English score.")

    factor = st.segmented_control(
        "Socioeconomic factor",
        ["Computer Access (%)", "Poverty Rate (%)", "Avg Income (VND '000)"],
        default="Computer Access (%)",
    )

    factor_map = {
        "Computer Access (%)":   ("computer_pct", "Household Computer Access (%)"),
        "Poverty Rate (%)":      ("poverty_pct",  "Multidimensional Poverty Rate (%)"),
        "Avg Income (VND '000)": ("avg_income",   "Average Monthly Income (VND '000)"),
    }
    col_key = factor_map[factor][0] if factor else "computer_pct"
    x_label = factor_map[factor][1] if factor else "Household Computer Access (%)"

    plot_data = merged[['Province', col_key, 'avg_score']].dropna().copy()
    slope, intercept, corr, X_vals, y_vals = ols(plot_data[col_key], plot_data['avg_score'])

    x_line = np.linspace(X_vals.min(), X_vals.max(), 100)
    reg_line_df = pd.DataFrame({'x': x_line, 'y': slope * x_line + intercept})

    y_pred = slope * X_vals + intercept
    ss_res = ((y_vals - y_pred) ** 2).sum()
    ss_tot = ((y_vals - y_vals.mean()) ** 2).sum()
    r2     = 1 - ss_res / ss_tot

    direction = "positive" if corr > 0 else "negative"
    strength  = "strong" if abs(corr) > 0.6 else ("moderate" if abs(corr) > 0.4 else "weak")

    # Residuals for ALL provinces
    plot_data = plot_data.copy()
    plot_data['predicted'] = y_pred
    plot_data['residual']  = y_vals - y_pred
    all_residuals = plot_data.sort_values('residual', ascending=False).copy()
    all_residuals['status'] = all_residuals['residual'].apply(
        lambda r: '▲ Over' if r > 0 else '▼ Under'
    )

    # ── Main layout: scatter left, stats right ─────────────────
    cols = st.columns([1.6, 1], border=True)

    with cols[0]:
        st.subheader(f"{factor} vs English Score")
        scatter = (
            alt.Chart(plot_data)
            .mark_point(filled=True, size=110, opacity=0.75, color='#c06f65')
            .encode(
                alt.X(f"{col_key}:Q", title=x_label, axis=alt.Axis(labelFontSize=13)),
                alt.Y("avg_score:Q", title="Avg English Score",
                      scale=alt.Scale(domain=[2, 9]),
                      axis=alt.Axis(labelFontSize=13)),
                tooltip=[alt.Tooltip("Province:N"),
                         alt.Tooltip(f"{col_key}:Q", title=x_label, format=".2f"),
                         alt.Tooltip("avg_score:Q", title="Avg Score", format=".2f"),
                         alt.Tooltip("residual:Q", title="Δ vs predicted", format="+.3f")]
            )
            .properties(height=400)
        )
        reg = (
            alt.Chart(reg_line_df)
            .mark_line(color='#538865', strokeWidth=2.5)
            .encode(alt.X("x:Q"), alt.Y("y:Q"))
        )
        st.altair_chart(scatter + reg, use_container_width=True)

    with cols[1]:
        st.subheader("What do these numbers mean?")

        stat_card(
            "Correlation (r)",
            f"{corr:.3f}",
            f"Measures the <strong>direction and strength</strong> of the linear relationship. "
            f"Ranges from −1 (perfect negative) to +1 (perfect positive). "
            f"r = {corr:.3f} means a <strong>{strength} {direction}</strong> relationship — "
            f"as {factor} {'increases' if corr > 0 else 'decreases'}, English scores tend to go {'up' if corr > 0 else 'down'}."
        )
        stat_card(
            "R² (coefficient of determination)",
            f"{r2:.3f}",
            f"Tells you <strong>how much of the variation</strong> in English scores is explained by {factor}. "
            f"R² = {r2:.3f} means this single factor accounts for <strong>{r2*100:.1f}% of the differences</strong> "
            f"in average scores across provinces. The remaining {100-r2*100:.1f}% is explained by other factors."
        )
        stat_card(
            "Slope",
            f"{slope:.4f}",
            f"For every <strong>1-unit increase</strong> in {factor}, the model predicts a "
            f"<strong>{abs(slope):.4f}-point {'increase' if slope > 0 else 'decrease'}</strong> "
            f"in average English score."
        )
        st.markdown(
            f"**Equation:** `Score = {slope:.4f} × [{factor}] + {intercept:.3f}`",
        )

    insight(
        f"There is a <strong>{strength} {direction} correlation (r = {corr:.2f})</strong> between "
        f"<em>{factor}</em> and provincial English scores. "
        f"The model explains <strong>{r2*100:.1f}%</strong> of the variance across provinces. "
        + (
            "Provinces with better digital infrastructure score significantly higher."
            if col_key == "computer_pct" else
            "Higher poverty rates are directly linked to weaker English performance."
            if col_key == "poverty_pct" else
            "Higher income provinces have meaningfully better English outcomes."
        )
    )

    # ── All provinces residual table ───────────────────────────
    st.subheader("All provinces: actual vs predicted")
    st.caption(
        "Every province ranked by how far its actual score deviates from the model's prediction. "
        "Positive = overperforming given its socioeconomic level. Negative = underperforming."
    )

    display_residuals = all_residuals[['Province','avg_score','predicted','residual','status']].copy()
    display_residuals.columns = ['Province','Actual Score','Predicted Score','Δ vs Predicted','Status']
    display_residuals['Actual Score']    = display_residuals['Actual Score'].round(3)
    display_residuals['Predicted Score'] = display_residuals['Predicted Score'].round(3)
    display_residuals['Δ vs Predicted']  = display_residuals['Δ vs Predicted'].round(3)

    st.dataframe(
        display_residuals,
        hide_index=True,
        use_container_width=True,
        height=400,
        column_config={
            'Actual Score':    st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.3f"),
            'Predicted Score': st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.3f"),
            'Δ vs Predicted':  st.column_config.NumberColumn(format="+.3f"),
            'Status':          st.column_config.TextColumn(),
        }
    )

    # ── All three factors side-by-side scatter plots ────────────
    st.subheader("All three factors at a glance")
    st.caption("Scatter plots and regression lines for all three socioeconomic predictors, shown side by side.")

    factor_configs = [
        ("computer_pct", "Computer Access (%)",     "Household Computer Access (%)"),
        ("poverty_pct",  "Poverty Rate (%)",         "Multidimensional Poverty Rate (%)"),
        ("avg_income",   "Avg Income (VND '000)",    "Average Monthly Income (VND '000)"),
    ]

    trio_cols = st.columns(3, border=True)
    for i, (fkey, flabel, fxtitle) in enumerate(factor_configs):
        fdata = merged[['Province', fkey, 'avg_score']].dropna().copy()
        fslope, fintercept, fcorr, fX, fy = ols(fdata[fkey], fdata['avg_score'])
        fr2   = 1 - ((fy - (fslope*fX + fintercept))**2).sum() / ((fy - fy.mean())**2).sum()
        fline = pd.DataFrame({'x': np.linspace(fX.min(), fX.max(), 80),
                              'y': fslope * np.linspace(fX.min(), fX.max(), 80) + fintercept})
        fdir  = "positive" if fcorr > 0 else "negative"
        fstr  = "strong" if abs(fcorr) > 0.6 else ("moderate" if abs(fcorr) > 0.4 else "weak")

        with trio_cols[i]:
            st.subheader(flabel)
            st.caption(f"r = **{fcorr:.3f}** · R² = **{fr2:.3f}** · {fstr} {fdir}")

            fscatter = (
                alt.Chart(fdata)
                .mark_point(filled=True, size=80, opacity=0.7, color='#c06f65')
                .encode(
                    alt.X(f"{fkey}:Q", title=fxtitle, axis=alt.Axis(labelFontSize=11)),
                    alt.Y("avg_score:Q", title="Avg Score",
                          scale=alt.Scale(domain=[2, 9]),
                          axis=alt.Axis(labelFontSize=11)),
                    tooltip=[alt.Tooltip("Province:N"),
                             alt.Tooltip(f"{fkey}:Q", title=flabel, format=".2f"),
                             alt.Tooltip("avg_score:Q", title="Avg Score", format=".2f")]
                )
                .properties(height=260)
            )
            freg = (
                alt.Chart(fline)
                .mark_line(color='#538865', strokeWidth=2)
                .encode(alt.X("x:Q"), alt.Y("y:Q"))
            )
            st.altair_chart(fscatter + freg, use_container_width=True)


    insight(
    f"💡 Across all three factors, the pattern is <strong>highly consistent</strong>: "
    f"<strong>wealthier, better-connected provinces</strong> achieve higher English scores. "
    f"<strong>Average income</strong> shows the <strong>strongest relationship</strong> "
    f"(r = <strong>0.854</strong>, R² = <strong>0.730</strong>), followed by "
    f"<strong>poverty rate</strong> (r = <strong>-0.758</strong>) and "
    f"<strong>computer access</strong> (r = <strong>0.692</strong>). "
    f"With up to <strong>73% of score variation explained</strong>, these results point to a "
    f"<strong>deep socioeconomic divide</strong> in Vietnam’s education system, "
    f"where access to resources, technology, and economic opportunity strongly aligns with performance. "
    f"While these are <strong>correlations rather than causation</strong>, the consistency across indicators "
    f"suggests structural inequalities play a major role in shaping outcomes."
)