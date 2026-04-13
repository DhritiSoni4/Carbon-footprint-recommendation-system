"""
app.py — Carbon Footprint Calculator & Recommender
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Footprint Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Background */
.stApp { background: #0d1117; color: #e6edf3; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #8b949e !important;
    font-size: 0.82rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
}
.metric-card .lbl {
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}

/* Impact badge */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.04em;
}
.badge-low    { background:#1a4731; color:#3fb950; border:1px solid #3fb950; }
.badge-medium { background:#3d2c00; color:#d29922; border:1px solid #d29922; }
.badge-high   { background:#3d1a1a; color:#f85149; border:1px solid #f85149; }

/* Rec card */
.rec-card {
    background: #161b22;
    border-left: 3px solid #238636;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.93rem;
    line-height: 1.55;
}
.rec-card .rec-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
    color: #3fb950;
    margin-bottom: 3px;
}

/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
}

/* Divider */
hr { border-color: #30363d !important; }

/* Buttons */
.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.8rem !important;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2ea043 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #30363d; }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px 6px 0 0;
    color: #8b949e;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] { color: #3fb950 !important; border-bottom: 2px solid #3fb950 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load("model/stacking_model.pkl")
    preprocessor = joblib.load("model/preprocessor.pkl")
    metrics      = joblib.load("model/model_metrics.pkl")
    return model, preprocessor, metrics

try:
    model, preprocessor, metrics = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"❌ Model not found. Run `python train_and_save.py` first.\n\n{e}")
    st.stop()

# ── Emission factors ──────────────────────────────────────────────────────────
TRANSPORT_FACTORS = {
    "Car":   0.21,
    "Bus":   0.10,
    "Train": 0.04,
    "EV":    0.05,
    "Walk":  0.00,
    "Cycle": 0.00,
    "Bike":  0.12,
}
INDIA_AVERAGE_KG = 8.0          # kg CO2e / day (approx from dataset mean)
GLOBAL_AVERAGE_KG = 13.0        # kg CO2e / day (approx global benchmark)

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_input_df(transport_mode, distance_km, electricity_kwh,
                  renewable_pct, food_type, screen_hours,
                  waste_kg, eco_actions, day_type):
    return pd.DataFrame([{
        "distance_km":        distance_km,
        "electricity_kwh":    electricity_kwh,
        "renewable_usage_pct": renewable_pct,
        "screen_time_hours":  screen_hours,
        "waste_generated_kg": waste_kg,
        "eco_actions":        eco_actions,
        "day_type":           day_type,
        "transport_mode":     transport_mode,
        "food_type":          food_type,
    }])

def compute_emissions(transport_mode, distance_km, electricity_kwh,
                      renewable_pct, screen_hours, waste_kg):
    t_factor = TRANSPORT_FACTORS.get(transport_mode, 0.21)
    net_elec = electricity_kwh * (1 - renewable_pct / 100)
    return {
        "Transport":   round(distance_km * t_factor, 3),
        "Electricity": round(net_elec * 0.82, 3),
        "Waste":       round(waste_kg * 0.45, 3),
        "Digital":     round(screen_hours * 0.02, 3),
    }

def get_impact_level(kg):
    if kg < 5:   return "Low",    "badge-low"
    if kg < 10:  return "Medium", "badge-medium"
    return "High", "badge-high"

def generate_recommendations(transport_mode, distance_km, electricity_kwh,
                              renewable_pct, food_type, screen_hours,
                              waste_kg, eco_actions, emissions):
    recs = []
    top2 = sorted(emissions.items(), key=lambda x: x[1], reverse=True)[:2]

    for src, val in top2:
        if src == "Transport" and val > 1.5:
            if transport_mode == "Car":
                recs.append(("🚌 Switch Transport Mode",
                    "You're using a car for " + f"{distance_km:.0f} km/day. Switching to public "
                    "transport 3×/week could cut transport emissions by ~50%."))
            elif transport_mode in ("Bike", "Bus"):
                recs.append(("🚂 Try Rail for Long Trips",
                    "For trips over 50 km, trains emit ~5× less than cars and are often faster city-to-city."))

        if src == "Electricity" and val > 2:
            if renewable_pct < 30:
                recs.append(("☀️ Adopt Renewable Energy",
                    f"Your renewable usage is only {renewable_pct}%. Even shifting 30% to solar "
                    "can reduce electricity emissions by ~25 kg CO₂e/month."))
            else:
                recs.append(("💡 Reduce Peak Consumption",
                    "Shift heavy appliances (AC, washing machine) to off-peak hours and use 5-star rated devices."))

        if src == "Waste" and val > 1:
            recs.append(("♻️ Waste Reduction",
                f"You generate {waste_kg:.1f} kg waste/day. Composting organic waste and segregating recyclables "
                "can cut waste emissions by ~40%."))

        if src == "Digital" and val > 0.5:
            recs.append(("📱 Digital Detox",
                f"With {screen_hours:.1f} hours of screen time, enabling battery-saver mode and "
                "reducing video streaming quality can meaningfully lower digital emissions."))

    if food_type == "Non-Veg":
        recs.append(("🥗 Dietary Shift",
            "Non-veg diets produce ~2.5× more emissions than plant-based. "
            "Adding 2–3 vegetarian days per week can save ~0.8 kg CO₂e/day."))
    elif food_type == "Mixed":
        recs.append(("🌱 Go More Plant-Based",
            "Reducing meat to once a week and favouring legumes and vegetables can lower dietary footprint by ~30%."))

    if renewable_pct < 20:
        recs.append(("🔋 Rooftop Solar",
            "A 1 kW rooftop solar panel in India generates ~4–5 kWh/day — enough to offset "
            "a significant fraction of household electricity needs."))

    if eco_actions <= 1:
        recs.append(("🛍️ Build Eco-Habits",
            "Carrying a reusable bag/bottle, avoiding single-use plastics, and choosing local produce "
            "are low-effort actions that compound over time."))

    return recs[:5]  # top 5

def plotly_donut(emissions, predicted):
    labels = list(emissions.keys())
    values = list(emissions.values())
    colors = ["#3fb950", "#58a6ff", "#d29922", "#f85149"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0d1117", width=2)),
        textinfo="label+percent",
        textfont=dict(family="DM Sans", size=13, color="#e6edf3"),
        hovertemplate="%{label}: %{value:.3f} kg CO₂e<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{predicted:.2f}</b><br>kg CO₂e",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#e6edf3", family="Syne"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(font=dict(color="#8b949e", size=12), bgcolor="rgba(0,0,0,0)"),
    )
    return fig

def plotly_comparison(predicted):
    categories = ["Your Footprint", "India Avg", "Global Avg"]
    values     = [predicted, INDIA_AVERAGE_KG, GLOBAL_AVERAGE_KG]
    colors     = ["#3fb950" if predicted < INDIA_AVERAGE_KG else "#f85149", "#d29922", "#58a6ff"]
    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker=dict(color=colors, line=dict(color="#0d1117", width=1)),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(family="Syne", size=14, color="#e6edf3"),
        hovertemplate="%{x}: %{y:.2f} kg CO₂e<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="#21262d", color="#8b949e", title="kg CO₂e / day"),
        xaxis=dict(color="#8b949e"),
        margin=dict(t=30, b=10, l=10, r=10),
        font=dict(family="DM Sans"),
    )
    return fig

def plotly_whatif(baseline, scenarios):
    names  = ["Baseline"] + [s["label"] for s in scenarios]
    values = [baseline]   + [s["value"] for s in scenarios]
    deltas = [0]          + [round(s["value"] - baseline, 3) for s in scenarios]
    colors = ["#58a6ff"] + ["#3fb950" if d <= 0 else "#f85149" for d in deltas[1:]]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(color=colors),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(family="Syne", size=13, color="#e6edf3"),
        hovertemplate="%{x}: %{y:.2f} kg CO₂e<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="#21262d", color="#8b949e", title="kg CO₂e / day"),
        xaxis=dict(color="#8b949e"),
        margin=dict(t=30, b=10, l=10, r=10),
        font=dict(family="DM Sans"),
    )
    return fig

# ── Sidebar — branding + model info only ─────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-family:Syne; font-size:1.5rem; font-weight:800; color:#3fb950;'>🌍 CarbonIQ</div>
        <div style='font-size:0.78rem; color:#8b949e; margin-top:4px;'>Personal Carbon Footprint Analyzer</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:1rem; background:#0d1117; border:1px solid #21262d; border-radius:8px;
                font-size:0.82rem; color:#8b949e; line-height:1.8;'>
        <b style='color:#e6edf3;'>ℹ️ How to use</b><br>
        Fill in your daily lifestyle habits in the <b style='color:#3fb950;'>inputs below</b> on the main page,
        then scroll down to see your footprint, breakdown, comparison, and what-if scenarios.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style='margin-top:1.5rem; padding:0.8rem; background:#0d1117; border:1px solid #21262d;
                border-radius:8px; font-size:0.75rem; color:#8b949e; text-align:center;'>
        Model: Stacking Ensemble (GB + ET → Ridge)<br>
        R² = {metrics['R2']:.4f} &nbsp;|&nbsp; MAE = {metrics['MAE']:.3f} kg CO₂e
    </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 1.5rem 0 0.5rem 0;'>
    <h1 style='font-family:Syne; font-size:2.1rem; font-weight:800; margin:0; color:#e6edf3;'>
        🌍 Carbon Footprint Analyzer
    </h1>
    <p style='color:#8b949e; margin-top:6px; font-size:0.95rem;'>
        ML-powered footprint prediction · personalised recommendations · what-if simulator
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ── INPUT SECTION (main page) ─────────────────────────────────────────────────
st.markdown("""
<div style='background:#161b22; border:1px solid #30363d; border-radius:14px; padding:1.5rem 1.8rem 0.5rem 1.8rem; margin-bottom:1.5rem;'>
    <div style='font-family:Syne; font-size:1.05rem; font-weight:700; color:#e6edf3; margin-bottom:1rem;'>
        📋 Enter Your Daily Lifestyle Data
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div style='background:#161b22; border:1px solid #30363d; border-radius:14px; padding:1.2rem 1.5rem 1.5rem 1.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        st.markdown("**📅 Day & Lifestyle**")
        day_type    = st.selectbox("Day Type", ["Weekday", "Weekend"], key="main_day")
        food_type   = st.selectbox("Diet Type", ["Veg", "Non-Veg", "Mixed"], key="main_food")
        eco_actions = st.slider("Eco Actions (per day)", 0, 10, 2, key="main_eco",
                                help="Reusing bags, skipping plastic, etc.")
    with r1c2:
        st.markdown("**🚗 Transport**")
        transport_mode = st.selectbox("Transport Mode",
                                      ["Car", "Bus", "Train", "EV", "Walk", "Cycle", "Bike"],
                                      key="main_transport")
        distance_km = st.slider("Distance (km/day)", 0.0, 150.0, 15.0, step=0.5, key="main_dist")
    with r1c3:
        st.markdown("**⚡ Energy**")
        electricity_kwh = st.slider("Electricity Used (kWh/day)", 0.0, 30.0, 6.0, step=0.5, key="main_elec")
        renewable_pct   = st.slider("Renewable Share (%)", 0, 100, 10, key="main_ren")
    with r1c4:
        st.markdown("**🗑️ Waste & Screen**")
        waste_kg     = st.slider("Waste Generated (kg/day)", 0.0, 5.0, 0.6, step=0.1, key="main_waste")
        screen_hours = st.slider("Screen Time (hours/day)", 0.0, 16.0, 4.0, step=0.5, key="main_screen")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Compute prediction ────────────────────────────────────────────────────────
input_df = make_input_df(transport_mode, distance_km, electricity_kwh,
                         renewable_pct, food_type, screen_hours,
                         waste_kg, eco_actions, day_type)
X_proc     = preprocessor.transform(input_df)
predicted  = float(model.predict(X_proc)[0])
emissions  = compute_emissions(transport_mode, distance_km, electricity_kwh,
                               renewable_pct, screen_hours, waste_kg)
impact_lbl, impact_cls = get_impact_level(predicted)
recs = generate_recommendations(transport_mode, distance_km, electricity_kwh,
                                renewable_pct, food_type, screen_hours,
                                waste_kg, eco_actions, emissions)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='val' style='color:#3fb950;'>{predicted:.2f}</div>
        <div class='lbl'>kg CO₂e Today</div>
    </div>""", unsafe_allow_html=True)

with k2:
    monthly = predicted * 30
    st.markdown(f"""
    <div class='metric-card'>
        <div class='val' style='color:#58a6ff;'>{monthly:.0f}</div>
        <div class='lbl'>kg CO₂e / Month</div>
    </div>""", unsafe_allow_html=True)

with k3:
    vs_india = ((predicted - INDIA_AVERAGE_KG) / INDIA_AVERAGE_KG) * 100
    clr = "#3fb950" if vs_india <= 0 else "#f85149"
    arrow = "▼" if vs_india <= 0 else "▲"
    st.markdown(f"""
    <div class='metric-card'>
        <div class='val' style='color:{clr};'>{arrow}{abs(vs_india):.1f}%</div>
        <div class='lbl'>vs India Average</div>
    </div>""", unsafe_allow_html=True)

with k4:
    trees = round(predicted * 365 / 21.77)
    st.markdown(f"""
    <div class='metric-card'>
        <div class='val' style='color:#d29922;'>{trees}</div>
        <div class='lbl'>Trees/yr to Offset</div>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style='margin: 1rem 0 0.3rem 0;'>
    Impact level: <span class='badge {impact_cls}'>{impact_lbl}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Breakdown", "🌏 Comparison", "🔮 What-If Simulator"])

# ── Tab 1 — Breakdown ─────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-label'>Emission Sources</div>", unsafe_allow_html=True)
        st.plotly_chart(plotly_donut(emissions, predicted), use_container_width=True)

    with col_right:
        st.markdown("<div class='section-label'>Personalised Recommendations</div>",
                    unsafe_allow_html=True)
        if recs:
            for title, body in recs:
                st.markdown(f"""
                <div class='rec-card'>
                    <div class='rec-title'>{title}</div>
                    {body}
                </div>""", unsafe_allow_html=True)
        else:
            st.success("🎉 Great job! Your footprint is already very low. Keep it up!")

    st.markdown("<div class='section-label' style='margin-top:1.5rem;'>Category Breakdown</div>",
                unsafe_allow_html=True)
    bdf = pd.DataFrame(list(emissions.items()), columns=["Category", "kg CO₂e"])
    bdf["% Share"] = (bdf["kg CO₂e"] / bdf["kg CO₂e"].sum() * 100).round(1)
    bdf["kg CO₂e"] = bdf["kg CO₂e"].round(3)
    st.dataframe(bdf, use_container_width=True, hide_index=True)

# ── Tab 2 — Comparison ────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-label'>Your footprint vs benchmarks</div>",
                unsafe_allow_html=True)
    st.plotly_chart(plotly_comparison(predicted), use_container_width=True)

    diff_india  = predicted - INDIA_AVERAGE_KG
    diff_global = predicted - GLOBAL_AVERAGE_KG

    c1, c2 = st.columns(2)
    with c1:
        clr = "#3fb950" if diff_india <= 0 else "#f85149"
        msg = "below 🎉" if diff_india <= 0 else "above ⚠️"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val' style='color:{clr};'>{abs(diff_india):.2f} kg</div>
            <div class='lbl'>{msg} India average ({INDIA_AVERAGE_KG} kg)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        clr = "#3fb950" if diff_global <= 0 else "#f85149"
        msg = "below 🎉" if diff_global <= 0 else "above ⚠️"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='val' style='color:{clr};'>{abs(diff_global):.2f} kg</div>
            <div class='lbl'>{msg} global average ({GLOBAL_AVERAGE_KG} kg)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1.5rem; padding:1rem; background:#161b22; border-radius:10px;
                border:1px solid #30363d; font-size:0.88rem; color:#8b949e; line-height:1.7;'>
        <b style='color:#e6edf3;'>📌 Context</b><br>
        India's per-capita carbon footprint is ~1.9 tonnes CO₂e/year (~5.2 kg/day).
        The dataset mean is ~8 kg/day reflecting daily activity-level tracking.
        The global average of ~13 kg/day reflects higher energy consumption in developed nations.
    </div>
    """, unsafe_allow_html=True)

# ── Tab 3 — What-If Simulator ─────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class='section-label'>Simulate habit changes — see the impact instantly</div>
    """, unsafe_allow_html=True)

    w1, w2 = st.columns(2, gap="large")

    with w1:
        st.markdown("**🔧 Adjust one habit at a time:**")

        # Smart defaults: pick a greener transport than current
        transport_options = list(TRANSPORT_FACTORS.keys())
        greener_transports = [t for t in transport_options
                              if TRANSPORT_FACTORS[t] < TRANSPORT_FACTORS.get(transport_mode, 0.21)]
        default_transport_idx = (transport_options.index(greener_transports[0])
                                 if greener_transports else 0)
        wi_transport = st.selectbox("What if I switch transport to…", transport_options,
                                    index=default_transport_idx, key="wi_t")

        # Default renewable to current + 40% (capped at 100)
        default_renewable = min(renewable_pct + 40, 100)
        wi_renewable = st.slider("Renewable energy share (%)", 0, 100, default_renewable, key="wi_r")

        # Default distance to half of current (min 1 km)
        default_distance = max(round(distance_km * 0.5, 1), 1.0)
        wi_distance = st.slider("Distance (km/day)", 0.0, 150.0, default_distance, step=0.5, key="wi_d")

        # Default diet: step toward Veg
        diet_order = ["Non-Veg", "Mixed", "Veg"]
        current_idx = diet_order.index(food_type) if food_type in diet_order else 0
        default_diet_idx = min(current_idx + 1, 2)
        wi_food = st.selectbox("Diet", diet_order, index=default_diet_idx, key="wi_f")

    with w2:
        st.markdown("**📈 Projected outcomes:**")

        # Capture current baseline values explicitly to avoid closure issues
        _base = {
            "distance_km": distance_km,
            "electricity_kwh": electricity_kwh,
            "renewable_usage_pct": renewable_pct,
            "screen_time_hours": screen_hours,
            "waste_generated_kg": waste_kg,
            "eco_actions": eco_actions,
            "day_type": day_type,
            "transport_mode": transport_mode,
            "food_type": food_type,
        }

        def predict_scenario(base, **overrides):
            row = dict(base)
            row.update(overrides)
            return float(model.predict(preprocessor.transform(pd.DataFrame([row])))[0])

        scenarios = [
            {"label": "Switch Transport",
             "value": predict_scenario(_base, transport_mode=wi_transport)},
            {"label": "More Renewables",
             "value": predict_scenario(_base, renewable_usage_pct=wi_renewable)},
            {"label": "Less Distance",
             "value": predict_scenario(_base, distance_km=wi_distance)},
            {"label": "Change Diet",
             "value": predict_scenario(_base, food_type=wi_food)},
            {"label": "All Combined",
             "value": predict_scenario(_base, transport_mode=wi_transport,
                                       renewable_usage_pct=wi_renewable,
                                       distance_km=wi_distance, food_type=wi_food)},
        ]

        st.plotly_chart(plotly_whatif(predicted, scenarios), use_container_width=True)

    # savings table
    st.markdown("<div class='section-label' style='margin-top:1rem;'>Savings Summary</div>",
                unsafe_allow_html=True)
    rows = []
    for s in scenarios:
        delta = s["value"] - predicted
        rows.append({
            "Scenario":   s["label"],
            "Footprint":  f"{s['value']:.3f} kg CO₂e",
            "Change":     f"{'▼' if delta<=0 else '▲'} {abs(delta):.3f} kg",
            "Monthly":    f"{'−' if delta<=0 else '+'} {abs(delta*30):.1f} kg/mo",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)