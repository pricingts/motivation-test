# dashboard_resultados.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from score_results import score_df, aggregate_summary

# ==========================
#   Configuración inicial
# ==========================
load_dotenv()

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
    f"@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}"
)

engine = create_engine(DB_URL)
st.set_page_config(page_title="Dashboard de Motivaciones", page_icon="📊", layout="wide")

st.title("📊 Dashboard de Motivaciones Laborales")
st.markdown("Explora los resultados del cuestionario de **Motivaciones**: Logro · Afiliación · Poder")

# ==========================
#   Cargar datos
# ==========================
@st.cache_data(ttl=60)
def cargar_datos():
    query = "SELECT * FROM respuestas ORDER BY timestamp_utc DESC;"
    df = pd.read_sql(query, con=engine)
    return df

df = cargar_datos()

if df.empty:
    st.warning("⚠️ No hay registros en la base de datos todavía.")
    st.stop()

# ==========================
#   Calcular puntajes
# ==========================
df.rename(columns={c: c.upper() for c in df.columns if c.startswith(("a", "f", "p"))}, inplace=True)
scored, _ = score_df(df)
summary = aggregate_summary(scored)

# ==========================
#   Vista general
# ==========================
st.subheader("📋 Resultados Globales")

col1, col2 = st.columns([5, 1])
with col1:
    st.dataframe(scored, use_container_width=True, height=400)
with col2:
    csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar CSV", data=csv, file_name="resultados_motivaciones.csv", mime="text/csv")

# ==========================
#   Filtros
# ==========================
persona = st.selectbox("Selecciona una persona:", ["(Todas)"] + list(scored["nombre"].unique()))

# ==========================
#   Análisis Individual
# ==========================
if persona != "(Todas)":
    st.subheader(f"🧠 Análisis Individual: {persona}")
    persona_df = scored[scored["nombre"] == persona].iloc[0]

    # Gráfico radar
    radar = go.Figure()
    categorias = ["Logros", "Afiliación", "Poder"]
    valores = [
        persona_df["Logros_media"],
        persona_df["Afiliación_media"],
        persona_df["Poder_media"],
    ]
    radar.add_trace(go.Scatterpolar(r=valores, theta=categorias, fill='toself', name=persona))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 5])), showlegend=False)
    st.plotly_chart(radar, use_container_width=True)

    # Perfil interpretativo
    perfil = persona_df["perfil_dominante"]
    interpretaciones = {
        "Logros": "Alta orientación al rendimiento, búsqueda de estándares exigentes y deseo de superación personal.",
        "Afiliación": "Motivado por las relaciones humanas, el trabajo en equipo y la armonía grupal.",
        "Poder": "Busca influir, liderar y tener impacto en los demás; disfruta de posiciones de responsabilidad.",
        "Mixto": "Presenta un equilibrio entre las tres motivaciones sin predominio claro."
    }
    st.info(f"**Perfil dominante:** {perfil}\n\n💬 {interpretaciones.get(perfil, 'Sin interpretación definida.')}")

# ==========================
#   Distribución Global
# ==========================
st.subheader("📊 Distribución general de puntajes")

melted = scored.melt(id_vars="nombre", value_vars=["Logros_media", "Afiliación_media", "Poder_media"],
                     var_name="aspecto", value_name="puntaje")
fig2 = px.box(melted, x="aspecto", y="puntaje", color="aspecto",
              title="Distribución general por aspecto", points="all")
st.plotly_chart(fig2, use_container_width=True)

# ==========================
#   Resumen organizacional
# ==========================
st.subheader("🏢 Resumen organizacional")

colA, colB = st.columns(2)
with colA:
    st.markdown("### Promedio general por escala")
    avg = summary[summary["metric"].str.endswith("_porc")].set_index("metric")
    st.bar_chart(avg["value"])
with colB:
    st.markdown("### Distribución de perfiles dominantes")
    dist = summary[summary["metric"].str.startswith("perfil_")]
    fig3 = px.pie(dist, names="metric", values="value", title="Perfiles en la organización")
    st.plotly_chart(fig3, use_container_width=True)
