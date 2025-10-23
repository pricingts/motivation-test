# app.py
# Gradio + PostgreSQL: Cuestionario (público) + Panel Admin (protegido)
# Requiere: gradio, pandas, numpy, filelock, psycopg2-binary, sqlalchemy
# Env vars: PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE, ADMIN_PASSWORD

import os
import argparse
from datetime import datetime
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
load_dotenv()


import pandas as pd
import numpy as np
import gradio as gr

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ==========================
#   Configuración & DB
# ==========================

def db_engine_from_env() -> Engine:
    required = ["PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Faltan variables de entorno para la BD: {', '.join(missing)}")
    url = (
        f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
        f"@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}"
    )
    return create_engine(url, pool_pre_ping=True)

engine = db_engine_from_env()

# Crear tabla si no existe
DDL = """
CREATE TABLE IF NOT EXISTS respuestas (
    id SERIAL PRIMARY KEY,
    timestamp_utc TIMESTAMP,
    nombre TEXT,
    cedula TEXT,
    A1 INT, A2 INT, A3 INT, A4 INT, A5 INT, A6 INT, A7 INT, A8 INT, A9 INT, A10 INT, A11 INT, A12 INT,
    F1 INT, F2 INT, F3 INT, F4 INT, F5 INT, F6 INT, F7 INT, F8 INT, F9 INT, F10 INT, F11 INT, F12 INT,
    P1 INT, P2 INT, P3 INT, P4 INT, P5 INT, P6 INT, P7 INT, P8 INT, P9 INT, P10 INT, P11 INT, P12 INT
);
"""
with engine.begin() as conn:
    conn.execute(text(DDL))

# ==========================
#   Ítems y utilidades
# ==========================

TITLE = "Cuestionario de Motivaciones (Logros · Afiliación · Poder)"
INSTRUCCION = (
    "Responde según tu trabajo actual. Escala Likert 1–5:\n"
    "1 = Totalmente en desacuerdo · 2 = En desacuerdo · 3 = Ni de acuerdo ni en desacuerdo · "
    "4 = De acuerdo · 5 = Totalmente de acuerdo."
)

PREGUNTAS: List[Tuple[int, str, str]] = [
    (1,  "A1",  "Me motiva superar metas desafiantes."),
    (2,  "F4",  "Prefiero trabajar aislado la mayor parte del tiempo."),
    (3,  "P4",  "Evito situaciones en las que deba influir en otras personas."),
    (4,  "A2",  "Prefiero tareas con retroalimentación clara sobre mi desempeño."),
    (5,  "F5",  "Los vínculos con mis compañeros me resultan indiferentes."),
    (6,  "P5",  "No me interesa asumir responsabilidad sobre el trabajo de otros."),
    (7,  "A3",  "Disfruto comparar mis resultados con estándares altos."),
    (8,  "F1",  "Me energiza trabajar en equipo."),
    (9,  "P1",  "Me interesa influir en decisiones que afectan al equipo."),
    (10, "A5",  "Me siento cómodo cuando las tareas no miden el desempeño."),
    (11, "F2",  "Procuro mantener relaciones armoniosas en el trabajo."),
    (12, "P2",  "Disfruto asumir liderazgo cuando es necesario."),
    (13, "A6",  "Evito objetivos cuantificables."),
    (14, "F6",  "No me preocupa el clima social del equipo."),
    (15, "P3",  "Me motiva tener impacto positivo en el desempeño de otros."),
    (16, "A4",  "Me esfuerzo por mejorar mis indicadores aun cuando ya cumplí el objetivo."),
    (17, "F3",  "Busco que los demás se sientan incluidos."),
    (18, "P6",  "Prefiero no exponer mis opiniones en foros o reuniones."),
    (19, "A10", "Prefiero actividades donde el resultado dependa más de la suerte que del esfuerzo."),
    (20, "F7",  "Disfruto colaborar con personas de otras áreas."),
    (21, "P9",  "Me incomoda dar retroalimentación directa a otras personas."),
    (22, "A7",  "Me atraen retos con un nivel de dificultad intermedio."),
    (23, "F9",  "Me incomoda pedir o recibir ayuda de otros."),
    (24, "P7",  "Me agrada negociar recursos y prioridades para lograr objetivos."),
    (25, "A11", "Evito competir contra estándares claros para no presionarme."),
    (26, "F10", "Evito participar en actividades que fortalezcan el sentido de equipo."),
    (27, "P8",  "Me motiva impulsar cambios que mejoren procesos."),
    (28, "A8",  "Si no alcanzo una meta, analizo datos para entender por qué."),
    (29, "F8",  "Me esfuerzo por escuchar activamente a mis compañeros."),
    (30, "P10", "Evito asumir la vocería del equipo aunque domine el tema."),
    (31, "A12", "Me conformo con comentarios generales como \"bien hecho\" sin detalles."),
    (32, "F11", "Celebro los logros de mis compañeros como si fueran propios."),
    (33, "P12", "Prefiero limitarme a ejecutar instrucciones aun cuando identifique mejores opciones."),
    (34, "A9",  "Me inquieta no saber cómo se medirá el éxito de una tarea."),
    (35, "F12", "Me cuesta ponerme en el lugar de los demás cuando atraviesan problemas."),
    (36, "P11", "Dedico tiempo a orientar o mentorear a otros."),
]

LIKERT_CHOICES = [1, 2, 3, 4, 5]
LIKERT_HINT = "1=Totalmente en desacuerdo · 5=Totalmente de acuerdo"

A_COLS = [f"A{i}" for i in range(1, 13)]
F_COLS = [f"F{i}" for i in range(1, 13)]
P_COLS = [f"P{i}" for i in range(1, 13)]

# ==========================
#   Persistencia (DB)
# ==========================

def validate_inputs(nombre: str, cedula: str, values: List[Any]) -> List[str]:
    errs = []
    if not nombre or not nombre.strip():
        errs.append("El nombre es obligatorio.")
    if not cedula or not cedula.strip():
        errs.append("La cédula/ID es obligatoria.")
    if any(v is None for v in values):
        errs.append("Debes responder todas las afirmaciones (1–5).")
    return errs

def save_to_db(nombre: str, cedula: str, values: List[Any]) -> None:
    ts = datetime.utcnow().isoformat(timespec="seconds")
    logical_codes = A_COLS + F_COLS + P_COLS
    # Mapear del orden mostrado → columnas lógicas
    code_to_value: Dict[str, Any] = {}
    for idx, (_pos, codigo, _texto) in enumerate(PREGUNTAS):
        code_to_value[codigo] = int(values[idx]) if values[idx] is not None else None

    record: Dict[str, Any] = {"timestamp_utc": ts, "nombre": nombre.strip(), "cedula": cedula.strip()}
    for c in logical_codes:
        record[c.lower()] = code_to_value.get(c)

    df = pd.DataFrame([record])
    df.to_sql("respuestas", con=engine, if_exists="append", index=False)

# ==========================
#   Scoring (import local)
# ==========================
# Usamos tus funciones de score_results.py
# Asegúrate de que score_results.py esté en el mismo directorio del proyecto.
from score_results import score_df  # devuelve (scored_df, reliability_dict)

def load_all_from_db() -> pd.DataFrame:
    q = "SELECT * FROM respuestas ORDER BY timestamp_utc ASC;"
    return pd.read_sql(q, con=engine)

def build_aggregate_bars(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DF 'largo' para gr.BarPlot con promedios de porcentaje por escala.
    Columnas: escala, porcentaje
    """
    cols = ["Logros_porc", "Afiliación_porc", "Poder_porc"]
    present = [c for c in cols if c in scored.columns]
    if not present:
        return pd.DataFrame(columns=["escala", "porcentaje"])
    means = scored[present].mean(numeric_only=True).round(2)
    data = pd.DataFrame({"escala": [c.replace("_porc", "") for c in means.index],
                         "porcentaje": list(means.values)})
    return data

def build_per_user_bars(scored: pd.DataFrame) -> pd.DataFrame:
    """
    DF largo por usuario: nombre, escala, porcentaje
    """
    if scored.empty:
        return pd.DataFrame(columns=["nombre", "escala", "porcentaje"])
    base_cols = [c for c in ["nombre", "cedula", "timestamp_utc"] if c in scored.columns]
    cols = [("Logros", "Logros_porc"), ("Afiliación", "Afiliación_porc"), ("Poder", "Poder_porc")]
    rows = []
    for _, row in scored.iterrows():
        for label, col in cols:
            if col in scored.columns and pd.notna(row[col]):
                rows.append({
                    "nombre": row.get("nombre", ""),
                    "escala": label,
                    "porcentaje": float(row[col])
                })
    return pd.DataFrame(rows)

# ==========================
#   Interfaz Gradio
# ==========================

with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as app:
    gr.Markdown(f"## 📝 {TITLE}")
    with gr.Tab("Cuestionario"):
        gr.Markdown(INSTRUCCION)

        with gr.Row():
            nombre = gr.Textbox(label="Nombre completo *", placeholder="Nombre Apellido", scale=1)
            cedula = gr.Textbox(label="Cédula / ID *", placeholder="12345678", scale=1)

        gr.Markdown("---")
        gr.Markdown("### Responde las 36 afirmaciones")

        question_components: List[gr.components.Radio] = []
        for pos, _codigo, texto in PREGUNTAS:
            comp = gr.Radio(
                choices=LIKERT_CHOICES,
                label=f"{pos}. {texto}",
                info=LIKERT_HINT
            )
            question_components.append(comp)

        with gr.Row():
            submit_btn = gr.Button("Enviar respuestas", variant="primary")
            reset_btn = gr.Button("Limpiar formulario")

        status = gr.Markdown()

        def on_submit(nombre_val: str, cedula_val: str, *answers):
            answers = list(answers)
            errs = validate_inputs(nombre_val, cedula_val, answers)
            if errs:
                return "❌ **Errores:**<br>- " + "<br>- ".join(errs)
            save_to_db(nombre_val, cedula_val, answers)
            return "✅ ¡Respuestas guardadas en la base de datos!"

        submit_btn.click(
            on_submit,
            inputs=[nombre, cedula] + question_components,
            outputs=[status]
        )

        reset_btn.click(
            lambda: ["", ""] + [None for _ in question_components] + [""],
            inputs=None,
            outputs=[nombre, cedula] + question_components + [status],
        )

    with gr.Tab("Panel Administrativo"):
        gr.Markdown("### 🔒 Acceso restringido")
        pwd = gr.Textbox(label="Contraseña de administrador", type="password")
        enter_btn = gr.Button("Entrar")
        admin_status = gr.Markdown()

        # Contenedores de datos/plots
        with gr.Group(visible=False) as admin_block:
            gr.Markdown("#### 👥 Respuestas registradas")
            df_users = gr.Dataframe(
                headers=None, interactive=False, wrap=True, label="Respuestas (raw)"
            )

            gr.Markdown("#### 📊 Calificaciones (por usuario)")
            df_scored = gr.Dataframe(
                headers=None, interactive=False, wrap=True, label="Resultados con puntajes"
            )

            with gr.Row():
                bar_avg = gr.BarPlot(
                    x="escala", y="porcentaje", title="Promedio por escala (%)", height=320
                )
                bar_per_user = gr.BarPlot(
                    x="nombre", y="porcentaje", color="escala",
                    title="Puntajes por usuario y escala (%)", height=320
                )

            download_file = gr.File(label="Descargar calificaciones (CSV)", visible=False)
            refresh_btn = gr.Button("Actualizar datos")

        def admin_login(p):
            admin_pass = os.getenv("ADMIN_PASSWORD", "")
            if not admin_pass:
                return "⚠️ Falta configurar ADMIN_PASSWORD en variables de entorno.", gr.update(visible=False)
            if p != admin_pass:
                return "❌ Contraseña incorrecta.", gr.update(visible=False)
            return "✅ Acceso concedido.", gr.update(visible=True)

        enter_btn.click(admin_login, inputs=[pwd], outputs=[admin_status, admin_block])

        def load_admin_data(_evt=None):
            # Cargar respuestas
            df = load_all_from_db()
            if df.empty:
                # limpiar vistas
                return (
                    "No hay respuestas aún.",
                    gr.update(value=pd.DataFrame()),
                    gr.update(value=pd.DataFrame()),
                    gr.update(value=pd.DataFrame(columns=["escala","porcentaje"])),
                    gr.update(value=pd.DataFrame(columns=["nombre","escala","porcentaje"])),
                    gr.update(visible=False, value=None),
                )

            # Calcular puntajes
            scored, _reliab = score_df(df)

            # CSV temporal para descarga
            csv_path = "/tmp/resultados_scoring.csv"
            scored.to_csv(csv_path, index=False, encoding="utf-8")

            # Graficar
            avg_df = build_aggregate_bars(scored)
            per_user_df = build_per_user_bars(scored)

            return (
                f"✅ {len(df)} respuestas cargadas.",
                df,                         # tabla raw
                scored,                     # tabla con puntajes
                avg_df,                     # barras promedio
                per_user_df,                # barras por usuario
                gr.update(visible=True, value=csv_path),
            )

        # Cargar datos al entrar y al refrescar
        enter_btn.click(load_admin_data, outputs=[admin_status, df_users, df_scored, bar_avg, bar_per_user, download_file])
        refresh_btn.click(load_admin_data, outputs=[admin_status, df_users, df_scored, bar_avg, bar_per_user, download_file])

# ==========================
#   Launcher
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app.queue(max_size=128).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", args.port)),
        show_error=True
    )
