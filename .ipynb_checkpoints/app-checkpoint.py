# /home/jupyter/Motivations_test/app.py
# Gradio app for 36-item Likert questionnaire (Forma A).
# Collects: timestamp, name, ID (cédula), A1..A12, F1..F12, P1..P12, and pos_1..pos_36 (display order)
# Run:
#   pip install gradio pandas filelock
#   python app.py --share
# or:
#   python app.py --port 7860

import os
import argparse
from datetime import datetime
from typing import List, Tuple, Dict, Any

import pandas as pd
from filelock import FileLock
import gradio as gr

# ---------------- Config ----------------
BASE_DIR = "/home/jupyter/Motivations_test"
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "respuestas_cuestionario.csv")
LOCK_PATH = CSV_PATH + ".lock"

TITLE = "Cuestionario de Motivaciones (Logros · Afiliación · Poder)"
INSTRUCCION = (
    "Responde según tu trabajo actual. Escala Likert 1–5:\n"
    "1 = Totalmente en desacuerdo · 2 = En desacuerdo · 3 = Ni de acuerdo ni en desacuerdo · "
    "4 = De acuerdo · 5 = Totalmente de acuerdo."
)

# Same order (Forma A) we agreed
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

# -------------- Helpers ---------------
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def save_row(nombre: str, cedula: str, values: List[Any]) -> str:
    """
    values = responses in the same order as PREGUNTAS (36 ints 1..5)
    """
    ensure_dirs()
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    record: Dict[str, Any] = {"timestamp_utc": ts, "nombre": nombre.strip(), "cedula": cedula.strip()}

    # Logical order columns (A1..A12, F1..F12, P1..P12)
    logical_codes = [f"A{i}" for i in range(1, 13)] + [f"F{i}" for i in range(1, 13)] + [f"P{i}" for i in range(1, 13)]
    # Initialize all as None
    for c in logical_codes:
        record[c] = None

    # Map responses (display order → code)
    for idx, (pos, codigo, _texto) in enumerate(PREGUNTAS):
        record[codigo] = int(values[idx]) if values[idx] is not None else None
        record[f"pos_{pos}"] = codigo  # track the presented order

    df = pd.DataFrame([record])

    # Concurrency-safe append
    write_header = not os.path.exists(CSV_PATH)
    with FileLock(LOCK_PATH, timeout=20):
        df.to_csv(CSV_PATH, mode="a", index=False, header=write_header, encoding="utf-8")

    return CSV_PATH

def validate_inputs(nombre: str, cedula: str, values: List[Any]) -> List[str]:
    errs = []
    if not nombre or not nombre.strip():
        errs.append("El nombre es obligatorio.")
    if not cedula or not cedula.strip():
        errs.append("La cédula/ID es obligatoria.")
    # Must answer all 36
    if any(v is None for v in values):
        errs.append("Debes responder todas las afirmaciones (1–5).")
    return errs

# -------------- UI / Gradio --------------
with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"## 📝 {TITLE}")
    gr.Markdown(INSTRUCCION)

    with gr.Row():
        nombre = gr.Textbox(label="Nombre completo *", placeholder="Nombre Apellido", scale=1)
        cedula = gr.Textbox(label="Cédula / ID *", placeholder="12345678", scale=1)

    gr.Markdown("---")
    gr.Markdown("### Responde las 36 afirmaciones")

    question_components: List[gr.components.Radio] = []
    for pos, codigo, texto in PREGUNTAS:
        comp = gr.Radio(
            choices=LIKERT_CHOICES,
            label=f"{pos}. {texto}",
            info=LIKERT_HINT
        )
        question_components.append(comp)

    with gr.Row():
        submit_btn = gr.Button("Enviar respuestas", variant="primary")
        reset_btn = gr.Button("Limpiar formulario")

    status = gr.Markdown()  # feedback for the user

    def on_submit(nombre_val: str, cedula_val: str, *answers):
        answers = list(answers)
        errs = validate_inputs(nombre_val, cedula_val, answers)
        if errs:
            return f"❌ **Errores:**<br>- " + "<br>- ".join(errs)

        path = save_row(nombre_val, cedula_val, answers)
        return f"✅ ¡Respuestas guardadas!<br>Archivo: `{path}`"

    # Wire the buttons
    submit_btn.click(
        on_submit,
        inputs=[nombre, cedula] + question_components,
        outputs=[status]
    )

    # Reset fields to None / empty
    reset_btn.click(
        lambda: ["", ""] + [None for _ in question_components] + [""],
        inputs=None,
        outputs=[nombre, cedula] + question_components + [status],
    )

# -------------- Launcher ---------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Get a public Gradio share link.")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on (if not using --share).")
    args = parser.parse_args()

    # queue() helps with concurrency (100 users is fine)
    demo.queue(max_size=128).launch(
        share=args.share,
        server_name="0.0.0.0",     # allows external access when firewall is open
        server_port=args.port,
        show_error=True
    )
