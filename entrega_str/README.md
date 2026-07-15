# STR discreto del péndulo invertido (Self-Tuning Regulator)

Entrega del trabajo práctico: regulador STR adaptativo indirecto, íntegramente en
tiempo discreto (RLS + ecuación Diofantina por matriz de Sylvester + ley RST).

## Contenido

- `memoria_str_pendulo.pdf` — memoria técnica.
- `respuesta_pid_pendulo_autotunning.py` — implementación del STR, simulaciones,
  métricas, figuras y las dos demos de autoajuste.
- `test_str_pendulo.py` — suite de validación automática.
- `animate_pendulum_run.py` — (opcional) animación de una corrida guardada.
- `requirements.txt` — dependencias.

## Requisitos

- Python 3.9 o superior.
- Paquetes de `requirements.txt` (numpy, scipy, matplotlib, pandas).

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Correr las simulaciones (genera figuras en
`presentacion_pid_pendulo/imagenes_pid_autotunning/` y CSV en
`saved_runs_pid_autotunning/`, ambos creados automáticamente):

```bash
python respuesta_pid_pendulo_autotunning.py
```

Imprime, por escenario (básico vs robusto), las métricas de desempeño y
estabilidad, y al final las dos demostraciones de autoajuste (carro 50% más
pesado y barra 10% más larga).

## Validación (tests)

```bash
python test_str_pendulo.py          # imprime un resumen y PASS/FAIL
# o con pytest:
pytest -q test_str_pendulo.py
```

## Animación (opcional)

Requiere haber corrido antes el script principal (para tener los CSV). Genera un
`.mp4` de una corrida:

```bash
python animate_pendulum_run.py
```
