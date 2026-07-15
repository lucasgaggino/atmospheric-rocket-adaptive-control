"""Suite de validación del STR discreto del péndulo invertido.

Valida:
  1. Correctitud del diseño (la Diofantina reproduce A_lc con el modelo nominal).
  2. Estabilidad del lazo cerrado (todos los polos dentro del círculo unitario)
     con el regulador final, en los 4 escenarios y en ambos modos (básico/robusto).
  3. Acotamiento de la salida (theta) en todos los casos.
  4. Convergencia del regulador RST (std de la ventana final) en modo robusto,
     y mejora respecto del básico en los casos con ruido/perturbación sinusoidal.
  5. Convergencia del estimador RLS al modelo real en modo robusto.

Ejecutable con pytest (`pytest -q test_str_pendulo.py`) o directamente
(`python test_str_pendulo.py`), en cuyo caso imprime un resumen.
"""
import numpy as np

from respuesta_pid_pendulo_autotunning import (
    THETA_NOMINAL, POLOS_LC, Ts,
    design_rst, closed_loop_poles, simulate_closed_loop_str,
)

SEED = 12345
T_SIM = 20.0
INITIAL_THETA = 0.1
WIN = 200  # ventana final (~4 s) para medir convergencia

ESCENARIOS = {
    "Sin_Perturbaciones": dict(),
    "Pert_Sinusoidal": dict(disturbance_func=lambda t: 0.005 * np.sin(2 * np.pi * 0.5 * t)),
    "Pert_Escalon": dict(disturbance_func=lambda t: 0.01 if 6.0 < t < 8.0 else 0.0),
    "Ruido_Medicion": dict(measurement_noise=0.005),
}

_CACHE = {}


def _run(nombre, robust):
    """Simula (con caché) un escenario en el modo pedido y devuelve métricas."""
    key = (nombre, robust)
    if key in _CACHE:
        return _CACHE[key]
    res = simulate_closed_loop_str(
        t_total=T_SIM, ref_func=lambda t: 0.0, initial_theta=INITIAL_THETA,
        robust=robust, seed=SEED, **ESCENARIOS[nombre])
    t, y, u, ref, theta_hist, rst_hist, redesign_k = res
    R_fin = np.concatenate(([1.0], rst_hist[-1, 0:3]))
    S_fin = rst_hist[-1, 3:6]
    metrics = dict(
        y=y, theta_final=theta_hist[-1],
        max_theta=float(np.max(np.abs(y))),
        max_pole=float(np.max(np.abs(closed_loop_poles(THETA_NOMINAL, R_fin, S_fin)))),
        s0_std=float(np.std(rst_hist[-WIN:, 3])),
        t0_std=float(np.std(rst_hist[-WIN:, 6])),
    )
    _CACHE[key] = metrics
    return metrics


# ----------------------------------------------------------------------
# 1. Correctitud del diseño: la Diofantina nominal reproduce A_lc
# ----------------------------------------------------------------------
def test_diophantine_nominal():
    R, S, t0 = design_rst(THETA_NOMINAL)
    poles = np.sort_complex(closed_loop_poles(THETA_NOMINAL, R, S))
    esperados = np.sort_complex(POLOS_LC.astype(complex))
    assert np.allclose(poles, esperados, atol=1e-6), \
        f"polos {poles} != deseados {esperados}"
    assert np.max(np.abs(poles)) < 1.0
    assert np.isfinite(t0)


# ----------------------------------------------------------------------
# 2. Estabilidad del lazo cerrado con el regulador final (ambos modos)
# ----------------------------------------------------------------------
def test_estabilidad_lazo():
    for nombre in ESCENARIOS:
        for robust in (False, True):
            m = _run(nombre, robust)
            assert m["max_pole"] < 1.0, \
                f"{nombre} robust={robust}: max|polo|={m['max_pole']:.4f} >= 1"


# ----------------------------------------------------------------------
# 3. Acotamiento de la salida
# ----------------------------------------------------------------------
def test_theta_acotado():
    for nombre in ESCENARIOS:
        for robust in (False, True):
            m = _run(nombre, robust)
            assert m["max_theta"] <= 0.12, \
                f"{nombre} robust={robust}: |theta|max={m['max_theta']:.4f}"


# ----------------------------------------------------------------------
# 4. Convergencia del regulador RST en modo robusto (+ mejora vs básico)
# ----------------------------------------------------------------------
def test_convergencia_RST_robusto():
    for nombre in ESCENARIOS:
        m = _run(nombre, True)
        assert m["s0_std"] < 20.0, \
            f"{nombre}: s0 no converge (std={m['s0_std']:.3e})"
        assert m["t0_std"] < 0.05, \
            f"{nombre}: t0 no converge (std={m['t0_std']:.3e})"


def test_robusto_mejora_convergencia():
    # En los casos donde el básico no converge, el robusto debe reducir la
    # dispersión del regulador en la ventana final.
    for nombre in ("Pert_Sinusoidal", "Ruido_Medicion"):
        b = _run(nombre, False)["s0_std"]
        r = _run(nombre, True)["s0_std"]
        assert r < b, f"{nombre}: robusto no mejora (std {r:.3e} !< {b:.3e})"


# ----------------------------------------------------------------------
# 5. Convergencia del estimador RLS al modelo real (modo robusto)
# ----------------------------------------------------------------------
def test_convergencia_RLS_robusto():
    a1n, a2n, b1n, b2n = THETA_NOMINAL
    bsum_nom = b1n + b2n
    for nombre in ESCENARIOS:
        a1, a2, b1, b2 = _run(nombre, True)["theta_final"]
        assert abs(a1 - a1n) < 0.05, f"{nombre}: a1={a1:.4f}"
        assert abs(a2 - a2n) < 0.05, f"{nombre}: a2={a2:.4f}"
        # el numerador debe conservar signo y orden de magnitud (va al denom. del diseño)
        assert (b1 + b2) < 0.0, f"{nombre}: b1+b2={b1 + b2:.2e} cambió de signo"
        assert abs((b1 + b2) - bsum_nom) / abs(bsum_nom) < 0.5, \
            f"{nombre}: b1+b2={b1 + b2:.2e} lejos del nominal {bsum_nom:.2e}"


# ----------------------------------------------------------------------
def _resumen():
    print(f"{'Escenario':<20}{'modo':<9}{'|th|max':>9}{'max|polo|':>11}"
          f"{'std(s0)':>11}{'std(t0)':>11}")
    print("-" * 71)
    for nombre in ESCENARIOS:
        for robust in (False, True):
            m = _run(nombre, robust)
            print(f"{nombre:<20}{('robusto' if robust else 'basico'):<9}"
                  f"{m['max_theta']:>9.4f}{m['max_pole']:>11.4f}"
                  f"{m['s0_std']:>11.3e}{m['t0_std']:>11.3e}")


if __name__ == "__main__":
    _resumen()
    print()
    fallas = 0
    for fn in [test_diophantine_nominal, test_estabilidad_lazo, test_theta_acotado,
               test_convergencia_RST_robusto, test_robusto_mejora_convergencia,
               test_convergencia_RLS_robusto]:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            fallas += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{'TODOS OK' if fallas == 0 else str(fallas) + ' test(s) fallaron'}")
    raise SystemExit(fallas)
