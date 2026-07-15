import numpy as np
from matplotlib import pyplot as plt
import os
import math
from scipy.integrate import solve_ivp
import pandas as pd

# ==============================================================================
# STR (SELF-TUNING REGULATOR) ADAPTATIVO INDIRECTO - 100% TIEMPO DISCRETO
# Péndulo invertido en posición vertical (inestable)
#
# Metodología (según ICA_STR.pdf):
#   1. RLS estima A(q), B(q) del modelo ARX en línea.
#   2. Se resuelve la ec. Diofantina (identidad de Bézout) A·R + B·S = A_lc
#      vía matriz de Sylvester, incluyendo un integrador en R (rechazo de
#      perturbación escalón) y SIN cancelar ceros (el cero está cerca de z=-1).
#   3. Se aplica la ley de control RST discreta: R(q)u = T(q)r - S(q)y.
#
# En NINGÚN punto del diseño del regulador se pasa a tiempo continuo.
# ==============================================================================

output_dir = "presentacion_pid_pendulo/imagenes_pid_autotunning"
save_dir = "saved_runs_pid_autotunning"

# ------------------------------------------------------------------
# Parámetros del péndulo (posición vertical - inestable)
# ------------------------------------------------------------------
M = 1.0   # masa del carro [kg]
m = 0.1   # masa del péndulo [kg]
l = 0.5   # semilongitud de la barra [m]
g = 9.81  # gravedad [m/s^2]
Ts = 0.02  # tiempo de muestreo [s]

# Constantes del modelo linealizado - POSICIÓN VERTICAL (inestable)
#   theta_ddot = A_theta * theta + B_theta * F
# A_theta > 0 (equilibrio inestable). B_theta < 0: un F > 0 (empuje del carro)
# reduce theta, por lo que la ganancia de la planta es negativa (coincide con
# el signo usado en pendulo.py y con la ODE no lineal cartpole_ode).
A_theta = 3.0 * g * (M + m) / (l * (4.0 * M + m))  # >0 (inestable)
B_theta = -3.0 / (l * (4.0 * M + m))               # <0 (signo real de la planta)

# Modelo discreto nominal (ARX) obtenido por discretización ZOH ANALÍTICA de
# G(s) = B_theta / (s^2 - A_theta). Se usa SOLO como prior/arranque del STR;
# el diseño del regulador es enteramente discreto (no se usan librerías de
# tiempo continuo ni transformaciones s <-> z).
#
# Para G(s) = B_theta/(s^2 - w^2) con w = sqrt(A_theta), la ZOH da:
#   A(z) = z^2 - 2 cosh(w Ts) z + 1
#   B(z) = [B_theta (cosh(w Ts) - 1)/A_theta] (z + 1)
# es decir A(q) = 1 + a1 q^-1 + a2 q^-2, B(q) = b1 q^-1 + b2 q^-2 con b1 = b2.

def _modelo_arx_nominal():
    w = math.sqrt(A_theta)
    ch = math.cosh(w * Ts)
    a1 = -2.0 * ch
    a2 = 1.0
    b = B_theta * (ch - 1.0) / A_theta
    return np.array([a1, a2, b, b])


THETA_NOMINAL = _modelo_arx_nominal()

# Polos deseados de lazo cerrado, especificados directamente en DISCRETO.
# Se mapean los polos continuos del diseño previo (-3, -3, -10) vía z = e^{s*Ts}
# y se agregan 2 polos auxiliares rápidos para completar el grado 5 requerido.
_POLOS_CONT = np.array([-3.0, -3.0, -10.0])
POLOS_LC = np.concatenate((np.exp(_POLOS_CONT * Ts), [0.2, 0.2]))

# Parámetros de la variante ROBUSTA del STR (ver simulate_closed_loop_str).
# La zona muerta se dimensiona al error de ecuación (el ruido de salida aparece
# amplificado por A(q): su desvío ~ sigma*||[1,a1,a2]|| ~ 2.4 sigma), por eso
# dz_factor ~ 10 equivale a ~4 sigma del error de ecuación.
ROBUST_DEFAULTS = dict(
    dz_factor=10.0,       # zona muerta = max(dz_factor*sigma_ruido, dz_floor)
    dz_floor=1.0e-2,      # piso de zona muerta (bloquea perturbación de salida)
    p_max=1.0e3,          # cota de trace(P) (anti covariance-windup)
    b_min=2.0e-4,         # |b1+b2| mínimo aceptable (|nominal| ~ 5.85e-4)
    t0_max=3.0,           # |t0| máximo aceptable (nominal ~ 0.67)
    s_norm_factor=2.0,    # ||S|| máx = s_norm_factor * ||S_nominal||
    redesign_rel_tol=0.02,  # cambio relativo mínimo de theta para rediseñar
    pole_margin=1e-4,     # exige max|polo LC estimado| < 1 - pole_margin
)


# ==============================================================================
# MODELO NO LINEAL DEL PÉNDULO (planta "real" a controlar)
# ==============================================================================

def cartpole_ode(t, y, F_control):
    """EDOs no lineales del cartpole. Estado: y = [x, xdot, theta, thetadot]."""
    x, xdot, th, thdot = y
    s, c = math.sin(th), math.cos(th)
    temp = (F_control + m * l * thdot * thdot * s) / (M + m)
    denom = l * (4.0 / 3.0 - (m * c * c) / (M + m))
    thddot = (g * s - c * temp) / denom
    xddot = temp - (m * l * thddot * c) / (M + m)
    return [xdot, xddot, thdot, thddot]


# ==============================================================================
# 1. ECUACIÓN DIOFANTINA DISCRETA (MATRIZ DE SYLVESTER)
# ==============================================================================

def solve_diophantine_sylvester(A_bar, B, Alc, deg_R, deg_S):
    """Resuelve la identidad de Bézout discreta:

        A_bar(q^-1) * R_bar(q^-1) + B(q^-1) * S(q^-1) = Alc(q^-1)

    con R_bar mónico (R_bar[0] = 1). Los polinomios se representan como arreglos
    de coeficientes en potencias crecientes de q^-1 (índice 0 = término q^0).

    Incógnitas: R_bar = [1, r1, ..., r_{deg_R}], S = [s0, ..., s_{deg_S}].
    Se arma la matriz de Sylvester (Toeplitz concatenadas) y se resuelve el
    sistema lineal resultante.
    """
    A_bar = np.asarray(A_bar, dtype=float)
    B = np.asarray(B, dtype=float)
    Alc = np.asarray(Alc, dtype=float)

    n_rows = len(Alc)                 # coeficientes de potencia 0 .. deg(Alc)
    n_unk = deg_R + (deg_S + 1)       # r1..r_degR  y  s0..s_degS

    M_syl = np.zeros((n_rows, n_unk))

    # Columnas asociadas a r_i (i = 1..deg_R): aporta A_bar desplazado i lugares
    for i in range(1, deg_R + 1):
        for k in range(len(A_bar)):
            M_syl[i + k, i - 1] += A_bar[k]

    # Columnas asociadas a s_j (j = 0..deg_S): aporta B desplazado j lugares
    off = deg_R
    for j in range(0, deg_S + 1):
        for k in range(len(B)):
            M_syl[j + k, off + j] += B[k]

    # Lado derecho: Alc menos el aporte del término mónico de R_bar (coef 1)
    rhs = Alc.copy()
    for k in range(len(A_bar)):
        rhs[k] -= A_bar[k]

    # La ecuación de potencia 0 es identidad (A_bar[0]*1 = Alc[0] = 1); se omite.
    M_solve = M_syl[1:1 + n_unk, :]
    rhs_solve = rhs[1:1 + n_unk]

    sol = np.linalg.solve(M_solve, rhs_solve)

    R_bar = np.concatenate(([1.0], sol[:deg_R]))
    S = sol[deg_R:]
    return R_bar, S


def design_rst(theta_hat, polos_lc=POLOS_LC):
    """Diseño del regulador RST discreto por colocación de polos, con integrador
    y sin cancelación de ceros.

    theta_hat = [a1, a2, b1, b2] (parámetros ARX estimados).
    Devuelve (R, S, t0) con:
        R = (1 - q^-1) * R_bar   (grado 3, incorpora el integrador)
        S = s0 + s1 q^-1 + s2 q^-2
        T = t0                   (ganancia estática unitaria)
    """
    a1, a2, b1, b2 = theta_hat
    A = np.array([1.0, a1, a2])       # A(q^-1)
    B = np.array([0.0, b1, b2])       # B(q^-1) = b1 q^-1 + b2 q^-2

    # Integrador: se diseña sobre A_bar = A * (1 - q^-1)  (grado 3)
    A_bar = np.convolve(A, [1.0, -1.0])

    # Polinomio característico deseado A_lc(q^-1) (grado 5)
    Alc = np.real(np.poly(polos_lc))  # [1, alc1, ..., alc5]

    # Controlador de mínimo orden: deg(S) = deg(A_bar) - 1 = 2, deg(R_bar) = 2
    R_bar, S = solve_diophantine_sylvester(A_bar, B, Alc, deg_R=2, deg_S=2)

    # R final incorpora el integrador
    R = np.convolve([1.0, -1.0], R_bar)  # grado 3

    # Feedforward para ganancia estática unitaria: y/r|_{q^-1=1} = B(1)T/Alc(1) = 1
    B1 = b1 + b2
    t0 = np.sum(Alc) / B1 if abs(B1) > 1e-12 else 0.0

    return R, S, t0


def closed_loop_poles(theta_plant, R, S):
    """Polos de lazo cerrado (plano z) al aplicar el regulador (R, S) a la planta
    ARX theta_plant = [a1, a2, b1, b2].

    El polinomio característico es A(q)R(q) + B(q)S(q). Devuelve sus raíces; el
    lazo es estable si todas tienen módulo < 1. Se usa tanto para validar el
    diseño (debe reproducir A_lc con el modelo nominal) como para medir el
    corrimiento de polos cuando el regulador se diseña con un modelo degradado.
    """
    a1, a2, b1, b2 = theta_plant
    A = np.array([1.0, a1, a2])
    B = np.array([0.0, b1, b2])
    la = np.convolve(A, R)
    lb = np.convolve(B, S)
    n = max(len(la), len(lb))
    cl = np.zeros(n)
    cl[:len(la)] += la
    cl[:len(lb)] += lb
    return np.roots(cl)


def candidato_valido(R, S, t0, theta_hat, s_norm_max, cfg=ROBUST_DEFAULTS,
                     plant_ref=None):
    """Gate de aceptación de un regulador candidato (modo robusto).

    Rechaza diseños provenientes de estimaciones degradadas:
      - b1+b2 con signo equivocado o magnitud demasiado chica (b va al
        denominador del diseño: amplifica el error);
      - |t0| o ||S|| desproporcionados respecto del nominal;
      - lazo cerrado INESTABLE evaluado sobre la planta de referencia CONFIABLE
        (el prior nominal), no sobre la estimación corrupta: así se garantiza
        que el candidato estabiliza una planta cercana a la real antes de
        aplicarlo. Validar contra el propio estimado ruidoso podía aceptar
        reguladores que en realidad desestabilizan.
    """
    if plant_ref is None:
        plant_ref = THETA_NOMINAL
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(S)) and np.isfinite(t0)):
        return False
    b1, b2 = theta_hat[2], theta_hat[3]
    bsum = b1 + b2
    if bsum >= 0.0 or abs(bsum) < cfg["b_min"]:   # nominal: bsum < 0
        return False
    if abs(t0) > cfg["t0_max"]:
        return False
    if np.linalg.norm(S) > s_norm_max:
        return False
    poles = closed_loop_poles(plant_ref, R, S)
    if np.max(np.abs(poles)) > 1.0 - cfg["pole_margin"]:
        return False
    return True


# ==============================================================================
# 2. CONTROLADOR RST Y ESTIMADOR RLS
# ==============================================================================

class RSTController:
    """Ley de control lineal general (RST) en tiempo discreto:

        R(q) u_k = T(q) r_k - S(q) y_k

    implementada de forma recursiva:

        u_k = (t0 r_k - S·[y_k, y_{k-1}, ...] - R[1:]·[u_{k-1}, u_{k-2}, ...]) / R[0]
    """

    def __init__(self, R, S, t0):
        self.set_params(R, S, t0)
        self.u_hist = np.zeros(max(len(R) - 1, 1))  # [u_{k-1}, u_{k-2}, ...]
        self.y_hist = np.zeros(max(len(S) - 1, 1))  # [y_{k-1}, y_{k-2}, ...]

    def set_params(self, R, S, t0):
        self.R = np.asarray(R, dtype=float)
        self.S = np.asarray(S, dtype=float)
        self.t0 = float(t0)

    def control(self, r, y):
        ny = len(self.S)
        nu = len(self.R)

        yvec = np.concatenate(([y], self.y_hist))[:ny]
        s_term = float(self.S @ yvec)
        r_term = float(self.R[1:] @ self.u_hist[:nu - 1]) if nu > 1 else 0.0

        u = (self.t0 * r - s_term - r_term) / self.R[0]

        # Actualizar historiales
        if len(self.y_hist) > 0:
            self.y_hist = np.concatenate(([y], self.y_hist[:-1]))
        if len(self.u_hist) > 0:
            self.u_hist = np.concatenate(([u], self.u_hist[:-1]))
        return u


def rls_step(theta, P, phi, y_meas, lambda_=0.995, dead_zone=0.0, p_max=None):
    """Un paso de mínimos cuadrados recursivos (RLS) con factor de olvido.

    Extensiones robustas (desactivadas por defecto -> modo básico idéntico):
      - dead_zone: si |err| <= dead_zone no se actualiza (evita adaptar sobre
        ruido/perturbación puros). Devuelve updated=False.
      - p_max: cota superior de trace(P) para frenar el *covariance windup*
        (con λ<1 y baja excitación la covarianza crece sin límite y provoca
        estallidos en la estimación).
    """
    y_pred = phi @ theta
    err = y_meas - y_pred
    if dead_zone > 0.0 and abs(err) <= dead_zone:
        return theta, P, err, False
    Pphi = P @ phi
    denom = lambda_ + phi @ Pphi
    K = Pphi / denom
    theta = theta + K * err
    P = (P - np.outer(K, Pphi)) / lambda_
    if p_max is not None:
        tr = np.trace(P)
        if tr > p_max:
            P = P * (p_max / tr)
    return theta, P, err, True


# ==============================================================================
# 3. SIMULACIÓN EN LAZO CERRADO CON STR ADAPTATIVO
# ==============================================================================

def simulate_closed_loop_str(t_total, ref_func, initial_theta=0.1,
                             disturbance_func=None, measurement_noise=0.0,
                             warmup_k=100, redesign_every=25, lambda_=0.995,
                             excite_thresh=1e-4, robust=False, seed=None,
                             cfg=ROBUST_DEFAULTS):
    """Simula el péndulo no lineal en lazo cerrado con un STR adaptativo indirecto.

    - Arranque: regulador RST diseñado con el modelo ARX nominal (prior).
    - En cada paso: RLS actualiza [a1, a2, b1, b2].
    - Tras el warmup y cada `redesign_every` pasos: se rediseña el RST resolviendo
      la ec. Diofantina con la estimación actual (certainty equivalence).

    Modo `robust=True` (mejoras para que R,S,T converjan con ruido/perturbación):
      - regresor consistente con la salida medida (errores-en-variables);
      - RLS con zona muerta (relativa al ruido) y anti covariance-windup;
      - gate de rediseño (candidato_valido) + rediseño sólo ante cambios
        significativos de theta, para no perseguir el ruido.
    """
    rng = np.random.default_rng(seed)

    N = int(t_total / Ts)
    t_sim = np.arange(N) * Ts

    # Estado no lineal completo: [x, xdot, theta, thetadot]
    y_full = np.zeros((4, N))
    y_full[2, 0] = initial_theta

    y = np.zeros(N)     # salida (theta) verdadera
    ym = np.zeros(N)    # salida medida (con ruido)
    u = np.zeros(N)     # acción de control
    ref = np.zeros(N)   # referencia
    y[0] = initial_theta

    # --- Regulador inicial: diseño con el modelo nominal (prior) para asegurar
    #     estabilidad desde t=0 (la planta es inestable a lazo abierto) ---
    R0, S0, t00 = design_rst(THETA_NOMINAL)
    ctrl_rst = RSTController(R0, S0, t00)
    s_norm_max = cfg["s_norm_factor"] * np.linalg.norm(S0)

    # --- Estimador RLS: arranca en el prior nominal. La planta se estabiliza
    #     rápido y la excitación se desvanece (persistencia de excitación, ver
    #     ICA_STR.pdf), por lo que partir del prior mantiene el estimador bien
    #     condicionado y el certainty-equivalence estable. ---
    theta_hat = THETA_NOMINAL.copy()
    P = 100.0 * np.eye(4)

    # Parámetros del modo robusto
    dead_zone = max(cfg["dz_factor"] * measurement_noise, cfg["dz_floor"]) if robust else 0.0
    p_max = cfg["p_max"] if robust else None
    last_design_theta = None

    # Historiales para graficar
    theta_est_hist = np.zeros((N, 4))          # [a1, a2, b1, b2]
    rst_hist = np.zeros((N, 7))                # [r1, r2, r3, s0, s1, s2, t0]
    redesign_k = None                          # instante del primer rediseño

    for k in range(N):
        theta_est_hist[k] = theta_hat
        rst_hist[k] = np.concatenate((ctrl_rst.R[1:4], ctrl_rst.S[:3], [ctrl_rst.t0]))

        ref[k] = ref_func(t_sim[k])
        y_measured = y[k] + (rng.normal(0, measurement_noise)
                             if measurement_noise > 0.0 else 0.0)
        ym[k] = y_measured

        # --- RLS: y[k] = -a1 y[k-1] - a2 y[k-2] + b1 u[k-1] + b2 u[k-2] ---
        # Modo robusto: regresor con la salida MEDIDA (consistente con el
        # target). Modo básico: regresor con la salida limpia (comportamiento
        # original, se conserva para comparación).
        if k >= 2:
            if robust:
                phi = np.array([-ym[k - 1], -ym[k - 2], u[k - 1], u[k - 2]])
            else:
                phi = np.array([-y[k - 1], -y[k - 2], u[k - 1], u[k - 2]])
            if np.linalg.norm(phi) > excite_thresh:  # freno por baja excitación
                theta_hat, P, _, _ = rls_step(theta_hat, P, phi, y_measured,
                                              lambda_, dead_zone=dead_zone,
                                              p_max=p_max)

        # --- Rediseño periódico del RST (certainty equivalence) ---
        do_redesign = k >= warmup_k and (k - warmup_k) % redesign_every == 0
        if do_redesign and robust and last_design_theta is not None:
            rel = (np.linalg.norm(theta_hat - last_design_theta)
                   / (np.linalg.norm(last_design_theta) + 1e-12))
            if rel < cfg["redesign_rel_tol"]:   # theta casi sin cambios -> congelar
                do_redesign = False
        if do_redesign:
            try:
                R_new, S_new, t0_new = design_rst(theta_hat)
                if robust:
                    ok = candidato_valido(R_new, S_new, t0_new, theta_hat,
                                          s_norm_max, cfg)
                else:
                    ok = (np.all(np.isfinite(R_new)) and np.all(np.isfinite(S_new))
                          and abs(t0_new) < 1e6)
                if ok:
                    ctrl_rst.set_params(R_new, S_new, t0_new)
                    last_design_theta = theta_hat.copy()
                    if redesign_k is None:
                        redesign_k = k
            except np.linalg.LinAlgError:
                pass  # matriz singular: se mantiene el regulador anterior

        # --- Acción de control RST ---
        u_control = ctrl_rst.control(ref[k], y_measured)
        u[k] = u_control

        # --- Guarda de divergencia: si el péndulo "cayó" (|theta| grande) se
        #     detiene la integración para no gastar tiempo en un caso inestable.
        if not np.isfinite(y[k]) or abs(y[k]) > math.pi / 2:
            print(f"  [aviso] divergencia en t={t_sim[k]:.2f}s (|theta|>90deg); "
                  f"se detiene la integración.")
            y[k:] = np.sign(y[k]) * (math.pi / 2) if np.isfinite(y[k]) else np.nan
            for j in range(k, N):
                theta_est_hist[j] = theta_hat
                rst_hist[j] = np.concatenate((ctrl_rst.R[1:4], ctrl_rst.S[:3], [ctrl_rst.t0]))
            break

        # --- Integración del modelo no lineal (ZOH sobre u) ---
        if k < N - 1:
            def ode(t, ys):
                return cartpole_ode(t, ys, u_control)
            sol = solve_ivp(ode, [t_sim[k], t_sim[k + 1]], y_full[:, k],
                            t_eval=[t_sim[k + 1]], method='RK45', rtol=1e-6,
                            max_step=Ts)
            if sol.success:
                y_full[:, k + 1] = sol.y[:, -1]
                if disturbance_func:
                    y_full[2, k + 1] += disturbance_func(t_sim[k + 1])
            else:
                y_full[:, k + 1] = y_full[:, k]
            y[k + 1] = y_full[2, k + 1]

    return t_sim, y, u, ref, theta_est_hist, rst_hist, redesign_k


# ==============================================================================
# 4. GRÁFICOS Y GUARDADO
# ==============================================================================

def plot_sim_results(t, y, u, ref, theta_hist, rst_hist, redesign_k, title, key=None):
    key = key if key is not None else title
    tr = t[redesign_k] if redesign_k is not None else None

    plt.figure(figsize=(12, 8))

    # Respuesta
    plt.subplot(2, 2, 1)
    plt.plot(t, y, 'b-', linewidth=1.5, label='theta')
    plt.plot(t, ref, 'k--', linewidth=1, label='Ref')
    if tr is not None:
        plt.axvline(x=tr, color='m', linestyle=':', label='Rediseño STR')
    plt.ylabel('Theta [rad]')
    plt.title(f'{title} - Respuesta')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    # Acción de control
    plt.subplot(2, 2, 2)
    plt.plot(t, u, 'r-', linewidth=1.0, label='Control u')
    if tr is not None:
        plt.axvline(x=tr, color='m', linestyle=':')
    plt.ylabel('u')
    plt.title('Acción de Control')
    plt.grid(True)

    # Parámetros estimados por RLS
    plt.subplot(2, 2, 3)
    labels = ['$a_1$', '$a_2$', '$b_1$', '$b_2$']
    colors = ['c', 'm', 'y', 'k']
    for i in range(4):
        plt.plot(t, theta_hist[:, i], color=colors[i], label=labels[i], linewidth=1)
    if tr is not None:
        plt.axvline(x=tr, color='r', linestyle='--')
    plt.ylabel('Valor')
    plt.xlabel('t [s]')
    plt.title('Estimación RLS de A(q), B(q)')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small', ncol=2)

    # Coeficientes del regulador RST
    plt.subplot(2, 2, 4)
    rst_labels = ['$r_1$', '$r_2$', '$r_3$', '$s_0$', '$s_1$', '$s_2$']
    for i in range(6):
        plt.plot(t, rst_hist[:, i], label=rst_labels[i], linewidth=1)
    if tr is not None:
        plt.axvline(x=tr, color='r', linestyle='--')
    plt.ylabel('Coef.')
    plt.xlabel('t [s]')
    plt.title('Coeficientes del regulador RST')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small', ncol=3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f'str_{key}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {fname}")
    plt.close()


def plot_comparacion(nombre, res_basico, res_robusto):
    """Superpone las trayectorias del STR básico vs robusto para un escenario:
    respuesta theta, coeficiente s0 (representativo de la ganancia del
    regulador), t0 y el parámetro b1+b2 (numerador, que va al denominador del
    diseño). Deja en evidencia la convergencia del modo robusto."""
    tb, yb, _, _, thb, rstb, _ = res_basico
    tr, yr, _, _, thr, rstr, _ = res_robusto

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(tb, yb, 'r-', lw=1.0, label='básico')
    plt.plot(tr, yr, 'b-', lw=1.0, label='robusto')
    plt.axhline(0, color='k', lw=0.5)
    plt.ylabel('Theta [rad]'); plt.title(f'{nombre} - Respuesta'); plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(2, 2, 2)
    plt.plot(tb, rstb[:, 3], 'r-', lw=1.0, label='$s_0$ básico')
    plt.plot(tr, rstr[:, 3], 'b-', lw=1.0, label='$s_0$ robusto')
    plt.ylabel('$s_0$'); plt.title('Coeficiente $s_0$ del regulador'); plt.grid(True)
    plt.legend(loc='best', fontsize='small')

    plt.subplot(2, 2, 3)
    plt.plot(tb, rstb[:, 6], 'r-', lw=1.0, label='$t_0$ básico')
    plt.plot(tr, rstr[:, 6], 'b-', lw=1.0, label='$t_0$ robusto')
    plt.ylabel('$t_0$'); plt.xlabel('t [s]'); plt.title('Prealimentación $t_0$')
    plt.grid(True); plt.legend(loc='best', fontsize='small')

    plt.subplot(2, 2, 4)
    plt.plot(tb, thb[:, 2] + thb[:, 3], 'r-', lw=1.0, label='$b_1+b_2$ básico')
    plt.plot(tr, thr[:, 2] + thr[:, 3], 'b-', lw=1.0, label='$b_1+b_2$ robusto')
    b_nom = THETA_NOMINAL[2] + THETA_NOMINAL[3]
    plt.axhline(b_nom, color='k', ls='--', lw=0.8, label='nominal')
    plt.ylabel('$b_1+b_2$'); plt.xlabel('t [s]')
    plt.title('Numerador estimado (va al denominador del diseño)')
    plt.grid(True); plt.legend(loc='best', fontsize='small')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f'str_cmp_{nombre}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {fname}")
    plt.close()


def plot_convergencia(nombres, std_basico, std_robusto):
    """Barras del desvío estándar de s0 en la ventana final (últimos ~4 s) por
    escenario, básico vs robusto. Métrica directa de convergencia del RST."""
    x = np.arange(len(nombres))
    w = 0.38
    floor = 1e-14  # piso para poder representar std=0 en escala logarítmica
    sb = np.maximum(np.asarray(std_basico), floor)
    sr = np.maximum(np.asarray(std_robusto), floor)
    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, sb, w, label='básico', color='tab:red')
    plt.bar(x + w / 2, sr, w, label='robusto', color='tab:blue')
    plt.yscale('log')
    plt.ylim(floor / 2, max(sb.max(), 1.0) * 3)
    plt.xticks(x, nombres, rotation=15, ha='right')
    plt.ylabel('std($s_0$) ventana final (log)')
    plt.title('Convergencia del regulador RST: básico vs robusto')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, 'str_convergencia.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {fname}")
    plt.close()


def plot_polos_ceros(theta=THETA_NOMINAL, polos_lc=POLOS_LC):
    """Mapa de polos y ceros DISCRETO (plano z).

    - Lazo abierto: polos del ARX nominal A(z) = z^2 + a1 z + a2 (uno inestable,
      |z|>1) y el cero de B(z) = b1 z + b2 (en z = -b2/b1 ~ -1, no fase mínima).
    - Lazo cerrado: polos deseados (POLOS_LC), todos dentro del círculo unitario.
    Todo el cálculo es discreto; no interviene el tiempo continuo.
    """
    a1, a2, b1, b2 = theta
    polos_la = np.roots([1.0, a1, a2])          # polos de A(z)
    ceros_la = np.roots([b1, b2]) if abs(b1) > 1e-12 else np.array([])

    ang = np.linspace(0, 2 * np.pi, 400)
    plt.figure(figsize=(7, 7))
    plt.plot(np.cos(ang), np.sin(ang), 'k--', alpha=0.6, label='|z| = 1')

    plt.plot(np.real(polos_la), np.imag(polos_la), 'rx', ms=13, mew=2.5,
             label='Polos LA (planta inestable)')
    if len(ceros_la) > 0:
        plt.plot(np.real(ceros_la), np.imag(ceros_la), 'bo', ms=11,
                 mfc='none', mew=2, label='Cero LA (z$\\approx$-1)')
    plt.plot(np.real(polos_lc), np.imag(polos_lc), 'g^', ms=11,
             label='Polos LC deseados ($A_{lc}$)')

    plt.axhline(0, color='gray', lw=0.6)
    plt.axvline(0, color='gray', lw=0.6)
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Mapa de polos y ceros discreto\nLazo abierto (inestable) vs polos de lazo cerrado (STR)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, 'str_polos_ceros.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en {fname}")
    plt.close()


def save_run_data(filename, t, y, u, ref, theta_hist, rst_hist):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        't': t, 'y': y, 'u': u, 'ref': ref,
        'a1_est': theta_hist[:, 0], 'a2_est': theta_hist[:, 1],
        'b1_est': theta_hist[:, 2], 'b2_est': theta_hist[:, 3],
        'r1': rst_hist[:, 0], 'r2': rst_hist[:, 1], 'r3': rst_hist[:, 2],
        's0': rst_hist[:, 3], 's1': rst_hist[:, 4], 's2': rst_hist[:, 5],
        't0': rst_hist[:, 6],
    })
    path = os.path.join(save_dir, filename)
    df.to_csv(path, index=False)
    print(f"Datos guardados en {path}")


# ==============================================================================
# PROGRAMA PRINCIPAL
# ==============================================================================

def main():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print("=== STR ADAPTATIVO INDIRECTO (TIEMPO DISCRETO) - PÉNDULO INVERTIDO ===")
    a1n, a2n, b1n, b2n = THETA_NOMINAL
    print(f"G(z) nominal = ({b1n:.6f} z + {b2n:.6f}) / (z^2 {a1n:+.4f} z {a2n:+.4f})")
    print(f"ARX nominal [a1, a2, b1, b2] = {np.round(THETA_NOMINAL, 6)}")
    print(f"Polos LC discretos deseados  = {np.round(POLOS_LC, 4)}")
    R0, S0, t00 = design_rst(THETA_NOMINAL)
    print(f"Regulador inicial: R = {np.round(R0, 4)}")
    print(f"                   S = {np.round(S0, 4)}")
    print(f"                   t0 = {t00:.4f}\n")

    # Mapa de polos y ceros discreto (lazo abierto vs lazo cerrado deseado)
    plot_polos_ceros()

    T_SIM = 20.0
    INITIAL_THETA = 0.1
    SEED = 12345
    WIN = 200   # ventana final (~4 s) para medir convergencia

    escenarios = [
        ("Sin_Perturbaciones", dict()),
        ("Pert_Sinusoidal", dict(disturbance_func=lambda t: 0.005 * np.sin(2 * np.pi * 0.5 * t))),
        ("Pert_Escalon", dict(disturbance_func=lambda t: 0.01 if 6.0 < t < 8.0 else 0.0)),
        ("Ruido_Medicion", dict(measurement_noise=0.005)),
    ]

    std_basico, std_robusto = [], []

    print(f"{'Escenario':<20}{'modo':<9}{'|th|max':>9}{'max|polo|':>11}"
          f"{'std(s0)fin':>12}{'estable':>9}")
    print("-" * 70)

    for nombre, kwargs in escenarios:
        res_b = simulate_closed_loop_str(
            t_total=T_SIM, ref_func=lambda t: 0.0, initial_theta=INITIAL_THETA,
            robust=False, seed=SEED, **kwargs)
        res_r = simulate_closed_loop_str(
            t_total=T_SIM, ref_func=lambda t: 0.0, initial_theta=INITIAL_THETA,
            robust=True, seed=SEED, **kwargs)

        for modo, res, key, title in [
            ("basico", res_b, nombre, nombre),
            ("robusto", res_r, f"robusto_{nombre}", f"{nombre} (robusto)"),
        ]:
            t, y, u, ref, theta_hist, rst_hist, redesign_k = res
            plot_sim_results(t, y, u, ref, theta_hist, rst_hist, redesign_k, title, key=key)
            save_run_data(f'{nombre.lower()}_{modo}.csv', t, y, u, ref, theta_hist, rst_hist)

            # métricas de validación: polos de LC del regulador FINAL aplicado a
            # la planta de referencia (nominal ~ real), que es lo relevante.
            R_fin = np.concatenate(([1.0], rst_hist[-1, 0:3]))
            S_fin = rst_hist[-1, 3:6]
            poles = closed_loop_poles(THETA_NOMINAL, R_fin, S_fin)
            maxpole = np.max(np.abs(poles))
            s0_std = np.std(rst_hist[-WIN:, 3])
            (std_basico if modo == "basico" else std_robusto).append(s0_std)
            print(f"{nombre:<20}{modo:<9}{np.max(np.abs(y)):>9.4f}{maxpole:>11.4f}"
                  f"{s0_std:>12.3e}{('sí' if maxpole < 1 else 'NO'):>9}")

        plot_comparacion(nombre, res_b, res_r)
        print()

    plot_convergencia([n for n, _ in escenarios], std_basico, std_robusto)


if __name__ == "__main__":
    main()
