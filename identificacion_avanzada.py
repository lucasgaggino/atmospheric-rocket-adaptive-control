import numpy as np
import matplotlib.pyplot as plt
import control as ctl
from scipy import signal
from scipy.integrate import solve_ivp
from scipy import stats
import math
from numpy.linalg import svd

# ========================================
# PARAMETROS DEL SISTEMA FISICO
# ========================================
M = 1.0    # masa del carro [kg]
m = 0.1    # masa del pendulo [kg]
l = 0.5    # longitud del pendulo [m]
g = 9.81   # gravedad [m/s^2]

# Parametros del sistema linealizado
A_theta = 3.0 * g * (M + m) / (l * (4.0 * M + m))  # polo inestable
B_theta = -3.0 / (l * (4.0 * M + m))

# Discretizacion ZOH
Ts = 0.02  # tiempo de muestreo [s]
num_c = [B_theta]
den_c = [1.0, 0.0, -A_theta]
sys_c = ctl.TransferFunction(num_c, den_c)
sys_d = ctl.c2d(sys_c, Ts, method='zoh')

# Obtener coeficientes del modelo discreto
num_d = np.squeeze(sys_d.num)
den_d = np.squeeze(sys_d.den)
A_d = -den_d[1] / den_d[0]  # coeficiente de y[k-1]
B_d = num_d[1] / den_d[0]   # coeficiente de u[k-1]

print("Sistema continuo: polos en ±{:.3f}j".format(np.sqrt(A_theta)))
print("Sistema discreto: A={:.6f}, B={:.6f}".format(A_d, B_d))

# ========================================
# FUNCIONES AVANZADAS DE IDENTIFICACION
# ========================================

def analyze_persistence_of_excitation(u, na=1, nb=1):
    """
    Analiza la persistencia de excitacion de la senal de entrada
    Calcula el numero de condicion de la matriz de autocorrelacion
    """
    N = len(u)
    k_start = max(na, nb)

    # Construir matriz de autocorrelacion
    phi = []
    for k in range(k_start, N):
        row = []
        for i in range(1, na+1):
            row.append(-u[k-i])  # para entrada
        for i in range(1, nb+1):
            row.append(u[k-i])
        phi.append(row)

    phi = np.array(phi)
    if phi.shape[0] < phi.shape[1]:
        return float('inf'), "Menos datos que parametros"

    # Numero de condicion
    cond = np.linalg.cond(phi)
    if cond > 1e12:
        quality = "Mala excitacion"
    elif cond > 1e6:
        quality = "Excitacion moderada"
    else:
        quality = "Buena excitacion"

    return cond, quality

def whiteness_test(residuals, lags=20, alpha=0.05):
    """
    Test de blancura de residuos usando autocorrelacion
    H0: residuos son blancos (autocorrelacion = 0 para lags > 0)
    """
    N = len(residuals)
    if N < lags * 2:
        return None, "Insuficientes datos para test"

    # Calcular autocorrelacion de manera simple
    residuals_centered = residuals - np.mean(residuals)
    autocorr = np.zeros(lags)

    for lag in range(lags):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            autocorr[lag] = np.sum(residuals_centered[:-lag] * residuals_centered[lag:]) / np.sum(residuals_centered**2)

    # Estadistico Q de Ljung-Box simplificado
    Q = N * np.sum(autocorr[1:]**2)

    # p-valor (aproximacion chi-cuadrado)
    try:
        p_value = 1 - stats.chi2.cdf(Q, lags-1)
    except:
        p_value = 0.0

    is_white = p_value > alpha

    return {
        'Q_statistic': Q,
        'p_value': p_value,
        'is_white': is_white,
        'autocorr': autocorr
    }, "Test completado"

def cross_correlation_test(residuals, inputs, lags=20, alpha=0.05):
    """
    Test de correlacion cruzada entre residuos y entradas
    H0: residuos no correlacionados con entradas
    """
    N = len(residuals)
    if N < lags * 2:
        return None, "Insuficientes datos"

    # Correlacion cruzada
    cross_corr = np.correlate(residuals - np.mean(residuals),
                             inputs - np.mean(inputs), mode='full')
    cross_corr = cross_corr[N-1:N+lags] / (np.std(residuals) * np.std(inputs) * N)

    # Para lags positivos, verificar si esta dentro de limites de confianza
    conf_limit = 1.96 / np.sqrt(N)  # 95% confidence

    significant_lags = []
    for lag in range(1, lags):
        if abs(cross_corr[lag]) > conf_limit:
            significant_lags.append(lag)

    is_uncorrelated = len(significant_lags) == 0

    return {
        'cross_corr': cross_corr,
        'significant_lags': significant_lags,
        'is_uncorrelated': is_uncorrelated,
        'conf_limit': conf_limit
    }, "Test completado"

def estimate_parameter_uncertainty(X, residuals, alpha=0.05):
    """
    Estima la incertidumbre de los parametros usando covarianza
    """
    N, p = X.shape
    if N <= p:
        return None, "Insuficientes grados de libertad"

    # Varianza de residuos
    sigma2 = np.sum(residuals**2) / (N - p)

    # Matriz de covarianza
    cov_theta = sigma2 * np.linalg.inv(X.T @ X)

    # Desvios estandar
    se = np.sqrt(np.diag(cov_theta))

    # Intervalos de confianza
    t_crit = stats.t.ppf(1 - alpha/2, N - p)
    ci_lower = se * t_crit
    ci_upper = se * t_crit

    return {
        'covariance': cov_theta,
        'standard_errors': se,
        'confidence_intervals': (ci_lower, ci_upper),
        't_critical': t_crit
    }, "Estimacion completada"

def model_order_selection(y, u, max_order=5):
    """
    Seleccion de orden del modelo usando criterios AIC y BIC
    """
    N = len(y)
    results = []

    for na in range(1, max_order+1):
        for nb in range(1, max_order+1):
            try:
                theta, y_pred = arx_identification(y, u, na=na, nb=nb)
                residuals = y - y_pred

                # Calcular AIC y BIC
                n_params = na + nb
                rss = np.sum(residuals**2)
                aic = N * np.log(rss/N) + 2 * n_params
                bic = N * np.log(rss/N) + n_params * np.log(N)

                results.append({
                    'na': na, 'nb': nb,
                    'n_params': n_params,
                    'aic': aic, 'bic': bic,
                    'rss': rss,
                    'theta': theta
                })
            except:
                continue

    # Encontrar mejores modelos
    if results:
        best_aic = min(results, key=lambda x: x['aic'])
        best_bic = min(results, key=lambda x: x['bic'])

        return {
            'all_results': results,
            'best_aic': best_aic,
            'best_bic': best_bic
        }, "Seleccion completada"
    else:
        return None, "Error en seleccion"

def subspace_identification(y, u, n=2, s=10):
    """
    Identificacion por metodos de subespacio (simplificado N4SID-like)
    """
    N = len(y)
    if N < 2*s + n:
        return None, "Insuficientes datos"

    # Construir matrices Hankel
    Y_p = np.zeros((n, N-2*s))
    Y_f = np.zeros((n, N-2*s))
    U_p = np.zeros((n, N-2*s))
    U_f = np.zeros((n, N-2*s))

    for i in range(n):
        Y_p[i, :] = y[s+i:s+i+(N-2*s)]
        Y_f[i, :] = y[s+i+1:s+i+1+(N-2*s)]
        U_p[i, :] = u[s+i:s+i+(N-2*s)]
        U_f[i, :] = u[s+i+1:s+i+1+(N-2*s)]

    # Matriz extendida
    W_p = np.vstack([U_p, Y_p])

    # SVD
    U, Sigma, Vt = svd(W_p @ Y_f.T / (N-2*s))

    # Orden del sistema
    if len(Sigma) < n:
        return None, "Sistema subdimensionado"

    # Estimacion de matrices del sistema (aproximacion)
    # Esta es una version muy simplificada - en la practica se necesitan mas pasos
    S_inv_sqrt = np.diag(1/np.sqrt(Sigma[:n]))
    U_s = U[:, :n] @ S_inv_sqrt

    # Estimacion basica de A (muy aproximada)
    A_est = U_s.T @ Y_f @ Vt[:n, :].T @ S_inv_sqrt

    return {
        'A_estimated': A_est,
        'singular_values': Sigma,
        'system_order': n
    }, "Identificacion por subespacio completada"

def stability_analysis(theta_arx, na=1, nb=1):
    """
    Analiza la estabilidad del modelo identificado
    """
    # Para ARX, los coeficientes de y[k-i] forman el polinomio caracteristico
    # y[k] = a1*y[k-1] + ... + an*y[k-n] + b1*u[k-1] + ...
    # El polinomio caracteristico es: z^n + a1*z^{n-1} + ... + an = 0
    # Para estabilidad discreta, raices dentro del circulo unidad

    poly_coeffs = np.zeros(na + 1)
    poly_coeffs[0] = 1.0  # z^na
    poly_coeffs[1:na+1] = theta_arx[:na]  # coeficientes a1, a2, ...

    # Calcular raices
    roots = np.roots(poly_coeffs)

    # Verificar estabilidad (|z| < 1)
    stable = np.all(np.abs(roots) < 1.0)

    return {
        'roots': roots,
        'is_stable': stable,
        'magnitudes': np.abs(roots)
    }, "Analisis completado"

# ========================================
# FUNCIONES DE SIMULACION
# ========================================
def cartpole_ode(t, y, u_func, Ts, K=0.0):
    """ODE del sistema no lineal completo con control opcional"""
    x, xdot, th, thdot = y
    # Encontrar u perturbacion actual
    k = int(t / Ts)
    if k >= len(u_func):
        k = len(u_func) - 1
    F_pert = u_func[k]

    # Control proporcional para estabilizacion (negativo para invertir)
    F_control = -K * th  # controlador simple P

    F = F_pert + F_control

    s, c = math.sin(th), math.cos(th)
    temp = (F + m * l * thdot * thdot * s) / (M + m)
    denom = l * (4.0/3.0 - (m * c * c) / (M + m))
    thddot = (g * s - c * temp) / denom
    xddot = temp - (m * l * thddot * c) / (M + m)
    return [xdot, xddot, thdot, thddot]

def simulate_nonlinear(u, Ts, y0=[0.0, 0.0, 0.0, 0.0], K=0.0):
    """Simula el modelo no lineal con control"""
    # Nota: Agregamos control P para estabilizar cerca del equilibrio inestable
    # Esto permite generar datos mas realistas sin divergencia inmediata
    N = len(u)
    t_eval = np.arange(N) * Ts
    sol = solve_ivp(lambda t, y: cartpole_ode(t, y, u, Ts, K=K),
                    [0, t_eval[-1]], y0, t_eval=t_eval)
    return sol.y[2]  # theta

# ========================================
# GENERACION DE DATOS
# ========================================
def generate_prbs(N, amplitude=5.0, switch_time=15):
    """Genera senal PRBS"""
    u = np.zeros(N)
    for i in range(0, N, switch_time):
        u[i:i+switch_time] = np.random.choice([-amplitude, amplitude])
    return u

def add_noise(signal, snr_db=20):
    """Agrega ruido blanco con SNR especificado"""
    # Evitar overflow calculando la potencia de manera mas robusta
    signal_power = np.var(signal)  # usar varianza en lugar de mean(signal^2)
    if signal_power == 0:
        signal_power = 1e-10  # evitar division por cero
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

# ========================================
# METODOS PARAMETRICOS
# ========================================
def arx_identification(y, u, na=1, nb=1):
    """Identificacion ARX por minimos cuadrados"""
    N = len(y)
    k_start = max(na, nb)

    X = []
    y_target = []

    for k in range(k_start, N):
        row = []
        # Terminos AR (salidas pasadas)
        for i in range(1, na+1):
            row.append(-y[k-i])
        # Terminos MA (entradas pasadas)
        for i in range(1, nb+1):
            row.append(u[k-i])
        X.append(row)
        y_target.append(y[k])

    X = np.array(X)
    y_target = np.array(y_target)

    # Minimos cuadrados
    theta_hat = np.linalg.pinv(X) @ y_target

    # Simulacion para validacion
    y_sim = np.zeros(N)
    y_sim[:k_start] = y[:k_start]

    for k in range(k_start, N):
        y_pred = 0
        idx = 0
        # Terminos AR
        for i in range(1, na+1):
            y_pred += theta_hat[idx] * y_sim[k-i]
            idx += 1
        # Terminos MA
        for i in range(1, nb+1):
            y_pred += theta_hat[idx] * u[k-i]
            idx += 1
        y_sim[k] = y_pred

    return theta_hat, y_sim

# ========================================
# ENSAYO AVANZADO
# ========================================
if __name__ == "__main__":
    # Crear directorio para imagenes si no existe
    import os
    if not os.path.exists('ident_imgs'):
        os.makedirs('ident_imgs')

    print("="*70)
    print("ENSAYO AVANZADO DE IDENTIFICACION DE SISTEMAS")
    print("Sistema: Pendulo Invertido - Analisis Completo")
    print("="*70)

    # Generar datos
    np.random.seed(42)
    N = 200  # reducir datos para evitar divergencia rapida en lazo abierto
    t = np.arange(N) * Ts

    # Senal de entrada con buena excitacion
    u = generate_prbs(N, amplitude=8.0, switch_time=8)  # mejor excitacion

    # Simulacion con control moderado
    K_stab = 0.0  # sin control - lazo abierto
    print(f"SISTEMA A LAZO ABIERTO (K={K_stab}) - identificacion directa")
    print("ADVERTENCIA: El sistema diverge rapidamente sin control")
    y = simulate_nonlinear(u, Ts, K=K_stab)

    # Agregar ruido
    clean = y.copy()  # guardar senal limpia para estadisticas
    y = add_noise(y, snr_db=25)

    print(f"Datos generados: N={N}, duracion={N*Ts:.1f}s")
    print(f"Estadisticas de entrada u: media={np.mean(u):.4f}, std={np.std(u):.4f}, min={np.min(u):.4f}, max={np.max(u):.4f}")
    print(f"Estadisticas de salida y (limpia): media={np.mean(clean):.4f}, std={np.std(clean):.4f}, min={np.min(clean):.4f}, max={np.max(clean):.4f}")
    print(f"Estadisticas de salida y (con ruido): media={np.mean(y):.4f}, std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")
    print(f"Relacion senal-ruido (SNR): {10*np.log10(np.var(clean)/np.var(y-clean)):.2f} dB")

    # ========================================
    # ANALISIS DE PERSISTENCIA DE EXCITACION
    # ========================================
    print("\n" + "="*50)
    print("ANALISIS DE PERSISTENCIA DE EXCITACION")
    print("="*50)

    cond, quality = analyze_persistence_of_excitation(u, na=1, nb=1)
    print(".2e")
    print(f"Calidad de excitacion: {quality}")

    if cond > 1e6:
        print("ADVERTENCIA: La senal de entrada puede no tener suficiente persistencia de excitacion")
        print("Considere usar una senal con mas contenido frecuencial")

    # ========================================
    # IDENTIFICACION Y VALIDACION
    # ========================================
    print("\n" + "="*40)
    print("IDENTIFICACION ARX")
    print("="*40)

    theta_arx, y_arx = arx_identification(y, u, na=1, nb=1)
    rmse_arx = np.sqrt(np.mean((y - y_arx)**2))
    print(".6f")
    print(f"Parametros ARX: a1={theta_arx[0]:.6f}, b1={theta_arx[1]:.6f}")
    print(f"Parametros teoricos: A_d={A_d:.6f}, B_d={B_d:.6f}")
    print(f"Diferencias absolutas: |a1_est - A_teor| = {abs(theta_arx[0] - A_d):.6f}")
    print(f"                      |b1_est - B_teor| = {abs(theta_arx[1] - B_d):.6f}")
    print(f"Errores relativos:    {abs(theta_arx[0] - A_d)/abs(A_d)*100:.2f}% para a1")
    print(f"                      {abs(theta_arx[1] - B_d)/abs(B_d)*100:.2f}% para b1")

    # ========================================
    # ANALISIS DE ESTABILIDAD
    # ========================================
    print("\n" + "="*40)
    print("ANALISIS DE ESTABILIDAD")
    print("="*40)

    stability_results, _ = stability_analysis(theta_arx, na=1, nb=1)
    print(f"Raices del polinomio caracteristico: {stability_results['roots']}")
    print(f"Magnitudes: {stability_results['magnitudes']}")
    print(f"¿Sistema estable?: {stability_results['is_stable']}")

    if not stability_results['is_stable']:
        print("ADVERTENCIA: El modelo identificado es inestable")
        print("Esto es esperado para el pendulo invertido sin control")

    # ========================================
    # VALIDACION ESTADISTICA
    # ========================================
    print("\n" + "="*50)
    print("VALIDACION ESTADISTICA DE MODELOS")
    print("="*50)

    residuals = y - y_arx

    # Test de blancura
    whiteness_result, _ = whiteness_test(residuals, lags=20)
    if whiteness_result:
        print("TEST DE BLANCURA:")
        print(".4f")
        print(".4f")
        print(f"¿Residuos son blancos? {whiteness_result['is_white']}")
        print(f"Primeros coeficientes de autocorrelacion: {whiteness_result['autocorr'][:5]}")
        print(f"Varianza de residuos: {np.var(residuals):.6f}")
        print(f"Media de residuos: {np.mean(residuals):.6f}")
        if whiteness_result['is_white']:
            print("[OK] Los residuos pasan el test de blancura")
        else:
            print("[FAIL] Los residuos no pasan el test de blancura - modelo inadecuado")

    # Test de correlacion cruzada
    cross_corr_result, _ = cross_correlation_test(residuals, u, lags=20)
    if cross_corr_result:
        print("\nTEST DE CORRELACION CRUZADA:")
        print(f"Lags significativos: {cross_corr_result['significant_lags']}")
        print(f"¿Residuos no correlacionados con entrada? {cross_corr_result['is_uncorrelated']}")
        if cross_corr_result['is_uncorrelated']:
            print("[OK] Los residuos pasan el test de independencia")
        else:
            print("[FAIL] Los residuos están correlacionados con la entrada - modelo inadecuado")

    # ========================================
    # INCERTIDUMBRE DE PARAMETROS
    # ========================================
    print("\n" + "="*50)
    print("INCERTIDUMBRE DE PARAMETROS")
    print("="*50)

    # Reconstruir matriz X para analisis de incertidumbre
    k_start = 2
    X = []
    for k in range(k_start, N):
        row = [-y[k-1], u[k-1]]  # para ARX(1,1)
        X.append(row)
    X = np.array(X)

    uncertainty_result, _ = estimate_parameter_uncertainty(X, residuals)
    if uncertainty_result:
        print("DESVIOS ESTANDAR:")
        print(f"  a1: ±{uncertainty_result['standard_errors'][0]:.6f}")
        print(f"  b1: ±{uncertainty_result['standard_errors'][1]:.6f}")
        print("INTERVALOS DE CONFIANZA (95%):")
        ci_lower = uncertainty_result['confidence_intervals'][0]
        ci_upper = uncertainty_result['confidence_intervals'][1]
        print(f"  a1: [{theta_arx[0] - ci_lower[0]:.6f}, {theta_arx[0] + ci_upper[0]:.6f}]")
        print(f"  b1: [{theta_arx[1] - ci_lower[1]:.6f}, {theta_arx[1] + ci_upper[1]:.6f}]")

        # Comparar con valores teoricos
        print("COMPARACION CON VALORES TEORICOS:")
        print(".6f")
        print(".6f")
        in_ci_a1 = (A_d >= theta_arx[0] - ci_lower[0] and A_d <= theta_arx[0] + ci_upper[0])
        in_ci_b1 = (B_d >= theta_arx[1] - ci_lower[1] and B_d <= theta_arx[1] + ci_upper[1])
        print(f"¿Valor teorico A_d dentro del CI? {in_ci_a1}")
        print(f"¿Valor teorico B_d dentro del CI? {in_ci_b1}")

    # ========================================
    # SELECCION DE ORDEN DEL MODELO
    # ========================================
    print("\n" + "="*50)
    print("SELECCION DE ORDEN DEL MODELO")
    print("="*50)

    order_selection, _ = model_order_selection(y, u, max_order=3)
    if order_selection:
        print("CRITERIOS DE SELECCION:")
        best_aic = order_selection['best_aic']
        best_bic = order_selection['best_bic']
        print(f"Mejor modelo por AIC: ARX({best_aic['na']},{best_aic['nb']}) - AIC={best_aic['aic']:.2f}")
        print(f"Mejor modelo por BIC: ARX({best_bic['na']},{best_bic['nb']}) - BIC={best_bic['bic']:.2f}")

        if best_aic['na'] == best_bic['na'] and best_aic['nb'] == best_bic['nb']:
            print("[OK] AIC y BIC coinciden en la eleccion del modelo")
        else:
            print("[WARN] AIC y BIC sugieren modelos diferentes")

    # ========================================
    # IDENTIFICACION POR SUBESPACIO
    # ========================================
    print("\n" + "="*50)
    print("IDENTIFICACION POR METODOS DE SUBESPACIO")
    print("="*50)

    try:
        subspace_result, _ = subspace_identification(y, u, n=1, s=10)
        if subspace_result:
            print("Matriz A estimada por subespacio:")
            print(f"{subspace_result['A_estimated']}")
            print(f"Valores singulares: {subspace_result['singular_values'][:3]}")
            print("Nota: Este es un metodo simplificado - en la practica se requieren mas pasos")
        else:
            print("Error en identificacion por subespacio")
    except Exception as e:
        print(f"Error en metodo de subespacio: {e}")
        print("Saltando este metodo por simplicidad")

    # ========================================
    # GRAFICOS DE VALIDACION AVANZADA
    # ========================================
    print("\n" + "="*50)
    print("GENERANDO GRAFICOS DE VALIDACION AVANZADA")
    print("="*50)

    # Grafico de autocorrelacion de residuos
    if whiteness_result:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.stem(range(len(whiteness_result['autocorr'])), whiteness_result['autocorr'])
        plt.axhline(y=1.96/np.sqrt(N), color='r', linestyle='--', alpha=0.7)
        plt.axhline(y=-1.96/np.sqrt(N), color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelación')
        plt.title('Autocorrelación de Residuos')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.stem(range(len(cross_corr_result['cross_corr'])), cross_corr_result['cross_corr'])
        plt.axhline(y=cross_corr_result['conf_limit'], color='r', linestyle='--', alpha=0.7)
        plt.axhline(y=-cross_corr_result['conf_limit'], color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Lag')
        plt.ylabel('Correlación Cruzada')
        plt.title('Correlación Cruzada Residuos vs Entrada')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(y, residuals, 'o', alpha=0.5, markersize=2)
        plt.xlabel('Salida y[k]')
        plt.ylabel('Residuos e[k]')
        plt.title('Residuos vs Salida')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, density=True)
        plt.xlabel('Residuos')
        plt.ylabel('Densidad')
        plt.title('Histograma de Residuos')
        plt.grid(True, alpha=0.3)

        plt.suptitle('Validación Estadística del Modelo ARX')
        plt.tight_layout()
        plt.savefig('ident_imgs/advanced_validation.png', dpi=150, bbox_inches='tight')
        # plt.show()  # Solo guardar, no mostrar

    # Grafico de criterios de informacion
    if order_selection:
        results = order_selection['all_results']
        orders = [f"({r['na']},{r['nb']})" for r in results]
        aic_values = [r['aic'] for r in results]
        bic_values = [r['bic'] for r in results]

        plt.figure(figsize=(10, 6))
        x = range(len(orders))
        plt.bar(x, aic_values, alpha=0.7, label='AIC', width=0.35)
        plt.bar([i+0.35 for i in x], bic_values, alpha=0.7, label='BIC', width=0.35)
        plt.xlabel('Orden del Modelo ARX(na,nb)')
        plt.ylabel('Valor del Criterio')
        plt.title('Selección de Orden del Modelo')
        plt.xticks([i+0.175 for i in x], orders)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ident_imgs/model_order_selection.png', dpi=150, bbox_inches='tight')
        # plt.show()  # Solo guardar, no mostrar

    print("\n" + "="*70)
    print("ANALISIS COMPLETADO")
    print("="*70)
    print("Imagenes guardadas en 'ident_imgs/':")
    print("- advanced_validation.png: Validacion estadistica")
    print("- model_order_selection.png: Seleccion de orden")

    print("\nCONCLUSIONES AVANZADAS:")
    if whiteness_result and whiteness_result['is_white']:
        print("[OK] El modelo pasa validacion estadistica basica")
    else:
        print("[WARN] El modelo no pasa todas las validaciones estadisticas")

    if order_selection and best_aic['na'] == best_bic['na'] and best_aic['nb'] == best_bic['nb']:
        print("[OK] AIC y BIC coinciden en el orden optimo del modelo")
    else:
        print("[WARN] Criterios de informacion sugieren diferentes ordenes")

    if stability_results['is_stable']:
        print("[OK] El modelo identificado es estable")
    else:
        print("[WARN] El modelo identificado es inestable (esperado para pendulo invertido)")

    print("\nEl analisis avanzado proporciona una validacion mas rigurosa del modelo identificado.")
