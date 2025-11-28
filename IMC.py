import numpy as np

def imc_pid_pendulum(K=1.4634, a=15.7917, beta=0.1, zeta=0.0):
    """
    Ajuste PID por IMC para el péndulo invertido (modelo 2º orden aproximado).

    Parámetros
    ----------
    K    : ganancia de la planta (1.4634)
    a    : valor del término constante en el denominador (≈ 15.7917)
           se usa para obtener la frecuencia natural: s^2 + a
    beta : parámetro IMC (más chico = respuesta más rápida)
    zeta : factor de amortiguamiento asumido del modelo (0 ~ sin amortiguamiento)

    Devuelve
    --------
    Kp, Ki, Kd : ganancias del PID continuo (paralelo)
    """
    # frecuencia natural y constante de tiempo "natural"
    wn = np.sqrt(a)
    tau_n = 1.0 / wn

    # Fórmulas IMC (caso planta 2º orden sin ceros)
    Ki = 1.0 / (2.0 * K * beta)
    Kd = tau_n**2 / (2.0 * K * beta)
    Kp = (tau_n * zeta) / (K * beta) - 1.0 / (4.0 * K)

    return Kp, Ki, Kd

# Ejemplo numérico:
Kp, Ki, Kd = imc_pid_pendulum(beta=0.1, zeta=0.0)
print("Kp =", Kp)
print("Ki =", Ki)
print("Kd =", Kd)