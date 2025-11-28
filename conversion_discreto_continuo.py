import numpy as np
import control as ctrl

def conversion_tustin_manual(a1, a2, b1, b2, Ts):
    """
    Realiza la conversión Discreto -> Continuo usando la aproximación bilineal (Tustin)
    manualmente, sin depender de librerías de control.
    
    H(z) = (b1*z + b2) / (z^2 + a1*z + a2)
    Sustitución: z = (1 + s*alpha) / (1 - s*alpha), con alpha = Ts/2
    
    Retorna coeficientes de G(s) = (num2*s^2 + num1*s + num0) / (den2*s^2 + den1*s + den0)
    """
    alpha = Ts / 2.0
    alpha_sq = alpha**2
    
    num_s2 = -b1 * alpha_sq + b2 * alpha_sq
    num_s1 = -2 * b2 * alpha
    num_s0 = b1 + b2
    
    # Denominador G(s)
    # Proviene de: (1+as)^2 + a1(1+as)(1-as) + a2(1-as)^2
    #            = (1 + 2as + a^2s^2) + a1(1 - a^2s^2) + a2(1 - 2as + a^2s^2)
    den_s2 = alpha_sq + a2 * alpha_sq - a1 * alpha_sq # OJO: a1(1 - a^2s^2) -> -a1*alpha_sq
    # Re-chequeo algebraico: 
    # Termino s^2: 1*alpha^2 - a1*alpha^2 + a2*alpha^2 = alpha^2 * (1 - a1 + a2)
    den_s2 = alpha_sq * (1 - a1 + a2)
    
    # Termino s^1: 2*alpha - 2*a2*alpha = 2*alpha * (1 - a2)
    den_s1 = 2 * alpha * (1 - a2)
    
    # Termino s^0: 1 + a1 + a2
    den_s0 = 1 + a1 + a2
    
    # Normalizar para que el coeficiente de mayor orden del denominador sea 1 (si no es 0)
    if abs(den_s2) > 1e-10:
        k = den_s2
    elif abs(den_s1) > 1e-10:
        k = den_s1
    else:
        k = 1.0
        
    return [num_s2/k, num_s1/k, num_s0/k], [den_s2/k, den_s1/k, den_s0/k]

if __name__ == "__main__":
    a1 = -2.0063
    a2 = 1.0000
    b1 = 0.0003
    b2 = 0.0003
    Ts = 0.02
    
    print(f"Parámetros Discretos: a1={a1}, a2={a2}, b1={b1}, b2={b2}, Ts={Ts}")
    
    # Conversión Manual
    num_c, den_c = conversion_tustin_manual(a1, a2, b1, b2, Ts)
    
    print("\n--- Resultado Conversión Manual (Tustin) ---")
    print(f"Numerador G(s): {num_c}")
    print(f"Denominador G(s): {den_c}")
    
    # Crear objeto para verificar
    sys_c_manual = ctrl.TransferFunction(num_c, den_c)
    print("\nSistema G(s):")
    print(sys_c_manual)
    
    print("\nPolos del sistema continuo obtenido:")
    print(sys_c_manual.poles())
    #1.4634 / (s^2 - 15.7917)