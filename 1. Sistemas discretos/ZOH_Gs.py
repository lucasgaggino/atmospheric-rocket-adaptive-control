# %% [markdown]
# # Discretización de una transferencia G(s) por ZOH
# 
# Para un sistema dado por un motor,
# 
# - Obtener el modelo de la dinámica expresado como una transferencia en $G(s)$ considerando el momento de inercia del rotor y un rozamiento hidrodinámico.
# 
# - Obtener la función de transferencia H(z) del sistema discretizado por ZOH.
# 
# - Con estos modelos calcular las transferencia en python-control, proponiendo valores para las constantes 
# 
# - Validar la discretización obtenida
# 
# - Qué particularidades se observan en las dinámicas de G(s) y H(z)

# %%
import sys


import sympy as sp
import control as ctl
import numpy as np
from matplotlib import pyplot as plt

# %% [markdown]
# # Descripción del sistema en G(s) y discretización

# %%
# Inicialización
sp.init_printing()
s, z, T = sp.symbols('s z T', real=True)
Km, taum = sp.symbols('K_m tau_m', real=True, positive=True)

# Planta de tiempo continuo (un motor)
G = Km / ( s*(taum*s + 1))
print("Planta de tiempo continuo\nG(s) =")
display(G)

# %%
# Respuesta al escalón
Y = G/s
print("\nRespuesta al escalón\nY(s) = G(s)/s =")
display(Y)

# Expresión a la que le vamos a sacar residuos
exp = Y / (1 - sp.exp(s*T) * z**(-1))

# Buscamos los polos de G/s
den = sp.denom(Y)
p_s = sp.roots(den, s)  # Esto te da s=0 (orden 2) y s=-1/taum

# Calculamos los residuos
residuos = []
for pole, mult in p_s.items():
    res = sp.residue(exp, s, pole)
    residuos.append(res)

# Sumamos residuos
Hz = (1 - z**(-1)) * sum(residuos)

print("\nPlanta discretizada con ZOH a partir de G(s)\nH(z) =")
display(sp.simplify(Hz))


# %% [markdown]
# # Validación
# Convertimos el sistema a una tf de python-control y luego evaluamos la respuesta al escalón comparando con la respuesta de G(s)

# %%
def sympy_to_tf(expr, z, Ts=None):
    # Extraemos los polinomios en z del numerador y el denominador
    num_sym, den_sym = sp.fraction(sp.simplify(expr))
    num_poly = sp.Poly(sp.expand(num_sym), z)
    den_poly = sp.Poly(sp.expand(den_sym), z)

    # Convierto coeficientes a array de flotantes
    num_coeffs = np.array([float(c.evalf()) for c in num_poly.all_coeffs()])
    den_coeffs = np.array([float(c.evalf()) for c in den_poly.all_coeffs()])

    # Queremos que el coeficiente del exponente más alto del denominador sea 1
    num_coeffs = num_coeffs/den_coeffs[0]
    den_coeffs = den_coeffs/den_coeffs[0]

    # Armamos el sistema en python-control. Si Ts no está definido, entonces es un sistema de tiempo continuo
    if Ts is None:
      return ctl.TransferFunction(num_coeffs, den_coeffs)
    else:
      return ctl.TransferFunction(num_coeffs, den_coeffs, Ts)


# Reemplazamos valores numéricos
Km_val = 2
taum_val = 0.5
Ts = 0.1

G_num = G.subs({Km: Km_val, taum: taum_val})
print("Planta de tiempo continuo\nG(s) =")
display(G_num)


Hz_num = Hz.subs({Km: Km_val, taum: taum_val, T: Ts})
Hz_num = sp.simplify(Hz_num)
print("\nPlanta de tiempo discreto\nH(z) =")
display(Hz_num)

# Obtengo los modelos tf de ambas plantas
tf_c = sympy_to_tf(G_num,s)
print("\nSistema continuo\n",tf_c)

tf_d = sympy_to_tf(Hz_num,z,Ts)
print("\nSistema discreto\n",tf_d)

# A fin de comparar los resultados discretizo usando la toolbox por c2d
tf_d2 = ctl.c2d(tf_c,Ts,method='zoh')
print("\nSistema discreto 2\n",tf_d2)


# %%

# Tiempo de simulación
t_final = 2.0
Ts = tf_d.dt  # tiempo de muestreo

# Impulso continuo
t_cont, y_cont = ctl.step_response(tf_c, T=np.linspace(0, t_final, 1000))

# Impulso discreto
t_disc, y_disc = ctl.step_response(tf_d, T=np.arange(0, t_final, Ts))

# Graficar
plt.figure(figsize=(10,5))
plt.plot(t_cont, y_cont, label="Respuesta de G(s)", linewidth=2)
plt.stem(t_disc, y_disc, linefmt='r-', markerfmt='ro', basefmt=' ', label="Respuesta de H(z)")
#plt.step(t_disc, y_disc, 'o-', where='post', label="Respuesta escalón (discreta)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Salida")
plt.title("Comparación de respuestas al escalón")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# # Ensayamos un control simple

# %%
# Polos de G
ctl.pzmap(tf_c, plot=True,title="Polos y ceros del sistema continuo")
print("\nPolos del sistema continuo:")
print(tf_c.poles())


# %%
# Polos del sistema discreto

ctl.pzmap(tf_d, plot=True, grid=True, title="Polos y ceros del sistema discreto H(z)")
print("\nPolos del sistema discreto:",tf_d.poles())
print("\nCeros del sistema discreto:",tf_d.zeros())

print("\nMapeo los polos continuos: ",np.exp(tf_c.poles()*tf_d.dt))



# %%
# Al parecer con un control proporcional aplicado al sistema discretizado podríamos obtener un sistema inestable!
ctl.rlocus(tf_d, kvect=np.linspace(0, 20, 500))
plt.title("Lugar de las raíces del sistema discreto")
plt.xlabel("Parte real")
plt.ylabel("Parte imaginaria")
plt.grid(True)
plt.show()

# %%
# Probamos un control proporcional con distintas ganancias
plt.figure(figsize=(10,5))

kp_s = [0.5,1,2,5,10]
for kp in kp_s:
  Gcl = ctl.feedback(kp*tf_d,1)
  t,y = ctl.step_response(Gcl,T=np.arange(0,5,Ts))
  plt.plot(t,y,'o-',label=f"kp={kp}")

plt.title("Respuesta al escalón con control proporcional")
plt.xlabel("Tiempo [s]")
plt.ylabel("Salida")
plt.legend()
plt.grid()
plt.show()


# %%
# En el sistema de tiempo continuo no tenemos este efecto
ctl.rlocus(tf_c)

plt.figure(figsize=(10,5))
kp_s = [0.5,1,2,5,100]
for kp in kp_s:
  Gcl = ctl.feedback(kp*tf_c,1)
  t,y = ctl.step_response(Gcl,T=np.linspace(0,5,100))
  plt.plot(t,y,'o-',label=f"kp={kp}")

plt.title("Respuesta al escalón con control proporcional")
plt.xlabel("Tiempo [s]")
plt.ylabel("Salida")
plt.legend()
plt.grid()
plt.show()


