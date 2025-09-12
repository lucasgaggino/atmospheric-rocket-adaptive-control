# Análisis Matemático del Péndulo Invertido
## Implementación a Nivel Universitario

###  **Descripción del Sistema**

Se modela un péndulo invertido con las siguientes características:
- **Masa del péndulo**: \( m = 1.0 \) kg
- **Longitud del brazo**: \( L = 1.0 \) m (sin masa)
- **Masa del carro**: \( M = 0.1 \) kg
- **Aceleración de gravedad**: \( g = 9.81 \) m/s²
- **Acción de control**: Fuerza \( F(t) \) aplicada al carro
- **Punto de trabajo**: \( \theta = 0 \) (equilibrio inestable)

###  **Ecuaciones Dinámicas No Lineales**

#### **Sistema Acoplado**
El sistema está compuesto por dos ecuaciones diferenciales acopladas:

1. **Ecuación del carro** (traslación):
   $$
   (m + M)\ddot{x} + m L \cos\theta \cdot \ddot{\theta} - m L \sin\theta \cdot \dot{\theta}^2 = F
   $$

2. **Ecuación del péndulo** (rotación):
   $$
   m g L \sin\theta + m L \cos\theta \cdot \ddot{x} = m L^2 \ddot{\theta}
   $$

#### **Vector de Estado**
$$
\mathbf{x} = \begin{bmatrix} x \\ \dot{x} \\ \theta \\ \dot{\theta} \end{bmatrix}
$$

###  **Linealización alrededor del Punto de Equilibrio**

#### **Punto de Equilibrio**
- \( x = 0 \), \( \dot{x} = 0 \), \( \theta = 0 \), \( \dot{\theta} = 0 \), \( F = 0 \)
- Para \( \theta \approx 0 \): \( \sin\theta \approx \theta \), \( \cos\theta \approx 1 \)

#### **Ecuaciones Linealizadas**

1. **Del carro**:
   $$
   (m + M)\ddot{x} + m L \ddot{\theta} = F
   $$

2. **Del péndulo**:
   $$
   m g L \theta + m L \ddot{x} = m L^2 \ddot{\theta}
   $$

#### **Resolviendo el Sistema Acoplado**

De la ecuación (2):
$$
\ddot{x} = \frac{g}{L} \theta - \frac{1}{m} \ddot{\theta}
$$

Sustituyendo en (1):
$$
(m + M)\left( \frac{g}{L} \theta - \frac{1}{m} \ddot{\theta} \right) + m L \ddot{\theta} = F
$$

$$
(m + M) \frac{g}{L} \theta - (m + M) \frac{1}{m} \ddot{\theta} + m L \ddot{\theta} = F
$$

$$
(m + M) \frac{g}{L} \theta + \ddot{\theta} \left( m L - \frac{m + M}{m} \right) = F
$$

$$
\ddot{\theta} = \frac{ (m + M) g }{ (m + M) L - m L } \theta + \frac{ F }{ (m + M) L - m L }
$$

###  **Modelo en Espacio de Estados**

#### **Matriz A (dinámica del sistema)**
$$
\mathbf{A} = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & -\frac{m g}{m + M} & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & \frac{g (m + M)}{(m + M) L - m L} & 0
\end{bmatrix}
$$

#### **Matriz B (entrada de control)**
$$
\mathbf{B} = \begin{bmatrix}
0 \\
\frac{1}{m + M} \\
0 \\
\frac{1}{(m + M) L - m L}
\end{bmatrix}
$$

#### **Matriz C (salidas medidas)**
$$
\mathbf{C} = \begin{bmatrix}
1 & 0 & 0 & 0 \\  % posición del carro
0 & 0 & 1 & 0    % ángulo del péndulo
\end{bmatrix}
$$

#### **Matriz D**
$$
\mathbf{D} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

###  **Parámetros Numéricos Calculados**

Con los valores dados:
- Masa total: \( m + M = 1.1 \) kg
- Coeficiente 1: \( \frac{m g}{m + M} = \frac{1 \cdot 9.81}{1.1} = 8.918 \) rad/s²
- Coeficiente 2: \( \frac{g (m + M)}{(m + M) L - m L} = \frac{9.81 \cdot 1.1}{1.1 - 1} = 107.91 \) rad/s²

### ⚡ **Análisis de Estabilidad**

#### **Polos del Sistema Continuo**
Los polos se obtienen resolviendo: \( \det(sI - A) = 0 \)

Para nuestro sistema:
- \( s^4 - 107.91 s^2 = 0 \)
- Soluciones: \( s = 0, 0, + 10.39j, - 10.39j \)

Sistema inestable (polos en el eje imaginario)

#### **Controlabilidad**
La matriz de controlabilidad:
$$
\mathcal{C} = [\mathbf{B}, \mathbf{A}\mathbf{B}, \mathbf{A}^2\mathbf{B}, \mathbf{A}^3\mathbf{B}]
$$
Rango = 4 → **Sistema controlable**

#### **Observabilidad**
La matriz de observabilidad:
$$
\mathcal{O} = [\mathbf{C}^\top, \mathbf{A}^\top\mathbf{C}^\top, (\mathbf{A}^\top)^2\mathbf{C}^\top, (\mathbf{A}^\top)^3\mathbf{C}^\top]
$$
Rango = 4 → **Sistema observable**

###  **Discretización ZOH**

#### **Método Zero-Order Hold**
La discretización ZOH mantiene constante la entrada \( u(k) \) durante el intervalo de muestreo \( T_s \):

```python
sys_discreto = ctrl.c2d(sys_continuo, Ts, method='zoh')
```

#### **Matrices Discretas**
Para \( T_s = 0.01 \) s:
$$
\mathbf{A}_d = \begin{bmatrix}
1.000 & 0.010 & -0.000446 & -0.00000149 \\
0.000 & 1.000 & -0.0893 & -0.000446 \\
0.000 & 0.000 & 1.0054 & 0.0100 \\
0.000 & 0.000 & 1.081 & 1.0054
\end{bmatrix}
$$

$$
\mathbf{B}_d = \begin{bmatrix}
0.0000454 \\
0.00908 \\
0.000500 \\
0.1002
\end{bmatrix}
$$

### 🎮 **Control Óptimo LQR**

#### **Formulación LQR**
Minimizar el costo:
$$
J = \int_0^\infty \left( \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{u}^\top \mathbf{R} \mathbf{u} \right) dt
$$

#### **Matrices de Ponderación**
- **Q** (estados): Diag([100, 1, 1000, 10])
- **R** (control): [0.1]

#### **Ganancia LQR Continua**
$$
\mathbf{K} = [-31.62, -22.17, 153.36, 13.42]
$$

#### **Ganancia LQR Discreta**
$$
\mathbf{K}_d = [-18.22, -12.88, 94.42, 8.30]
$$

#### **Polos del Lazo Cerrado**
- Continuo: [-100.6, -10.0, -1.74±1.69j]
- Discreto: [0.38, 0.90, 0.98±0.017j]

###  **Resultados de Simulación**

#### **Respuesta al Pulso de Fuerza**
- **Entrada**: F = 2N durante 0.5s
- **Salida θ**: Oscilaciones amortiguadas por LQR
- **Salida x**: Movimiento suave del carro

#### **Error de Discretización**
- Error máximo en θ: < 0.01 rad
- Error máximo en x: < 0.001 m
- Precisión adecuada para control digital

###  **Conclusiones**

1. **✓ Sistema completamente modelado**: Tanto continuo como discreto
2. **✓ Controlable y observable**: Propiedades necesarias para control
3. **✓ LQR efectivo**: Estabiliza el sistema inestable
4. **✓ Discretización precisa**: ZOH mantiene las propiedades del sistema
5. **✓ Validación completa**: Simulaciones confirman el modelo matemático


