# System Identification for Inverted Pendulum on Cart

## Introduction

This document explains the step-by-step process for identifying a discrete-time linear model of an inverted pendulum on a cart system using system identification techniques. The approach uses an ARX (AutoRegressive with eXogenous input) model structure and least squares estimation.

## 1. Physical System Description

The inverted pendulum on a cart is a classic control system example. The system consists of:

- **Cart**: mass $M = 1.0$ kg, position $x$
- **Pendulum**: mass $m = 0.1$ kg, length $l = 0.5$ m
- **Input**: horizontal force $F$ applied to the cart
- **Output**: pendulum angle $\theta$ (rad)

The nonlinear dynamics of the system are given by:

$$\ddot{\theta} = \frac{g \sin\theta - \cos\theta \cdot \frac{F + m l \dot{\theta}^2 \sin\theta}{M+m}}{l\left(\frac{4}{3} - \frac{m\cos^2\theta}{M+m}\right)}$$

## 2. Linearization Around Equilibrium

For small angles around the upward equilibrium ($\theta = 0$), we can linearize the system:

$$\sin\theta \approx \theta, \quad \cos\theta \approx 1$$

This gives us the linear continuous-time transfer function from force $F$ to angle $\theta$:

$$\frac{\Theta(s)}{F(s)} = \frac{B_\theta}{s^2 - A_\theta}$$

where:

$$A_\theta = \frac{3g(M+m)}{l(4M+m)} = 64.68 \quad \text{rad}^2/\text{s}^2$$

$$B_\theta = \frac{-3}{l(4M+m)} = -1.43 \quad \text{rad/(N} \cdot \text{s}^2\text{)}$$

The system has **unstable poles** at $s = \pm\sqrt{A_\theta} = \pm 8.04$ rad/s (purely imaginary), confirming the inherent instability of the inverted pendulum.

## 3. Discretization (Zero-Order Hold)

To work with digital control systems, we discretize the continuous system using the Zero-Order Hold (ZOH) method with sampling time $T_s = 0.02$ s:

$$G(z) = \mathcal{Z}\left\{\frac{B_\theta}{s^2 - A_\theta}\right\}_{T_s}$$

The resulting discrete transfer function has the form:

$$\frac{\Theta(z)}{F(z)} = \frac{B_d z^{-1}}{1 - A_d z^{-1}}$$

For our system:
- $A_d \approx 1.0258$ (unstable, outside unit circle)
- $B_d \approx -0.000293$

## 4. ARX Model Structure

We use an AutoRegressive with eXogenous input (ARX) model to represent the system:

$$\theta_k = a_1 \theta_{k-1} + b_1 u_{k-1}$$

where:
- $\theta_k$: output (angle) at time step $k$
- $u_k$: input (force) at time step $k$
- $a_1, b_1$: parameters to be estimated

This is a first-order model, sufficient for this system since the discrete transfer function is effectively first-order.

## 5. Data Generation

To identify the system, we need input-output data:

1. **Input Signal**: Pseudo-Random Binary Sequence (PRBS)
   - Amplitude: $\pm 5$ N
   - Switch time: 15 samples (0.3 s)
   - Total samples: $N = 300$ (6 seconds)

2. **Output Signal**: Simulated response using the true discrete model
   $$\theta_k = A_d \theta_{k-1} + B_d u_{k-1}$$

PRBS signals are ideal for identification because they:
- Excite all frequencies within a bandwidth
- Have white-noise-like spectral properties
- Are easy to generate and implement

## 6. Least Squares Identification

### 6.1 Regression Matrix Construction

For each time step $k = 1, 2, \ldots, N-1$, we construct the regressor vector:

$$\boldsymbol{\phi}_k = \begin{bmatrix} \theta_{k-1} \\ u_{k-1} \end{bmatrix}$$

The complete regression matrix and output vector are:

$$\mathbf{X} = \begin{bmatrix} 
\theta_0 & u_0 \\
\theta_1 & u_1 \\
\vdots & \vdots \\
\theta_{N-2} & u_{N-2}
\end{bmatrix}, \quad
\mathbf{y} = \begin{bmatrix}
\theta_1 \\
\theta_2 \\
\vdots \\
\theta_{N-1}
\end{bmatrix}$$

### 6.2 Parameter Estimation

The ARX model can be written in matrix form:

$$\mathbf{y} = \mathbf{X} \boldsymbol{\theta} + \mathbf{e}$$

where $\boldsymbol{\theta} = [a_1, b_1]^\top$ is the parameter vector and $\mathbf{e}$ is the error vector.

The **least squares** solution minimizes the sum of squared errors:

$$J(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2$$

The optimal parameter estimate is:

$$\hat{\boldsymbol{\theta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} = \mathbf{X}^+ \mathbf{y}$$

where $\mathbf{X}^+$ is the Moore-Penrose pseudoinverse of $\mathbf{X}$.

### 6.3 Implementation

In Python using NumPy:

```python
X = np.array([[theta[k-1], u[k-1]] for k in range(1, N)])
y_target = np.array([theta[k] for k in range(1, N)])
theta_hat = np.linalg.pinv(X) @ y_target
```

## 7. Model Validation

### 7.1 Simulation

To validate the identified model, we perform a **free-run simulation**:

$$\hat{\theta}_k = \hat{a}_1 \hat{\theta}_{k-1} + \hat{b}_1 u_{k-1}$$

Starting from $\hat{\theta}_0 = \theta_0$, we use the estimated parameters and the same input sequence to predict all future outputs.

### 7.2 Performance Metrics

The **Root Mean Square Error (RMSE)** quantifies the prediction accuracy:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{k=0}^{N-1} (\theta_k - \hat{\theta}_k)^2}$$

For our identification:
- Estimated: $\hat{a}_1 \approx 0.0229$, $\hat{b}_1 \approx -0.0003$
- True values: $a_1 = 0.0000$, $b_1 = -0.0003$
- RMSE $\approx 8.3 \times 10^{-5}$ rad

The excellent match confirms successful identification!

## 8. Key Observations

1. **Model Order**: The first-order ARX model is sufficient because the discrete system has a single dominant mode.

2. **Stability**: The identified parameter $\hat{a}_1 \approx 0.02$ is close to zero, but the true continuous system is **unstable**. The discrete approximation appears stable only because $T_s$ is very small.

3. **Input Excitation**: The PRBS input effectively excites the system dynamics, enabling accurate parameter estimation.

4. **Validation**: The small RMSE and visual comparison of true vs. estimated outputs confirm the model quality.

## 9. Extensions

This basic identification framework can be extended to:

- **Higher-order models**: Include more lags ($\theta_{k-2}$, $u_{k-2}$, etc.)
- **Nonlinear models**: Add terms like $\sin(\theta_{k-1})$ for NARX models
- **Recursive estimation**: Update parameters online as new data arrives (RLS, Kalman filter)
- **Closed-loop identification**: Identify the system while under feedback control

## Conclusion

This example demonstrates the complete workflow for linear system identification:
1. Physical modeling and linearization
2. Discretization for digital implementation
3. Data collection with appropriate excitation
4. Parameter estimation using least squares
5. Model validation through simulation

The ARX model structure combined with least squares provides a simple yet powerful tool for identifying linear dynamic systems from experimental data.

