Suponga que la relación entre dos variables $$x, y ∈ R $$ está dada por la ecuación $$y = g(x;\theta) = \theta_0 + \theta_1 x + \theta_2 x^2 + ... + \theta_n X^n $$, donde los $$\theta_i ∈ R $$ son los parámetros del modelo. Considere los datos en el archivo `datosPolinomio.txt`. Cada fila de este archivo contiene una observación de $$x $$ y $$y $$, respectivamente.

a) Como un primer ejercicio de estimación, encuentre un estimado de todos los $$\theta$$, utilizando regresión lineal minimizando el error cuadrático basado en los datos para $$n = 1,2,5,6 $$. Cada solución se debe encontrar de forma analítica utilizando la ecuación encontrada en clase, y no utilizando funciones predefinidas de algún software para resolver el problema de regresión lineal. Para el proceso de estimación identifique la matriz H, el vector Y, y la función de costo en términos de la norma de un vector. Grafique cada una de las tres funciones estimadas (graficada como curvas continuas), y las observaciones traslapadas sobre estas funciones. La figura debe identificar muy bien cada uno de los casos. Para cada modelo, calcule los cuatro indicadores vistos en clase (RMSE, MAL ...). Analice los resultados a partir de las gráficas y los indicadores calculados.

Para resolver el problema descrito, seguiremos estos pasos:

1. **Formulación del Problema:**
   La relación entre las variables $$x$$ y $$y$$ está dada por el modelo polinómico:

$$
   y = g(x; \theta) = \theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_n x^n
$$

donde $$\theta = (\theta_0, \theta_1, \theta_2, \ldots, \theta_n)$$ son los parámetros a estimar.

2. **Preparación de los Datos:**
   Primero, cargamos los datos del archivo `datosPolinomio.txt`. Suponiendo que cada fila del archivo contiene una observación de $$x$$ y $$y $$, almacenamos estos datos en un formato que podamos procesar.

3. **Construcción de la Matriz de Diseño $$H $$ y el Vector $$Y $$:**
   - La matriz de diseño $$H$$ es una matriz en la que cada fila corresponde a una observación y cada columna representa una potencia de $$x$$.
   - El vector $$Y$$ contiene los valores observados de $$y$$.

   Para un modelo polinómico de grado $$n$$, $$H$$ será una matriz de $$m \times (n+1)$$ donde $$m$$ es el número de observaciones.

   Si tenemos $$m$$ observaciones $$(x_i, y_i)$$:

$$
   H = \begin{bmatrix}
   1 & x_1 & x_1^2 & \cdots & x_1^n \\
   1 & x_2 & x_2^2 & \cdots & x_2^n \\
   \vdots & \vdots & \vdots & \ddots & \vdots \\
   1 & x_m & x_m^2 & \cdots & x_m^n
   \end{bmatrix}
$$
   
   y

$$
   Y = \begin{bmatrix}
   y_1 \\
   y_2 \\
   \vdots \\
   y_m
   \end{bmatrix}
$$

5. **Estimación de los Parámetros $$\theta$$:**
   Utilizamos la fórmula de estimación de los parámetros en regresión lineal:

$$
   \hat{\theta} = (H^T H)^{-1} H^T Y
$$

   donde $$\hat{\theta}$$ es el vector de parámetros estimados.

5. **Cálculo de la Función de Costo:**
   La función de costo es el error cuadrático medio (ECM) y se define como:

$$
   J(\theta) = \frac{1}{m} \sum_{i=1}^m (y_i - H_i \theta)^2
$$

   donde $$H_i$$ es la fila $$i$$-ésima de $$H$$.

6. **Graficación de los Resultados:**
   - Para cada valor de $$n$$, graficamos la función estimada y superponemos los datos observados.

7. **Cálculo de Indicadores de Error:**
   - **RMSE (Root Mean Squared Error):** 

$$
     \text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2}
$$

   - **MAL (Mean Absolute Loss):** 

$$
     \text{MAL} = \frac{1}{m} \sum_{i=1}^m |y_i - \hat{y}_i|
$$

   - **R2 (Coeficiente de Determinación):** 

$$
     R^2 = 1 - \frac{\sum_{i=1}^m (y_i - \hat{y}_i)^2}{\sum_{i=1}^m (y_i - \bar{y})^2}
$$

   - **MAE (Mean Absolute Error):**

$$
     \text{MAE} = \frac{1}{m} \sum_{i=1}^m |y_i - \hat{y}_i|
$$

     (Nota: En algunos contextos, MAE y MAL se usan indistintamente).

8. **Análisis de Resultados:**
   - Compara las curvas estimadas con los datos observados.
   - Evalúa los indicadores de error para entender qué tan bien se ajusta el modelo a los datos.

### Ejemplo Detallado (n = 1, 2, 5, 6)

Para ilustrar, supongamos que tienes el archivo `datosPolinomio.txt` con los siguientes datos:

| x   | y   |
|-----|-----|
| 1   | 2   |
| 2   | 3   |
| 3   | 5   |
| 4   | 7   |
| 5   | 11  |

Para cada valor de $$n$$:

1. **n = 1:**
   - $$H = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_5 \end{bmatrix}$$
   - Estimamos $$\theta$$ usando $$\hat{\theta} = (H^T H)^{-1} H^T Y$$.
   - Graficamos la recta estimada y los puntos.

2. **n = 2:**
   - $$H = \begin{bmatrix} 1 & x_1 & x_1^2 \\ 1 & x_2 & x_2^2 \\ \vdots & \vdots & \vdots \\ 1 & x_5 & x_5^2 \end{bmatrix}$$
   - Procedemos igual que en $$n = 1$$ y graficamos la parábola estimada y los puntos.

3. **n = 5 y n = 6:**
   - Similarmente, construimos $$H$$ con términos hasta $$x^5$$ o $$x^6$$ y repetimos el proceso.

### Cálculo y Graficación

El código en Python para realizar estos cálculos podría ser algo así:

```python
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
data = np.loadtxt('datosPolinomio.txt')
x = data[:, 0]
y = data[:, 1]

# Función para ajustar modelo polinómico
def fit_polynomial(x, y, degree):
    H = np.vstack([x**i for i in range(degree + 1)]).T
    theta = np.linalg.inv(H.T @ H) @ H.T @ y
    return theta

# Función para graficar
def plot_fit(x, y, theta, degree, ax):
    H = np.vstack([x**i for i in range(degree + 1)]).T
    x_fit = np.linspace(min(x), max(x), 100)
    H_fit = np.vstack([x_fit**i for i in range(degree + 1)]).T
    y_fit = H_fit @ theta
    ax.plot(x_fit, y_fit, label=f'Degree {degree}')
    ax.scatter(x, y, color='red')
    ax.set_title(f'Polynomial Fit (degree={degree})')
    ax.legend()

# Gráfico para cada grado
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
degrees = [1, 2, 5, 6]

for degree, ax in zip(degrees, axs.flatten()):
    theta = fit_polynomial(x, y, degree)
    plot_fit(x, y, theta, degree, ax)

plt.tight_layout()
plt.show()
```

### Conclusiones
- **Para n = 1 y 2:** Los modelos lineales y cuadráticos suelen ajustarse razonablemente si los datos tienen una tendencia simple.
- **Para n = 5 y 6:** Los modelos de mayor grado pueden sobreajustar los datos y captar ruido.

Estos pasos te permitirán realizar una regresión polinómica y analizar el ajuste del modelo a los datos. La interpretación de los resultados dependerá de cómo se comporten los indicadores de error y las gráficas obtenidas.
