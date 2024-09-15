import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função de perturbação
def perturb(x, e, x_l, x_u):
    x_cand = np.random.uniform(low=x-e, high=x+e, size=x.shape)
    return np.clip(x_cand, x_l, x_u)

# Função f(x1, x2)
def f(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))

# Configuração da superfície 3D
x1_axis = np.linspace(-2, 4, 1000)
x2_axis = np.linspace(-2, 5, 1000)
X, Y = np.meshgrid(x1_axis, x2_axis)
Z = f(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Definir ponto inicial
x_opt = np.array([0.0, 0.0])  # Pode começar em qualquer ponto dentro do domínio
f_opt = f(x_opt[0], x_opt[1])

# Definir limites
x_l = np.array([-2, -2])  # Limite inferior
x_u = np.array([4, 5])    # Limite superior

# Exibir o ponto inicial
ax.scatter(x_opt[0], x_opt[1], f_opt, marker='x', color='r', s=100)
print(f"Current x_opt: {x_opt}, f_opt: {f_opt}")
# Parâmetros do Hill Climbing
max_it = 10000
max_viz = 20
e = 0.1

i = 0
melhoria = True

# Loop de otimização
while i < max_it and melhoria:
    melhoria = False
    for j in range(max_viz):
        # Gerar novo candidato
        x_cand = perturb(x_opt, e, x_l, x_u)
        f_cand = f(x_cand[0], x_cand[1])

        # Verificar se houve melhoria (maximização)
        if f_cand > f_opt:
            x_opt = x_cand
            f_opt = f_cand
            melhoria = True
            break
    i += 1

# Exibir ponto final
print(f"Current x_opt: {x_opt}, f_opt: {f_opt}")
ax.scatter(x_opt[0], x_opt[1], f_opt, marker='x', color='g', s=100)
plt.show()
