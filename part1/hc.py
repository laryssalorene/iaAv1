import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função de perturbação
def perturb(x, e, x_l, x_u):
    x_cand = np.random.uniform(low=x-e, high=x+e, size=x.shape)
    return np.clip(x_cand, x_l, x_u)

# Função f(x1, x2) = x1^2 + x2^2
def f(x1, x2):
    return x1**2 + x2**2

# Configuração da superfície 3D
x_axis = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(x_axis, x_axis)
Z = f(X, Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Definir ponto inicial
x_opt = np.array([-100, -100])
f_opt = f(x_opt[0], x_opt[1])

# Definir limites
x_l = -100  # Limite inferior
x_u = 100   # Limite superior

# Exibir o ponto inicial
ax.scatter(x_opt[0], x_opt[1], f_opt, marker='x', color='r', s=100)


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

     
        # Verificar se houve melhoria (minimização)
        if f_cand < f_opt:
            x_opt = x_cand
            f_opt = f_cand
            melhoria = True
                

            break
    i += 1

# Exibir ponto final

print(f"Current x_opt: {x_opt}, f_opt: {f_opt}")
print(f"x_opt: {x_opt[0]},{x_opt[1]} ")

ax.scatter(x_opt[0], x_opt[1], f_opt, marker='o', color='g', s=100, linewidth=20)
plt.show()


