import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função a ser minimizada
def f(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - \
           np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + \
           20 + np.e

# Função para gerar um novo candidato com perturbação
def perturb(x, sigma):
    return x + np.random.normal(0, sigma, size=x.shape)

# Inicializando o algoritmo de Busca Local Aleatória (LRS)
def lrs(f, bounds, sigma, Nmax):
    x_best = np.random.uniform(bounds[0], bounds[1], 2)
    f_best = f(x_best)
    
    all_candidates = [x_best]
    
    for _ in range(Nmax):
        x_cand = perturb(x_best, sigma)
        x_cand = np.clip(x_cand, bounds[0], bounds[1])
        
        f_cand = f(x_cand)
        
        if f_cand < f_best:  # Minimização
            x_best = x_cand
            f_best = f_cand
        
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = np.array([-8, -8]), np.array([8, 8])
sigma = 0.1
Nmax = 1000

# Rodar o algoritmo LRS
x_opt, f_opt, candidates = lrs(f, bounds, sigma, Nmax)

# Resultados
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor mínimo da função: f(x) = {f_opt}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0][0], bounds[1][0], 100)
x2_vals = np.linspace(bounds[0][1], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X1**2 + X2**2))) - \
    np.exp(0.5 * (np.cos(2 * np.pi * X1) + np.cos(2 * np.pi * X2))) + \
    20 + np.e

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax.scatter(candidates[:, 0], candidates[:, 1], f(candidates.T), color='blue', s=10, label="Candidatos")
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

ax.set_title('Busca Local Aleatória (LRS) - Minimização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()
