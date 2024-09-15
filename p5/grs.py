import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função a ser maximizada
def f(x):
    return (x[0] * np.cos(x[0]) / 20) + 2 * np.exp(-x[0]**2 - (x[1] - 1)**2) + 0.01 * x[0] * x[1]

# Inicializando o algoritmo de Busca Aleatória Global (GRS)
def grs(f, bounds, N):
    # Gerar N amostras uniformemente dentro dos limites
    candidates = np.random.uniform(bounds[0], bounds[1], (N, 2))
    
    # Avaliar a função objetivo para todas as amostras
    values = np.array([f(cand) for cand in candidates])
    
    # Encontrar a melhor solução
    best_idx = np.argmax(values)
    x_best = candidates[best_idx]
    f_best = values[best_idx]
    
    return x_best, f_best, candidates

# Parâmetros do problema
bounds = np.array([-10, -10]), np.array([10, 10])
N = 1000  # Número de amostras

# Rodar o algoritmo GRS
x_opt, f_opt, candidates = grs(f, bounds, N)

# Resultados
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor máximo da função: f(x) = {f_opt}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0][0], bounds[1][0], 100)
x2_vals = np.linspace(bounds[0][1], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = (X1 * np.cos(X1) / 20) + 2 * np.exp(-X1**2 - (X2 - 1)**2) + 0.01 * X1 * X2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos
ax.scatter(candidates[:, 0], candidates[:, 1], f(candidates.T), color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Busca Aleatória Global (GRS) - Maximização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

# Definindo o intervalo dos eixos para cobrir o intervalo completo
ax.set_xlim(bounds[0][0], bounds[1][0])
ax.set_ylim(bounds[0][1], bounds[1][1])

plt.show()
