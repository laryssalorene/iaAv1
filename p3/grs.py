import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função a ser minimizada
def f(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - \
           np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + \
           20 + np.e

# Inicializando o algoritmo de Busca Aleatória Global (GRS)
def grs(f, bounds, N, Nmax):
    # Gerar N pontos iniciais uniformemente dentro dos limites
    x_best = np.random.uniform(bounds[0], bounds[1], (N, 2))
    f_best = np.array([f(x) for x in x_best])
    
    # Encontrar o melhor ponto inicial
    min_index = np.argmin(f_best)
    x_best = x_best[min_index]
    f_best = f_best[min_index]
    
    # Lista para armazenar todos os pontos candidatos
    all_candidates = [x_best]
    
    # Rodar o algoritmo por Nmax iterações
    for _ in range(Nmax):
        x_cand = np.random.uniform(bounds[0], bounds[1], (N, 2))
        f_cand = np.array([f(x) for x in x_cand])
        
        # Encontrar o melhor ponto candidato
        min_index = np.argmin(f_cand)
        x_cand_best = x_cand[min_index]
        f_cand_best = f_cand[min_index]
        
        # Se o candidato for melhor, atualiza a melhor solução
        if f_cand_best < f_best:  # Minimização
            x_best = x_cand_best
            f_best = f_cand_best
        
        # Armazenar o melhor ponto encontrado até agora
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = (np.array([-8, -8]), np.array([8, 8]))
N = 50  # Número de candidatos por iteração
Nmax = 1000  # Número máximo de iterações

# Rodar o algoritmo GRS
x_opt, f_opt, candidates = grs(f, bounds, N, Nmax)

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

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos em todas as iterações
ax.scatter(candidates[:, 0], candidates[:, 1], f(candidates.T), color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Busca Aleatória Global (GRS) - Minimização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

# Definindo o intervalo dos eixos para cobrir o intervalo completo
ax.set_xlim(bounds[0][0], bounds[1][0])
ax.set_ylim(bounds[0][1], bounds[1][1])

plt.show()
