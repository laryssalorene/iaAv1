import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função objetivo
def f(x1, x2):
    term1 = -np.sin(x1) * (np.sin(x1**2 / np.pi)**(2 * 10))
    term2 = -np.sin(x2) * (np.sin(2 * x2**2 / np.pi)**(2 * 10))
    return term1 - term2

# Algoritmo Global Random Search
def global_random_search(f, bounds, N, Nmax):
    # Gerar N pontos iniciais uniformemente dentro dos limites
    x1_best = np.random.uniform(bounds[0], bounds[1], N)
    x2_best = np.random.uniform(bounds[0], bounds[1], N)
    
    # Avaliar a função objetivo para todos os pontos iniciais
    f_best = np.array([f(x1, x2) for x1, x2 in zip(x1_best, x2_best)])
    
    # Encontrar o melhor ponto inicial
    min_index = np.argmin(f_best)
    x1_best, x2_best = x1_best[min_index], x2_best[min_index]
    f_best = f_best[min_index]
    
    # Lista para armazenar todos os pontos candidatos
    all_candidates = [(x1_best, x2_best)]
    
    # Rodar o algoritmo por Nmax iterações
    for _ in range(Nmax):
        # Gerar N novos candidatos
        x1_cand = np.random.uniform(bounds[0], bounds[1], N)
        x2_cand = np.random.uniform(bounds[0], bounds[1], N)
        
        # Avaliar a função objetivo para os novos candidatos
        f_cand = np.array([f(x1, x2) for x1, x2 in zip(x1_cand, x2_cand)])
        
        # Encontrar o melhor ponto candidato
        min_index = np.argmin(f_cand)
        x1_cand_best, x2_cand_best = x1_cand[min_index], x2_cand[min_index]
        f_cand_best = f_cand[min_index]
        
        # Se o candidato for melhor, atualiza a melhor solução
        if f_cand_best < f_best:  # Minimização
            x1_best, x2_best = x1_cand_best, x2_cand_best
            f_best = f_cand_best
        
        # Armazenar o melhor ponto encontrado até agora
        all_candidates.append((x1_best, x2_best))
    
    return (x1_best, x2_best), f_best, np.array(all_candidates)

# Parâmetros do Global Random Search
bounds = [0, np.pi]
N = 50  # Número de candidatos por iteração
Nmax = 1000  # Número máximo de iterações

# Rodar o algoritmo GRS
(x_opt, y_opt), f_opt, candidates = global_random_search(f, bounds, N, Nmax)

print(f"Global Random Search - Solução ótima encontrada: x1 = {x_opt}, x2 = {y_opt}")
print(f"Global Random Search - Valor mínimo da função: f(x1, x2) = {f_opt}")

# Visualização
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1_vals = np.linspace(bounds[0], bounds[1], 100)
x2_vals = np.linspace(bounds[0], bounds[1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = -np.sin(X1) * (np.sin(X1**2 / np.pi)**(2 * 10)) - np.sin(X2) * (np.sin(2 * X2**2 / np.pi)**(2 * 10))

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos durante a busca
ax.scatter(candidates[:, 0], candidates[:, 1], [f(x1, x2) for x1, x2 in candidates], color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt, y_opt, f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Global Random Search - Minimização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()
