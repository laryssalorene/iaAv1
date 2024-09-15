import numpy as np
import matplotlib.pyplot as plt

# Função objetivo
def f(x1, x2):
    term1 = -np.sin(x1) * (np.sin(x1**2 / np.pi)**(2 * 10))
    term2 = -np.sin(x2) * (np.sin(2 * x2**2 / np.pi)**(2 * 10))
    return term1 - term2

# Função de perturbação
def perturb(x1, x2, epsilon):
    x1_new = x1 + np.random.uniform(-epsilon, epsilon)
    x2_new = x2 + np.random.uniform(-epsilon, epsilon)
    return np.clip(x1_new, 0, np.pi), np.clip(x2_new, 0, np.pi)

# Algoritmo Hill Climbing
def hill_climbing(f, bounds, epsilon, max_it):
    x1_best, x2_best = np.random.uniform(bounds[0], bounds[1], 2)
    f_best = f(x1_best, x2_best)
    
    candidates = [(x1_best, x2_best)]
    
    for _ in range(max_it):
        x1_cand, x2_cand = perturb(x1_best, x2_best, epsilon)
        f_cand = f(x1_cand, x2_cand)
        
        if f_cand < f_best:  # Minimização
            x1_best, x2_best = x1_cand, x2_cand
            f_best = f_cand
        
        candidates.append((x1_best, x2_best))
    
    return (x1_best, x2_best), f_best, np.array(candidates)

# Parâmetros do Hill Climbing
bounds = [0, np.pi]
epsilon = 0.1
max_it = 1000

(x_opt, y_opt), f_opt, candidates = hill_climbing(f, bounds, epsilon, max_it)

print(f"Hill Climbing - Solução ótima encontrada: x1 = {x_opt}, x2 = {y_opt}")
print(f"Hill Climbing - Valor mínimo da função: f(x1, x2) = {f_opt}")

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
ax.set_title('Hill Climbing - Minimização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()
