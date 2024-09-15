import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função objetivo
def f(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

# Local Random Search
def local_random_search(f, bounds, sigma, max_iter):
    x_best = np.random.uniform(bounds[0], bounds[1], 2)
    f_best = f(x_best[0], x_best[1])
    all_candidates = [x_best]
    
    for _ in range(max_iter):
        x_cand = x_best + np.random.normal(0, sigma, size=2)
        x_cand = np.clip(x_cand, bounds[0], bounds[1])
        f_cand = f(x_cand[0], x_cand[1])
        if f_cand < f_best:
            x_best = x_cand
            f_best = f_cand
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros
bounds = np.array([[-200, 20], [-200, 20]])
sigma = 0.5
max_iter = 1000

# Rodar Local Random Search
x_opt_lrs, f_opt_lrs, candidates_lrs = local_random_search(f, bounds, sigma, max_iter)

# Plotagem
x1_vals = np.linspace(bounds[0][0], bounds[0][1], 100)
x2_vals = np.linspace(bounds[1][0], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax.scatter(candidates_lrs[:, 0], candidates_lrs[:, 1], f(candidates_lrs[:, 0], candidates_lrs[:, 1]), color='blue', s=10, label="Candidatos LRS")
ax.scatter(x_opt_lrs[0], x_opt_lrs[1], f_opt_lrs, color='red', s=50, label="Ótimo LRS")
ax.set_title('Local Random Search')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()
