import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função objetivo
def f(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

# Global Random Search
def global_random_search(f, bounds, N, max_iter):
    x_best = np.random.uniform(bounds[0], bounds[1], (N, 2))
    f_best = np.array([f(x[0], x[1]) for x in x_best])
    
    min_index = np.argmin(f_best)
    x_best = x_best[min_index]
    f_best = f_best[min_index]
    all_candidates = [x_best]
    
    for _ in range(max_iter):
        x_cand = np.random.uniform(bounds[0], bounds[1], (N, 2))
        f_cand = np.array([f(x[0], x[1]) for x in x_cand])
        
        min_index = np.argmin(f_cand)
        x_cand_best = x_cand[min_index]
        f_cand_best = f_cand[min_index]
        
        if f_cand_best < f_best:
            x_best = x_cand_best
            f_best = f_cand_best
        
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros
bounds = np.array([[-200, 20], [-200, 20]])
N = 50
max_iter = 1000

# Rodar Global Random Search
x_opt_grs, f_opt_grs, candidates_grs = global_random_search(f, bounds, N, max_iter)

# Plotagem
x1_vals = np.linspace(bounds[0][0], bounds[0][1], 100)
x2_vals = np.linspace(bounds[1][0], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax.scatter(candidates_grs[:, 0], candidates_grs[:, 1], f(candidates_grs[:, 0], candidates_grs[:, 1]), color='blue', s=10, label="Candidatos GRS")
ax.scatter(x_opt_grs[0], x_opt_grs[1], f_opt_grs, color='red', s=50, label="Ótimo GRS")
ax.set_title('Global Random Search')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()
