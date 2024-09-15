import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função objetivo
def f(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

# Hill Climbing
def hill_climbing(f, bounds, epsilon, max_iter):
    x_best = np.array([bounds[0][0], bounds[1][0]])
    f_best = f(x_best[0], x_best[1])
    all_candidates = [x_best]
    
    for _ in range(max_iter):
        candidate_found = False
        for _ in range(10):  # 10 tentativas por iteração
            x_cand = x_best + np.random.uniform(-epsilon, epsilon, size=2)
            x_cand = np.clip(x_cand, bounds[0], bounds[1])
            f_cand = f(x_cand[0], x_cand[1])
            if f_cand < f_best:
                x_best = x_cand
                f_best = f_cand
                candidate_found = True
                break
        all_candidates.append(x_best)
        if not candidate_found:
            break
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros
bounds = np.array([[-200, 20], [-200, 20]])
epsilon = 0.1
max_iter = 1000

# Rodar Hill Climbing
x_opt_hc, f_opt_hc, candidates_hc = hill_climbing(f, bounds, epsilon, max_iter)

# Plotagem
x1_vals = np.linspace(bounds[0][0], bounds[0][1], 100)
x2_vals = np.linspace(bounds[1][0], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax.scatter(candidates_hc[:, 0], candidates_hc[:, 1], f(candidates_hc[:, 0], candidates_hc[:, 1]), color='blue', s=10, label="Candidatos HC")
ax.scatter(x_opt_hc[0], x_opt_hc[1], f_opt_hc, color='red', s=50, label="Ótimo HC")
ax.set_title('Hill Climbing')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()
