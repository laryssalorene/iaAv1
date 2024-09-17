import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Função objetivo
def f(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

# Local Random Search
def lrs(f, bounds, sigma, max_iter):
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
max_iter = 10000

# Rodar Local Random Search
x_opt_lrs, f_opt_lrs, candidates_lrs = lrs(f, bounds, sigma, max_iter)

# Encontrar os primeiros 100 candidatos
num_candidates_to_display = min(100, len(candidates_lrs))
candidates_to_display = candidates_lrs[:num_candidates_to_display]

# Calcular a moda das coordenadas x1 e x2
mode_x1_result = stats.mode(candidates_to_display[:, 0], keepdims=True)
mode_x2_result = stats.mode(candidates_to_display[:, 1], keepdims=True)

mode_x1 = mode_x1_result.mode[0]
count_x1 = mode_x1_result.count[0]
mode_x2 = mode_x2_result.mode[0]
count_x2 = mode_x2_result.count[0]

# Print no terminal com a solução ótima e valor mínimo da função
print(f"Solução ótima encontrada: x1 = {x_opt_lrs[0]:.1f}, x2 = {x_opt_lrs[1]:.1f}")
print(f"Valor mínimo da função: f(x1, x2) = {f_opt_lrs:.6f}")
print(f"Moda das coordenadas x1: {mode_x1:.3f} com contagem {count_x1}")
print(f"Moda das coordenadas x2: {mode_x2:.3f} com contagem {count_x2}")

# Plotagem da função
x1_vals = np.linspace(bounds[0][0], bounds[0][1], 100)
x2_vals = np.linspace(bounds[1][0], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
ax.scatter(candidates_lrs[:, 0], candidates_lrs[:, 1], f(candidates_lrs[:, 0], candidates_lrs[:, 1]), color='blue', s=10, label="Candidatos LRS")
ax.scatter(x_opt_lrs[0], x_opt_lrs[1], f_opt_lrs, color='red', s=50, label="Ótimo LRS")
ax.set_title('LRS - Min')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()

# Plotagem da tabela
fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(14, 10))
ax2_left.axis('tight')
ax2_left.axis('off')
ax2_right.axis('tight')
ax2_right.axis('off')

# Dados para as tabelas formatados com 3 casas decimais
table_data_left = [["Índice", "x1", "x2"]] + [[i+1, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(candidates_to_display[:50])]
table_data_right = [["Índice", "x1", "x2"]] + [[i+51, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(candidates_to_display[50:])]

# Criando as tabelas
table_left = ax2_left.table(cellText=table_data_left, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table_right = ax2_right.table(cellText=table_data_right, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

# Ajustando o espaçamento das linhas
for (i, key) in enumerate(table_left.get_celld().keys()):
    cell = table_left.get_celld()[key]
    if key[0] == 0 or key[1] == 0:  # Header row
        cell.set_fontsize(10)
    cell.set_height(0.4)  # Ajusta a altura das linhas

for (i, key) in enumerate(table_right.get_celld().keys()):
    cell = table_right.get_celld()[key]
    if key[0] == 0 or key[1] == 0:  # Header row
        cell.set_fontsize(10)
    cell.set_height(0.4)  # Ajusta a altura das linhas

# Ajustar o layout da figura para garantir que a tabela seja visível
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.show()
