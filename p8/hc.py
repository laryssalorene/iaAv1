import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Função objetivo
def f(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

# Hill Climbing
def hill_climbing(f, bounds, epsilon, max_iter, min_iter):
    x_best = np.array([bounds[0][0], bounds[1][0]])
    f_best = f(x_best[0], x_best[1])
    all_candidates = [x_best]
    
    iter_count = 0
    for _ in range(max_iter):
        candidate_found = False
        for _ in range(20):  # 20 tentativas por iteração
            x_cand = x_best + np.random.uniform(-epsilon, epsilon, size=2)
            x_cand = np.clip(x_cand, bounds[0], bounds[1])
            f_cand = f(x_cand[0], x_cand[1])
            if f_cand < f_best:
                x_best = x_cand
                f_best = f_cand
                candidate_found = True
                break
        all_candidates.append(x_best)
        iter_count += 1
        if not candidate_found and iter_count >= min_iter:
            break
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros
bounds = np.array([[-200, 20], [-200, 20]])
epsilon = 0.1
max_iter = 1000
min_iter = 100  # Garantir pelo menos 100 iterações

# Rodar Hill Climbing
x_opt_hc, f_opt_hc, candidates_hc = hill_climbing(f, bounds, epsilon, max_iter, min_iter)

# Encontrar os primeiros 100 candidatos
num_candidates_to_display = min(100, len(candidates_hc))
candidates_to_display = candidates_hc[:num_candidates_to_display]

# Calcular a moda das coordenadas x1 e x2
mode_x1_result = stats.mode(candidates_to_display[:, 0], keepdims=True)
mode_x2_result = stats.mode(candidates_to_display[:, 1], keepdims=True)

mode_x1 = mode_x1_result.mode[0]
count_x1 = mode_x1_result.count[0]
mode_x2 = mode_x2_result.mode[0]
count_x2 = mode_x2_result.count[0]

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
ax.scatter(candidates_hc[:, 0], candidates_hc[:, 1], f(candidates_hc[:, 0], candidates_hc[:, 1]), color='blue', s=10, label="Candidatos HC")
ax.scatter(x_opt_hc[0], x_opt_hc[1], f_opt_hc, color='red', s=50, label="Ótimo HC")
ax.set_title('Hill Climbing')
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
#garantia reforçada de iterar pelo menos 100 vezes devido as 100 rodadas necessarias