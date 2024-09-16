import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
    
    iteration = 0
    while iteration < max_it:
        improvement = False
        for _ in range(20):  # Número de candidatos por iteração
            x1_cand, x2_cand = perturb(x1_best, x2_best, epsilon)
            f_cand = f(x1_cand, x2_cand)
            
            if f_cand < f_best:  # Minimização
                x1_best, x2_best = x1_cand, x2_cand
                f_best = f_cand
                improvement = True
                break
        
        if not improvement and iteration >= 100:
            break
        
        candidates.append((x1_best, x2_best))
        iteration += 1
    
    return (x1_best, x2_best), f_best, np.array(candidates)

# Parâmetros do Hill Climbing
bounds = [0, np.pi]
epsilon = 0.1
max_it = 1000

(x_opt, y_opt), f_opt, candidates = hill_climbing(f, bounds, epsilon, max_it)

# Selecionar os primeiros 100 candidatos (ou menos se houver menos de 100 candidatos)
if len(candidates) > 100:
    first_100_candidates = candidates[:100]
else:
    first_100_candidates = candidates

# Calcular a moda das coordenadas
mode_x1_result = stats.mode(first_100_candidates[:, 0], keepdims=True)
mode_x2_result = stats.mode(first_100_candidates[:, 1], keepdims=True)

# Acesso correto aos resultados
mode_x1 = mode_x1_result.mode[0]
count_x1 = mode_x1_result.count[0]
mode_x2 = mode_x2_result.mode[0]
count_x2 = mode_x2_result.count[0]

# Resultados no terminal
print(f"Hill Climbing - Solução ótima encontrada: x1 = {x_opt}, x2 = {y_opt}")
print(f"Hill Climbing - Valor mínimo da função: f(x1, x2) = {f_opt}")
print(f"Moda das coordenadas x1: {mode_x1:.3f} com contagem {count_x1}")
print(f"Moda das coordenadas x2: {mode_x2:.3f} com contagem {count_x2}")

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
ax.scatter(x_opt, y_opt, f(x_opt, y_opt), color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Hill Climbing - Min de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()

# Visualização da tabela com os primeiros 100 valores de candidates
fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(14, 10))
ax2_left.axis('tight')
ax2_left.axis('off')
ax2_right.axis('tight')
ax2_right.axis('off')

# Dados para as tabelas formatados com 3 casas decimais
table_data_left = [["Índice", "x1", "x2"]] + [[i+1, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(first_100_candidates[:50])]
table_data_right = [["Índice", "x1", "x2"]] + [[i+51, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(first_100_candidates[50:])]

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
