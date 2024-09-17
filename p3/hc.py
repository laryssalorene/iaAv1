import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Definindo a função a ser minimizada
def f(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - \
           np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + \
           20 + np.e

# Função para gerar candidatos na vizinhança
def generate_neighbor(x_best, epsilon, bounds):
    x_cand = np.random.uniform(x_best - epsilon, x_best + epsilon, size=x_best.shape)
    return np.clip(x_cand, bounds[0], bounds[1])

# Inicializando o algoritmo de Hill Climbing
def hill_climbing(f, bounds, epsilon, Nmax):
    x_best = np.array([bounds[0][0], bounds[1][0]])
    f_best = f(x_best)
    
    all_candidates = [x_best]
    
    for _ in range(Nmax):
        for _ in range(20):  # Adicionado para gerar 20 vizinhos
            x_cand = generate_neighbor(x_best, epsilon, bounds)
            f_cand = f(x_cand)
            
            if f_cand < f_best:  # Minimização
                x_best = x_cand
                f_best = f_cand
                break
        
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = (np.array([-8, -8]), np.array([8, 8]))
epsilon = 0.1
Nmax = 10000

# Rodar o algoritmo Hill Climbing uma vez
x_opt, f_opt, all_candidates = hill_climbing(f, bounds, epsilon, Nmax)

# Utilizando os primeiros 100 valores
if len(all_candidates) > 100:
    first_100_candidates = all_candidates[:100]
else:
    first_100_candidates = all_candidates

# Calcular a moda das soluções
mode_x1_result = stats.mode(first_100_candidates[:, 0], keepdims=True)
mode_x2_result = stats.mode(first_100_candidates[:, 1], keepdims=True)

# Acesso correto aos resultados
mode_x1 = mode_x1_result.mode[0]
count_x1 = mode_x1_result.count[0]
mode_x2 = mode_x2_result.mode[0]
count_x2 = mode_x2_result.count[0]

# Resultados no terminal
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor mínimo da função: f(x) = {f_opt}")
print(f"Moda das coordenadas x1: {mode_x1:.3f} com contagem {count_x1}")
print(f"Moda das coordenadas x2: {mode_x2:.3f} com contagem {count_x2}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0][0], bounds[1][0], 100)
x2_vals = np.linspace(bounds[0][1], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X1**2 + X2**2))) - \
    np.exp(0.5 * (np.cos(2 * np.pi * X1) + np.cos(2 * np.pi * X2))) + \
    20 + np.e

# Criando o gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo")

# Ajustando os rótulos e o título
ax.set_title('Hill Climbing - Min de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

# Definindo o intervalo dos eixos para cobrir o intervalo completo
ax.set_xlim(bounds[0][0], bounds[1][0])
ax.set_ylim(bounds[0][1], bounds[1][1])

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
