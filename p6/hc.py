import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Definindo a função a ser maximizada
def f(x):
    return x[0] * np.sin(4 * np.pi * x[0]) - x[1] * np.sin(4 * np.pi * x[1] + np.pi) + 1

# Função para gerar um novo candidato com perturbação
def perturb(x, epsilon):
    return x + np.random.uniform(-epsilon, epsilon, size=x.shape)

# Inicializando o algoritmo Hill Climbing (HC)
def hc(f, bounds, epsilon, Nmax):
    # Gerar ponto inicial uniformemente dentro dos limites
    x_best = np.random.uniform(bounds[0], bounds[1], 2)
    f_best = f(x_best)
    
    # Armazenar todos os pontos candidatos para visualização
    all_candidates = [x_best]
    
    iteration = 0
    # Rodar o algoritmo por Nmax iterações ou até o número mínimo de iterações
    while iteration < Nmax:
        improvement = False
        for _ in range(20):  # Número de candidatos por iteração
            x_cand = perturb(x_best, epsilon)
            x_cand = np.clip(x_cand, bounds[0], bounds[1])
            f_cand = f(x_cand)
            
            # Se o candidato for melhor, atualiza a melhor solução
            if f_cand > f_best:
                x_best = x_cand
                f_best = f_cand
                improvement = True
                break
        
        # Se não houver melhoria, continue iterando até completar 100 iterações
        if not improvement and iteration >= 100:
            break
        
        # Armazenar o candidato atual
        all_candidates.append(x_best)
        iteration += 1
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = np.array([-1, -1]), np.array([3, 3])
epsilon = 0.1  # Perturbação
Nmax = 1000    # Número máximo de iterações

# Rodar o algoritmo HC
x_opt, f_opt, candidates = hc(f, bounds, epsilon, Nmax)

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
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor máximo da função: f(x) = {f_opt}")
print(f"Moda das coordenadas x1: {mode_x1:.3f} com contagem {count_x1}")
print(f"Moda das coordenadas x2: {mode_x2:.3f} com contagem {count_x2}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0][0], bounds[1][0], 100)
x2_vals = np.linspace(bounds[0][1], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = X1 * np.sin(4 * np.pi * X1) - X2 * np.sin(4 * np.pi * X2 + np.pi) + 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos
ax.scatter(candidates[:, 0], candidates[:, 1], f(candidates.T), color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Hill Climbing - Maximização de f(x1, x2)')
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

#garantia de q o algoritmo vai iterar ao menos 100 vezes