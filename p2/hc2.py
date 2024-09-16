import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Definindo a função a ser maximizada
def f(x):
    return np.exp(-(x[0]**2 + x[1]**2)) + 2 * np.exp(-((x[0] - 1.7)**2 + (x[1] - 1.7)**2))

# Função para gerar candidatos na vizinhança
def generate_neighbor(x_best, epsilon, bounds):
    # Gera um novo candidato dentro da vizinhança definida
    x_cand = np.random.uniform(x_best - epsilon, x_best + epsilon, size=x_best.shape)
    # Certifica que o candidato está dentro dos limites
    return np.clip(x_cand, bounds[0], bounds[1])

# Inicializando o algoritmo de Hill Climbing
def hill_climbing(f, bounds, epsilon, Nmax):
    # Ponto inicial é o limite inferior
    x_best = np.array([bounds[0][0], bounds[1][0]])
    f_best = f(x_best)
    
    # Armazenar todos os pontos candidatos para visualização começando pelo ponto inicial
    all_candidates = [x_best]
    
    # Rodar o algoritmo por Nmax iterações
    for _ in range(Nmax):
        x_cand = generate_neighbor(x_best, epsilon, bounds)
        f_cand = f(x_cand)
        
        # Se o candidato for melhor, atualiza a melhor solução (maximização)
        if f_cand > f_best:
            x_best = x_cand
            f_best = f_cand
        
        # Armazenar o candidato atual
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = (np.array([-2, -2]), np.array([4, 5]))  # Limites ajustados para x1 e x2
epsilon = 0.1  # Perturbação para vizinhança
Nmax = 1000  # Número máximo de iterações
num_rounds = 100  # Número de rodadas

# Listas para armazenar as soluções obtidas em cada rodada
all_solutions = []

# Rodar o algoritmo Hill Climbing para cada uma das 100 rodadas
for _ in range(num_rounds):
    x_opt, f_opt, candidates = hill_climbing(f, bounds, epsilon, Nmax)
    all_solutions.append(x_opt)

# Convertendo soluções para array numpy
all_solutions = np.array(all_solutions)

# Calcular a moda das soluções
mode_x1_result = stats.mode(all_solutions[:, 0], keepdims=True)
mode_x2_result = stats.mode(all_solutions[:, 1], keepdims=True)

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
Z = np.exp(-(X1**2 + X2**2)) + 2 * np.exp(-((X1 - 1.7)**2 + (X2 - 1.7)**2))

# Criando o gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos durante a busca
for sol in all_solutions:
    ax.scatter(sol[0], sol[1], f(sol), color='blue', s=10)

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Hill Climbing - Maximização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

# Visualização da tabela com soluções
fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(14, 10))
ax2_left.axis('tight')
ax2_left.axis('off')
ax2_right.axis('tight')
ax2_right.axis('off')

# Dados para as tabelas formatados com 3 casas decimais
table_data_left = [["Rodada", "x1", "x2"]] + [[i+1, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(all_solutions[:50])]
table_data_right = [["Rodada", "x1", "x2"]] + [[i+51, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(all_solutions[50:])]

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
