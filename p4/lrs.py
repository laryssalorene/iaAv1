import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Definindo a função a ser minimizada
def f(x):
    return (x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10) + \
           (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) + 10)

# Função para gerar um novo candidato com perturbação
def perturb(x, sigma):
    return x + np.random.normal(0, sigma, size=x.shape)

# Inicializando o algoritmo de busca aleatória local (LRS) com critério de parada
def lrs(f, bounds, sigma, Nmax, stop_criteria):
    # Gerar x inicial uniformemente dentro dos limites bounds
    x_best = np.random.uniform(bounds[0], bounds[1], 2)
    f_best = f(x_best)
    
    # Armazenar todos os pontos candidatos para visualização começando pelo ponto inicial
    all_candidates = [x_best]
    
    # Contador de iterações sem melhoria
    no_improvement_count = 0
    
    # Rodar o algoritmo por Nmax iterações com critério de parada por melhoria
    for i in range(Nmax):
        x_cand = perturb(x_best, sigma)
        
        # Verificar se o novo candidato está dentro dos limites (restrição de caixa)/bounds
        x_cand = np.clip(x_cand, bounds[0], bounds[1])
        
        f_cand = f(x_cand)
        
        # Se o candidato for melhor, atualiza a melhor solução
        if f_cand < f_best:
            x_best = x_cand
            f_best = f_cand
            no_improvement_count = 0  # Resetar o contador de sem melhoria
        else:
            no_improvement_count += 1
        
        # Armazenar o melhor ponto encontrado até agora
        all_candidates.append(x_best)
        
        # Critério de parada se não houver melhoria por um número de iterações
        if no_improvement_count >= stop_criteria:
            print(f"Parada antecipada após {i+1} iterações devido a falta de melhorias.")
            break
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = (np.array([-5.12, -5.12]), np.array([5.12, 5.12]))
sigma = 0.1  # Desvio padrão da perturbação
Nmax = 10000  # Número máximo de iterações
stop_criteria = 50  # Número de iterações sem melhoria para parar

# Rodar o algoritmo LRS
x_opt, f_opt, all_candidates = lrs(f, bounds, sigma, Nmax, stop_criteria)

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
Z = (X1**2 - 10 * np.cos(2 * np.pi * X1) + 10) + \
    (X2**2 - 10 * np.cos(2 * np.pi * X2) + 10)

# Criando o gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos durante a busca
for sol in first_100_candidates:
    ax.scatter(sol[0], sol[1], f(sol), color='blue', s=10)

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo")

# Ajustando os rótulos e o título
ax.set_title('LRS - Min de f(x1, x2)')
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
