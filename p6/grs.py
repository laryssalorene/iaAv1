import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Função objetivo
def f(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

# Algoritmo Global Random Search (GRS) com critério de parada
def grs(f, bounds, N, Nmax, t):
    # Inicialização
    x_best = np.random.uniform(bounds[0], bounds[1], (N, 2))
    f_best = np.array([f(x[0], x[1]) for x in x_best])
    
    min_index = np.argmax(f_best)
    x_best = x_best[min_index]
    f_best = f_best[min_index]
    
    all_candidates = [x_best]
    no_improvement_count = 0
    
    for _ in range(Nmax):
        # Gerar novos candidatos
        x_cand = np.random.uniform(bounds[0], bounds[1], (N, 2))
        f_cand = np.array([f(x[0], x[1]) for x in x_cand])
        
        # Encontrar o melhor candidato
        min_index = np.argmax(f_cand)
        x_cand_best = x_cand[min_index]
        f_cand_best = f_cand[min_index]
        
        # Atualizar a melhor solução se houver melhoria
        if f_cand_best > f_best:  # Maximização
            x_best = x_cand_best
            f_best = f_cand_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Verificar critério de parada
        if no_improvement_count >= t:
            print(f"Critério de parada atingido após {_} iterações.")
            break
        
        # Armazenar o melhor ponto encontrado até agora
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = np.array([-1, -1]), np.array([3, 3])
N = 50  # Número de candidatos por iteração
Nmax = 10000  # Número máximo de iterações
t = 100  # Número de iterações sem melhoria para critério de parada

# Rodar o algoritmo GRS
x_opt, f_opt, candidates = grs(f, bounds, N, Nmax, t)

# Resultados
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor máximo da função: f(x) = {f_opt}")

# Calculando a moda das coordenadas x1 e x2 dos primeiros 100 candidatos
mode_x1 = stats.mode(candidates[:100, 0], keepdims=True)
mode_x2 = stats.mode(candidates[:100, 1], keepdims=True)

# Acesso correto aos resultados
mode_x1_value = mode_x1.mode[0]
mode_x2_value = mode_x2.mode[0]
count_x1 = mode_x1.count[0]
count_x2 = mode_x2.count[0]

print(f"Moda das coordenadas x1: {mode_x1_value:.3f} com contagem {count_x1}")
print(f"Moda das coordenadas x2: {mode_x2_value:.3f} com contagem {count_x2}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0][0], bounds[1][0], 100)
x2_vals = np.linspace(bounds[0][1], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos
ax.scatter(candidates[:, 0], candidates[:, 1], f(candidates[:, 0], candidates[:, 1]), color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('GRS - Max de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()

# Visualização da tabela com os primeiros 100 pontos candidatos
# Dados para as tabelas formatados com 3 casas decimais
table_data_left = [["Candidato", "x1", "x2"]] + [[i+1, f"{cand[0]:.3f}", f"{cand[1]:.3f}"] for i, cand in enumerate(candidates[:50])]
table_data_right = [["Candidato", "x1", "x2"]] + [[i+51, f"{cand[0]:.3f}", f"{cand[1]:.3f}"] for i, cand in enumerate(candidates[50:100])]

# Criando as tabelas
fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(14, 10))
ax2_left.axis('tight')
ax2_left.axis('off')
ax2_right.axis('tight')
ax2_right.axis('off')

# Adicionando as tabelas
table_left = ax2_left.table(cellText=table_data_left, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table_right = ax2_right.table(cellText=table_data_right, colLabels=None, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

# Ajustando o espaçamento das linhas
for (i, key) in enumerate(table_left.get_celld().keys()):
    cell = table_left.get_celld()[key]
    cell.set_height(0.6)  # Ajusta a altura das linhas

for (i, key) in enumerate(table_right.get_celld().keys()):
    cell = table_right.get_celld()[key]
    cell.set_height(0.6)  # Ajusta a altura das linhas

# Ajustar o layout da figura para garantir que a tabela seja visível
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.show()
