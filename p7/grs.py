import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Função objetivo
def f(x1, x2):
    term1 = -np.sin(x1) * (np.sin(x1**2 / np.pi)**(2 * 10))
    term2 = -np.sin(x2) * (np.sin(2 * x2**2 / np.pi)**(2 * 10))
    return term1 - term2

# Algoritmo Global Random Search
def global_random_search(f, bounds, Nmax):
    # Gerar Nmax pontos iniciais uniformemente dentro dos limites
    x1_best = np.random.uniform(bounds[0], bounds[1], Nmax)
    x2_best = np.random.uniform(bounds[0], bounds[1], Nmax)
    
    # Avaliar a função objetivo para todos os pontos iniciais
    f_best = np.array([f(x1, x2) for x1, x2 in zip(x1_best, x2_best)])
    
    # Encontrar o melhor ponto inicial
    min_index = np.argmin(f_best)
    x1_best, x2_best = x1_best[min_index], x2_best[min_index]
    f_best = f_best[min_index]
    
    # Lista para armazenar todos os pontos candidatos
    all_candidates = [(x1_best, x2_best)]
    
    # Critério de parada
    no_improvement_count = 0
    best_value = f_best
    
    # Rodar o algoritmo por Nmax iterações
    for _ in range(Nmax):
        # Gerar novos candidatos
        x1_cand = np.random.uniform(bounds[0], bounds[1], Nmax)
        x2_cand = np.random.uniform(bounds[0], bounds[1], Nmax)
        
        # Avaliar a função objetivo para os novos candidatos
        f_cand = np.array([f(x1, x2) for x1, x2 in zip(x1_cand, x2_cand)])
        
        # Encontrar o melhor ponto candidato
        min_index = np.argmin(f_cand)
        x1_cand_best, x2_cand_best = x1_cand[min_index], x2_cand[min_index]
        f_cand_best = f_cand[min_index]
        
        # Se o candidato for melhor, atualiza a melhor solução
        if f_cand_best < best_value:  # Minimização
            x1_best, x2_best = x1_cand_best, x2_cand_best
            f_best = f_cand_best
            best_value = f_best
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Parar se não houver melhoria por 1000 iterações
        if no_improvement_count >= 100:
            break
        
        # Armazenar o melhor ponto encontrado até agora
        all_candidates.append((x1_best, x2_best))
    
    return (x1_best, x2_best), f_best, np.array(all_candidates)

# Parâmetros do Global Random Search
bounds = [0, np.pi]
Nmax = 10000 # 10000 demora mto pra rodar; sugestão diminuir pra 1000 q roda rapidin

# Rodar o algoritmo GRS
(x_opt, y_opt), f_opt, candidates = global_random_search(f, bounds, Nmax)

print(f" Solução ótima encontrada: x1 = {x_opt}, x2 = {y_opt}")
print(f"Valor mínimo da função: f(x1, x2) = {f_opt}")

# Calculando a moda das coordenadas x1 e x2 dos primeiros 100 pontos candidatos
mode_x1 = stats.mode(candidates[:100, 0], keepdims=True)
mode_x2 = stats.mode(candidates[:100, 1], keepdims=True)

mode_x1_value = mode_x1.mode[0]
mode_x2_value = mode_x2.mode[0]
count_x1 = mode_x1.count[0]
count_x2 = mode_x2.count[0]

print(f"Moda das coordenadas x1: {mode_x1_value:.3f} com contagem {count_x1}")
print(f"Moda das coordenadas x2: {mode_x2_value:.3f} com contagem {count_x2}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
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
ax.set_title('GRS - Min de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()

# Visualização da tabela com os primeiros 100 pontos candidatos
import pandas as pd

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
