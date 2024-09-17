import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Definindo a função a ser maximizada
def f(x):
    return np.exp(-(x[0]**2 + x[1]**2)) + 2 * np.exp(-((x[0] - 1.7)**2 + (x[1] - 1.7)**2))

# Inicializando o algoritmo de Busca Aleatória Global (GRS)
def grs(f, bounds, N, Nmax):
    # Gerar N pontos iniciais uniformemente dentro dos limites
    x_best = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(N, 2))
    f_best = np.array([f(x) for x in x_best])
    
    # Encontrar o melhor ponto inicial
    max_index = np.argmax(f_best)
    x_best = x_best[max_index]
    f_best = f_best[max_index]
    
    # Lista para armazenar todos os pontos candidatos
    all_candidates = [x_best]
    
    # Rodar o algoritmo por Nmax iterações
    for i in range(Nmax):
        x_cand = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(N, 2))
        f_cand = np.array([f(x) for x in x_cand])
        
        # Encontrar o melhor ponto candidato
        max_index = np.argmax(f_cand)
        x_cand_best = x_cand[max_index]
        f_cand_best = f_cand[max_index]
        
        # Se o candidato for melhor, atualiza a melhor solução
        if f_cand_best > f_best:
            x_best = x_cand_best
            f_best = f_cand_best
        
        # Armazenar o melhor ponto encontrado até agora
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = np.array([[-2, 4], [-2, 5]])  # Ajustando os limites para cada variável
N = 50  # Número de candidatos por iteração
Nmax = 10000  # Número máximo de iterações

# Executando o algoritmo GRS
x_opt, f_opt, all_candidates = grs(f, bounds, N, Nmax)

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0, 0], bounds[0, 1], 100)
x2_vals = np.linspace(bounds[1, 0], bounds[1, 1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.exp(-(X1**2 + X2**2)) + 2 * np.exp(-((X1 - 1.7)**2 + (X2 - 1.7)**2))

# Criando o gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os candidatos como pontos azuis
ax.scatter(all_candidates[:, 0], all_candidates[:, 1], f(all_candidates.T), color='blue', s=20, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo")

# Ajustando os rótulos e o título
ax.set_title('GRS - Max de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

# Criando uma nova figura para a tabela com os pontos candidatos
fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(14, 10))
ax2_left.axis('tight')
ax2_left.axis('off')
ax2_right.axis('tight')
ax2_right.axis('off')

# Dados para as tabelas formatados com 3 casas decimais
table_data_left = [["Rodada", "x1", "x2"]] + [[i + 1, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(all_candidates[:50])]
table_data_right = [["Rodada", "x1", "x2"]] + [[i + 51, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(all_candidates[50:100])]

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

plt.show()
