import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função a ser minimizada
def f(x):
    return (x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10) + \
           (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) + 10)

# Função de perturbação
def perturb(x, epsilon):
    return x + np.random.uniform(-epsilon, epsilon, size=x.shape)

# Inicializando o algoritmo de Hill Climbing (HC)
def hc(f, bounds, epsilon, Nmax):
    x_best = np.array([bounds[0][0], bounds[0][1]])
    f_best = f(x_best)
    
    all_candidates = [x_best]
    
    for _ in range(Nmax):
        improvement = False
        for _ in range(20):
            x_cand = perturb(x_best, epsilon)
            x_cand = np.clip(x_cand, bounds[0], bounds[1])
            f_cand = f(x_cand)
            
            if f_cand < f_best:
                x_best = x_cand
                f_best = f_cand
                improvement = True
                break
        
        if not improvement:
            break
        
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = (np.array([-5.12, -5.12]), np.array([5.12, 5.12]))
epsilon = 0.1
Nmax = 1000

# Rodar o algoritmo HC
x_opt, f_opt, candidates = hc(f, bounds, epsilon, Nmax)

# Resultados
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor mínimo da função: f(x) = {f_opt}")

# Calculando a moda das coordenadas x1 e x2 das soluções obtidas
mode_x1_result = stats.mode(candidates[:, 0], keepdims=True)
mode_x2_result = stats.mode(candidates[:, 1], keepdims=True)

# Extraindo a moda diretamente
mode_x1 = mode_x1_result.mode[0] if mode_x1_result.mode.size > 0 else None
mode_x2 = mode_x2_result.mode[0] if mode_x2_result.mode.size > 0 else None

print(f"Moda das coordenadas x1: {mode_x1}")
print(f"Moda das coordenadas x2: {mode_x2}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0][0], bounds[1][0], 100)
x2_vals = np.linspace(bounds[0][1], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = (X1**2 - 10 * np.cos(2 * np.pi * X1) + 10) + \
    (X2**2 - 10 * np.cos(2 * np.pi * X2) + 10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos em todas as iterações
ax.scatter(candidates[:, 0], candidates[:, 1], f(candidates.T), color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Hill Climbing (HC) - Minimização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

# Definindo o intervalo dos eixos para cobrir o intervalo completo
ax.set_xlim(bounds[0][0], bounds[1][0])
ax.set_ylim(bounds[0][1], bounds[1][1])

plt.show()

# Verificar o número de candidatos para criar as tabelas
num_candidates = len(candidates)
split_point = min(num_candidates, 50)

# Visualização da tabela com soluções
fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(14, 10))
ax2_left.axis('tight')
ax2_left.axis('off')
ax2_right.axis('tight')
ax2_right.axis('off')

# Dados para as tabelas formatados com 3 casas decimais
table_data_left = [["Rodada", "x1", "x2"]] + [[i+1, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(candidates[:split_point])]
table_data_right = [["Rodada", "x1", "x2"]] + [[i+split_point+1, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(candidates[split_point:])]

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
