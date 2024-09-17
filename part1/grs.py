import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Definindo a função a ser minimizada
def f(x):
    return x[0]**2 + x[1]**2

# Inicializando o algoritmo de Busca Aleatória Global (GRS)
def grs(f, bounds, N, Nmax, t):
    # Gerar N pontos iniciais uniformemente dentro dos limites
    x_best = np.random.uniform(bounds[0], bounds[1], (N, 2)) 
    f_best = np.array([f(x) for x in x_best])
    
    # Encontrar o melhor ponto inicial
    min_index = np.argmin(f_best) 
    x_best = x_best[min_index]
    f_best = f_best[min_index]
    
    # Lista para armazenar todos os pontos candidatos.
    all_candidates = [x_best]
    
    # Critério de parada
    no_improvement_count = 0
    
    # Rodar o algoritmo por Nmax iterações
    for i in range(Nmax): 
        x_cand = np.random.uniform(bounds[0], bounds[1], (N, 2)) 
        f_cand = np.array([f(x) for x in x_cand]) 
        
        # Encontrar o melhor ponto candidato
        min_index = np.argmin(f_cand) 
        
        x_cand_best = x_cand[min_index]
        f_cand_best = f_cand[min_index]
        
        # Se o candidato for melhor, atualiza a melhor solução
        if f_cand_best < f_best:
            x_best = x_cand_best
            f_best = f_cand_best
            no_improvement_count = 0  # Resetar contador quando houver melhoria
        else:
            no_improvement_count += 1
        
        # Critério de parada baseado na ausência de melhoria
        if no_improvement_count >= t:
            break
        
        # Armazenar o melhor ponto encontrado até agora
        all_candidates.append(x_best)
    
    # Retornar apenas os primeiros 100 valores
    return x_best, f_best, np.array(all_candidates[:100])

# Parâmetros do problema
bounds = np.array([-100, 100])
N = 50  # Número de candidatos por iteração
Nmax = 10000  # Número máximo de iterações
t = 100  # Critério de parada

# Executar o algoritmo GRS
x_opt, f_opt, all_candidates = grs(f, bounds, N, Nmax, t)

# Calculando a moda das soluções para cada coordenada separadamente
mode_x1_result = stats.mode(all_candidates[:, 0], keepdims=True)
mode_x2_result = stats.mode(all_candidates[:, 1], keepdims=True)

# Acesso correto aos resultados
mode_x1 = mode_x1_result.mode[0]
count_x1 = mode_x1_result.count[0]
mode_x2 = mode_x2_result.mode[0]
count_x2 = mode_x2_result.count[0]

# Resultados
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor mínimo da função: f(x) = {f_opt}")
print(f"Moda das coordenadas x1: {mode_x1:.3f} com contagem {count_x1}")
print(f"Moda das coordenadas x2: {mode_x2:.3f} com contagem {count_x2}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0], bounds[1], 100)
x2_vals = np.linspace(bounds[0], bounds[1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = X1**2 + X2**2

# Criando o gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos durante a busca
ax.scatter(all_candidates[:, 0], all_candidates[:, 1], [f(x) for x in all_candidates], color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('GRS - Min de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

# Adicionando duas tabelas com as soluções obtidas e a moda
fig2, (ax2_left, ax2_right) = plt.subplots(1, 2, figsize=(14, 10))
ax2_left.axis('tight')
ax2_left.axis('off')
ax2_right.axis('tight')
ax2_right.axis('off')

# Dados para as tabelas formatados com 3 casas decimais
table_data_left = [["Rodada", "x1", "x2"]] + [[i+1, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(all_candidates[:50])]
table_data_right = [["Rodada", "x1", "x2"]] + [[i+51, f"{sol[0]:.3f}", f"{sol[1]:.3f}"] for i, sol in enumerate(all_candidates[50:])]

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
