import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Definindo a função a ser minimizada
def f(x):
    return (x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10) + \
           (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) + 10)

# Inicializando o algoritmo de Busca Aleatória Global (GRS)
def grs(f, bounds, N, Nmax, num_rounds):
    all_solutions = []
    
    for _ in range(num_rounds):
        # Gerar N pontos iniciais uniformemente dentro dos limites
        x_best = np.random.uniform(bounds[0], bounds[1], (N, 2))
        f_best = np.array([f(x) for x in x_best])
        
        # Encontrar o melhor ponto inicial
        min_index = np.argmin(f_best)
        x_best = x_best[min_index]
        f_best = f_best[min_index]
        
        # Rodar o algoritmo por Nmax iterações
        for _ in range(Nmax):
            x_cand = np.random.uniform(bounds[0], bounds[1], (N, 2))
            f_cand = np.array([f(x) for x in x_cand])
            
            # Encontrar o melhor ponto candidato
            min_index = np.argmin(f_cand)
            x_cand_best = x_cand[min_index]
            f_cand_best = f_cand[min_index]
            
            # Se o candidato for melhor, atualiza a melhor solução
            if f_cand_best < f_best:  # Minimização
                x_best = x_cand_best
                f_best = f_cand_best
        
        # Armazenar a melhor solução encontrada nesta rodada
        all_solutions.append(x_best)
    
    return np.array(all_solutions)

# Parâmetros do problema
bounds = (np.array([-5.12, -5.12]), np.array([5.12, 5.12]))
N = 50  # Número de candidatos por iteração
Nmax = 1000  # Número máximo de iterações por rodada
num_rounds = 100  # Número de rodadas

# Rodar o algoritmo GRS
all_solutions = grs(f, bounds, N, Nmax, num_rounds)

# Encontrando a melhor solução e seu valor
f_values = np.array([f(sol) for sol in all_solutions])  # Avaliando a função em todas as soluções
min_index = np.argmin(f_values)  # Índice da solução com menor valor da função

x_opt = all_solutions[min_index]  # Melhor solução encontrada
f_opt = f_values[min_index]  # Valor da função na melhor solução

print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor mínimo da função: f(x) = {f_opt}")

# Calculando a moda das coordenadas x1 e x2 das soluções obtidas
mode_x1 = stats.mode(all_solutions[:, 0], keepdims=True)
mode_x2 = stats.mode(all_solutions[:, 1], keepdims=True)

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
Z = (X1**2 - 10 * np.cos(2 * np.pi * X1) + 10) + \
    (X2**2 - 10 * np.cos(2 * np.pi * X2) + 10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando o ponto da melhor solução encontrada
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=100, label="Melhor Solução", marker='x')

# Ajustando os rótulos e o título
ax.set_title('Busca Aleatória Global (GRS) - Minimização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

# Definindo o intervalo dos eixos para cobrir o intervalo completo
ax.set_xlim(bounds[0][0], bounds[1][0])
ax.set_ylim(bounds[0][1], bounds[1][1])

plt.show()

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
