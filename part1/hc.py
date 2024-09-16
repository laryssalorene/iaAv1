import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Função de perturbação
def perturb(x, e, x_l, x_u):
    x_cand = np.random.uniform(low=x-e, high=x+e, size=x.shape)
    return np.clip(x_cand, x_l, x_u)

# Função f(x1, x2) = x1^2 + x2^2
def f(x1, x2):
    return x1**2 + x2**2

# Configuração da superfície 3D
x_axis = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(x_axis, x_axis)
Z = f(X, Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Definir limites
x_l = -100  # Limite inferior
x_u = 100   # Limite superior

# Parâmetros do Hill Climbing
max_it = 1000
max_viz = 20
e = 0.1
num_rounds = 100  # Número de rodadas

# Inicializar variáveis
all_solutions = []  # Lista para armazenar as soluções de todas as rodadas

# Executar o algoritmo para cada uma das 100 rodadas
for round in range(num_rounds):
    x_opt = np.array([-100, -100])
    f_opt = f(x_opt[0], x_opt[1])
    
    i = 0
    melhoria = True

    # Loop de otimização para cada rodada
    while i < max_it and melhoria:
        melhoria = False
        for j in range(max_viz):
            # Gerar novo candidato
            x_cand = perturb(x_opt, e, x_l, x_u)
            f_cand = f(x_cand[0], x_cand[1])

            # Verificar se houve melhoria (minimização)
            if f_cand < f_opt:
                x_opt = x_cand
                f_opt = f_cand
                melhoria = True
                break
        i += 1

    # Armazenar a solução final desta rodada
    all_solutions.append(x_opt)

    # Exibir ponto final
    if round == num_rounds - 1:
        ax.scatter(x_opt[0], x_opt[1], f_opt, marker='o', color='g', s=100, linewidth=20)

# Converta a lista de soluções para numpy array
all_solutions = np.array(all_solutions)

# Calcular a moda das soluções
mode_x1_result = stats.mode(all_solutions[:, 0], keepdims=True)
mode_x2_result = stats.mode(all_solutions[:, 1], keepdims=True)

# Acesso correto aos resultados
mode_x1 = mode_x1_result.mode[0]
count_x1 = mode_x1_result.count[0]
mode_x2 = mode_x2_result.mode[0]
count_x2 = mode_x2_result.count[0]

print(f"Moda das coordenadas x1: {mode_x1:.3f} com contagem {count_x1}")
print(f"Moda das coordenadas x2: {mode_x2:.3f} com contagem {count_x2}")

# Visualizar a tabela com soluções
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

print(f"Current x_opt: {x_opt}, f_opt: {f_opt}")
print(f"x_opt: {x_opt[0]},{x_opt[1]} ")

# Ajustar o layout da figura para garantir que a tabela seja visível
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.show()
