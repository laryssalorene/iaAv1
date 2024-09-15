import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função a ser minimizada
def f(x):
    return (x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10) + \
           (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) + 10)

# Função para gerar um novo candidato com perturbação
def perturb(x, sigma):
    return x + np.random.normal(0, sigma, size=x.shape)

# Inicializando o algoritmo de Busca Local Aleatória (LRS)
def lrs(f, bounds, sigma, Nmax):
    # Gerar x inicial uniformemente dentro dos limites
    x_best = np.random.uniform(bounds[0], bounds[1], 2)
    f_best = f(x_best)
    
    # Armazenar todos os pontos candidatos para visualização começando pelo ponto inicial
    all_candidates = [x_best]
    
    # Rodar o algoritmo por Nmax iterações
    for _ in range(Nmax):
        x_cand = perturb(x_best, sigma)  # Gera um novo candidato perturbado
        
        # Verificar se o novo candidato está dentro dos limites
        x_cand = np.clip(x_cand, bounds[0], bounds[1])
        
        f_cand = f(x_cand)  # Avalia a função objetivo no novo candidato
        
        # Se o candidato for melhor, atualiza a melhor solução
        if f_cand < f_best:
            x_best = x_cand
            f_best = f_cand
        
        # Armazenar o candidato atual
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = np.array([-5.12, -5.12]), np.array([5.12, 5.12])
sigma = 0.1  # Desvio padrão da perturbação
Nmax = 1000  # Número máximo de iterações

# Rodar o algoritmo LRS
x_opt, f_opt, candidates = lrs(f, bounds, sigma, Nmax)

# Resultados
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor mínimo da função: f(x) = {f_opt}")

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
ax.set_title('Busca Local Aleatória (LRS) - Minimização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

# Definindo o intervalo dos eixos para cobrir o intervalo completo
ax.set_xlim(bounds[0][0], bounds[1][0])
ax.set_ylim(bounds[0][1], bounds[1][1])

plt.show()
