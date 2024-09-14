import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função a ser minimizada
def f(x):
    return x[0]**2 + x[1]**2

# Função para gerar um novo candidato com perturbação
def perturb(x, sigma):
    return x + np.random.normal(0, sigma, size=x.shape)
#Gera um vetor de perturbação com distribuição normal (gaussiana) de média 0 e desvio padrão sigma, com a mesma forma que x.

# Inicializando o algoritmo de busca aleatória local (LRS)
def lrs(f, bounds, sigma, Nmax):
    # Gerar x inicial uniformemente dentro dos limites bounds
    x_best = np.random.uniform(bounds[0], bounds[1], 2)
    f_best = f(x_best) #Avalia o valor da função objetivo para o ponto inicial.
    
    # Armazenar todos os pontos candidatos para visualização começando pelo ponto inicial
    all_candidates = [x_best]
    
    # Rodar o algoritmo por Nmax iterações
    for i in range(Nmax):
        x_cand = perturb(x_best, sigma) #Gera um novo candidato perturbado.
        
        # Verificar se o novo candidato está dentro dos limites (restrição de caixa)/bounds
        x_cand = np.clip(x_cand, bounds[0], bounds[1])
        
        f_cand = f(x_cand) #Avalia a função objetivo no novo candidato.
        
        # Se o candidato for melhor, atualiza a melhor solução
        if f_cand < f_best:
            x_best = x_cand
            f_best = f_cand
        
        # Armazenar o candidato atual
        all_candidates.append(x_best) # Adiciona o ponto atual x_best à lista de candidatos
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = np.array([-100, 100])
sigma = 1.0  # Desvio padrão da perturbação
Nmax = 1000  # Número máximo de iterações

# Rodar o algoritmo LRS
x_opt, f_opt, candidates = lrs(f, bounds, sigma, Nmax)

# Resultados
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor mínimo da função: f(x) = {f_opt}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0], bounds[1], 100)
x2_vals = np.linspace(bounds[0], bounds[1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = X1**2 + X2**2

# Criando o gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos em todas as iterações
ax.scatter(candidates[:, 0], candidates[:, 1], f(np.array(candidates).T), color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Busca Aleatória Local (LRS) - Minimização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()
