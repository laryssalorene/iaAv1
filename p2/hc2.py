import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função a ser maximizada
def f(x):
    return np.exp(-(x[0]**2 + x[1]**2)) + 2 * np.exp(-((x[0] - 1.7)**2 + (x[1] - 1.7)**2))

# Função para gerar candidatos na vizinhança
def generate_neighbor(x_best, epsilon, bounds):
    # Gera um novo candidato dentro da vizinhança definida
    x_cand = np.random.uniform(x_best - epsilon, x_best + epsilon, size=x_best.shape)
    # Certifica que o candidato está dentro dos limites
    return np.clip(x_cand, bounds[0], bounds[1])

# Inicializando o algoritmo de Hill Climbing
def hill_climbing(f, bounds, epsilon, Nmax):
    # Ponto inicial é o limite inferior
    x_best = np.array([bounds[0][0], bounds[1][0]])
    f_best = f(x_best)
    
    # Armazenar todos os pontos candidatos para visualização começando pelo ponto inicial
    all_candidates = [x_best]
    
    # Rodar o algoritmo por Nmax iterações
    for _ in range(Nmax):
        x_cand = generate_neighbor(x_best, epsilon, bounds)
        f_cand = f(x_cand)
        
        # Se o candidato for melhor, atualiza a melhor solução (maximização)
        if f_cand > f_best:
            x_best = x_cand
            f_best = f_cand
        
        # Armazenar o candidato atual
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = (np.array([-2, -2]), np.array([4, 5]))  # Limites ajustados para x1 e x2
epsilon = 0.1  # Perturbação para vizinhança
Nmax = 1000  # Número máximo de iterações

# Rodar o algoritmo Hill Climbing
x_opt, f_opt, candidates = hill_climbing(f, bounds, epsilon, Nmax)

# Resultados
print(f"Solução ótima encontrada: x = {x_opt}")
print(f"Valor máximo da função: f(x) = {f_opt}")

# Visualização da função e dos pontos candidatos (Gráfico 3D)
x1_vals = np.linspace(bounds[0][0], bounds[1][0], 100)
x2_vals = np.linspace(bounds[0][1], bounds[1][1], 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = np.exp(-(X1**2 + X2**2)) + 2 * np.exp(-((X1 - 1.7)**2 + (X2 - 1.7)**2))

# Criando o gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície da função
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

# Plotando os pontos candidatos em todas as iterações
ax.scatter(candidates[:, 0], candidates[:, 1], f(candidates.T), color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Hill Climbing - Maximização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()
