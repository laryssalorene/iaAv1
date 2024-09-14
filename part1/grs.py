import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definindo a função a ser minimizada
def f(x):
    return x[0]**2 + x[1]**2

# Inicializando o algoritmo de Busca Aleatória Global (GRS)
def grs(f, bounds, N, Nmax):
    # Gerar N pontos iniciais uniformemente dentro dos limites
    x_best = np.random.uniform(bounds[0], bounds[1], (N, 2)) 
    #x_best : Gera N candidatos iniciais uniformemente distribuídos dentro dos limites.
    f_best = np.array([f(x) for x in x_best])
    #f best: Avalia a função objetivo para todos os candidatos iniciais.
    
    # Encontrar o melhor ponto inicial
    min_index = np.argmin(f_best) # Encontra o índice do melhor candidato inicial.
    x_best = x_best[min_index]
    f_best = f_best[min_index]
    #x_best = x_best[min_index], 
    # f_best = f_best[min_index]: Atualiza o melhor candidato inicial.
    
    # Lista para armazenar todos os pontos candidatos.
    all_candidates = [x_best]
    
    # Rodar o algoritmo por Nmax iterações
    for i in range(Nmax): #Loop para iterações de busca
        x_cand = np.random.uniform(bounds[0], bounds[1], (N, 2)) #Gera N novos candidatos.
        f_cand = np.array([f(x) for x in x_cand]) #Avalia a função objetivo para os novos candidatos.
        
        # Encontrar o melhor ponto candidato
        min_index = np.argmin(f_cand) #Encontra o índice do melhor candidato atual.

        #x_cand_best = x_cand[min_index], 
        #f_cand_best = f_cand[min_index]: Atualiza o melhor candidato encontrado.
        x_cand_best = x_cand[min_index]
        f_cand_best = f_cand[min_index]
        
        # Se o candidato for melhor, atualiza a melhor solução
        if f_cand_best < f_best:
            x_best = x_cand_best
            f_best = f_cand_best
        
        # Armazenar o melhor ponto encontrado até agora/
        #Adiciona o melhor ponto encontrado na iteração à lista.
        all_candidates.append(x_best)
    
    return x_best, f_best, np.array(all_candidates)

# Parâmetros do problema
bounds = np.array([-100, 100])
N = 50  # Número de candidatos por iteração
Nmax = 1000  # Número máximo de iterações

# Rodar o algoritmo GRS
x_opt, f_opt, candidates = grs(f, bounds, N, Nmax)

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

# Plotando os pontos candidatos durante a busca
ax.scatter(candidates[:, 0], candidates[:, 1], f(np.array(candidates).T), color='blue', s=10, label="Candidatos")

# Plotando o ponto ótimo encontrado
ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=50, label="Ótimo encontrado")

# Ajustando os rótulos e o título
ax.set_title('Busca Aleatória Global (GRS) - Minimização de f(x1, x2)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.legend()

plt.show()
