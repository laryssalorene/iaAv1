import numpy as np
import pandas as pd

def rastrigin(x):
    A = 10
    p = len(x)
    return A * p + np.sum([xi**2 - A * np.cos(2 * np.pi * xi) for xi in x])

# Funções para Representação Cromossômica Canônica (Binária)
def initial_population_binary(pop_size, dimensions):
    return np.random.randint(2, size=(pop_size, dimensions))

def binary_to_float(binary_array, bounds):
    # Converte array binário para valores em ponto flutuante dentro do intervalo especificado
    integer_value = int("".join(str(int(bit)) for bit in binary_array), 2)
    max_integer = 2**len(binary_array) - 1
    return bounds[0] + (bounds[1] - bounds[0]) * (integer_value / max_integer)

def fitness_binary(individual, bounds):
    # Converte o cromossomo binário para ponto flutuante
    real_values = np.array([binary_to_float(individual, bounds)])
    return rastrigin(real_values) + 1

def roulette_wheel_selection(population, fitnesses):
    total_fitness = np.sum(1 / fitnesses)
    probabilities = (1 / fitnesses) / total_fitness
    r = np.random.rand()
    cumulative_sum = 0
    for i in range(len(population)):
        cumulative_sum += probabilities[i]
        if cumulative_sum >= r:
            return population[i]
    return population[-1]

def mutation(offspring):
    mutation_rate = 0.1
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] = 1 - offspring[i]  # Flip bit
    return offspring

def two_point_crossover(parent1, parent2):
    # Seleciona dois pontos de crossover aleatórios
    length = len(parent1)
    point1, point2 = np.sort(np.random.choice(range(1, length), 2, replace=False))
    
    # Inicializa a máscara com zeros
    mask = np.zeros(length, dtype=int)
    
    # Define a máscara para a região entre os dois pontos de crossover
    mask[point1:point2] = 1
    
    # Aplica a máscara para criar os descendentes
    offspring1 = np.where(mask == 1, parent2, parent1)
    offspring2 = np.where(mask == 1, parent1, parent2)
    
    return offspring1, offspring2

def genetic_algorithm_roleta(pop_size, dimensions, bounds, generations, convergence_window=10, recombination_percent=0.85):
    population = initial_population_binary(pop_size, dimensions)
    best_solution = None
    best_fitness = float('inf')
    no_improvement_counter = 0  # Contador de gerações sem melhoria
    converged = False  # Sinalizador de convergência

    for generation in range(generations):
        if converged:
            break  # Interrompe o loop se a convergência for detectada

        new_population = []
        fitnesses = np.array([fitness_binary(individual, bounds) for individual in population])
        
        current_best_fitness = np.min(fitnesses)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmin(fitnesses)]
            no_improvement_counter = 0  # Reseta o contador se houver melhoria
        else:
            no_improvement_counter += 1  # Incrementa o contador se não houver melhoria

        # Verifica se a convergência foi atingida
        if no_improvement_counter >= convergence_window:
            converged = True  # Sinaliza que a convergência foi alcançada

        num_recombination = int(pop_size * recombination_percent)
        num_random = pop_size - num_recombination
        
        # Criação de indivíduos via recombinação
        while len(new_population) < num_recombination:
            parent1 = roulette_wheel_selection(population, fitnesses)
            parent2 = roulette_wheel_selection(population, fitnesses)
            offspring1, offspring2 = two_point_crossover(parent1, parent2)
            offspring1 = mutation(offspring1)
            offspring2 = mutation(offspring2)
            new_population.append(offspring1)
            new_population.append(offspring2)

        # Criação de indivíduos aleatórios, se necessário
        while len(new_population) < pop_size:
            num_needed = pop_size - len(new_population)  # Calcule o número necessário
            for _ in range(num_needed):
                individual = np.random.randint(2, size=dimensions)
                individual = mutation(individual)
                new_population.append(individual)

        population = np.array(new_population[:pop_size])

    return best_solution, best_fitness

# Funções para Representação Cromossômica em Ponto Flutuante e Seleção por Torneio
def initial_population(pop_size, dimensions, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, dimensions))

def fitness_float(individual):
    return rastrigin(individual) + 1

def tournament_selection(population, fitnesses, tournament_size=3):
    selected = np.random.choice(len(population), tournament_size)
    best_individual = selected[0]
    for i in selected[1:]:
        if fitnesses[i] < fitnesses[best_individual]:
            best_individual = i
    return population[best_individual]

def sbx_crossover(parent1, parent2, eta=2):
    u = np.random.rand()
    if u <= 0.5:
        beta = (2 * u)**(1 / (eta + 1))
    else:
        beta = (1 / (2 * (1 - u)))**(1 / (eta + 1))
    offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    return offspring1, offspring2

def gaussian_mutation(offspring, bounds, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        mutation_value = np.random.normal(0, 1)
        offspring += mutation_value
        offspring = np.clip(offspring, bounds[0], bounds[1])
    return offspring

def genetic_algorithm_torneio(pop_size, dimensions, bounds, generations, convergence_window=10, recombination_percent=0.85):
    population = initial_population(pop_size, dimensions, bounds)
    best_solution = None
    best_fitness = float('inf')
    no_improvement_counter = 0  # Contador de gerações sem melhoria
    converged = False  # Sinalizador de convergência

    for generation in range(generations):
        if converged:
            break  # Interrompe o loop se a convergência for detectada

        new_population = []
        fitnesses = np.array([fitness_float(individual) for individual in population])
        
        current_best_fitness = np.min(fitnesses)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmin(fitnesses)]
            no_improvement_counter = 0  # Reseta o contador se houver melhoria
        else:
            no_improvement_counter += 1  # Incrementa o contador se não houver melhoria

        # Verifica se a convergência foi atingida
        if no_improvement_counter >= convergence_window:
            converged = True  # Sinaliza que a convergência foi alcançada

        num_recombination = int(pop_size * recombination_percent)
        num_random = pop_size - num_recombination
        
        # Criação de indivíduos via recombinação
        while len(new_population) < num_recombination:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            offspring1, offspring2 = sbx_crossover(parent1, parent2)
            offspring1 = gaussian_mutation(offspring1, bounds)
            offspring2 = gaussian_mutation(offspring2, bounds)
            new_population.append(offspring1)
            new_population.append(offspring2)

        # Criação de indivíduos aleatórios, se necessário
        while len(new_population) < pop_size:
            individual = np.random.uniform(bounds[0], bounds[1], dimensions)
            individual = gaussian_mutation(individual, bounds)
            new_population.append(individual)

        population = np.array(new_population[:pop_size])

    return best_solution, best_fitness

# Função para Executar o Experimento
def run_experiment(pop_size, dimensions, bounds, generations, runs=100):
    results_roleta = []
    results_torneio = []

    for _ in range(runs):
        _, best_fitness_roleta = genetic_algorithm_roleta(pop_size, dimensions, bounds, generations)
        _, best_fitness_torneio = genetic_algorithm_torneio(pop_size, dimensions, bounds, generations)
        results_roleta.append(best_fitness_roleta)
        results_torneio.append(best_fitness_torneio)

    return results_roleta, results_torneio

# Parâmetros do Experimento
pop_size = 50
dimensions = 10
bounds = [-5.12, 5.12]
generations = 100

# Execução do Experimento
results_roleta, results_torneio = run_experiment(pop_size, dimensions, bounds, generations)

# Criação do DataFrame para Exibição dos Resultados
data = {
    "Métrica": ["Menor Valor", "Maior Valor", "Média", "Desvio Padrão"],
    "Roleta": [np.min(results_roleta), np.max(results_roleta), np.mean(results_roleta), np.std(results_roleta)],
    "Torneio": [np.min(results_torneio), np.max(results_torneio), np.mean(results_torneio), np.std(results_torneio)]
}

df = pd.DataFrame(data)
print(df)
