import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

def f(board):
    """
    Função de avaliação f(x) que retorna 28 - número de conflitos em um tabuleiro.
    """
    conflitos = 0
    n = len(board)
    # Contar conflitos para cada par de rainhas
    for i in range(n):
        for j in range(i + 1, n):
            # Rainhas na mesma linha ou na mesma diagonal
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                conflitos += 1
    # A função de aptidão é 28 menos o número de conflitos
    return 28 - conflitos

def perturba_solucao(board):
    """
    Função de perturbação controlada da solução.
    Move uma rainha aleatoriamente para uma nova linha na mesma coluna.
    """
    n = len(board)
    nova_board = board.copy()
    col = random.randint(0, n - 1)  # Seleciona uma coluna aleatória
    nova_linha = random.randint(0, n - 1)  # Nova linha para a rainha
    nova_board[col] = nova_linha
    return nova_board

def tempera_simulada(board_inicial, T0, T_final, nt, metodo, decaimento):
    """
    Algoritmo de Têmpera Simulada para resolver o Problema das 8 Rainhas.
    """
    board_atual = board_inicial
    T = T0
    conflitos_atual = f(board_atual)
    conflitos_hist = [conflitos_atual]
    
    for iteracao in range(nt):
        nova_board = perturba_solucao(board_atual)
        conflitos_novo = f(nova_board)
        delta = conflitos_novo - conflitos_atual
        
        # Aceita a nova solução com base na temperatura
        if delta > 0 or random.uniform(0, 1) < math.exp(delta / T):
            board_atual = nova_board
            conflitos_atual = conflitos_novo
        
        conflitos_hist.append(conflitos_atual)
        
        # Atualizar a temperatura de acordo com o método escolhido
        if metodo == 1:
            T = decaimento * T
        elif metodo == 2:
            T = T / (1 + decaimento * math.sqrt(T))
        elif metodo == 3:
            delta_T = (T0 - T_final) / nt
            T = T - delta_T
        
        # Parar se encontramos uma solução ótima
        if conflitos_atual == 28:  # Máxima função de aptidão, nenhuma rainha se atacando. valor ótimo 
            break
    
    return board_atual, conflitos_atual, conflitos_hist

def resolver_8_rainhas(T0, T_final, nt, decaimento):
    """
    Resolver o problema das 8 rainhas utilizando Têmpera Simulada
    com três métodos de escalonamento de temperatura.
    """
    max_solucoes = 92
    solucoes = []
    
    for metodo in range(1, 4):
        print(f"Iniciando teste com método de escalonamento {metodo}...")
        solucoes_unicas = set()
        start_time = time.time()
        
        while len(solucoes_unicas) < max_solucoes:
            board_inicial = np.random.randint(0, 8, size=8)
            solucao, conflitos, conflitos_hist = tempera_simulada(board_inicial, T0, T_final, nt, metodo, decaimento)
            
            # Adiciona a solução ao conjunto de soluções únicas se não estiver duplicada
            solucao_tuple = tuple(solucao)
            if solucao_tuple not in solucoes_unicas:
                solucoes_unicas.add(solucao_tuple)
                solucoes.append((solucao, conflitos, time.time() - start_time))
            
            # Verifica se atingiu o número máximo de soluções
            if len(solucoes_unicas) >= max_solucoes:
                break
        
        end_time = time.time()
        tempo_execucao = end_time - start_time
        
        # Exibir resultados
        print(f"Teste com método de escalonamento {metodo}:")
        print(f"Número de soluções únicas encontradas: {len(solucoes_unicas)}")
        print(f"Tempo de execução total: {tempo_execucao:.4f} segundos")
        print(f"Número total de iterações realizadas: {len(solucoes)}")
        print()
        
        # Plotar o gráfico do histórico da função de aptidão
        plt.plot(conflitos_hist, label=f'Método {metodo}')
    
    plt.xlabel('Iterações')
    plt.ylabel('Valor da Função de Aptidão (28 - Número de Conflitos)')
    plt.title('Histórico da Função de Aptidão durante a Têmpera Simulada')
    plt.legend()
    plt.show()
    
    return solucoes

# Parâmetros
T0 = 1000  # Temperatura inicial
T_final = 1  # Temperatura final
nt = 1000  # Número de iterações máximas
decaimento = 0.99  # Fator de decaimento para o escalonamento

# Resolver o problema das 8 rainhas
solucoes = resolver_8_rainhas(T0, T_final, nt, decaimento)
