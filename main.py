from probability import BayesNet, prior_sample, likelihood_weighting, gibbs_ask, enumeration_ask, ProbDist
from prettytable import PrettyTable
import numpy as np
import random


def prior(X, e, bn, N):

    all_observations = [prior_sample(bn) for x in range(N)]

    var_X = [
        observation for observation in all_observations if observation[X] == True]

    var_e = [
        observation for observation in all_observations if observation[e] == True]

    X_and_e = [
        observation for observation in var_X if observation[e] == True]

    ans = len(X_and_e) / len(var_e)

    return ans


def prior_double(X, e1, e2, bn, N):

    all_observations = [prior_sample(bn) for x in range(N)]

    var_X = [
        observation for observation in all_observations if observation[X] == True]

    X_and_e1 = [
        observation for observation in var_X if observation[e1] == True]

    X_and_e1_and_e2 = [
        observation for observation in X_and_e1 if observation[e2] == True]

    var_e1 = [
        observation for observation in all_observations if observation[e1] == True]

    e1_and_e2 = [
        observation for observation in var_e1 if observation[e2] == True]

    ans = len(X_and_e1_and_e2) / len(var_e1)*(len(e1_and_e2)/len(var_X))

    return ans


if __name__ == "__main__":

    """
    Exercício 14.21

    Interpretação para os nós da rede bayesiana:

    X

    T: Prob. do time fazer gols (qualidade)
    
    ...

    XY.resultado:

    T,F: Time X venceu
    F,T: Time Y venceu
    T,T: Empate com gols
    F,F: Empate sem gols 
    
    ...
    
    X.win:

    T,F: Fez gols no 1o jogo e nada no 2o
    F,T: Nenhum gol no 1o jogo e fez gols no 2o jogo
    T,T: Fez gols em ambos os jogos
    F,F: Nenhum gol em ambas as partidas
    
    """

    """ Repostas:

    a) As classes são Team, com instâncias A, B e C, e Match, com instâncias AB,
    BC e CA. Cada equipe tem uma qualidade Q e cada partida tem um time X e e um time Y. 
    Os nomes das equipes para cada partida são, é claro, fixados antecipadamente. A priori
    sobre a qualidade dos timespode ser uniforme e a probabilidade de uma vitória para a 
    equipe 1 deve aumentar com Q (Team1) - Q (Team2).

    b) Priori: A, B, C
    Pais(AB.resultado) = A, B
    Pais(BC.resultado) = B, C
    Pais(CA.resultado) = C, A

    c) Codigo abaixo
    d) O custo de inferência em tal modelo será O (2n) devido a quantidade de pais do resultado
    e) O MCMC parece se dar bem com esse problema, desde que as probabilidades não sejam muito
    enviesadas. Os resultados mostram um comportamento aparentemente linear no número de
    equipes, embora não tenha investigado para um n grande.
    
    """
    T, F = True, False

    quality = np.arange(4)
    normed = [i/sum(quality) for i in quality]
    q4, q3, q2, q1 = normed

    team_matches = BayesNet([
        ('A', '', q1),
        ('B', '', q2),
        ('C', '', q3),
        ('AB.resultado', 'A B', {(T, T): 0.3,
                                 (T, F): 0.5, (F, T): 0.2, (F, F): 0.3}),
        ('BC.resultado', 'B C', {(T, T): 0.2,
                                 (T, F): 0.4, (F, T): 0.1, (F, F): 0.2}),
        ('CA.resultado', 'C A', {(T, T): 0.3,
                                 (T, F): 0.2, (F, T): 0.5, (F, F): 0.3}),
        ('Awin', 'AB.resultado CA.resultado', {(T, T): 0.6,
                                                (T, F): 0.3, (F, T): 0.3, (F, F): 0.1}),
        ('Bwin', 'AB.resultado BC.resultado', {(T, T): 0.4,
                                                (T, F): 0.3, (F, T): 0.3, (F, F): 0.2}),
        ('Cwin', 'CA.resultado BC.resultado', {(T, T): 0.2,
                                                (T, F): 0.3, (F, T): 0.3, (F, F): 0.6}),
    ])


    ans = enumeration_ask(
        'BC.resultado', {'Awin': T, 'Bwin': F, 'Cwin': T}, team_matches).show_approx()
    ans1 = gibbs_ask('BC.resultado', dict(Awin=T, Bwin=F, Cwin=T), team_matches, 1000).show_approx()

    """ Item (c) e (e) A probabilidade de B vencer o jogo com dada a evidencia que A ganhou o 
    primeiro jogo com B e empatou com C = 'False: 0.783, True: 0.217' para enumeration_ask e 
    'False: 0.777, True: 0.223' para gibbs_ask, ou seja, B tem maior probabilidade de perder dado os jogos anteriores """

    """ Exercício 14.20: 
    
    Rede bayesiana com variáveis Booleanas B = BrokeElectionLaw,
    I = Indicted, M = PoliticallyMotivatedProsecutor , G = FoundGuilty, J = Jailed """

    jailed = BayesNet([
        ('B', '', 0.9),
        ('M', '', 0.1),
        ('I', 'B M',
         {(T, T): 0.9, (T, F): 0.5, (F, T): 0.5, (F, F): 0.1}),
        ('G', 'B I M',
         {(T, T, T): 0.9, (T, T, F): 0.8, (T, F, T): 0.0, (T, F, F): 0.0, (F, T, T): 0.2, (F, T, F): 0.1, (F, F, T): 0.0, (F, F, F): 0.0}),
        ('J', 'G', {T: 0.9, F: 0.0})
    ])

    N = 1000

    # Prior Sample: Performs a query P(G|I,J) = P(G,I,J)P(J)/(P(I)P(I and J), such that P(I|J) = P(I and J)/P(J)

    p1 = prior("J", "B", jailed, 1000)
    p2 = prior_double("G", "I", "J", jailed, 1000)

    # Likelihood Weighting: Performs a query P(G|I,J) and P(J|B)

    # random.seed(1017)
    p3 = likelihood_weighting("J", dict(B=T), jailed, 1000).show_approx()
    p4 = likelihood_weighting("G", dict(I=T, J=T), jailed, 1000).show_approx()

    # Gibbs Ask: Performs a query P(G|I,J) and P(J|B)

    p5 = gibbs_ask("J", dict(B=T), jailed, 1000).show_approx()
    p6 = gibbs_ask("G", dict(I=F, J=F), jailed, 1000).show_approx()

"""     table = PrettyTable()

    table.field_names("type of sampling", "P(J|B)")
    table.add_row(["Prior Sample", str(p1)])
    table.add_row(["Likelihood Sampling", str(p3[0])])

    print(table) """
