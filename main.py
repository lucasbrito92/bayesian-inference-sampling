from probability import BayesNet, prior_sample, likelihood_weighting, gibbs_ask
from prettytable import PrettyTable
import random

""" A simple Bayes net with Boolean variables B = BrokeElectionLaw,
I = Indicted, M = PoliticallyMotivatedProsecutor , G = FoundGuilty, J = Jailed """

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
    
    T, F = True, False

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

    random.seed(1017)
    p3 = likelihood_weighting("J", dict(B=T), jailed, 1000).show_approx()
    p4 = likelihood_weighting("G", dict(I=T, J=T), jailed, 1000).show_approx()

    # Gibbs Ask: Performs a query P(G|I,J) and P(J|B)

    p5 = gibbs_ask('J', dict(B=T), jailed, 1000).show_approx()
    p6 = gibbs_ask("G", dict(I=F, J=F), jailed, 1000).show_approx()

    table = PrettyTable()

    table.field_names("type of sampling", "P(J|B)")
    table.add_row(["Prior Sample", str(p1)])
    table.add_row(["Likelihood Sampling", str(p3[0])])

    print(table)


