from probability import *

""" A simple Bayes net with Boolean variables B = BrokeElectionLaw,
I = Indicted, M = PoliticallyMotivatedProsecutor , G = FoundGuilty, J = Jailed """

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
all_observations = [prior_sample(jailed) for x in range(N)]

# Algorithm: Prior Sample
# Performs a query P(J|B) = P(J and B)/P(B) in prior sample

j = [
    observation for observation in all_observations if observation['J'] == True]

b = [
    observation for observation in all_observations if observation['B'] == True]

j_and_b = [
    observation for observation in j if observation['B'] == True]

ans_prior = len(j_and_b) / len(b)

# Algorithm: Prior Sample
# Performs a query P(G|I,J) = P(G,I,J)P(J)/(P(I)P(I and J), such that P(I|J) = P(I and J)/P(J)

g = [
    observation for observation in all_observations if observation['G'] == True]

g_and_i = [
    observation for observation in g if observation['I'] == True]

g_and_i_and_j = [
    observation for observation in g_and_i if observation['J'] == True]

i = [
    observation for observation in all_observations if observation['I'] == True]

i_and_j = [
    observation for observation in i if observation['J'] == True]

ans_prior2 = len(g_and_i_and_j) / len(i)*(len(i_and_j)/len(j))

print(ans_prior)
print(ans_prior2)
