"""Paramètres par défaut de la simulation et de l'algorithme génétique."""

# Phéromones
PHEROMONE_MIN = 0.1  # Plancher après évaporation
TAUX_EVAPORATION = 0.1
Q = 100.0  # Quantité de phéromone de référence
FACTEUR_DEPOT = 1.5  # Renforcement des chemins trouvés pendant le tour
FACTEUR_DEPOT_ELITE = 2.0  # Renforcement supplémentaire du meilleur chemin

# Déplacement des fourmis
VITESSE_PAR_DEFAUT = 0.02
VITESSE_MAX = 0.1  # Au-delà, la vitesse est ramenée à VITESSE_PAR_DEFAUT
PIXELS_PAR_ITERATION_MIN = 1
PIXELS_PAR_ITERATION_MAX = 3

# Population
GAMMA_MIN, GAMMA_MAX = -2, 2
DELAI_INITIAL_MAX = 10  # Délai aléatoire avant le premier départ
DELAI_POST_GENETIQUE_MAX = 15  # Délai de départ progressif après l'optimisation

# Algorithme génétique
ITERATIONS_EVALUATION = 50
PROPORTION_ELITE = 5  # L'élite représente 1/5 de la population
TAILLE_ELITE_MIN = 2
PROBA_CROISEMENT = 0.3
PROBA_MUTATION_APRES_CROISEMENT = 0.1
MUTATION_INTENSITE_MIN = 0.8  # Intensité tirée dans [0.8, 1.2[
MUTATION_INTENSITE_AMPLITUDE = 0.4
MUTATION_GAMMA = 0.5  # Décalage de gamma tiré dans [-0.5, 0.5]
