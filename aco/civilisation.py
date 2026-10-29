"""Environnement de simulation et algorithme génétique."""

import copy
import math
import random
from typing import List, Optional, Tuple

from . import config
from .ant import Ant
from .models import Route, Ville


class Civilisation:
    """Gère l'environnement et la simulation avec algorithme génétique"""

    def __init__(self):
        self.src_nourriture = None
        self.nid = None
        self.routes = []
        self.villes = []
        self.fourmis = []
        self.meilleur_chemin = None
        self.meilleure_distance = float("inf")
        self.taux_evaporation = config.TAUX_EVAPORATION
        self.Q = config.Q
        self.iteration = 0
        self.generation = 0
        self.meilleure_fourmi = None
        self.algo_genetique_applique = (
            False  # Flag pour indiquer si l'algo génétique a été appliqué
        )

    # ------------------------------------------------------------------
    # Construction du graphe
    # ------------------------------------------------------------------

    def ajouter_ville(self, x: int, y: int) -> Ville:
        ville = Ville(x, y, len(self.villes))
        self.villes.append(ville)
        return ville

    def trouver_route(self, v1: Ville, v2: Ville) -> Optional[Route]:
        """Retourne la route reliant v1 et v2, ou None si elle n'existe pas"""
        return next((route for route in self.routes if route.relie(v1, v2)), None)

    def ajouter_route(self, v1: Ville, v2: Ville) -> Route:
        route_existante = self.trouver_route(v1, v2)
        if route_existante is not None:
            return route_existante

        longueur = math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)
        route = Route(longueur=longueur, premiere_ville=v1, seconde_ville=v2)
        self.routes.append(route)
        v1.routes.append(route)
        v2.routes.append(route)
        return route

    # ------------------------------------------------------------------
    # Population et algorithme génétique
    # ------------------------------------------------------------------

    def initialiser_fourmis(
        self, nombre_fourmis: int, alpha_base: float, beta_base: float
    ):
        """Initialise la population de fourmis avec des paramètres variés"""
        self.fourmis = []
        for _ in range(nombre_fourmis):
            # Variation aléatoire autour des paramètres de base
            alpha = alpha_base * (0.5 + random.random())  # ±50% variation
            beta = beta_base * (0.5 + random.random())
            gamma = random.uniform(config.GAMMA_MIN, config.GAMMA_MAX)

            fourmi = Ant(alpha=alpha, beta=beta, gamma=gamma)
            fourmi.repartir_du_nid(self.nid)
            # Ajouter un délai initial aléatoire pour éviter les départs synchronisés
            fourmi.delai_initial = random.randint(0, config.DELAI_INITIAL_MAX)
            self.fourmis.append(fourmi)

        self.iteration = 0
        self.generation = 0

    def evaluer_population(self, iterations_test=config.ITERATIONS_EVALUATION):
        """Évalue la performance des fourmis sur un nombre limité d'itérations"""
        # Sauvegarde des valeurs actuelles
        iterations_orig = self.iteration

        # Pour chaque fourmi, faire un certain nombre d'itérations pour évaluer sa performance
        for fourmi in self.fourmis:
            fourmi.reinitialiser_stats()
            fourmi.repartir_du_nid(self.nid)

            for _ in range(iterations_test):
                self.evaluer_fourmi(fourmi)

        # Restaurer les valeurs
        self.iteration = iterations_orig

        # Mettre à jour les scores de performance
        for fourmi in self.fourmis:
            fourmi.mettre_a_jour_performance()

    def evaluer_fourmi(self, fourmi: Ant):
        """Évalue une fourmi sur une itération"""
        if fourmi.en_deplacement and fourmi.mettre_a_jour_position():
            # Si la fourmi atteint la source de nourriture
            if (
                fourmi.ville_actuelle == self.src_nourriture
                and not fourmi.porte_nourriture
            ):
                fourmi.charger_nourriture()
                # Toujours démarrer le chemin de retour immédiatement
                self._entamer_retour(fourmi)

            # Si la fourmi revient au nid ET porte de la nourriture
            elif fourmi.ville_actuelle == self.nid and fourmi.porte_nourriture:
                fourmi.deposer_nourriture(self.nid)
                fourmi.initialiser_position(self.nid)
                # Sélectionner immédiatement la prochaine destination
                fourmi.explorer()
            return

        # Vérifier si une fourmi immobile est à la source de nourriture
        if (
            fourmi.ville_actuelle == self.src_nourriture
            and not fourmi.en_deplacement
            and not fourmi.porte_nourriture
        ):
            # Forcer la collecte de nourriture et le départ si bloquée à la source
            fourmi.charger_nourriture()
            self._entamer_retour(fourmi)
            return

        # Si porte de la nourriture et en chemin de retour - ne pas reinitialiser porte_nourriture!
        if fourmi.porte_nourriture and not fourmi.en_deplacement:
            # Si au nid, déposer la nourriture et repartir
            if fourmi.ville_actuelle == self.nid:
                fourmi.deposer_nourriture(self.nid)
                fourmi.explorer()
                return

            # Sinon continuer le chemin de retour
            if fourmi.chemin_retour:
                prochaine_ville = fourmi.chemin_retour[0]
                if fourmi.est_reliee_a(prochaine_ville):
                    fourmi.demarrer_deplacement(prochaine_ville)
                    fourmi.chemin_retour.pop(0)
                    return
            # Si plus de chemin retour mais pas encore au nid, diriger vers le nid
            elif fourmi.est_reliee_a(self.nid):
                fourmi.demarrer_deplacement(self.nid)
                return

        # Sélection de la prochaine destination pour l'exploration (uniquement si pas de nourriture)
        if not fourmi.porte_nourriture and not fourmi.explorer():
            # Si aucune ville n'est disponible, retour au nid
            fourmi.reinitialiser_au_nid(self.nid)

    def _entamer_retour(self, fourmi: Ant):
        """Démarre le retour vers la première ville du chemin retour si elle est reliée"""
        if fourmi.chemin_retour:
            prochaine_ville = fourmi.chemin_retour.pop(0)
            if fourmi.est_reliee_a(prochaine_ville):
                fourmi.demarrer_deplacement(prochaine_ville)

    def mutation(self, fourmi: Ant):
        """Applique une mutation aux paramètres de la fourmi"""
        # Mutations avec intensité variable
        intensite = (
            random.random() * config.MUTATION_INTENSITE_AMPLITUDE
            + config.MUTATION_INTENSITE_MIN
        )  # 80-120% du paramètre original
        fourmi.alpha *= intensite
        fourmi.beta *= intensite
        fourmi.gamma += random.uniform(-config.MUTATION_GAMMA, config.MUTATION_GAMMA)

    def croisement(self, parent1: Ant, parent2: Ant) -> Ant:
        """Crée une nouvelle fourmi en combinant les paramètres de deux parents"""
        # Croisement avec ratio aléatoire
        ratio = random.random()
        alpha = parent1.alpha * ratio + parent2.alpha * (1 - ratio)
        beta = parent1.beta * ratio + parent2.beta * (1 - ratio)
        gamma = parent1.gamma * ratio + parent2.gamma * (1 - ratio)

        return Ant(alpha=alpha, beta=beta, gamma=gamma)

    def algorithme_genetique(self, generations=10):
        """Applique l'algorithme génétique pour optimiser les paramètres des fourmis"""
        if self.algo_genetique_applique:
            # Si l'algorithme a déjà été appliqué, ne rien faire
            return

        for _ in range(generations):
            self.generation += 1

            # Évaluer les performances sur un nombre limité d'itérations
            self.evaluer_population()

            # Trier les fourmis par performance
            self.fourmis.sort(key=lambda f: f.performance_score, reverse=True)

            # Sauvegarder la meilleure fourmi
            if (
                not self.meilleure_fourmi
                or self.fourmis[0].performance_score
                > self.meilleure_fourmi.performance_score
            ):
                self.meilleure_fourmi = copy.deepcopy(self.fourmis[0])

            # Sélectionner l'élite (20% meilleurs) et créer la nouvelle génération
            nb_elite = max(
                config.TAILLE_ELITE_MIN, len(self.fourmis) // config.PROPORTION_ELITE
            )
            elite = self.fourmis[:nb_elite]
            self.fourmis = self._nouvelle_generation(elite, len(self.fourmis))

            # Réinitialiser les fourmis pour l'évaluation suivante
            for fourmi in self.fourmis:
                fourmi.repartir_du_nid(self.nid)
                fourmi.reinitialiser_stats()

        # Marquer que l'algorithme génétique a été appliqué
        self.algo_genetique_applique = True

        # Utiliser les paramètres de la meilleure fourmi pour toutes les fourmis
        if self.meilleure_fourmi:
            for fourmi in self.fourmis:
                fourmi.alpha = self.meilleure_fourmi.alpha
                fourmi.beta = self.meilleure_fourmi.beta
                fourmi.gamma = self.meilleure_fourmi.gamma

                # Réinitialiser pour la simulation
                fourmi.reinitialiser_au_nid(self.nid)
                # Délai pour départ progressif
                fourmi.delai_initial = random.randint(
                    0, config.DELAI_POST_GENETIQUE_MAX
                )

    def _nouvelle_generation(self, elite: List[Ant], taille: int) -> List[Ant]:
        """Construit une génération à partir de l'élite (conservée telle quelle)"""
        nouvelle_generation = [copy.deepcopy(f) for f in elite]

        # Compléter la population
        while len(nouvelle_generation) < taille:
            if random.random() < config.PROBA_CROISEMENT:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                enfant = self.croisement(parent1, parent2)
                if random.random() < config.PROBA_MUTATION_APRES_CROISEMENT:
                    self.mutation(enfant)
            else:  # Sinon, copie mutée d'un membre de l'élite
                modele = random.choice(elite)
                enfant = copy.deepcopy(modele)
                self.mutation(enfant)
            nouvelle_generation.append(enfant)

        return nouvelle_generation

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def tour_suivant(self):
        """Effectue un tour de simulation. Retourne les actions effectuées et le meilleur chemin trouvé."""
        self.iteration += 1
        actions_effectuees = False

        # Chemins complets (nid -> nourriture) trouvés pendant ce tour, avec leur distance
        chemins_valides: List[Tuple[List[Ville], float]] = []

        # Mise à jour des positions des fourmis
        for fourmi in self.fourmis:
            # Gérer le délai initial
            if fourmi.delai_initial > 0:
                fourmi.delai_initial -= 1
                continue

            if fourmi.en_deplacement:
                if fourmi.mettre_a_jour_position():
                    actions_effectuees = True
                    self._traiter_arrivee(fourmi, chemins_valides)
            elif self._deplacer_fourmi_immobile(fourmi):
                actions_effectuees = True

        self._mettre_a_jour_pheromones(chemins_valides)
        self._mettre_a_jour_meilleure_fourmi()
        return actions_effectuees, self.meilleur_chemin, self.meilleure_distance

    def _traiter_arrivee(
        self, fourmi: Ant, chemins_valides: List[Tuple[List[Ville], float]]
    ):
        """Gère une fourmi qui vient d'atteindre sa ville de destination"""
        fourmi.en_deplacement = False
        fourmi.ville_actuelle = fourmi.ville_destination
        fourmi.progress = 0.0

        # Arrivée à la source de nourriture
        if fourmi.ville_actuelle == self.src_nourriture and not fourmi.porte_nourriture:
            fourmi.charger_nourriture()
            # Évaluer le chemin pour les phéromones
            self._enregistrer_chemin(fourmi, chemins_valides)
            # La fourmi commence immédiatement son trajet retour
            if fourmi.chemin_retour:
                fourmi.demarrer_deplacement(fourmi.chemin_retour.pop(0))

        # Retour au nid avec de la nourriture : seul endroit où elle est déposée
        elif fourmi.ville_actuelle == self.nid and fourmi.porte_nourriture:
            fourmi.deposer_nourriture(self.nid)
            fourmi.explorer()

        # En chemin de retour avec de la nourriture
        elif fourmi.porte_nourriture:
            self._poursuivre_retour(fourmi)

        # Exploration normale : réinitialiser si bloquée
        elif not fourmi.explorer():
            fourmi.repartir_du_nid(self.nid)
            fourmi.porte_nourriture = False

    def _deplacer_fourmi_immobile(self, fourmi: Ant) -> bool:
        """Fait repartir une fourmi immobile. Retourne True si une action a eu lieu."""
        # Fourmi immobile à la source de nourriture : forcer la collecte et le départ
        if fourmi.ville_actuelle == self.src_nourriture and not fourmi.porte_nourriture:
            fourmi.charger_nourriture()
            if fourmi.chemin_retour:
                fourmi.demarrer_deplacement(fourmi.chemin_retour.pop(0))
                return True
            return False

        # Fourmi immobile avec de la nourriture (en chemin de retour)
        if fourmi.porte_nourriture:
            if fourmi.ville_actuelle == self.nid:
                fourmi.deposer_nourriture(self.nid)
                fourmi.explorer()
            else:
                self._poursuivre_retour(fourmi)
            return True

        # Fourmi immobile en exploration (sans nourriture)
        return fourmi.explorer()

    def _poursuivre_retour(self, fourmi: Ant):
        """Avance d'une étape sur le chemin retour, ou se dirige vers le nid s'il est voisin"""
        if fourmi.chemin_retour:
            fourmi.demarrer_deplacement(fourmi.chemin_retour.pop(0))
        elif fourmi.est_reliee_a(self.nid):
            fourmi.demarrer_deplacement(self.nid)

    def _enregistrer_chemin(
        self, fourmi: Ant, chemins_valides: List[Tuple[List[Ville], float]]
    ):
        """Enregistre le chemin aller de la fourmi et met à jour le meilleur chemin"""
        if len(fourmi.chemin_parcouru) <= 1:
            return

        distance = self.calculer_distance_chemin(fourmi.chemin_parcouru)
        if distance < float("inf"):
            chemins_valides.append((fourmi.chemin_parcouru.copy(), distance))
            fourmi.chemins_uniques.add(tuple(fourmi.chemin_parcouru))
            if distance < self.meilleure_distance:
                self.meilleure_distance = distance
                self.meilleur_chemin = fourmi.chemin_parcouru.copy()

    def _mettre_a_jour_pheromones(
        self, chemins_valides: List[Tuple[List[Ville], float]]
    ):
        """Évaporation puis dépôt de phéromones (chemins du tour + renfort élitiste)"""
        for route in self.routes:
            route.evaporer_pheromone(self.taux_evaporation)

        # Augmenter le dépôt pour les plus courts chemins
        for chemin, distance in chemins_valides:
            depot = (self.Q / distance) * config.FACTEUR_DEPOT
            self.deposer_pheromone(chemin, depot)

        # Renforcement élitiste encore plus fort
        if self.meilleur_chemin:
            depot_elite = (self.Q / self.meilleure_distance) * config.FACTEUR_DEPOT_ELITE
            self.deposer_pheromone(self.meilleur_chemin, depot_elite)

    def _mettre_a_jour_meilleure_fourmi(self):
        for fourmi in self.fourmis:
            fourmi.mettre_a_jour_performance()
            if (
                not self.meilleure_fourmi
                or fourmi.performance_score > self.meilleure_fourmi.performance_score
            ):
                self.meilleure_fourmi = copy.deepcopy(fourmi)

    def deposer_pheromone(self, chemin: List[Ville], depot: float):
        """Dépose `depot` phéromones sur chaque route du chemin"""
        for v1, v2 in zip(chemin, chemin[1:]):
            route = self.trouver_route(v1, v2)
            if route is not None:
                route.pheromone += depot

    def calculer_distance_chemin(self, chemin: List[Ville]) -> float:
        if len(chemin) < 2:
            return float("inf")

        distance = 0
        for v1, v2 in zip(chemin, chemin[1:]):
            route = self.trouver_route(v1, v2)
            if route is None:
                return float("inf")
            distance += route.longueur
        return distance
