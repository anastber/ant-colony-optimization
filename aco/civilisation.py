"""Environnement de simulation et algorithme génétique."""

import copy
import math
import random
from typing import List

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
        self.taux_evaporation = 0.1
        self.Q = 100.0
        self.iteration = 0
        self.generation = 0
        self.meilleure_fourmi = None
        self.algo_genetique_applique = (
            False  # Flag pour indiquer si l'algo génétique a été appliqué
        )

    def ajouter_ville(self, x: int, y: int) -> Ville:
        ville = Ville(x, y, len(self.villes))
        self.villes.append(ville)
        return ville

    def ajouter_route(self, v1: Ville, v2: Ville) -> Route:
        for route in self.routes:
            if (route.premiere_ville == v1 and route.seconde_ville == v2) or (
                route.premiere_ville == v2 and route.seconde_ville == v1
            ):
                return route

        longueur = math.sqrt((v1.x - v2.x) ** 2 + (v1.y - v2.y) ** 2)
        route = Route(longueur=longueur, premiere_ville=v1, seconde_ville=v2)
        self.routes.append(route)
        v1.routes.append(route)
        v2.routes.append(route)
        return route

    def initialiser_fourmis(
        self, nombre_fourmis: int, alpha_base: float, beta_base: float
    ):
        """Initialise la population de fourmis avec des paramètres variés"""
        self.fourmis = []
        for _ in range(nombre_fourmis):
            # Variation aléatoire autour des paramètres de base
            alpha = alpha_base * (0.5 + random.random())  # ±50% variation
            beta = beta_base * (0.5 + random.random())
            gamma = random.uniform(-2, 2)

            fourmi = Ant(alpha=alpha, beta=beta, gamma=gamma)
            fourmi.initialiser_position(self.nid)
            fourmi.chemin_parcouru = [self.nid]
            # Ajouter un délai initial aléatoire pour éviter les départs synchronisés
            fourmi.delai_initial = random.randint(0, 10)
            self.fourmis.append(fourmi)

        self.iteration = 0
        self.generation = 0

    def evaluer_population(self, iterations_test=50):
        """Évalue la performance des fourmis sur un nombre limité d'itérations"""
        # Sauvegarde des valeurs actuelles
        iterations_orig = self.iteration

        # Pour chaque fourmi, faire un certain nombre d'itérations pour évaluer sa performance
        for fourmi in self.fourmis:
            fourmi.reinitialiser_stats()
            fourmi.initialiser_position(self.nid)
            fourmi.chemin_parcouru = [self.nid]

            for _ in range(iterations_test):
                self.evaluer_fourmi(fourmi)

        # Restaurer les valeurs
        self.iteration = iterations_orig

        # Mettre à jour les scores de performance
        for fourmi in self.fourmis:
            fourmi.mettre_a_jour_performance()

    def evaluer_fourmi(self, fourmi):
        """Évalue une fourmi sur une itération"""
        if fourmi.en_deplacement:
            if fourmi.mettre_a_jour_position():
                # Si la fourmi atteint la source de nourriture
                if (
                    fourmi.ville_actuelle == self.src_nourriture
                    and not fourmi.porte_nourriture
                ):
                    fourmi.porte_nourriture = True
                    # Sauvegarder le chemin de retour (dans l'ordre inverse)
                    fourmi.chemin_retour = [
                        ville for ville in reversed(fourmi.chemin_parcouru[1:])
                    ]

                    # Toujours démarrer le chemin de retour immédiatement
                    if fourmi.chemin_retour:
                        prochaine_ville = fourmi.chemin_retour.pop(0)
                        for route in fourmi.ville_actuelle.routes:
                            destination = (
                                route.seconde_ville
                                if route.premiere_ville == fourmi.ville_actuelle
                                else route.premiere_ville
                            )
                            if destination == prochaine_ville:
                                fourmi.demarrer_deplacement(prochaine_ville)
                                break
                    return

                # Si la fourmi revient au nid ET porte de la nourriture
                elif fourmi.ville_actuelle == self.nid and fourmi.porte_nourriture:
                    fourmi.qte_nourriture_collectee += 1
                    fourmi.porte_nourriture = False  # Réinitialiser UNIQUEMENT ici

                    # Réinitialiser la fourmi pour un nouveau voyage
                    fourmi.initialiser_position(self.nid)
                    fourmi.chemin_parcouru = [self.nid]
                    fourmi.chemin_retour = []

                    # Sélectionner immédiatement la prochaine destination
                    result = fourmi.selectionner_prochaine_ville(
                        fourmi.ville_actuelle.routes
                    )
                    if result:
                        ville, route = result
                        fourmi.demarrer_deplacement(ville)
                        fourmi.chemin_parcouru.append(ville)
                return

        # Vérifier si une fourmi immobile est à la source de nourriture
        if (
            fourmi.ville_actuelle == self.src_nourriture
            and not fourmi.en_deplacement
            and not fourmi.porte_nourriture
        ):
            # Forcer la collecte de nourriture et le départ si bloquée à la source
            fourmi.porte_nourriture = True
            fourmi.chemin_retour = list(reversed(fourmi.chemin_parcouru[1:]))

            # S'assurer qu'elle démarre le retour
            if fourmi.chemin_retour:
                prochaine_ville = fourmi.chemin_retour.pop(0)
                for route in fourmi.ville_actuelle.routes:
                    destination = (
                        route.seconde_ville
                        if route.premiere_ville == fourmi.ville_actuelle
                        else route.premiere_ville
                    )
                    if destination == prochaine_ville:
                        fourmi.demarrer_deplacement(prochaine_ville)
                        break
            return

        # Si porte de la nourriture et en chemin de retour - ne pas reinitialiser porte_nourriture!
        if fourmi.porte_nourriture and not fourmi.en_deplacement:
            # Si au nid, déposer la nourriture
            if fourmi.ville_actuelle == self.nid:
                fourmi.qte_nourriture_collectee += 1
                fourmi.porte_nourriture = False
                fourmi.chemin_parcouru = [self.nid]
                fourmi.chemin_retour = []

                # Sélectionner nouvelle destination
                result = fourmi.selectionner_prochaine_ville(
                    fourmi.ville_actuelle.routes
                )
                if result:
                    ville, route = result
                    fourmi.demarrer_deplacement(ville)
                    fourmi.chemin_parcouru.append(ville)
                return

            # Sinon continuer le chemin de retour
            if fourmi.chemin_retour:
                prochaine_ville = fourmi.chemin_retour[0]
                for route in fourmi.ville_actuelle.routes:
                    destination = (
                        route.seconde_ville
                        if route.premiere_ville == fourmi.ville_actuelle
                        else route.premiere_ville
                    )
                    if destination == prochaine_ville:
                        fourmi.demarrer_deplacement(prochaine_ville)
                        fourmi.chemin_retour.pop(0)
                        return
            else:
                # Si plus de chemin retour mais pas encore au nid, diriger vers le nid
                for route in fourmi.ville_actuelle.routes:
                    destination = (
                        route.seconde_ville
                        if route.premiere_ville == fourmi.ville_actuelle
                        else route.premiere_ville
                    )
                    if destination == self.nid:
                        fourmi.demarrer_deplacement(self.nid)
                        return

        # Sélection de la prochaine destination pour l'exploration (uniquement si pas de nourriture)
        if not fourmi.porte_nourriture:
            result = fourmi.selectionner_prochaine_ville(fourmi.ville_actuelle.routes)
            if result:
                ville, route = result
                fourmi.demarrer_deplacement(ville)
                fourmi.chemin_parcouru.append(ville)
            else:
                # Si aucune ville n'est disponible, retour au nid
                fourmi.initialiser_position(self.nid)
                fourmi.chemin_parcouru = [self.nid]
                fourmi.chemin_retour = []
                fourmi.porte_nourriture = False

    def mutation(self, fourmi: Ant):
        """Applique une mutation aux paramètres de la fourmi"""
        # Mutations avec intensité variable
        intensite = random.random() * 0.4 + 0.8  # 80-120% du paramètre original
        fourmi.alpha *= intensite
        fourmi.beta *= intensite
        fourmi.gamma += random.uniform(-0.5, 0.5)

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

            # Sélectionner l'élite (20% meilleurs)
            nb_elite = max(2, len(self.fourmis) // 5)
            elite = self.fourmis[:nb_elite]

            # Créer nouvelle génération
            nouvelle_generation = []

            # Garder l'élite
            nouvelle_generation.extend(copy.deepcopy(f) for f in elite)

            # Compléter la population
            while len(nouvelle_generation) < len(self.fourmis):
                if random.random() < 0.3:  # 30% chance de croisement
                    parent1 = random.choice(elite)
                    parent2 = random.choice(elite)
                    enfant = self.croisement(parent1, parent2)
                    if random.random() < 0.1:  # 10% chance de mutation après croisement
                        self.mutation(enfant)
                else:  # 70% chance de mutation
                    modele = random.choice(elite)
                    enfant = copy.deepcopy(modele)
                    self.mutation(enfant)
                nouvelle_generation.append(enfant)

            # Mettre à jour la population
            self.fourmis = nouvelle_generation

            # Réinitialiser les fourmis pour l'évaluation suivante
            for fourmi in self.fourmis:
                fourmi.initialiser_position(self.nid)
                fourmi.chemin_parcouru = [self.nid]
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
                fourmi.initialiser_position(self.nid)
                fourmi.chemin_parcouru = [self.nid]
                fourmi.chemin_retour = []
                fourmi.porte_nourriture = False
                fourmi.delai_initial = random.randint(
                    0, 15
                )  # Délai pour départ progressif
            self.meilleure_fourmi.performance_score = (
                self.meilleure_fourmi.performance_score
            )

    def tour_suivant(self):
        """Effectue un tour de simulation. Retourne les actions effectuées et le meilleur chemin trouvé."""
        self.iteration += 1
        actions_effectuees = False

        chemins_valides = []
        distances_chemins = []

        # Mise à jour des positions des fourmis
        for fourmi in self.fourmis:
            # Gérer le délai initial
            if fourmi.delai_initial > 0:
                fourmi.delai_initial -= 1
                continue

            if fourmi.en_deplacement:
                est_arrivee = fourmi.mettre_a_jour_position()
                if est_arrivee:
                    actions_effectuees = True
                    fourmi.en_deplacement = False
                    fourmi.ville_actuelle = fourmi.ville_destination
                    fourmi.progress = 0.0

                    # Arrivée à la source de nourriture
                    if (
                        fourmi.ville_actuelle == self.src_nourriture
                        and not fourmi.porte_nourriture
                    ):
                        # La fourmi trouve de la nourriture
                        fourmi.porte_nourriture = True
                        # Inverser le chemin parcouru pour le retour
                        fourmi.chemin_retour = list(
                            reversed(fourmi.chemin_parcouru[1:])
                        )

                        # CORRECTION IMPORTANTE: Évaluer le chemin pour les phéromones
                        if len(fourmi.chemin_parcouru) > 1:
                            distance = self.calculer_distance_chemin(
                                fourmi.chemin_parcouru
                            )
                            if distance < float("inf"):
                                chemins_valides.append(fourmi.chemin_parcouru.copy())
                                distances_chemins.append(distance)
                                fourmi.chemins_uniques.add(
                                    tuple(fourmi.chemin_parcouru)
                                )
                                if distance < self.meilleure_distance:
                                    self.meilleure_distance = distance
                                    self.meilleur_chemin = fourmi.chemin_parcouru.copy()

                        # CORRECTION CRITIQUE: S'assurer que la fourmi commence immédiatement son trajet retour
                        if fourmi.chemin_retour:
                            # Obtenir la première ville du chemin de retour
                            prochaine_ville = fourmi.chemin_retour.pop(0)
                            # Démarrer le déplacement vers cette ville
                            fourmi.demarrer_deplacement(prochaine_ville)
                        continue  # Passer à la fourmi suivante après avoir démarré le retour

                    # Si la fourmi revient au nid ET porte de la nourriture
                    elif fourmi.ville_actuelle == self.nid and fourmi.porte_nourriture:
                        # La fourmi est revenue au nid - UNIQUEMENT ICI on réinitialise porte_nourriture
                        fourmi.qte_nourriture_collectee += 1
                        fourmi.porte_nourriture = (
                            False  # Réinitialisation seulement au nid
                        )

                        # Réinitialiser pour un nouveau trajet
                        fourmi.chemin_parcouru = [self.nid]
                        fourmi.chemin_retour = []

                        # Sélectionner une nouvelle destination immédiatement
                        result = fourmi.selectionner_prochaine_ville(
                            fourmi.ville_actuelle.routes
                        )
                        if result:
                            ville, route = result
                            fourmi.demarrer_deplacement(ville)
                            fourmi.chemin_parcouru.append(ville)
                        continue  # Passer à la fourmi suivante

                    # Si la fourmi est en chemin de retour et porte TOUJOURS de la nourriture
                    elif fourmi.porte_nourriture:
                        # Continuer le trajet de retour si des villes restent
                        if fourmi.chemin_retour:
                            prochaine_ville = fourmi.chemin_retour.pop(0)
                            fourmi.demarrer_deplacement(prochaine_ville)
                        else:
                            # Si c'est la dernière étape avant le nid, se diriger vers le nid
                            for route in fourmi.ville_actuelle.routes:
                                destination = (
                                    route.seconde_ville
                                    if route.premiere_ville == fourmi.ville_actuelle
                                    else route.premiere_ville
                                )
                                if destination == self.nid:
                                    fourmi.demarrer_deplacement(self.nid)
                                    break
                        continue  # Passer à la fourmi suivante

                    # Si la fourmi est en exploration normale (sans nourriture)
                    else:
                        # La fourmi continue son exploration
                        result = fourmi.selectionner_prochaine_ville(
                            fourmi.ville_actuelle.routes
                        )
                        if result:
                            ville, route = result
                            fourmi.demarrer_deplacement(ville)
                            fourmi.chemin_parcouru.append(ville)
                        else:
                            # Réinitialiser si bloquée
                            fourmi.initialiser_position(self.nid)
                            fourmi.chemin_parcouru = [self.nid]
                            fourmi.porte_nourriture = False

                continue  # Continue à la prochaine fourmi si en déplacement

            # Pour les fourmis immobiles

            # Vérifier si une fourmi immobile est à la source de nourriture
            if (
                fourmi.ville_actuelle == self.src_nourriture
                and not fourmi.en_deplacement
                and not fourmi.porte_nourriture
            ):
                # Forcer la collecte de nourriture et le départ
                fourmi.porte_nourriture = True
                fourmi.chemin_retour = list(reversed(fourmi.chemin_parcouru[1:]))

                # S'assurer qu'elle démarre le retour
                if fourmi.chemin_retour:
                    prochaine_ville = fourmi.chemin_retour.pop(0)
                    fourmi.demarrer_deplacement(prochaine_ville)
                    actions_effectuees = True
                continue  # Passer à la fourmi suivante

            # Gérer une fourmi immobile avec de la nourriture (en chemin de retour)
            if fourmi.porte_nourriture and not fourmi.en_deplacement:
                # Si la fourmi est au nid avec de la nourriture, la déposer
                if fourmi.ville_actuelle == self.nid:
                    fourmi.qte_nourriture_collectee += 1
                    fourmi.porte_nourriture = False
                    fourmi.chemin_parcouru = [self.nid]
                    fourmi.chemin_retour = []

                    # Sélectionner une nouvelle destination
                    result = fourmi.selectionner_prochaine_ville(
                        fourmi.ville_actuelle.routes
                    )
                    if result:
                        ville, route = result
                        fourmi.demarrer_deplacement(ville)
                        fourmi.chemin_parcouru.append(ville)
                    actions_effectuees = True
                    continue

                # Sinon continuer le chemin de retour
                if fourmi.chemin_retour:
                    prochaine_ville = fourmi.chemin_retour.pop(0)
                    fourmi.demarrer_deplacement(prochaine_ville)
                else:
                    # Si plus de villes dans le chemin de retour, diriger vers le nid
                    for route in fourmi.ville_actuelle.routes:
                        destination = (
                            route.seconde_ville
                            if route.premiere_ville == fourmi.ville_actuelle
                            else route.premiere_ville
                        )
                        if destination == self.nid:
                            fourmi.demarrer_deplacement(self.nid)
                            break
                actions_effectuees = True
                continue

            # Gérer une fourmi immobile en exploration (sans nourriture)
            if not fourmi.porte_nourriture and not fourmi.en_deplacement:
                # Sélection de la prochaine destination pour exploration
                result = fourmi.selectionner_prochaine_ville(
                    fourmi.ville_actuelle.routes
                )
                if result:
                    ville, route = result
                    fourmi.demarrer_deplacement(ville)
                    fourmi.chemin_parcouru.append(ville)
                    actions_effectuees = True

        # Évaporation des phéromones
        for route in self.routes:
            route.evaporer_pheromone(self.taux_evaporation)

        # Dépôt de phéromones
        for chemin, distance in zip(chemins_valides, distances_chemins):
            # Augmenter le dépôt pour les plus courts chemins
            depot = (self.Q / distance) * 1.5  # Facteur de renforcement
            for i in range(len(chemin) - 1):
                v1, v2 = chemin[i], chemin[i + 1]
                for route in self.routes:
                    if (route.premiere_ville == v1 and route.seconde_ville == v2) or (
                        route.premiere_ville == v2 and route.seconde_ville == v1
                    ):
                        route.pheromone += depot
                        break

        # Renforcement élitiste encore plus fort
        if self.meilleur_chemin:
            depot_elite = (
                self.Q / self.meilleure_distance
            ) * 2.0  # Renforcement supplémentaire
            for i in range(len(self.meilleur_chemin) - 1):
                v1, v2 = self.meilleur_chemin[i], self.meilleur_chemin[i + 1]
                for route in self.routes:
                    if (route.premiere_ville == v1 and route.seconde_ville == v2) or (
                        route.premiere_ville == v2 and route.seconde_ville == v1
                    ):
                        route.pheromone += depot_elite
                        break
        for fourmi in self.fourmis:
            fourmi.mettre_a_jour_performance()
            # Mettre à jour la meilleure fourmi si nécessaire
            if (
                not self.meilleure_fourmi
                or fourmi.performance_score > self.meilleure_fourmi.performance_score
            ):
                self.meilleure_fourmi = copy.deepcopy(fourmi)
        return actions_effectuees, self.meilleur_chemin, self.meilleure_distance

    def calculer_distance_chemin(self, chemin: List[Ville]) -> float:
        if len(chemin) < 2:
            return float("inf")

        distance = 0
        for i in range(len(chemin) - 1):
            v1, v2 = chemin[i], chemin[i + 1]
            route = next(
                (
                    r
                    for r in self.routes
                    if (r.premiere_ville == v1 and r.seconde_ville == v2)
                    or (r.premiere_ville == v2 and r.seconde_ville == v1)
                ),
                None,
            )
            if route:
                distance += route.longueur
            else:
                return float("inf")
        return distance
