import copy
import math
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import List, Optional, Tuple


@dataclass
class Route:
    """Route entre deux villes"""

    longueur: float
    pheromone: float = 0
    premiere_ville: Optional["Ville"] = None
    seconde_ville: Optional["Ville"] = None

    def evaporer_pheromone(self, taux_evaporation: float):
        """Simulation de l'évaporation de la phéromone"""
        self.pheromone = max(0.1, (1 - taux_evaporation) * self.pheromone)


@dataclass
class Ville:
    """Une ville (nœud) dans l'environnement"""

    x: int
    y: int
    id: int
    routes: List[Route] = None

    def __post_init__(self):
        if self.routes is None:
            self.routes = []

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if not isinstance(other, Ville):
            return False
        return self.id == other.id


class Ant:
    """Représente une fourmi avec des paramètres évolutifs"""

    def __init__(self, alpha: float, beta: float, gamma: float):
        self.alpha = alpha  # Importance des phéromones
        self.beta = beta  # Importance de la distance
        self.gamma = gamma  # Paramètre d'exploration
        self.porte_nourriture = (
            False  # Indique si la fourmi transporte de la nourriture (= trajet retour)
        )
        self.qte_nourriture_collectee = 0
        self.ville_actuelle = None
        self.ville_destination = None
        self.chemin_parcouru = []
        self.chemin_retour = []  # Pour stocker le chemin de retour
        self.chemins_uniques = set()  # Pour tracking de l'exploration
        self.performance_score = 0.0  # Score global de performance
        self.delai_initial = 0  # Délai avant le premier déplacement

        # Attributs pour l'animation
        self.x = 0  # Position actuelle en x
        self.y = 0  # Position actuelle en y
        self.progress = 0.0  # Progression sur l'arête actuelle (0.0 à 1.0)
        self.vitesse = 0.02  # Vitesse de déplacement en pixels/itération
        self.en_deplacement = False

    def initialiser_position(self, ville: Ville):
        """Initialise la position de la fourmi à une ville donnée"""
        self.ville_actuelle = ville
        self.x = ville.x
        self.y = ville.y
        self.en_deplacement = False
        self.progress = 0.0

    def selectionner_prochaine_ville(
        self, routes_disponibles: List[Route]
    ) -> Optional[Tuple[Ville, Route]]:
        if not routes_disponibles:
            return None

        # Si la fourmi porte de la nourriture et a un chemin de retour, elle suit ce chemin
        if self.porte_nourriture and self.chemin_retour:
            prochaine_ville = self.chemin_retour.pop(0)
            for route in routes_disponibles:
                destination = (
                    route.seconde_ville
                    if route.premiere_ville == self.ville_actuelle
                    else route.premiere_ville
                )
                if destination == prochaine_ville:
                    return prochaine_ville, route
            # Si la route n'est pas trouvée (cas rare), continue avec la sélection normale

        villes_non_visitees = []
        probabilities = []

        for route in routes_disponibles:
            ville_destination = (
                route.seconde_ville
                if route.premiere_ville == self.ville_actuelle
                else route.premiere_ville
            )

            # Facteur d'exploration (gamma) influençant la sélection des nouvelles routes
            exploration_factor = 1.0
            if ville_destination not in self.chemin_parcouru:
                exploration_factor = 1.0 + abs(self.gamma)

            # Éviter de revisiter les villes (sauf si pas d'autre choix)
            if (
                ville_destination in self.chemin_parcouru
                and len(routes_disponibles) > 1
            ):
                continue

            villes_non_visitees.append((ville_destination, route))

            # Formule modifiée intégrant l'exploration
            probability = (
                (route.pheromone**self.alpha)
                * ((1.0 / route.longueur) ** self.beta)
                * exploration_factor
            )
            probabilities.append(probability)

        if not villes_non_visitees:
            return None

        # Normalisation et sélection
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
            r = random.random()
            cumul = 0
            for i, (ville, route) in enumerate(villes_non_visitees):
                cumul += probabilities[i]
                if r <= cumul:
                    return ville, route

        return villes_non_visitees[-1]

    def mettre_a_jour_position(self):
        """Met à jour la position de la fourmi lors de son déplacement sur une arête"""
        if not self.en_deplacement or not self.ville_destination:
            return False

        # Vérification de sécurité pour la vitesse
        if self.vitesse > 0.1:  # Valeur maximale raisonnable
            self.vitesse = 0.02  # Réinitialiser à une valeur par défaut sûre

        # Calculer la distance totale entre les deux villes
        distance_totale = math.sqrt(
            (self.ville_destination.x - self.ville_actuelle.x) ** 2
            + (self.ville_destination.y - self.ville_actuelle.y) ** 2
        )

        # Calculer le pas de déplacement en pixels par itération (vitesse constante en pixels)
        pixels_par_iteration = max(1, min(3, self.vitesse * 100))

        # Calculer la progression comme un ratio de la distance parcourue
        increment = pixels_par_iteration / max(1, distance_totale)
        self.progress += increment

        # Limiter la progression à 1.0
        if self.progress >= 1.0:
            self.progress = 1.0
            self.x = self.ville_destination.x
            self.y = self.ville_destination.y
            return True

        # Calculer la nouvelle position par interpolation linéaire
        self.x = (
            self.ville_actuelle.x
            + (self.ville_destination.x - self.ville_actuelle.x) * self.progress
        )
        self.y = (
            self.ville_actuelle.y
            + (self.ville_destination.y - self.ville_actuelle.y) * self.progress
        )

        return False

    def demarrer_deplacement(self, ville_destination: Ville):
        """Démarre le déplacement vers une nouvelle ville"""
        self.ville_destination = ville_destination
        self.en_deplacement = True
        self.progress = 0.0

        # Stocke la position de départ pour l'interpolation
        self.start_x = self.x
        self.start_y = self.y

    def mettre_a_jour_performance(self):
        """Calcule le score de performance global de la fourmi"""
        # Mesure de performance basée sur l'exploration et l'exploitation
        exploration_score = len(self.chemins_uniques)
        exploitation_score = self.qte_nourriture_collectee

        # Le score global peut être une combinaison pondérée ou simplement la somme
        self.performance_score = exploration_score + exploitation_score

    def reinitialiser_stats(self):
        """Réinitialise les statistiques de la fourmi pour la nouvelle génération"""
        self.qte_nourriture_collectee = 0
        self.chemins_uniques.clear()
        self.performance_score = 0.0


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


class ACOVisualizer(tk.Tk):
    """Interface graphique avec support pour l'algorithme génétique"""

    def __init__(self):
        super().__init__()
        self.title("Simulation Colonie de Fourmis avec Algorithme Génétique")
        self.geometry("1200x800")

        self.civilisation = Civilisation()
        self.selected_city = None
        self.mode = "add_city"
        self.simulation_running = False
        self.animation_counter = 0
        self.post_genetic_delay = 0
        self.vitesse_de_base = 0.02
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panneau de contrôle
        control_frame = ttk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Modes
        modes_frame = ttk.LabelFrame(control_frame, text="Mode")
        modes_frame.pack(fill=tk.X, padx=5, pady=5)

        self.mode_var = tk.StringVar(value="add_city")
        modes = [
            ("Ajouter villes", "add_city"),
            ("Créer routes", "add_edge"),
            ("Placer nid", "set_nest"),
            ("Placer nourriture", "set_food"),
        ]
        for text, value in modes:
            ttk.Radiobutton(
                modes_frame,
                text=text,
                value=value,
                variable=self.mode_var,
                command=self.change_mode,
            ).pack(anchor=tk.W, padx=5, pady=2)

        # Paramètres de base
        params_frame = ttk.LabelFrame(control_frame, text="Paramètres de base")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(params_frame, text="Nombre de fourmis:").pack(anchor=tk.W, padx=5)
        self.ant_count = ttk.Entry(params_frame)
        self.ant_count.insert(0, "20")
        self.ant_count.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(params_frame, text="Alpha de base (phéromone):").pack(
            anchor=tk.W, padx=5
        )
        self.alpha_var = tk.StringVar(value="1.0")
        alpha_values = ["0.0", "0.5", "1.0", "2.0", "5.0"]
        alpha_combo = ttk.Combobox(
            params_frame, textvariable=self.alpha_var, values=alpha_values
        )
        alpha_combo.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(params_frame, text="Beta de base (distance):").pack(
            anchor=tk.W, padx=5
        )
        self.beta_var = tk.StringVar(value="2.0")
        beta_values = ["0.0", "1.0", "2.0", "5.0"]
        beta_combo = ttk.Combobox(
            params_frame, textvariable=self.beta_var, values=beta_values
        )
        beta_combo.pack(fill=tk.X, padx=5, pady=2)

        # Vitesse de simulation
        ttk.Label(params_frame, text="Vitesse des fourmis:").pack(anchor=tk.W, padx=5)
        self.speed_var = tk.StringVar(value="0.02")
        speed_values = ["0.01", "0.02", "0.05", "0.1", "0.2"]
        speed_combo = ttk.Combobox(
            params_frame, textvariable=self.speed_var, values=speed_values
        )
        speed_combo.pack(fill=tk.X, padx=5, pady=2)

        # Ajout d'un slider pour ajuster la vitesse en temps réel
        ttk.Label(params_frame, text="Ajustement en temps réel:").pack(
            anchor=tk.W, padx=5
        )
        self.speed_slider = ttk.Scale(
            params_frame,
            from_=0.005,
            to=0.1,
            orient="horizontal",
            command=self.adjust_speed,
        )
        self.speed_slider.set(float(self.speed_var.get()))
        self.speed_slider.pack(fill=tk.X, padx=5, pady=2)

        # Paramètres de l'algorithme génétique
        genetic_frame = ttk.LabelFrame(control_frame, text="Paramètres Génétiques")
        genetic_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(genetic_frame, text="Taux d'évaporation:").pack(anchor=tk.W, padx=5)
        self.evap_var = tk.StringVar(value="0.1")
        evap_values = ["0.01", "0.05", "0.1", "0.2", "0.5"]
        evap_combo = ttk.Combobox(
            genetic_frame, textvariable=self.evap_var, values=evap_values
        )
        evap_combo.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(genetic_frame, text="Nombre de générations:").pack(
            anchor=tk.W, padx=5
        )
        self.gen_count = ttk.Entry(genetic_frame)
        self.gen_count.insert(0, "10")
        self.gen_count.pack(fill=tk.X, padx=5, pady=2)

        # Boutons
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill=tk.X, padx=5, pady=10)

        self.genetic_button = ttk.Button(
            buttons_frame, text="Optimiser (GA)", command=self.apply_genetic_algorithm
        )
        self.genetic_button.pack(fill=tk.X, pady=2)

        self.start_button = ttk.Button(
            buttons_frame, text="Démarrer Simulation", command=self.toggle_simulation
        )
        self.start_button.pack(fill=tk.X, pady=2)

        ttk.Button(
            buttons_frame, text="Réinitialiser", command=self.reset_simulation
        ).pack(fill=tk.X, pady=2)

        # Bouton pour générer un graphe test
        ttk.Button(
            buttons_frame, text="Générer Graphe Test", command=self.generer_graphe_test
        ).pack(fill=tk.X, pady=2)

        # Canvas et infos
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Infos de base
        self.info_frame = ttk.LabelFrame(canvas_frame, text="Informations")
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.info_label = ttk.Label(self.info_frame, text="Mode: Ajouter des villes")
        self.info_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Légende pour les couleurs
        self.legend_frame = ttk.LabelFrame(canvas_frame, text="Légende")
        self.legend_frame.pack(fill=tk.X, padx=5, pady=5)

        legend_items = [
            ("Nid", "yellow"),
            ("Nourriture", "green"),
            ("Ville", "lightblue"),
            ("Arête", "#AAAAFF"),
            ("Meilleur chemin", "red"),
            ("Fourmi sans nourriture", "black"),
            ("Fourmi avec nourriture", "orange"),
        ]

        for text, color in legend_items:
            item_frame = ttk.Frame(self.legend_frame)
            item_frame.pack(side=tk.LEFT, padx=10, pady=5)

            canvas = tk.Canvas(item_frame, width=15, height=15, bg=color)
            canvas.pack(side=tk.LEFT)

            ttk.Label(item_frame, text=text).pack(side=tk.LEFT, padx=3)

        # Statistiques algorithme génétique
        self.stats_frame = ttk.LabelFrame(canvas_frame, text="Statistiques")
        self.stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.iter_label = ttk.Label(self.stats_frame, text="Itération: 0")
        self.iter_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.gen_label = ttk.Label(self.stats_frame, text="Génération: 0")
        self.gen_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.path_label = ttk.Label(self.stats_frame, text="Meilleur chemin: -")
        self.path_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.dist_label = ttk.Label(self.stats_frame, text="Distance: -")
        self.dist_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Statistiques de la meilleure fourmi
        self.best_ant_frame = ttk.LabelFrame(canvas_frame, text="Meilleure Fourmi")
        self.best_ant_frame.pack(fill=tk.X, padx=5, pady=5)

        self.best_alpha_label = ttk.Label(self.best_ant_frame, text="Alpha: -")
        self.best_alpha_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.best_beta_label = ttk.Label(self.best_ant_frame, text="Beta: -")
        self.best_beta_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.best_gamma_label = ttk.Label(self.best_ant_frame, text="Gamma: -")
        self.best_gamma_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.best_score_label = ttk.Label(self.best_ant_frame, text="Score: -")
        self.best_score_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bindings
        self.canvas.bind("<Button-1>", self.canvas_clicked)

    def adjust_speed(self, value):
        """Ajuste la vitesse des fourmis en temps réel"""
        speed = float(value)
        for fourmi in self.civilisation.fourmis:
            fourmi.vitesse = speed
        # Mettre à jour l'affichage de la vitesse
        self.speed_var.set(f"{speed:.3f}")
        self.vitesse_de_base = speed

    def change_mode(self):
        self.mode = self.mode_var.get()
        self.selected_city = None
        self.info_label.config(text=f"Mode: {self.mode}")
        self.redraw()

    def canvas_clicked(self, event):
        if self.simulation_running:
            return

        x, y = event.x, event.y
        clicked_city = self.find_closest_city(x, y)

        if self.mode == "add_city" and not clicked_city:
            self.civilisation.ajouter_ville(x, y)
        elif self.mode == "add_edge":
            if clicked_city:
                if not self.selected_city:
                    self.selected_city = clicked_city
                elif clicked_city != self.selected_city:
                    self.civilisation.ajouter_route(self.selected_city, clicked_city)
                    self.selected_city = None
        elif self.mode == "set_nest" and clicked_city:
            self.civilisation.nid = clicked_city
        elif self.mode == "set_food" and clicked_city:
            self.civilisation.src_nourriture = clicked_city

        self.redraw()

    def find_closest_city(self, x, y, threshold=20):
        for ville in self.civilisation.villes:
            dist = math.sqrt((ville.x - x) ** 2 + (ville.y - y) ** 2)
            if dist < threshold:
                return ville
        return None

    def redraw(self):
        self.canvas.delete("all")

        # Routes normales
        for route in self.civilisation.routes:
            # Couleur uniforme pour toutes les arêtes normales
            color = "#AAAAFF"  # Bleu clair
            width = (
                2
                if self.selected_city
                and (
                    route.premiere_ville == self.selected_city
                    or route.seconde_ville == self.selected_city
                )
                else 1
            )

            # Dessiner la ligne
            self.canvas.create_line(
                route.premiere_ville.x,
                route.premiere_ville.y,
                route.seconde_ville.x,
                route.seconde_ville.y,
                fill=color,
                width=width,
                tags="route",
            )

            # Calcul du milieu de la ligne pour positionner le texte
            mid_x = (route.premiere_ville.x + route.seconde_ville.x) / 2
            mid_y = (route.premiere_ville.y + route.seconde_ville.y) / 2

            # Afficher la phéromone et la longueur
            info_text = f"P: {route.pheromone:.1f}\nL: {route.longueur:.1f}"

            # Créer un fond blanc pour rendre le texte lisible
            self.canvas.create_rectangle(
                mid_x - 30,
                mid_y - 15,
                mid_x + 30,
                mid_y + 15,
                fill="white",
                outline="gray",
                tags="info_bg",
            )

            # Afficher le texte
            self.canvas.create_text(
                mid_x,
                mid_y,
                text=info_text,
                fill="black",
                font=("Arial", 8),
                tags="route_info",
            )

        # Meilleur chemin - Mise en évidence avec une ligne rouge plus épaisse
        if (
            self.civilisation.meilleur_chemin
            and len(self.civilisation.meilleur_chemin) > 1
        ):
            for i in range(len(self.civilisation.meilleur_chemin) - 1):
                v1 = self.civilisation.meilleur_chemin[i]
                v2 = self.civilisation.meilleur_chemin[i + 1]
                self.canvas.create_line(
                    v1.x, v1.y, v2.x, v2.y, fill="red", width=4, tags="optimal_path"
                )

        # Villes
        for ville in self.civilisation.villes:
            # Fond blanc pour la ville
            self.canvas.create_oval(
                ville.x - 12,
                ville.y - 12,
                ville.x + 12,
                ville.y + 12,
                fill="white",
                outline="white",
                tags="city_background",
            )

            # Dessin de la ville
            color = (
                "yellow"
                if ville == self.civilisation.nid
                else (
                    "green"
                    if ville == self.civilisation.src_nourriture
                    else "lightblue"
                )
            )
            outline = "red" if ville == self.selected_city else "black"
            width = 2 if ville == self.selected_city else 1

            self.canvas.create_oval(
                ville.x - 10,
                ville.y - 10,
                ville.x + 10,
                ville.y + 10,
                fill=color,
                outline=outline,
                width=width,
                tags="city",
            )

            # ID de la ville
            self.canvas.create_text(
                ville.x,
                ville.y - 20,
                text=str(ville.id),
                fill="black",
                font=("Arial", 10, "bold"),
                tags="city_label",
            )

        # Fourmis
        if hasattr(self.civilisation, "fourmis"):
            # Dessiner les fourmis elles-mêmes
            for fourmi in self.civilisation.fourmis:
                # Taille augmentée des fourmis
                taille_fourmi = 5

                # Différencier les fourmis selon leur état
                if fourmi.porte_nourriture:
                    # Fourmi qui transporte de la nourriture (trajet retour)
                    color = "orange"
                    outline = "red"
                    taille_fourmi = 6  # Encore plus grand pour les fourmis chargées
                else:
                    # Fourmi en recherche
                    color = "black"
                    outline = "gray"

                # Dessiner la fourmi avec une taille plus grande
                self.canvas.create_oval(
                    fourmi.x - taille_fourmi,
                    fourmi.y - taille_fourmi,
                    fourmi.x + taille_fourmi,
                    fourmi.y + taille_fourmi,
                    fill=color,
                    outline=outline,
                    width=1.5,
                    tags="ant",
                )

                # Direction de déplacement
                if fourmi.en_deplacement and fourmi.ville_destination:
                    # Calculer la direction
                    dx = fourmi.ville_destination.x - fourmi.ville_actuelle.x
                    dy = fourmi.ville_destination.y - fourmi.ville_actuelle.y
                    longueur = math.sqrt(dx * dx + dy * dy)
                    if longueur > 0:
                        dx /= longueur
                        dy /= longueur

                        # Dessiner une petite ligne indiquant la direction
                        self.canvas.create_line(
                            fourmi.x,
                            fourmi.y,
                            fourmi.x + dx * taille_fourmi * 1.5,
                            fourmi.y + dy * taille_fourmi * 1.5,
                            fill=color,
                            width=1.5,
                            tags="ant_direction",
                        )

        # Mise à jour des labels de la meilleure fourmi
        if self.civilisation.meilleure_fourmi:
            self.best_alpha_label.config(
                text=f"Alpha: {self.civilisation.meilleure_fourmi.alpha:.2f}"
            )
            self.best_beta_label.config(
                text=f"Beta: {self.civilisation.meilleure_fourmi.beta:.2f}"
            )
            self.best_gamma_label.config(
                text=f"Gamma: {self.civilisation.meilleure_fourmi.gamma:.2f}"
            )
            self.best_score_label.config(
                text=f"Score: {self.civilisation.meilleure_fourmi.performance_score:.2f}"
            )

    def apply_genetic_algorithm(self):
        """Applique l'algorithme génétique une seule fois avant la simulation"""
        if self.simulation_running:
            messagebox.showwarning(
                "Attention",
                "Arrêtez la simulation avant d'appliquer l'algorithme génétique!",
            )
            return

        if not self.civilisation.nid or not self.civilisation.src_nourriture:
            messagebox.showwarning(
                "Attention", "Définissez d'abord le nid et la source de nourriture!"
            )
            return

        try:
            num_ants = int(self.ant_count.get())
            alpha = float(self.alpha_var.get())
            beta = float(self.beta_var.get())
            generations = int(self.gen_count.get())
            evap_rate = float(self.evap_var.get())
            self.vitesse_de_base = min(0.05, float(self.speed_var.get()))
            self.speed_var.set(str(self.vitesse_de_base))
            self.speed_slider.set(self.vitesse_de_base)
            self.civilisation.taux_evaporation = evap_rate
        except ValueError:
            messagebox.showerror("Erreur", "Paramètres invalides")
            return

        # Afficher qu'on lance l'optimisation
        self.info_label.config(text="Optimisation génétique en cours...")
        self.update()  # Force la mise à jour de l'interface

        # Initialiser les fourmis si nécessaire
        if not self.civilisation.fourmis:
            self.civilisation.initialiser_fourmis(num_ants, alpha, beta)

        # Mise à jour de la vitesse des fourmis
        for fourmi in self.civilisation.fourmis:
            fourmi.vitesse = self.vitesse_de_base

        # Appliquer l'algorithme génétique
        self.civilisation.algorithme_genetique(generations)

        # Mettre à jour les statistiques
        self.gen_label.config(text=f"Génération: {self.civilisation.generation}")
        if self.civilisation.meilleure_fourmi:
            self.best_alpha_label.config(
                text=f"Alpha: {self.civilisation.meilleure_fourmi.alpha:.2f}"
            )
            self.best_beta_label.config(
                text=f"Beta: {self.civilisation.meilleure_fourmi.beta:.2f}"
            )
            self.best_gamma_label.config(
                text=f"Gamma: {self.civilisation.meilleure_fourmi.gamma:.2f}"
            )
            self.best_score_label.config(
                text=f"Score: {self.civilisation.meilleure_fourmi.performance_score:.2f}"
            )

        # Informer que l'optimisation est terminée
        self.info_label.config(
            text="Optimisation génétique terminée! Vous pouvez lancer la simulation."
        )

        # Redessiner
        self.redraw()

    def toggle_simulation(self):
        if not self.simulation_running:
            if not self.civilisation.nid or not self.civilisation.src_nourriture:
                messagebox.showwarning(
                    "Attention", "Définissez d'abord le nid et la source de nourriture!"
                )
                return

            try:
                num_ants = int(self.ant_count.get())
                alpha = float(self.alpha_var.get())
                beta = float(self.beta_var.get())
                evap_rate = float(self.evap_var.get())

                # Limiter la vitesse pour éviter les problèmes
                self.vitesse_de_base = min(0.05, float(self.speed_var.get()))
                self.speed_var.set(str(self.vitesse_de_base))
                self.speed_slider.set(self.vitesse_de_base)

                self.civilisation.taux_evaporation = evap_rate
            except ValueError:
                messagebox.showerror("Erreur", "Paramètres invalides")
                return

            self.simulation_running = True
            self.animation_counter = 0
            self.post_genetic_delay = 0
            self.start_button.config(text="Arrêter Simulation")

            # Si aucune fourmi n'existe encore ou si l'algo génétique n'a pas été appliqué
            if not self.civilisation.fourmis:
                self.civilisation.initialiser_fourmis(num_ants, alpha, beta)

            # Si l'algorithme génétique n'a pas encore été appliqué, suggérer de le faire
            if not self.civilisation.algo_genetique_applique:
                if messagebox.askyesno(
                    "Question",
                    "L'algorithme génétique n'a pas été appliqué. Souhaitez-vous l'appliquer maintenant ?",
                ):
                    self.apply_genetic_algorithm()

            # Application de la vitesse à toutes les fourmis
            for fourmi in self.civilisation.fourmis:
                fourmi.vitesse = self.vitesse_de_base

            self.update_simulation()
        else:
            self.simulation_running = False
            self.start_button.config(text="Démarrer Simulation")

    def update_simulation(self):
        if not self.simulation_running:
            return

        # Gérer la période post-génétique avec un délai
        if self.post_genetic_delay > 0:
            self.post_genetic_delay -= 1
            self.redraw()  # Juste redessiner sans mettre à jour la simulation
            self.after(33, self.update_simulation)
            return

        # Exécuter la logique de simulation
        actions_effectuees, meilleur_chemin, distance = self.civilisation.tour_suivant()

        # Mettre à jour les statistiques
        self.iter_label.config(text=f"Itération: {self.civilisation.iteration}")
        self.gen_label.config(text=f"Génération: {self.civilisation.generation}")
        if self.civilisation.meilleure_fourmi:
            self.best_score_label.config(
                text=f"Score: {self.civilisation.meilleure_fourmi.performance_score:.2f}"
            )

        if meilleur_chemin:
            path_str = " → ".join(str(v.id) for v in meilleur_chemin)
            self.path_label.config(text=f"Meilleur chemin: {path_str}")
            self.dist_label.config(text=f"Distance: {distance:.2f}")

            # Log pour debug
            if self.civilisation.iteration % 10 == 0:
                print(
                    f"Itération {self.civilisation.iteration}, "
                    f"Génération {self.civilisation.generation}: "
                    f"Meilleur chemin = {path_str}, "
                    f"Distance = {distance:.2f}"
                )

                # Afficher des statistiques sur les fourmis avec nourriture
                fourmis_avec_nourriture = sum(
                    1 for f in self.civilisation.fourmis if f.porte_nourriture
                )
                print(
                    f"Fourmis transportant de la nourriture: {fourmis_avec_nourriture}/{len(self.civilisation.fourmis)}"
                )

        # Redessiner la scène
        self.redraw()

        # Animation toujours fluide à 30 FPS
        self.after(33, self.update_simulation)

    def reset_simulation(self):
        """Réinitialise complètement la simulation"""
        self.simulation_running = False
        self.start_button.config(text="Démarrer Simulation")
        self.civilisation = Civilisation()
        self.selected_city = None
        self.post_genetic_delay = 0

        # Réinitialiser tous les labels
        self.iter_label.config(text="Itération: 0")
        self.gen_label.config(text="Génération: 0")
        self.path_label.config(text="Meilleur chemin: -")
        self.dist_label.config(text="Distance: -")
        self.best_alpha_label.config(text="Alpha: -")
        self.best_beta_label.config(text="Beta: -")
        self.best_gamma_label.config(text="Gamma: -")
        self.best_score_label.config(text="Score: -")
        self.info_label.config(text=f"Mode: {self.mode}")

        self.redraw()

    def generer_graphe_test(self):
        """Génère un graphe de test avec des nœuds et des arêtes aléatoires"""
        # Réinitialiser d'abord
        self.reset_simulation()

        # Paramètres du graphe
        nb_villes = 10
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Assurer que le canvas a une taille
        if canvas_width < 50 or canvas_height < 50:
            canvas_width = 800
            canvas_height = 600

        # Générer des villes avec un espacement minimal
        min_distance = 70
        for _ in range(nb_villes):
            max_attempts = 50
            for attempt in range(max_attempts):
                x = random.randint(50, canvas_width - 50)
                y = random.randint(50, canvas_height - 50)

                # Vérifier si cette position est à une distance minimale des autres villes
                too_close = False
                for ville in self.civilisation.villes:
                    dist = math.sqrt((ville.x - x) ** 2 + (ville.y - y) ** 2)
                    if dist < min_distance:
                        too_close = True
                        break

                if not too_close or attempt == max_attempts - 1:
                    self.civilisation.ajouter_ville(x, y)
                    break

        # Créer un arbre couvrant (pour s'assurer que le graphe est connecté)
        villes_non_connectees = self.civilisation.villes[1:]
        villes_connectees = [self.civilisation.villes[0]]

        while villes_non_connectees:
            v1 = random.choice(villes_connectees)
            v2 = random.choice(villes_non_connectees)
            self.civilisation.ajouter_route(v1, v2)
            villes_connectees.append(v2)
            villes_non_connectees.remove(v2)

        # Ajouter quelques routes supplémentaires
        nb_routes_supplementaires = nb_villes // 2
        for _ in range(nb_routes_supplementaires):
            v1 = random.choice(self.civilisation.villes)
            v2 = random.choice(self.civilisation.villes)
            if v1 != v2:
                self.civilisation.ajouter_route(v1, v2)

        # Définir le nid et la source de nourriture
        self.civilisation.nid = self.civilisation.villes[0]
        self.civilisation.src_nourriture = self.civilisation.villes[-1]

        # Mettre à jour l'affichage
        self.redraw()
        messagebox.showinfo(
            "Succès",
            f"Graphe généré avec {nb_villes} villes et {len(self.civilisation.routes)} routes",
        )


if __name__ == "__main__":
    app = ACOVisualizer()
    app.mainloop()
