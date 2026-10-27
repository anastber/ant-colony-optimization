"""Fourmi aux paramètres évolutifs (alpha, beta, gamma)."""

import math
import random
from typing import List, Optional, Tuple

from .models import Route, Ville


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

    def repartir_du_nid(self, nid: Ville):
        """Replace la fourmi au nid avec un chemin parcouru vierge"""
        self.initialiser_position(nid)
        self.chemin_parcouru = [nid]

    def reinitialiser_au_nid(self, nid: Ville):
        """Replace la fourmi au nid et efface son trajet et sa charge"""
        self.repartir_du_nid(nid)
        self.chemin_retour = []
        self.porte_nourriture = False

    def charger_nourriture(self):
        """Prend la nourriture et prépare le retour par le chemin inverse"""
        self.porte_nourriture = True
        self.chemin_retour = list(reversed(self.chemin_parcouru[1:]))

    def deposer_nourriture(self, nid: Ville):
        """Dépose la nourriture au nid et prépare un nouveau trajet"""
        self.qte_nourriture_collectee += 1
        self.porte_nourriture = False
        self.chemin_parcouru = [nid]
        self.chemin_retour = []

    def est_reliee_a(self, ville: Ville) -> bool:
        """Indique si une route relie la ville actuelle à `ville`"""
        return any(
            route.autre_extremite(self.ville_actuelle) == ville
            for route in self.ville_actuelle.routes
        )

    def explorer(self) -> bool:
        """Choisit la prochaine ville et s'y dirige. Retourne False si aucune n'est disponible."""
        result = self.selectionner_prochaine_ville(self.ville_actuelle.routes)
        if not result:
            return False
        ville, _ = result
        self.demarrer_deplacement(ville)
        self.chemin_parcouru.append(ville)
        return True

    def selectionner_prochaine_ville(
        self, routes_disponibles: List[Route]
    ) -> Optional[Tuple[Ville, Route]]:
        if not routes_disponibles:
            return None

        # Si la fourmi porte de la nourriture et a un chemin de retour, elle suit ce chemin
        if self.porte_nourriture and self.chemin_retour:
            prochaine_ville = self.chemin_retour.pop(0)
            for route in routes_disponibles:
                destination = route.autre_extremite(self.ville_actuelle)
                if destination == prochaine_ville:
                    return prochaine_ville, route
            # Si la route n'est pas trouvée (cas rare), continue avec la sélection normale

        villes_non_visitees = []
        probabilities = []

        for route in routes_disponibles:
            ville_destination = route.autre_extremite(self.ville_actuelle)

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
