"""Structures du graphe : villes (nœuds) et routes (arêtes)."""

from dataclasses import dataclass
from typing import List, Optional


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
