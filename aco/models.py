"""Structures du graphe : villes (nœuds) et routes (arêtes)."""

from dataclasses import dataclass
from typing import List, Optional

from .config import PHEROMONE_MIN


@dataclass
class Route:
    """Route entre deux villes"""

    longueur: float
    pheromone: float = 0
    premiere_ville: Optional["Ville"] = None
    seconde_ville: Optional["Ville"] = None

    def evaporer_pheromone(self, taux_evaporation: float):
        """Simulation de l'évaporation de la phéromone"""
        self.pheromone = max(
            PHEROMONE_MIN, (1 - taux_evaporation) * self.pheromone
        )

    def autre_extremite(self, ville: "Ville") -> Optional["Ville"]:
        """Retourne la ville située à l'autre bout de la route"""
        return self.seconde_ville if self.premiere_ville == ville else self.premiere_ville

    def relie(self, v1: "Ville", v2: "Ville") -> bool:
        """Indique si la route relie v1 et v2 (dans un sens ou dans l'autre)"""
        return (self.premiere_ville == v1 and self.seconde_ville == v2) or (
            self.premiere_ville == v2 and self.seconde_ville == v1
        )


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
