"""Point d'entrée de la simulation de colonie de fourmis.

Les classes sont réexportées ici pour conserver la compatibilité avec
``from Rendu_ACO import Ant, Civilisation, Route, Ville``.
"""

from aco import Ant, Civilisation, Route, Ville

__all__ = ["Ant", "Civilisation", "Route", "Ville", "main"]


def main():
    # Import local : Tkinter n'est requis que pour l'interface graphique
    from aco.gui import ACOVisualizer

    app = ACOVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()
