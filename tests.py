import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np

# Importer les classes de Rendu_ACO.py
try:
    from Rendu_ACO import Ant, Civilisation, Route, Ville
except ImportError:
    print("ERREUR: Impossible de trouver ou d'importer le fichier Rendu_ACO.py")
    print(
        "Assurez-vous que le fichier existe dans le même répertoire et qu'il est correctement nommé."
    )
    exit(1)


def test_cercle_complexe():
    """
    Test avec un graphe circulaire plus complexe
    """
    print("\n----- Test sur graphe circulaire complexe -----")

    # Créer une instance de Civilisation
    civ = Civilisation()

    # Créer un graphe circulaire avec des raccourcis
    villes = []
    rayon = 200
    centre_x, centre_y = 300, 300
    nb_villes = 8

    for i in range(nb_villes):
        angle = 2 * math.pi * i / nb_villes
        x = centre_x + rayon * math.cos(angle)
        y = centre_y + rayon * math.sin(angle)
        ville = civ.ajouter_ville(int(x), int(y))
        villes.append(ville)

    # Créer des routes entre villes adjacentes (forme un cercle)
    for i in range(nb_villes):
        civ.ajouter_route(villes[i], villes[(i + 1) % nb_villes])

    # Ajouter des routes diagonales (chemins alternatifs)
    for i in range(nb_villes):
        civ.ajouter_route(villes[i], villes[(i + 2) % nb_villes])
        # Ajouter encore plus de connexions pour rendre le problème plus complexe
        civ.ajouter_route(villes[i], villes[(i + 3) % nb_villes])

    # Définir le nid et la source de nourriture
    civ.nid = villes[0]
    civ.src_nourriture = villes[nb_villes // 2]  # À l'opposé du nid

    # Initialiser les fourmis avec des valeurs par défaut
    nb_fourmis = 30
    civ.initialiser_fourmis(nb_fourmis, alpha_base=1.0, beta_base=2.0)

    # Définir le taux d'évaporation
    civ.taux_evaporation = 0.1

    # Appliquer l'algorithme génétique
    print("Application de l'algorithme génétique...")
    civ.algorithme_genetique(generations=5)

    # Si une meilleure fourmi a été identifiée
    if civ.meilleure_fourmi:
        print("\nParamètres de la meilleure fourmi:")
        print(f"Alpha: {civ.meilleure_fourmi.alpha:.2f}")
        print(f"Beta: {civ.meilleure_fourmi.beta:.2f}")
        print(f"Gamma: {civ.meilleure_fourmi.gamma:.2f}")

    # Réinitialiser les statistiques pour la simulation
    for fourmi in civ.fourmis:
        fourmi.initialiser_position(civ.nid)
        fourmi.chemin_parcouru = [civ.nid]
        fourmi.chemin_retour = []
        fourmi.porte_nourriture = False
        fourmi.delai_initial = random.randint(0, 10)

    # Stocker les données pour le suivi
    iterations = []
    distances = []

    # Exécuter la simulation
    print("\nExécution de la simulation...")
    nb_iterations = 15000

    for i in range(nb_iterations):
        actions_effectuees, meilleur_chemin, meilleure_distance = civ.tour_suivant()

        iterations.append(i)
        distances.append(
            meilleure_distance if meilleure_distance != float("inf") else None
        )

        if (i + 1) % 500 == 0:
            print(f"Itération {i+1}:")
            if meilleur_chemin:
                chemin_ids = [ville.id for ville in meilleur_chemin]
                print(f"  Meilleur chemin: {' -> '.join(map(str, chemin_ids))}")
                print(f"  Distance: {meilleure_distance}")

    if civ.meilleur_chemin:
        chemin_ids = [ville.id for ville in civ.meilleur_chemin]
        print(f"Meilleur chemin: {' -> '.join(map(str, chemin_ids))}")
        print(f"Distance: {civ.meilleure_distance}")
    else:
        print("Aucun chemin trouvé!")

    # Visualiser uniquement le graphe et le meilleur chemin
    plt.figure(figsize=(10, 8))

    # Tracer toutes les routes
    for route in civ.routes:
        x = [route.premiere_ville.x, route.seconde_ville.x]
        y = [route.premiere_ville.y, route.seconde_ville.y]
        plt.plot(x, y, "k-", alpha=0.2)

    # Tracer le meilleur chemin
    if civ.meilleur_chemin and len(civ.meilleur_chemin) > 1:
        x_best = [ville.x for ville in civ.meilleur_chemin]
        y_best = [ville.y for ville in civ.meilleur_chemin]
        plt.plot(x_best, y_best, "r-", linewidth=2, label="Meilleur chemin")

    # Placer les villes
    for ville in civ.villes:
        if ville == civ.nid:
            plt.plot(ville.x, ville.y, "yo", markersize=10)  # Nid en jaune
        elif ville == civ.src_nourriture:
            plt.plot(ville.x, ville.y, "go", markersize=10)  # Nourriture en vert
        else:
            plt.plot(ville.x, ville.y, "bo", markersize=8)  # Autres villes en bleu
        plt.text(ville.x + 5, ville.y + 5, str(ville.id))  # Ajouter IDs

    plt.title("Visualisation du graphe circulaire et du meilleur chemin")
    plt.axis("equal")

    # Ajouter une légende
    plt.plot([], [], "yo", markersize=10, label="Nid")
    plt.plot([], [], "go", markersize=10, label="Nourriture")
    plt.plot([], [], "bo", markersize=8, label="Villes")
    plt.plot([], [], "r-", linewidth=2, label="Meilleur chemin")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Ajuster la mise en page et sauvegarder
    plt.tight_layout()
    plt.savefig("graphe_cercle_complexe.png")
    plt.close()

    return civ


def test_graphe_pieges():
    """
    Test avec un graphe contenant des pièges (cul-de-sacs, chemins plus longs mais plus attractifs)
    """
    print("\n----- Test sur graphe avec pièges -----")

    # Créer une instance de Civilisation
    civ = Civilisation()

    # Créer la grille principale
    nb_lignes, nb_colonnes = 5, 5
    villes = [[None for _ in range(nb_colonnes)] for _ in range(nb_lignes)]

    # Créer les villes en grille
    for i in range(nb_lignes):
        for j in range(nb_colonnes):
            villes[i][j] = civ.ajouter_ville(100 + j * 100, 100 + i * 100)

    # Créer les routes de base (horizontales et verticales)
    for i in range(nb_lignes):
        for j in range(nb_colonnes):
            if j < nb_colonnes - 1:  # Route horizontale
                civ.ajouter_route(villes[i][j], villes[i][j + 1])
            if i < nb_lignes - 1:  # Route verticale
                civ.ajouter_route(villes[i][j], villes[i + 1][j])

    # Définir le nid et la source de nourriture
    civ.nid = villes[0][0]  # Coin supérieur gauche
    civ.src_nourriture = villes[nb_lignes - 1][nb_colonnes - 1]  # Coin inférieur droit

    # Créer des pièges - Tous sous forme d'impasses
    # Créer des villes de piège
    ville_piege1 = civ.ajouter_ville(250, 250)  # Entre (1,1) et (2,2) - Impasse
    ville_piege2 = civ.ajouter_ville(350, 350)  # Entre (2,2) et (3,3) - Impasse
    ville_piege3 = civ.ajouter_ville(450, 250)  # Proche de (2,4) - Déjà une impasse

    # Créer des routes qui mènent aux impasses
    civ.ajouter_route(villes[1][1], ville_piege1)  # Entrée de l'impasse 1
    civ.ajouter_route(villes[2][2], ville_piege2)  # Entrée de l'impasse 2
    civ.ajouter_route(villes[2][3], ville_piege3)  # Entrée de l'impasse 3

    # Nous ne relions plus les pièges entre eux ni au reste du graphe pour qu'ils soient de vraies impasses

    # Initialiser les fourmis
    nb_fourmis = 30
    civ.initialiser_fourmis(nb_fourmis, alpha_base=1.0, beta_base=2.0)
    civ.taux_evaporation = 0.1

    # Appliquer l'algorithme génétique
    print("Application de l'algorithme génétique...")
    civ.algorithme_genetique(generations=5)

    # Réinitialiser les fourmis pour la simulation
    for fourmi in civ.fourmis:
        fourmi.initialiser_position(civ.nid)
        fourmi.chemin_parcouru = [civ.nid]
        fourmi.chemin_retour = []
        fourmi.porte_nourriture = False
        fourmi.delai_initial = random.randint(0, 10)

    # Stocker les données pour le suivi
    iterations = []
    distances = []

    # Suivre l'utilisation des pièges
    visites_pieges = {ville_piege1.id: 0, ville_piege2.id: 0, ville_piege3.id: 0}

    # Exécuter la simulation
    print("\nExécution de la simulation...")
    nb_iterations = 800

    for i in range(nb_iterations):
        actions_effectuees, meilleur_chemin, meilleure_distance = civ.tour_suivant()

        # Collecter les statistiques standard
        iterations.append(i)
        distances.append(
            meilleure_distance if meilleure_distance != float("inf") else None
        )

        # Compter les fourmis qui sont dans les pièges
        for fourmi in civ.fourmis:
            if fourmi.ville_actuelle:
                ville_id = fourmi.ville_actuelle.id
                if ville_id in visites_pieges:
                    visites_pieges[ville_id] += 1

        if (i + 1) % 100 == 0:
            print(f"Itération {i+1}:")
            if meilleur_chemin:
                chemin_ids = [ville.id for ville in meilleur_chemin]
                print(f"  Meilleur chemin: {' -> '.join(map(str, chemin_ids))}")
                print(f"  Distance: {meilleure_distance}")

    if civ.meilleur_chemin:
        chemin_ids = [ville.id for ville in civ.meilleur_chemin]
        print(f"Meilleur chemin: {' -> '.join(map(str, chemin_ids))}")
        print(f"Distance: {civ.meilleure_distance}")
    else:
        print("Aucun chemin trouvé!")

    print("\nStatistiques des pièges:")
    for ville_id, visites in visites_pieges.items():
        print(f"Ville {ville_id}: {visites} visites")

    # Visualiser uniquement le graphe et le meilleur chemin
    plt.figure(figsize=(10, 8))

    # Tracer toutes les routes
    for route in civ.routes:
        x = [route.premiere_ville.x, route.seconde_ville.x]
        y = [route.premiere_ville.y, route.seconde_ville.y]
        plt.plot(x, y, "k-", alpha=0.2)

    # Tracer le meilleur chemin
    if civ.meilleur_chemin and len(civ.meilleur_chemin) > 1:
        x_best = [ville.x for ville in civ.meilleur_chemin]
        y_best = [ville.y for ville in civ.meilleur_chemin]
        plt.plot(x_best, y_best, "r-", linewidth=2, label="Meilleur chemin")

    # Placer les villes
    for ville in civ.villes:
        if ville == civ.nid:
            plt.plot(ville.x, ville.y, "yo", markersize=10)  # Nid en jaune
        elif ville == civ.src_nourriture:
            plt.plot(ville.x, ville.y, "go", markersize=10)  # Nourriture en vert
        elif ville.id in visites_pieges:
            plt.plot(ville.x, ville.y, "ro", markersize=8)  # Pièges en rouge
        else:
            plt.plot(ville.x, ville.y, "bo", markersize=8)  # Autres villes en bleu
        plt.text(ville.x + 5, ville.y + 5, str(ville.id))  # Ajouter IDs

    plt.title("Visualisation du graphe avec pièges et meilleur chemin")
    plt.axis("equal")

    # Ajouter une légende
    plt.plot([], [], "yo", markersize=10, label="Nid")
    plt.plot([], [], "go", markersize=10, label="Nourriture")
    plt.plot([], [], "bo", markersize=8, label="Villes")
    plt.plot([], [], "ro", markersize=8, label="Pièges")
    plt.plot([], [], "r-", linewidth=2, label="Meilleur chemin")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Ajuster la mise en page et sauvegarder
    plt.tight_layout()
    plt.savefig("graphe_pieges.png")
    plt.close()

    return civ


# Programme principal
if __name__ == "__main__":
    print("===== TESTS DE L'ALGORITHME DE COLONIE DE FOURMIS =====")
    print("Ce script exécute des tests pour vérifier le bon fonctionnement du code.")

    try:
        # Test 1: Graphe circulaire complexe
        print("\nDémarrage du test sur graphe circulaire complexe...")
        civ_cercle = test_cercle_complexe()

        # Test 2: Graphe avec pièges
        print("\nDémarrage du test sur graphe avec pièges...")
        civ_pieges = test_graphe_pieges()

        print("\n===== TOUS LES TESTS SONT TERMINÉS =====")
        print("Les graphiques ont été générés dans le répertoire courant:")
        print("- graphe_cercle_complexe.png")
        print("- graphe_pieges.png")

    except Exception as e:
        print(f"\nERREUR: Une exception s'est produite pendant les tests: {e}")
        import traceback

        traceback.print_exc()
