import math
import random

import matplotlib.pyplot as plt

# Importer les classes de Rendu_ACO.py
try:
    from Rendu_ACO import Civilisation
except ImportError:
    print("ERREUR: Impossible de trouver ou d'importer le fichier Rendu_ACO.py")
    print(
        "Assurez-vous que le fichier existe dans le même répertoire et qu'il est correctement nommé."
    )
    exit(1)


def formater_chemin(chemin):
    return " -> ".join(str(ville.id) for ville in chemin)


def initialiser_et_optimiser(civ, nb_fourmis=30, generations=5):
    """Crée la population, applique l'algorithme génétique et prépare la simulation"""
    civ.initialiser_fourmis(nb_fourmis, alpha_base=1.0, beta_base=2.0)
    civ.taux_evaporation = 0.1

    print("Application de l'algorithme génétique...")
    civ.algorithme_genetique(generations=generations)


def reinitialiser_fourmis(civ):
    """Replace toutes les fourmis au nid avant la simulation"""
    for fourmi in civ.fourmis:
        fourmi.reinitialiser_au_nid(civ.nid)
        fourmi.delai_initial = random.randint(0, 10)


def executer_simulation(civ, nb_iterations, intervalle_affichage, apres_tour=None):
    """Exécute la simulation en affichant régulièrement le meilleur chemin"""
    print("\nExécution de la simulation...")

    for i in range(nb_iterations):
        _, meilleur_chemin, meilleure_distance = civ.tour_suivant()

        if apres_tour:
            apres_tour(civ)

        if (i + 1) % intervalle_affichage == 0:
            print(f"Itération {i+1}:")
            if meilleur_chemin:
                print(f"  Meilleur chemin: {formater_chemin(meilleur_chemin)}")
                print(f"  Distance: {meilleure_distance}")

    if civ.meilleur_chemin:
        print(f"Meilleur chemin: {formater_chemin(civ.meilleur_chemin)}")
        print(f"Distance: {civ.meilleure_distance}")
    else:
        print("Aucun chemin trouvé!")


def tracer_graphe(civ, titre, fichier, pieges=()):
    """Sauvegarde une image du graphe avec le meilleur chemin (et les pièges éventuels)"""
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
        elif ville.id in pieges:
            plt.plot(ville.x, ville.y, "ro", markersize=8)  # Pièges en rouge
        else:
            plt.plot(ville.x, ville.y, "bo", markersize=8)  # Autres villes en bleu
        plt.text(ville.x + 5, ville.y + 5, str(ville.id))  # Ajouter IDs

    plt.title(titre)
    plt.axis("equal")

    # Ajouter une légende
    plt.plot([], [], "yo", markersize=10, label="Nid")
    plt.plot([], [], "go", markersize=10, label="Nourriture")
    plt.plot([], [], "bo", markersize=8, label="Villes")
    if pieges:
        plt.plot([], [], "ro", markersize=8, label="Pièges")
    plt.plot([], [], "r-", linewidth=2, label="Meilleur chemin")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Ajuster la mise en page et sauvegarder
    plt.tight_layout()
    plt.savefig(fichier)
    plt.close()


def test_cercle_complexe():
    """
    Test avec un graphe circulaire plus complexe
    """
    print("\n----- Test sur graphe circulaire complexe -----")

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
        villes.append(civ.ajouter_ville(int(x), int(y)))

    # Créer des routes entre villes adjacentes (forme un cercle)
    for i in range(nb_villes):
        civ.ajouter_route(villes[i], villes[(i + 1) % nb_villes])

    # Ajouter des routes diagonales (chemins alternatifs)
    for i in range(nb_villes):
        civ.ajouter_route(villes[i], villes[(i + 2) % nb_villes])
        civ.ajouter_route(villes[i], villes[(i + 3) % nb_villes])

    # Définir le nid et la source de nourriture
    civ.nid = villes[0]
    civ.src_nourriture = villes[nb_villes // 2]  # À l'opposé du nid

    initialiser_et_optimiser(civ)

    if civ.meilleure_fourmi:
        print("\nParamètres de la meilleure fourmi:")
        print(f"Alpha: {civ.meilleure_fourmi.alpha:.2f}")
        print(f"Beta: {civ.meilleure_fourmi.beta:.2f}")
        print(f"Gamma: {civ.meilleure_fourmi.gamma:.2f}")

    reinitialiser_fourmis(civ)
    executer_simulation(civ, nb_iterations=15000, intervalle_affichage=500)

    tracer_graphe(
        civ,
        "Visualisation du graphe circulaire et du meilleur chemin",
        "graphe_cercle_complexe.png",
    )
    return civ


def test_graphe_pieges():
    """
    Test avec un graphe contenant des pièges (cul-de-sacs, chemins plus longs mais plus attractifs)
    """
    print("\n----- Test sur graphe avec pièges -----")

    civ = Civilisation()

    # Créer la grille principale
    nb_lignes, nb_colonnes = 5, 5
    villes = [
        [civ.ajouter_ville(100 + j * 100, 100 + i * 100) for j in range(nb_colonnes)]
        for i in range(nb_lignes)
    ]

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

    # Créer des pièges sous forme d'impasses, reliés à un seul point de la grille
    ville_piege1 = civ.ajouter_ville(250, 250)  # Entre (1,1) et (2,2)
    ville_piege2 = civ.ajouter_ville(350, 350)  # Entre (2,2) et (3,3)
    ville_piege3 = civ.ajouter_ville(450, 250)  # Proche de (2,4)

    civ.ajouter_route(villes[1][1], ville_piege1)
    civ.ajouter_route(villes[2][2], ville_piege2)
    civ.ajouter_route(villes[2][3], ville_piege3)

    initialiser_et_optimiser(civ)
    reinitialiser_fourmis(civ)

    # Suivre l'utilisation des pièges
    visites_pieges = {ville_piege1.id: 0, ville_piege2.id: 0, ville_piege3.id: 0}

    def compter_visites_pieges(civ):
        for fourmi in civ.fourmis:
            if fourmi.ville_actuelle and fourmi.ville_actuelle.id in visites_pieges:
                visites_pieges[fourmi.ville_actuelle.id] += 1

    executer_simulation(
        civ,
        nb_iterations=800,
        intervalle_affichage=100,
        apres_tour=compter_visites_pieges,
    )

    print("\nStatistiques des pièges:")
    for ville_id, visites in visites_pieges.items():
        print(f"Ville {ville_id}: {visites} visites")

    tracer_graphe(
        civ,
        "Visualisation du graphe avec pièges et meilleur chemin",
        "graphe_pieges.png",
        pieges=visites_pieges,
    )
    return civ


# Programme principal
if __name__ == "__main__":
    print("===== TESTS DE L'ALGORITHME DE COLONIE DE FOURMIS =====")
    print("Ce script exécute des tests pour vérifier le bon fonctionnement du code.")

    try:
        print("\nDémarrage du test sur graphe circulaire complexe...")
        civ_cercle = test_cercle_complexe()

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
