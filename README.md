# Simulation de Colonie de Fourmis avec Algorithme Génétique

**Auteur**: Anas Tber (anas.tber@etu.ec-lyon.fr)  
**Contexte**: Projet développé dans le cadre de l'électif S8 "Algorithmes collaboratifs et applications"

## Description

Ce projet implémente une simulation visuelle d'optimisation par colonie de fourmis (ACO - Ant Colony Optimization) combinée avec des algorithmes génétiques. Le programme permet de visualiser comment les fourmis virtuelles trouvent le chemin optimal entre leur nid et une source de nourriture à travers un réseau de chemins, tout en optimisant leurs paramètres comportementaux grâce à une approche évolutionnaire.

## Fonctionnalités

- Interface graphique interactive pour créer et manipuler le graphe
- Optimisation des paramètres des fourmis via un algorithme génétique
- Visualisation en temps réel du déplacement des fourmis
- Traçage des chemins optimaux découverts
- Ajustement des paramètres de simulation (alpha, beta, gamma, taux d'évaporation)
- Génération automatique de graphes de test

## Tutoriel d'utilisation

1. **Lancement du programme**:
   ```
   python Rendu_ACO.py
   ```

2. **Création de l'environnement**:
   - Utilisez le mode "Ajouter villes" pour placer des nœuds
   - Passez au mode "Créer routes" pour relier les villes entre elles
   - Définissez le nid (colonie) et la source de nourriture avec les modes correspondants
   - Alternativement, utilisez le bouton "Générer Graphe Test" pour une configuration rapide

3. **Configuration et lancement de la simulation**:
   - Ajustez les paramètres de base (nombre de fourmis, alpha, beta)
   - Cliquez sur "Optimiser (GA)" pour appliquer l'algorithme génétique
   - Lancez la simulation avec "Démarrer Simulation"
   - Ajustez la vitesse en temps réel avec le curseur



## Aspects techniques

Le projet est composé de plusieurs classes principales:
- `Ville`: Représente un nœud dans le graphe
- `Route`: Gère les connexions entre les villes
- `Ant`: Implémente le comportement des fourmis avec leurs paramètres évolutifs
- `Civilisation`: Orchestre l'environnement et les mécanismes d'algorithme génétique
- `ACOVisualizer`: Interface graphique permettant de visualiser et contrôler la simulation

Les fourmis utilisent trois paramètres principaux qui évoluent grâce à l'algorithme génétique:
- **Alpha**: Importance accordée aux phéromones
- **Beta**: Importance accordée à la distance
- **Gamma**: Paramètre d'exploration des nouvelles routes

## Tests

Un script `tests.py` est fourni pour tester l'algorithme sur des configurations spécifiques:
- Test avec un graphe circulaire complexe
- Test avec un graphe contenant des pièges (impasses)

Pour exécuter les tests:
```
python tests.py
```

## Ressources supplémentaires

Une vidéo tutoriel est disponible dans le dossier du projet pour une démonstration complète des fonctionnalités.

## Prérequis

- Python 3.7+
- Tkinter (pour l'interface graphique)
- Matplotlib (pour les tests et visualisations)
- NumPy (pour certaines opérations mathématiques)