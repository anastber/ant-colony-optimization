import math
import random
import tkinter as tk
from tkinter import messagebox, ttk

from aco.ant import Ant
from aco.civilisation import Civilisation
from aco.models import Route, Ville


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
