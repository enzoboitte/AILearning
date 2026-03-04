from test_loading_ai import ChargeurModele
import tkinter as tk
import os
import json
import time

class ApplicationIA:
    def __init__(self, fenetre):
        self.fenetre = fenetre
        self.fenetre.title("Créateur de Dataset IA & Inférence")

        # ── Modèle ──────────────────────────────────────────────────────────
        chargeur = ChargeurModele("model.txt")
        self.modele = chargeur.charger()
        # layers = [input, h1, h2, ..., output]
        # On affiche les colonnes : h1, h2, ..., output  (layers[1:])
        self.couches_visu = self.modele.layers[1:]   # ex: [128, 20, 10]
        self.nb_colonnes  = len(self.couches_visu)

        self.taille_entree = self.modele.taille_entree
        self.cote_image    = int(self.taille_entree ** 0.5)
        self.image_pixels  = [0.0] * self.taille_entree

        # ── Dossier entrainement ─────────────────────────────────────────────
        self.dossier_entrainement = "entrainement"
        self.fichier_json = os.path.join(self.dossier_entrainement, "dataset.json")
        if not os.path.exists(self.dossier_entrainement):
            os.makedirs(self.dossier_entrainement)
        if not os.path.exists(self.fichier_json):
            with open(self.fichier_json, "w") as f:
                json.dump([], f)

        # ── Interface ────────────────────────────────────────────────────────
        self.cadre_gauche = tk.Frame(fenetre)
        self.cadre_gauche.pack(side=tk.LEFT, padx=20, pady=20)
        self.cadre_droite = tk.Frame(fenetre, bg="#111111")
        self.cadre_droite.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Zone Dessin
        self.taille_pixel = max(2, 256 // self.cote_image)
        cote_canvas = self.cote_image * self.taille_pixel
        self.canvas_dessin = tk.Canvas(self.cadre_gauche, width=cote_canvas, height=cote_canvas, bg="black")
        self.canvas_dessin.pack()
        self.canvas_dessin.bind("<B1-Motion>", self.dessiner)

        self.label_resultat = tk.Label(self.cadre_gauche, text="Dessinez un chiffre", font=("Arial", 16, "bold"))
        self.label_resultat.pack(pady=10)

        self.bouton_effacer = tk.Button(self.cadre_gauche, text="Effacer le dessin", command=self.effacer_image)
        self.bouton_effacer.pack(pady=5)

        topo_str = " → ".join(str(s) for s in self.modele.layers)
        tk.Label(self.cadre_gauche, text=f"Topologie : {topo_str}", font=("Arial", 9), fg="gray").pack()

        # Boutons label
        nb_sorties = self.couches_visu[-1]
        frame_ent = tk.LabelFrame(self.cadre_gauche, text="Sauvegarder l'image", font=("Arial", 10))
        frame_ent.pack(pady=15, fill="x")
        for i in range(nb_sorties):
            tk.Button(frame_ent, text=str(i), width=3, font=("Arial", 12, "bold"),
                      command=lambda v=i: self.clic_apprentissage(v)).grid(
                row=i // 5, column=i % 5, padx=5, pady=5)

        # ── Canvas réseau ─────────────────────────────────────────────────────
        # Largeur : miniature + (nb_colonnes colonnes) + marges
        HAUTEUR   = 580
        MINI_W    = max(64, self.cote_image * max(1, 120 // self.cote_image))
        COL_W     = 130     # largeur par colonne de neurones
        MARGE     = 15
        largeur_canvas = MINI_W + MARGE + self.nb_colonnes * COL_W + MARGE + 30

        self.canvas_reseau = tk.Canvas(
            self.cadre_droite, width=largeur_canvas, height=HAUTEUR,
            bg="#111111", highlightthickness=0)
        self.canvas_reseau.pack()

        self.HAUTEUR    = HAUTEUR
        self.MINI_W     = MINI_W
        self.COL_W      = COL_W
        self.MARGE      = MARGE

        # ── Miniature pixels ──────────────────────────────────────────────────
        self.mini_taille = max(1, MINI_W // self.cote_image)
        self.mini_x0     = MARGE
        self.mini_y0     = (HAUTEUR - self.cote_image * self.mini_taille) // 2
        self.mini_rects  = {}
        self._creer_miniature()

        # ── Nœuds et liaisons ─────────────────────────────────────────────────
        # noeuds[c]  = liste d'IDs canvas  (c = index dans couches_visu)
        # liaisons[c][j][i] = ID ligne entre couche c et c+1
        self.noeuds    = [[] for _ in range(self.nb_colonnes)]
        self.liaisons   = []   # liaisons[c] = dict (j,i) -> id
        self.textes_proba = []

        self._creer_colonnes()

    # ─────────────────────────────────────────────────────────────────────────
    # Construction graphique
    # ─────────────────────────────────────────────────────────────────────────

    def _x_colonne(self, c):
        """Retourne le X central de la colonne c (0 = première couche cachée)."""
        return self.MINI_W + self.MARGE * 2 + c * self.COL_W + self.COL_W // 2

    def _creer_miniature(self):
        m, x0, y0, c = self.mini_taille, self.mini_x0, self.mini_y0, self.cote_image
        for py in range(c):
            for px in range(c):
                rid = self.canvas_reseau.create_rectangle(
                    x0 + px * m, y0 + py * m,
                    x0 + px * m + m - 1, y0 + py * m + m - 1,
                    fill="black", outline="")
                self.mini_rects[(px, py)] = rid
        # Label
        self.canvas_reseau.create_text(
            x0 + c * m // 2, y0 + c * m + 10,
            text=f"Entrée ({self.taille_entree})", fill="gray", font=("Arial", 8))
        # Flèche entrée → 1re colonne
        ax = x0 + c * m + 4
        self.canvas_reseau.create_line(
            ax, self.HAUTEUR // 2, ax + self.MARGE, self.HAUTEUR // 2,
            fill="#555555", arrow=tk.LAST, width=2)

    def _rayon(self, n):
        """Rayon dynamique des cercles selon nb de neurones."""
        return max(2, min(12, 220 // n))

    def _creer_colonnes(self):
        HAUTEUR = self.HAUTEUR
        nb_sorties = self.couches_visu[-1]

        for c, n in enumerate(self.couches_visu):
            x   = self._x_colonne(c)
            esp = (HAUTEUR - 40) / (n + 1)
            r   = self._rayon(n)
            is_output = (c == self.nb_colonnes - 1)

            # Liaisons vers la couche PRÉCÉDENTE
            if c > 0:
                n_prev = self.couches_visu[c - 1]
                x_prev = self._x_colonne(c - 1)
                esp_prev = (HAUTEUR - 40) / (n_prev + 1)
                liens = {}
                for j in range(n):
                    y_j = 20 + (j + 1) * esp
                    for i in range(n_prev):
                        y_i = 20 + (i + 1) * esp_prev
                        lid = self.canvas_reseau.create_line(
                            x_prev, y_i, x, y_j, fill="#1a1a1a", width=1)
                        liens[(j, i)] = lid
                self.liaisons.append(liens)
            else:
                self.liaisons.append(None)   # pas de liaisons à gauche de la 1re colonne

            # Nœuds
            for i in range(n):
                y = 20 + (i + 1) * esp
                oid = self.canvas_reseau.create_oval(
                    x - r, y - r, x + r, y + r,
                    fill="black", outline="white")
                self.noeuds[c].append(oid)
                # Etiquette de sortie
                if is_output:
                    self.canvas_reseau.create_text(
                        x + r + 14, y, text=str(i), fill="white", font=("Arial", 12, "bold"))
                    tp = self.canvas_reseau.create_text(
                        x - r - 20, y, text="0%", fill="gray", font=("Arial", 9))
                    self.textes_proba.append(tp)

            # Flèche entre colonnes (sauf après la dernière)
            if c < self.nb_colonnes - 1:
                x_next = self._x_colonne(c + 1)
                mid_y  = HAUTEUR // 2
                self.canvas_reseau.create_line(
                    x + r + 2, mid_y, x_next - r - 2, mid_y,
                    fill="#555555", arrow=tk.LAST, width=2)

            # Légende
            nom = f"{'Sortie' if is_output else f'Cachée {c+1}'} ({n}n)"
            self.canvas_reseau.create_text(
                x, HAUTEUR - 10, text=nom, fill="gray", font=("Arial", 8))

    # ─────────────────────────────────────────────────────────────────────────
    # Dessin / effacement
    # ─────────────────────────────────────────────────────────────────────────

    def dessiner(self, event):
        x_g = event.x // self.taille_pixel
        y_g = event.y // self.taille_pixel
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x_g + dx, y_g + dy
                if 0 <= nx < self.cote_image and 0 <= ny < self.cote_image:
                    idx = ny * self.cote_image + nx
                    if self.image_pixels[idx] == 0.0:
                        self.image_pixels[idx] = 1.0
                        p1, p2 = nx * self.taille_pixel, ny * self.taille_pixel
                        self.canvas_dessin.create_rectangle(
                            p1, p2, p1 + self.taille_pixel, p2 + self.taille_pixel,
                            fill="white", outline="white")
        self.predire_en_direct()

    def effacer_image(self):
        self.canvas_dessin.delete("all")
        self.image_pixels = [0.0] * self.taille_entree
        self.predire_en_direct()

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────────────────────────────────────────

    def sauvegarder_image_et_label(self, vrai_chiffre):
        timestamp = int(time.time() * 1000)
        nom = f"chiffre_{vrai_chiffre}_{timestamp}.pgm"
        chemin = os.path.join(self.dossier_entrainement, nom)
        with open(chemin, "w") as f:
            f.write(f"P2\n{self.cote_image} {self.cote_image}\n255\n")
            for i in range(self.taille_entree):
                f.write(f"{int(self.image_pixels[i] * 255)} ")
                if (i + 1) % self.cote_image == 0:
                    f.write("\n")
        with open(self.fichier_json, "r") as f:
            data = json.load(f)
        data.append({"chemin": chemin, "label": vrai_chiffre})
        with open(self.fichier_json, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✅ Sauvegardé : {nom} (Label: {vrai_chiffre})")

    def clic_apprentissage(self, vrai_chiffre):
        self.sauvegarder_image_et_label(vrai_chiffre)
        self.predire_en_direct()

    # ─────────────────────────────────────────────────────────────────────────
    # Inférence + mise à jour visuelle
    # ─────────────────────────────────────────────────────────────────────────

    def _maj_miniature(self):
        c = self.cote_image
        for py in range(c):
            for px in range(c):
                v = self.image_pixels[py * c + px]
                iv = int(v * 255)
                self.canvas_reseau.itemconfig(
                    self.mini_rects[(px, py)],
                    fill=f"#{iv:02x}{iv:02x}{iv:02x}")

    def predire_en_direct(self):
        self._maj_miniature()
        chiffre, precision, activations = self.modele.predire(self.image_pixels)
        # activations[0] = entrée, activations[1..L] = couches

        self.label_resultat.config(
            text=f"Résultat : {chiffre} ({precision:.1f}%)",
            fg="blue" if precision > 50 else "black")

        HAUTEUR = self.HAUTEUR

        for c, n in enumerate(self.couches_visu):
            act = activations[c + 1]   # activation de la couche c+1
            mx  = max(act) if max(act) > 0 else 1.0
            esp = (HAUTEUR - 40) / (n + 1)
            is_output = (c == self.nb_colonnes - 1)

            # Couleur des nœuds
            for i in range(n):
                if is_output:
                    iv = min(255, int(act[i] * 255))
                    fill = f"#{0:02x}{iv:02x}{0:02x}"
                else:
                    iv = min(255, int((act[i] / mx) * 255))
                    fill = f"#{iv:02x}{iv:02x}{iv:02x}"
                self.canvas_reseau.itemconfig(self.noeuds[c][i], fill=fill)

            # Probabilités (couche de sortie uniquement)
            if is_output:
                for i in range(n):
                    pct = act[i] * 100
                    self.canvas_reseau.itemconfig(
                        self.textes_proba[i],
                        text=f"{pct:.0f}%",
                        fill="white" if pct > 10 else "gray")

            # Couleur des liaisons (entre colonne c-1 et c)
            if c > 0 and self.liaisons[c] is not None:
                act_prev = activations[c]   # activation couche précédente
                n_prev   = self.couches_visu[c - 1]
                # poids de la couche c (index c dans modele.weights)
                W = self.modele.weights[c]
                for j in range(n):
                    for i in range(n_prev):
                        w      = W[j * n_prev + i]
                        signal = act_prev[i] * w
                        if abs(signal) < 0.005:
                            couleur = "#1a1a1a"
                        elif signal > 0:
                            iv     = min(255, int(signal * 1500))
                            couleur = f"#{0:02x}{iv:02x}{0:02x}"
                        else:
                            iv     = min(255, int(abs(signal) * 1500))
                            couleur = f"#{iv:02x}{0:02x}{0:02x}"
                        self.canvas_reseau.itemconfig(self.liaisons[c][(j, i)], fill=couleur)


if __name__ == "__main__":
    fenetre_principale = tk.Tk()
    app = ApplicationIA(fenetre_principale)
    fenetre_principale.mainloop()