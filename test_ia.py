from test_loading_ai import ChargeurModele, Modele
import tkinter as tk
import os
import json
import time

class ApplicationIA:
    def __init__(self, fenetre):
        self.fenetre = fenetre
        self.fenetre.title("Créateur de Dataset IA & Inférence")
        
        # --- INITIALISATION DE L'OBJET MODELE ---
        chargeur = ChargeurModele("model.txt")
        self.modele = chargeur.charger() # C'est maintenant un objet Modele !
        self.image_pixels = [0.0] * 4096
        
        # --- PRÉPARATION DU DOSSIER ---
        self.dossier_entrainement = "entrainement"
        self.fichier_json = os.path.join(self.dossier_entrainement, "dataset.json")
        if not os.path.exists(self.dossier_entrainement): os.makedirs(self.dossier_entrainement)
        if not os.path.exists(self.fichier_json):
            with open(self.fichier_json, "w") as f: json.dump([], f)
        
        # --- INTERFACE ---
        self.cadre_gauche = tk.Frame(fenetre)
        self.cadre_gauche.pack(side=tk.LEFT, padx=20, pady=20)
        self.cadre_droite = tk.Frame(fenetre, bg='#111111')
        self.cadre_droite.pack(side=tk.RIGHT, padx=20, pady=20)

        # Zone Dessin
        self.taille_pixel = 4 
        self.canvas_dessin = tk.Canvas(self.cadre_gauche, width=64*self.taille_pixel, height=64*self.taille_pixel, bg='black')
        self.canvas_dessin.pack()
        self.canvas_dessin.bind("<B1-Motion>", self.dessiner)
        
        self.label_resultat = tk.Label(self.cadre_gauche, text="Dessinez un chiffre", font=("Arial", 16, "bold"))
        self.label_resultat.pack(pady=10)
        
        self.bouton_effacer = tk.Button(self.cadre_gauche, text="Effacer le dessin", command=self.effacer_image)
        self.bouton_effacer.pack(pady=5)

        # Zone Entraînement
        frame_entrainement = tk.LabelFrame(self.cadre_gauche, text="Sauvegarder l'image", font=("Arial", 10))
        frame_entrainement.pack(pady=15, fill="x")
        
        for i in range(10):
            btn = tk.Button(frame_entrainement, text=str(i), width=3, font=("Arial", 12, "bold"), 
                            command=lambda val=i: self.clic_apprentissage(val))
            btn.grid(row=i//5, column=i%5, padx=5, pady=5)

        # Zone Réseau Visuel
        self.canvas_reseau = tk.Canvas(self.cadre_droite, width=450, height=550, bg='#111111', highlightthickness=0)
        self.canvas_reseau.pack()
        
        self.noeuds_caches = []
        self.noeuds_sortie = []
        
        # NOUVEAU : On utilise la taille du modèle dynamique
        nb_caches = self.modele.nb_c1 
        self.lignes_liaisons = [[None for _ in range(nb_caches)] for _ in range(self.modele.nb_c2)]
        self.textes_proba = []
        
        self.dessiner_architecture_reseau()

    def sauvegarder_image_et_label(self, vrai_chiffre):
        timestamp = int(time.time() * 1000)
        nom_fichier = f"chiffre_{vrai_chiffre}_{timestamp}.pgm"
        chemin_complet = os.path.join(self.dossier_entrainement, nom_fichier)
        
        with open(chemin_complet, "w") as f:
            f.write("P2\n64 64\n255\n")
            for i in range(4096):
                f.write(f"{int(self.image_pixels[i] * 255)} ")
                if (i + 1) % 64 == 0: f.write("\n")
                    
        with open(self.fichier_json, "r") as f: donnees_json = json.load(f)
        donnees_json.append({"chemin": chemin_complet, "label": vrai_chiffre})
        with open(self.fichier_json, "w") as f: json.dump(donnees_json, f, indent=4)
        print(f"✅ Sauvegardé : {nom_fichier} (Label: {vrai_chiffre})")

    def dessiner_architecture_reseau(self):
        x_cache = 100
        x_sortie = 350
        nb_caches = self.modele.nb_c1
        nb_sorties = self.modele.nb_c2
        
        espacement_cache = 550 / (nb_caches + 1)
        espacement_sortie = 550 / (nb_sorties + 1)
        
        # Taille dynamique des cercles pour éviter qu'ils se superposent
        rayon_cache = max(2, int(200 / nb_caches))

        for j in range(nb_sorties): 
            y_s = (j + 1) * espacement_sortie
            for i in range(nb_caches): 
                y_c = (i + 1) * espacement_cache
                ligne = self.canvas_reseau.create_line(x_cache, y_c, x_sortie, y_s, fill="#222222", width=1)
                self.lignes_liaisons[j][i] = ligne

        for i in range(nb_caches):
            y = (i + 1) * espacement_cache
            cercle = self.canvas_reseau.create_oval(x_cache-rayon_cache, y-rayon_cache, x_cache+rayon_cache, y+rayon_cache, fill="black", outline="white")
            self.noeuds_caches.append(cercle)

        for i in range(nb_sorties):
            y = (i + 1) * espacement_sortie
            cercle = self.canvas_reseau.create_oval(x_sortie-15, y-15, x_sortie+15, y+15, fill="black", outline="white")
            self.noeuds_sortie.append(cercle)
            self.canvas_reseau.create_text(x_sortie+25, y, text=str(i), fill="white", font=("Arial", 14, "bold"))
            texte_p = self.canvas_reseau.create_text(x_sortie-45, y, text="0%", fill="gray", font=("Arial", 10))
            self.textes_proba.append(texte_p)

    def dessiner(self, event):
        x_g = event.x // self.taille_pixel
        y_g = event.y // self.taille_pixel
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x_g + dx, y_g + dy
                if 0 <= nx < 64 and 0 <= ny < 64:
                    index = ny * 64 + nx
                    if self.image_pixels[index] == 0.0:
                        self.image_pixels[index] = 1.0 
                        p1, p2 = nx * self.taille_pixel, ny * self.taille_pixel
                        self.canvas_dessin.create_rectangle(p1, p2, p1+self.taille_pixel, p2+self.taille_pixel, fill="white", outline="white")
        self.predire_en_direct()

    def effacer_image(self):
        self.canvas_dessin.delete("all")
        self.image_pixels = [0.0] * 4096
        self.predire_en_direct() 

    def clic_apprentissage(self, vrai_chiffre):
        self.sauvegarder_image_et_label(vrai_chiffre)
        self.predire_en_direct()

    def predire_en_direct(self):
        # APPEL DIRECT À L'OBJET MODELE
        chiffre, precision, _, cachee, probas = self.modele.predire(self.image_pixels)
        
        self.label_resultat.config(text=f"Résultat : {chiffre} ({precision:.1f}%)", fg="blue" if precision > 50 else "black")

        nb_caches = self.modele.nb_c1
        max_cachee = max(cachee) if max(cachee) > 0 else 1.0
        
        for i in range(nb_caches):
            intensite = min(255, int((cachee[i] / max_cachee) * 255))
            self.canvas_reseau.itemconfig(self.noeuds_caches[i], fill=f"#{intensite:02x}{intensite:02x}{intensite:02x}")

        for j in range(10): 
            for i in range(nb_caches): 
                signal = cachee[i] * self.modele.poids_c2[j][i] 
                if abs(signal) < 0.01:
                    couleur_ligne = "#222222" 
                elif signal > 0:
                    intensite = min(255, int(signal * 2000)) 
                    couleur_ligne = f"#{0:02x}{intensite:02x}{0:02x}" 
                else:
                    intensite = min(255, int(abs(signal) * 2000))
                    couleur_ligne = f"#{intensite:02x}{0:02x}{0:02x}" 
                self.canvas_reseau.itemconfig(self.lignes_liaisons[j][i], fill=couleur_ligne)

        for i in range(10):
            intensite = min(255, int(probas[i] * 255))
            couleur = f"#{0:02x}{intensite:02x}{0:02x}" 
            self.canvas_reseau.itemconfig(self.noeuds_sortie[i], fill=couleur)
            pourcentage = probas[i] * 100
            self.canvas_reseau.itemconfig(self.textes_proba[i], text=f"{pourcentage:.0f}%", fill="white" if pourcentage > 10 else "gray")

if __name__ == "__main__":
    fenetre_principale = tk.Tk()
    app = ApplicationIA(fenetre_principale)
    fenetre_principale.mainloop()