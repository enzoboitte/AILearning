from test_loading_ai import ChargeurModele, OutputMode
import tkinter as tk
from tkinter import simpledialog, messagebox
import os
import json
import time
import threading
import struct

try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

from PIL import Image, ImageTk

# ─── Constantes ────────────────────────────────────────────────────────────────
FACE_SIZE   = 64          # taille des images de visage (64×64)
INPUT_SIZE  = FACE_SIZE * FACE_SIZE   # 4096
WEBCAM_W    = 400
WEBCAM_H    = 300


class ApplicationFace:
    def __init__(self, fenetre):
        self.fenetre = fenetre
        self.fenetre.title("Reconnaissance Faciale IA")
        self.fenetre.configure(bg="#1a1a2e")

        if not CV2_OK:
            messagebox.showerror("Erreur", "OpenCV non installé.\nLance : pip install opencv-python")
            fenetre.destroy()
            return

        # ── Modèle ─────────────────────────────────────────────────────────────
        chargeur = ChargeurModele("model.txt")
        self.modele = chargeur.charger(INPUT_SIZE)
        self.layers  = self.modele.layers
        self.mode    = self.modele.mode
        self.nb_sorties = self.layers[-1]

        # Noms des classes (chargés depuis un fichier JSON si disponible)
        self.labels = self._charger_labels()

        # ── Détecteur de visages ────────────────────────────────────────────────
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detecteur = cv2.CascadeClassifier(cascade_path)

        # ── Webcam ─────────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        self.frame_actuelle = None     # BGR numpy array
        self.face_img       = None     # float list[4096] normalisé
        self.face_rect      = None     # (x,y,w,h) du visage détecté
        self.running        = True

        # ── Dataset ────────────────────────────────────────────────────────────
        self.dossier = "entrainement"
        self.fichier_json = os.path.join(self.dossier, "dataset.json")
        if not os.path.exists(self.dossier):
            os.makedirs(self.dossier)
        if not os.path.exists(self.fichier_json):
            with open(self.fichier_json, "w") as f: json.dump([], f)

        # ── Interface ───────────────────────────────────────────────────────────
        self._build_ui()

        # ── Thread webcam ────────────────────────────────────────────────────────
        self.thread = threading.Thread(target=self._boucle_cam, daemon=True)
        self.thread.start()

        self.fenetre.protocol("WM_DELETE_WINDOW", self._quitter)
        self._boucle_affichage()

    # ─────────────────────────────────────────────────────────────────────────────
    # Construction de l'interface
    # ─────────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        STYLE_LBL   = {"bg": "#1a1a2e", "fg": "white",  "font": ("Arial", 11)}
        STYLE_BTN   = {"bg": "#16213e", "fg": "#e94560", "font": ("Arial", 10, "bold"),
                       "relief": "flat", "padx": 8, "pady": 4}
        STYLE_FRAME = {"bg": "#16213e", "relief": "flat", "bd": 0}

        # ── Panneau gauche : webcam + contrôles ───────────────────────────────
        g = tk.Frame(self.fenetre, bg="#1a1a2e")
        g.pack(side=tk.LEFT, padx=15, pady=15)

        # Webcam canvas
        self.canvas_cam = tk.Canvas(g, width=WEBCAM_W, height=WEBCAM_H,
                                    bg="black", highlightthickness=2,
                                    highlightbackground="#e94560")
        self.canvas_cam.pack()

        # Résultat inférence
        self.label_resultat = tk.Label(g, text="— aucun visage détecté —",
                                       font=("Arial", 14, "bold"),
                                       bg="#1a1a2e", fg="#e94560")
        self.label_resultat.pack(pady=8)

        # Mode affiché
        mode_str = "Classification" if self.mode == OutputMode.Classification else "Régression"
        topo_str = " → ".join(str(s) for s in self.layers)
        tk.Label(g, text=f"Topologie : {topo_str}  |  Mode : {mode_str}",
                 font=("Arial", 8), bg="#1a1a2e", fg="#555577").pack()

        # ── Aperçu visage détecté ───────────────────────────────────────────────
        face_frame = tk.Frame(g, **STYLE_FRAME, padx=10, pady=5)
        face_frame.pack(pady=8)
        tk.Label(face_frame, text="Visage détecté", **STYLE_LBL).pack()
        self.canvas_face = tk.Canvas(face_frame, width=FACE_SIZE * 3, height=FACE_SIZE * 3,
                                     bg="#0f3460", highlightthickness=0)
        self.canvas_face.pack()

        # ── Enregistrement dataset ──────────────────────────────────────────────
        cap_frame = tk.LabelFrame(g, text=" Ajouter au dataset ",
                                  font=("Arial", 10, "bold"),
                                  bg="#1a1a2e", fg="white", bd=1)
        cap_frame.pack(pady=10, fill="x")

        # Boutons de labels dynamiques
        btn_grid = tk.Frame(cap_frame, bg="#1a1a2e")
        btn_grid.pack(pady=6)
        self.boutons_label = []
        for i in range(min(self.nb_sorties, 10)):
            nom = self.labels.get(i, f"Personne {i}")
            btn = tk.Button(btn_grid, text=nom, **STYLE_BTN,
                            command=lambda idx=i: self.capturer_visage(idx))
            btn.grid(row=i // 5, column=i % 5, padx=4, pady=3)
            self.boutons_label.append(btn)

        # Bouton renommer
        tk.Button(cap_frame, text="✏️ Renommer un label",
                  bg="#1a1a2e", fg="#aaaaff", font=("Arial", 9),
                  relief="flat", command=self._renommer_label).pack(pady=4)

        # ── Panneau droit : réseau visuel ───────────────────────────────────────
        d = tk.Frame(self.fenetre, bg="#111111")
        d.pack(side=tk.RIGHT, padx=10, pady=10)

        self.couches_visu  = self.layers[1:]
        self.nb_colonnes   = len(self.couches_visu)
        HAUTEUR   = 570
        COL_W     = 120
        MARGE     = 12
        MINI_W    = FACE_SIZE * 2   # miniature 128×128

        largeur_canvas = MINI_W + MARGE + self.nb_colonnes * COL_W + MARGE + 30
        self.canvas_reseau = tk.Canvas(d, width=largeur_canvas, height=HAUTEUR,
                                       bg="#111111", highlightthickness=0)
        self.canvas_reseau.pack()

        self.HAUTEUR, self.COL_W, self.MARGE, self.MINI_W = HAUTEUR, COL_W, MARGE, MINI_W

        # Miniature pixel par pixel dans le canvas réseau
        self.mini_taille = 2            # 2px par pixel → 128×128
        self.mini_x0     = MARGE
        self.mini_y0     = (HAUTEUR - FACE_SIZE * self.mini_taille) // 2
        self.mini_rects  = {}
        self._creer_miniature_reseau()

        self.noeuds      = [[] for _ in range(self.nb_colonnes)]
        self.liaisons    = []
        self.textes_proba = []
        self._creer_colonnes()

    # ─────────────────────────────────────────────────────────────────────────────
    # Canvas réseau
    # ─────────────────────────────────────────────────────────────────────────────

    def _x_col(self, c):
        return self.MINI_W + self.MARGE * 2 + c * self.COL_W + self.COL_W // 2

    def _creer_miniature_reseau(self):
        m, x0, y0 = self.mini_taille, self.mini_x0, self.mini_y0
        for py in range(FACE_SIZE):
            for px in range(FACE_SIZE):
                rid = self.canvas_reseau.create_rectangle(
                    x0 + px * m, y0 + py * m,
                    x0 + px * m + m - 1, y0 + py * m + m - 1,
                    fill="#0f3460", outline="")
                self.mini_rects[(px, py)] = rid
        self.canvas_reseau.create_text(
            x0 + FACE_SIZE * m // 2, y0 + FACE_SIZE * m + 8,
            text=f"Entrée ({INPUT_SIZE})", fill="gray", font=("Arial", 8))
        ax = x0 + FACE_SIZE * m + 4
        self.canvas_reseau.create_line(
            ax, self.HAUTEUR // 2, ax + self.MARGE, self.HAUTEUR // 2,
            fill="#e94560", arrow=tk.LAST, width=2)

    def _creer_colonnes(self):
        HAUTEUR = self.HAUTEUR
        for c, n in enumerate(self.couches_visu):
            x   = self._x_col(c)
            esp = (HAUTEUR - 40) / (n + 1)
            r   = max(2, min(12, 220 // n))
            is_out = (c == self.nb_colonnes - 1)

            if c > 0:
                n_prev   = self.couches_visu[c - 1]
                x_prev   = self._x_col(c - 1)
                esp_prev = (HAUTEUR - 40) / (n_prev + 1)
                liens = {}
                for j in range(n):
                    y_j = 20 + (j + 1) * esp
                    for i in range(n_prev):
                        lid = self.canvas_reseau.create_line(
                            x_prev, 20 + (i + 1) * esp_prev, x, y_j,
                            fill="#1a1a1a", width=1)
                        liens[(j, i)] = lid
                self.liaisons.append(liens)
            else:
                self.liaisons.append(None)

            for i in range(n):
                y   = 20 + (i + 1) * esp
                oid = self.canvas_reseau.create_oval(
                    x - r, y - r, x + r, y + r, fill="black", outline="white")
                self.noeuds[c].append(oid)
                if is_out:
                    nom = self.labels.get(i, str(i))
                    self.canvas_reseau.create_text(
                        x + r + 5, y, text=nom, fill="white",
                        font=("Arial", 8), anchor="w")
                    tp = self.canvas_reseau.create_text(
                        x - r - 22, y, text="0%", fill="gray", font=("Arial", 9))
                    self.textes_proba.append(tp)

            if c < self.nb_colonnes - 1:
                xn = self._x_col(c + 1)
                self.canvas_reseau.create_line(
                    x + r + 2, HAUTEUR // 2, xn - r - 2, HAUTEUR // 2,
                    fill="#555", arrow=tk.LAST, width=2)

            nom_col = f"{'Sortie' if is_out else f'Cachée {c+1}'} ({n}n)"
            self.canvas_reseau.create_text(
                x, HAUTEUR - 8, text=nom_col, fill="gray", font=("Arial", 7))

    # ─────────────────────────────────────────────────────────────────────────────
    # Thread webcam
    # ─────────────────────────────────────────────────────────────────────────────

    def _boucle_cam(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detecteur.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

            self.face_rect = None
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                self.face_rect = (x, y, w, h)
                # Recadre et redimensionne à 64×64 en niveaux de gris
                face_gray = gray[y:y+h, x:x+w]
                face_64   = cv2.resize(face_gray, (FACE_SIZE, FACE_SIZE))
                self.face_img = [face_64[py, px] / 255.0
                                 for py in range(FACE_SIZE)
                                 for px in range(FACE_SIZE)]
                # Aperçu dans canvas_face
                face_rgb = cv2.cvtColor(face_64, cv2.COLOR_GRAY2RGB)
                face_big = cv2.resize(face_rgb, (FACE_SIZE * 3, FACE_SIZE * 3),
                                      interpolation=cv2.INTER_NEAREST)
                self._photo_face = ImageTk.PhotoImage(Image.fromarray(face_big))
                self.canvas_face.create_image(0, 0, anchor="nw", image=self._photo_face)
            else:
                self.face_img = None

            # Dessin du rectangle sur le flux
            frame_disp = cv2.resize(frame, (WEBCAM_W, WEBCAM_H))
            if self.face_rect is not None:
                fx = int(self.face_rect[0] * WEBCAM_W / frame.shape[1])
                fy = int(self.face_rect[1] * WEBCAM_H / frame.shape[0])
                fw = int(self.face_rect[2] * WEBCAM_W / frame.shape[1])
                fh = int(self.face_rect[3] * WEBCAM_H / frame.shape[0])
                cv2.rectangle(frame_disp, (fx, fy), (fx+fw, fy+fh), (233, 69, 96), 2)

            frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            self.frame_actuelle = frame_rgb
            time.sleep(0.03)

    # ─────────────────────────────────────────────────────────────────────────────
    # Boucle d'affichage Tkinter (~30 fps)
    # ─────────────────────────────────────────────────────────────────────────────

    def _boucle_affichage(self):
        if self.frame_actuelle is not None:
            img = ImageTk.PhotoImage(Image.fromarray(self.frame_actuelle))
            self.canvas_cam.create_image(0, 0, anchor="nw", image=img)
            self._photo_cam = img   # garder la référence

        if self.face_img is not None:
            self._predire_et_afficher()
        else:
            self.label_resultat.config(text="— aucun visage détecté —", fg="#e94560")

        self.fenetre.after(33, self._boucle_affichage)

    # ─────────────────────────────────────────────────────────────────────────────
    # Inférence + mise à jour réseau visuel
    # ─────────────────────────────────────────────────────────────────────────────

    def _predire_et_afficher(self):
        pred, conf, activations = self.modele.predire(self.face_img)
        nom_predit = self.labels.get(pred, f"#{pred}")

        if self.mode == OutputMode.Classification:
            self.label_resultat.config(
                text=f"👤 {nom_predit}  ({conf:.1f}%)",
                fg="#00d4aa" if conf > 60 else "#e94560")
        else:
            self.label_resultat.config(
                text=f"Sortie : {conf:.3f}  (classe estimée : {nom_predit})",
                fg="#f5a623")

        # Miniature dans le canvas réseau
        c = FACE_SIZE
        for py in range(c):
            for px in range(c):
                v  = self.face_img[py * c + px]
                iv = int(v * 255)
                self.canvas_reseau.itemconfig(
                    self.mini_rects[(px, py)],
                    fill=f"#{iv:02x}{iv:02x}{iv:02x}")

        HAUTEUR = self.HAUTEUR
        for col, n in enumerate(self.couches_visu):
            act    = activations[col + 1]
            mx     = max(act) if max(act) > 0 else 1.0
            esp    = (HAUTEUR - 40) / (n + 1)
            is_out = (col == self.nb_colonnes - 1)

            for i in range(n):
                if is_out and self.mode == OutputMode.Classification:
                    iv   = min(255, int(act[i] * 255))
                    fill = f"#{0:02x}{iv:02x}{int(iv*0.5):02x}"
                elif is_out:
                    v    = min(1.0, max(0.0, act[i]))
                    iv   = int(v * 255)
                    fill = f"#{iv:02x}{int(iv*0.7):02x}{0:02x}"
                else:
                    iv   = min(255, int((act[i] / mx) * 255))
                    fill = f"#{iv:02x}{iv:02x}{iv:02x}"
                self.canvas_reseau.itemconfig(self.noeuds[col][i], fill=fill)

            if is_out:
                for i in range(n):
                    pct = act[i] * 100 if self.mode == OutputMode.Classification else act[i]
                    fmt = f"{pct:.0f}%" if self.mode == OutputMode.Classification else f"{pct:.2f}"
                    self.canvas_reseau.itemconfig(
                        self.textes_proba[i], text=fmt,
                        fill="white" if abs(pct) > 5 else "gray")

            if col > 0 and self.liaisons[col]:
                act_prev = activations[col]
                n_prev   = self.couches_visu[col - 1]
                W        = self.modele.weights[col]
                for j in range(n):
                    for i in range(n_prev):
                        s = act_prev[i] * W[j * n_prev + i]
                        if abs(s) < 0.005:
                            c_ligne = "#1a1a1a"
                        elif s > 0:
                            iv = min(255, int(s * 1500))
                            c_ligne = f"#{0:02x}{iv:02x}{0:02x}"
                        else:
                            iv = min(255, int(abs(s) * 1500))
                            c_ligne = f"#{iv:02x}{0:02x}{0:02x}"
                        self.canvas_reseau.itemconfig(self.liaisons[col][(j, i)], fill=c_ligne)

    # ─────────────────────────────────────────────────────────────────────────────
    # Capture dans le dataset
    # ─────────────────────────────────────────────────────────────────────────────

    def capturer_visage(self, label_idx):
        if self.face_img is None:
            messagebox.showwarning("Aucun visage", "Aucun visage détecté dans le flux webcam.")
            return
        ts   = int(time.time() * 1000)
        nom  = f"face_{label_idx}_{ts}.pgm"
        path = os.path.join(self.dossier, nom)
        with open(path, "w") as f:
            f.write(f"P2\n{FACE_SIZE} {FACE_SIZE}\n255\n")
            for i, v in enumerate(self.face_img):
                f.write(f"{int(v * 255)} ")
                if (i + 1) % FACE_SIZE == 0: f.write("\n")

        with open(self.fichier_json, "r") as f: data = json.load(f)
        data.append({"chemin": path, "label": label_idx})
        with open(self.fichier_json, "w") as f: json.dump(data, f, indent=4)

        nom_label = self.labels.get(label_idx, f"Personne {label_idx}")
        print(f"✅ Visage capturé → {nom}  (label: {nom_label})")

    # ─────────────────────────────────────────────────────────────────────────────
    # Labels / noms des personnes
    # ─────────────────────────────────────────────────────────────────────────────

    LABELS_FILE = "entrainement/labels.json"

    def _charger_labels(self):
        if os.path.exists(self.LABELS_FILE):
            with open(self.LABELS_FILE, "r") as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        return {}

    def _sauvegarder_labels(self):
        os.makedirs("entrainement", exist_ok=True)
        with open(self.LABELS_FILE, "w") as f:
            json.dump({str(k): v for k, v in self.labels.items()}, f, indent=2)

    def _renommer_label(self):
        idx_str = simpledialog.askstring("Numéro", f"Index (0 à {self.nb_sorties-1}) :")
        if idx_str is None: return
        try: idx = int(idx_str)
        except ValueError: return
        if not (0 <= idx < self.nb_sorties): return
        nom = simpledialog.askstring("Nom", f"Nom pour la personne {idx} :")
        if nom:
            self.labels[idx] = nom
            self._sauvegarder_labels()
            if idx < len(self.boutons_label):
                self.boutons_label[idx].config(text=nom)
            # Mettre à jour les labels dans le canvas réseau
            self._refresh_labels_reseau()

    def _refresh_labels_reseau(self):
        # Recréer les textes de labels de sortie
        # (simple: on recrée entièrement les colonnes n'est pas nécessaire,
        #  les textes de noms sont des items séparés qu'on peut retagger)
        pass  # les labels sont lus dynamiquement dans la prédiction

    # ─────────────────────────────────────────────────────────────────────────────
    # Fermeture
    # ─────────────────────────────────────────────────────────────────────────────

    def _quitter(self):
        self.running = False
        self.cap.release()
        self.fenetre.destroy()


if __name__ == "__main__":
    if not CV2_OK:
        print("❌ OpenCV non installé. Lance : pip install opencv-python")
        exit(1)
    fenetre = tk.Tk()
    app = ApplicationFace(fenetre)
    fenetre.mainloop()