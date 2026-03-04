import os
import struct
import math
import numpy as np

# ─── Fonctions mathématiques ──────────────────────────────────────────────────
def relu(x): return max(0.0, x)

def softmax(vecteur):
    val_max = max(vecteur)
    exps = [math.exp(min(v - val_max, 10)) for v in vecteur]
    somme = sum(exps)
    return [v / somme for v in exps]

def produit_matriciel_vecteur(W_flat, out_size, in_size, vecteur):
    resultat = []
    for i in range(out_size):
        s = sum(W_flat[i * in_size + j] * vecteur[j] for j in range(in_size))
        resultat.append(s)
    return resultat

def addition_vecteurs(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

# ─── Mode de sortie ───────────────────────────────────────────────────────────
class OutputMode:
    Classification = 0   # Softmax + Cross-Entropy
    Regression     = 1   # Linéaire + MSE

# ─── Classe Modele (N couches) ────────────────────────────────────────────────
class Modele:
    """
    Réseau neuronal N couches.
    layers  = [inputSize, hidden1, ..., outputSize]
    mode    = OutputMode.Classification ou OutputMode.Regression
    weights[l] : liste plate float, taille = layers[l+1] * layers[l]
    biases[l]  : liste float,       taille = layers[l+1]
    """
    def __init__(self, layers, mode=OutputMode.Classification):
        self.layers  = layers
        self.mode    = mode
        self.weights = []
        self.biases  = []
        for l in range(len(layers) - 1):
            n_in  = layers[l]
            n_out = layers[l + 1]
            self.weights.append([0.0] * (n_out * n_in))
            self.biases.append([0.0] * n_out)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def predire(self, image_pixels):
        """
        Retourne :
          chiffre_predit (int)   → indice du max de la sortie
          confiance (float)      → % si Classification, valeur brute si Regression
          activations (list)     → list[list[float]], activations[0]=entrée, ...[-1]=sortie
        """
        activations = [list(image_pixels)]
        L = len(self.layers) - 1

        for l in range(L):
            n_in  = self.layers[l]
            n_out = self.layers[l + 1]
            z = addition_vecteurs(
                produit_matriciel_vecteur(self.weights[l], n_out, n_in, activations[l]),
                self.biases[l])

            if l < L - 1:
                a = [relu(v) for v in z]           # couches cachées → ReLU
            elif self.mode == OutputMode.Classification:
                a = softmax(z)                     # sortie → Softmax
            else:
                a = list(z)                        # sortie → linéaire

            activations.append(a)

        output = activations[-1]
        pred   = output.index(max(output))
        conf   = max(output) * (100.0 if self.mode == OutputMode.Classification else 1.0)
        return pred, conf, activations

    # ── Propriétés de compatibilité ───────────────────────────────────────────
    @property
    def nb_c1(self):
        return self.layers[1] if len(self.layers) > 2 else self.layers[-1]

    @property
    def nb_c2(self):
        return self.layers[-1]

    @property
    def taille_entree(self):
        return self.layers[0]


# ─── Classe ChargeurModele ────────────────────────────────────────────────────
class ChargeurModele:
    """
    Lit le format binaire C++ :
      [int]   nb_layers
      [int]*  layers[0..nb_layers-1]
      [int]   mode  (0=Classification, 1=Regression)
      Pour chaque l :
        [float]* weights[l]
        [float]* biases[l]
    """
    def __init__(self, chemin_fichier="model.txt"):
        self.chemin_fichier = chemin_fichier

    def charger(self, input_size_defaut=4096):
        if not os.path.exists(self.chemin_fichier):
            print(f"⚠️  {self.chemin_fichier} introuvable → modèle aléatoire par défaut.")
            return Modele([input_size_defaut, 128, 10], OutputMode.Classification)

        with open(self.chemin_fichier, "rb") as f:
            def lire_ints(n):
                return list(struct.unpack(f"{n}i", f.read(n * 4)))
            def lire_floats(n):
                return list(struct.unpack(f"{n}f", f.read(n * 4)))

            nb_layers = lire_ints(1)[0]
            layers    = lire_ints(nb_layers)

            # Lecture du mode (fichiers anciens sans mode → défaut Classification)
            try:
                mode_int = lire_ints(1)[0]
                mode = mode_int  # 0 ou 1
            except struct.error:
                mode = OutputMode.Classification

            modele = Modele(layers, mode)
            for l in range(nb_layers - 1):
                n_in  = layers[l]
                n_out = layers[l + 1]
                modele.weights[l] = lire_floats(n_out * n_in)
                modele.biases[l]  = lire_floats(n_out)

        topo      = " → ".join(str(s) for s in layers)
        mode_str  = "Classification" if mode == OutputMode.Classification else "Régression"
        print(f"🧠 Modèle chargé : {topo}  |  Mode : {mode_str}")
        return modele