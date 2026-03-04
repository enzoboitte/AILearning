import os
import struct
import numpy as np

# ─── Fonctions mathématiques ──────────────────────────────────────────────────
def relu(x): return max(0.0, x)

def softmax(vecteur):
    val_max = max(vecteur)
    exps = [2.718281828 ** min(v - val_max, 10) for v in vecteur]
    somme = sum(exps)
    return [v / somme for v in exps]

def produit_matriciel_vecteur(W_flat, out_size, in_size, vecteur):
    """Calcule W @ v où W est stocké à plat (row-major)."""
    resultat = []
    for i in range(out_size):
        s = sum(W_flat[i * in_size + j] * vecteur[j] for j in range(in_size))
        resultat.append(s)
    return resultat

def addition_vecteurs(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

# ─── Classe Modele (N couches) ────────────────────────────────────────────────
class Modele:
    """
    Réseau neuronal entièrement connecté, N couches.
    layers = [inputSize, hidden1, ..., outputSize]
    weights[l] : liste plate float, taille = layers[l+1] * layers[l]
    biases[l]  : liste float,       taille = layers[l+1]
    """
    def __init__(self, layers):
        self.layers  = layers          # ex : [784, 128, 64, 10]
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
          chiffre_predit (int),
          confiance (float, %),
          activations (list[list[float]]) : toutes les activations,
            activations[0] = entrée, activations[-1] = softmax sortie
        """
        activations = [list(image_pixels)]
        L = len(self.layers) - 1

        for l in range(L):
            n_in  = self.layers[l]
            n_out = self.layers[l + 1]
            z = addition_vecteurs(
                produit_matriciel_vecteur(self.weights[l], n_out, n_in, activations[l]),
                self.biases[l]
            )
            if l < L - 1:
                a = [relu(v) for v in z]   # couches cachées → ReLU
            else:
                a = softmax(z)             # couche de sortie → Softmax
            activations.append(a)

        probas = activations[-1]
        chiffre_predit = probas.index(max(probas))
        confiance = max(probas) * 100
        return chiffre_predit, confiance, activations

    # ── Propriétés de compatibilité ───────────────────────────────────────────
    # (pour que test_ia.py puisse accéder facilement aux dimensions)
    @property
    def nb_c1(self):
        """Taille de la première couche cachée (layers[1])."""
        return self.layers[1] if len(self.layers) > 2 else self.layers[-1]

    @property
    def nb_c2(self):
        """Taille de la couche de sortie (layers[-1])."""
        return self.layers[-1]

    @property
    def taille_entree(self):
        return self.layers[0]


# ─── Classe ChargeurModele ────────────────────────────────────────────────────
class ChargeurModele:
    """
    Lit le format binaire produit par CModel::F_vSave() :
      [int]   nb_layers
      [int]*  layers[0..nb_layers-1]
      Pour chaque l = 0..nb_layers-2 :
        [float]* weights[l]  (layers[l+1] * layers[l] floats)
        [float]* biases[l]   (layers[l+1] floats)
    """
    def __init__(self, chemin_fichier="model.txt"):
        self.chemin_fichier = chemin_fichier

    def charger(self):
        if not os.path.exists(self.chemin_fichier):
            print(f"⚠️  {self.chemin_fichier} introuvable → modèle aléatoire par défaut.")
            return Modele([4096, 128, 10])

        with open(self.chemin_fichier, "rb") as f:
            def lire_ints(n):
                return list(struct.unpack(f"{n}i", f.read(n * 4)))
            def lire_floats(n):
                return list(struct.unpack(f"{n}f", f.read(n * 4)))

            nb_layers = lire_ints(1)[0]
            layers = lire_ints(nb_layers)

            modele = Modele(layers)
            for l in range(nb_layers - 1):
                n_in  = layers[l]
                n_out = layers[l + 1]
                modele.weights[l] = lire_floats(n_out * n_in)
                modele.biases[l]  = lire_floats(n_out)

        topo = " → ".join(str(s) for s in layers)
        print(f"🧠 Modèle chargé : {topo}")
        return modele