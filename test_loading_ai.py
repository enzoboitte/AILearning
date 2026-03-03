import random
import os
import struct
import numpy as np

# --- FONCTIONS MATHÉMATIQUES STATIQUES ---
def relu(x): return x if x > 0 else 0.0
def exponentielle(x): return 2.718281828 ** min(x, 10)
def softmax(vecteur):
    val_max = max(vecteur)
    exps = [exponentielle(v - val_max) for v in vecteur]
    somme = sum(exps)
    return [v / somme for v in exps]

def produit_matriciel(matrice, vecteur):
    resultat = []
    for ligne in matrice:
        somme = sum(ligne[i] * vecteur[i] for i in range(len(vecteur)))
        resultat.append(somme)
    return resultat

def addition_vecteurs(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

# --- CLASSE MODELE (LE CERVEAU) ---
class Modele:
    def __init__(self, nb_c1=64, nb_c2=10):
        self.nb_c1 = nb_c1
        self.nb_c2 = nb_c2
        self.taille_entree = 4096
        
        # Initialisation aléatoire par défaut
        self.poids_c1 = [[random.uniform(-0.1, 0.1) for _ in range(self.taille_entree)] for _ in range(self.nb_c1)] 
        self.biais_c1 = [random.uniform(-0.1, 0.1) for _ in range(self.nb_c1)]
        self.poids_c2 = [[random.uniform(-0.1, 0.1) for _ in range(self.nb_c1)] for _ in range(self.nb_c2)] 
        self.biais_c2 = [random.uniform(-0.1, 0.1) for _ in range(self.nb_c2)]

    def predire(self, image_pixels):
        e_cachee = addition_vecteurs(produit_matriciel(self.poids_c1, image_pixels), self.biais_c1)
        s_cachee = [relu(x) for x in e_cachee]
        s_bruts = addition_vecteurs(produit_matriciel(self.poids_c2, s_cachee), self.biais_c2)
        probas = softmax(s_bruts)
        
        chiffre_predit = probas.index(max(probas))
        confiance = max(probas) * 100
        
        # On renvoie tout pour l'interface graphique
        return chiffre_predit, confiance, e_cachee, s_cachee, probas

    def entrainer(self, image_pixels, vrai_chiffre, taux_apprentissage):
        # 1. Forward Pass interne
        _, _, e_cachee, s_cachee, probas = self.predire(image_pixels)

        # 2. Erreur sortie
        erreur_sortie = [probas[i] - (1.0 if i == vrai_chiffre else 0.0) for i in range(self.nb_c2)]
        
        # 3. Erreur cachée
        erreur_cachee = [0.0] * self.nb_c1
        for i in range(self.nb_c1):
            somme_erreurs = sum(erreur_sortie[j] * self.poids_c2[j][i] for j in range(self.nb_c2))
            erreur_cachee[i] = somme_erreurs * (1.0 if e_cachee[i] > 0 else 0.0)

        # 4. Maj Couche 2
        for j in range(self.nb_c2):
            for i in range(self.nb_c1):
                self.poids_c2[j][i] -= taux_apprentissage * erreur_sortie[j] * s_cachee[i]
            self.biais_c2[j] -= taux_apprentissage * erreur_sortie[j]

        # 5. Maj Couche 1
        for i in range(self.nb_c1):
            for k in range(self.taille_entree):
                if image_pixels[k] > 0:
                    self.poids_c1[i][k] -= taux_apprentissage * erreur_cachee[i] * image_pixels[k]
            self.biais_c1[i] -= taux_apprentissage * erreur_cachee[i]

    def sauvegarder(self, chemin="model.txt"):
        with open(chemin, "w") as f:
            f.write(f"{self.nb_c1} {self.nb_c2}\n")
            
            valeurs_l1 = []
            for i in range(self.nb_c1):
                valeurs_l1.extend(self.poids_c1[i])
                valeurs_l1.append(self.biais_c1[i])
            f.write(" ".join(map(str, valeurs_l1)) + "\n")
            
            valeurs_l2 = []
            for j in range(self.nb_c2):
                valeurs_l2.extend(self.poids_c2[j])
                valeurs_l2.append(self.biais_c2[j])
            f.write(" ".join(map(str, valeurs_l2)) + "\n")

# --- CLASSE CHARGEUR ---
class ChargeurModele:
    def __init__(self, chemin_fichier="model.txt"):
        self.chemin_fichier = chemin_fichier

    def charger(self):
        if not os.path.exists(self.chemin_fichier):
            print(f"⚠️ {self.chemin_fichier} introuvable.")
            return Modele(nb_c1=64)

        with open(self.chemin_fichier, "rb") as f:
            nb_c1, nb_c2, input_size = struct.unpack("iii", f.read(12))

            def f_lLireFloats(n):
                return list(struct.unpack(f"{n}f", f.read(n * 4)))

            w1 = f_lLireFloats(nb_c1 * input_size)
            b1 = f_lLireFloats(nb_c1)
            w2 = f_lLireFloats(nb_c2 * nb_c1)
            b2 = f_lLireFloats(nb_c2)

        print(f"🧠 {nb_c1} neurones cachés, {nb_c2} sorties, input={input_size}")

        modele = Modele(nb_c1, nb_c2)
        modele.taille_entree = input_size
        modele.poids_c1 = np.array(w1).reshape(nb_c1, input_size).tolist()
        modele.biais_c1 = list(b1)
        modele.poids_c2 = np.array(w2).reshape(nb_c2, nb_c1).tolist()
        modele.biais_c2 = list(b2)
        return modele