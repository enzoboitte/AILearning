#ifndef CMODEL_HPP
#define CMODEL_HPP

#include <cblas.h>
#include <string>
#include <vector>

// ─── Fonctions utilitaires ───────────────────────────────────────────────────
float relu(float x);
float sigmoid(float x);
std::vector<float> softmax(std::vector<float> x);
std::vector<float> add_vectors(const std::vector<float> &a,
                               const std::vector<float> &b);
std::vector<float> sub_vectors(const std::vector<float> &a,
                               const std::vector<float> &b);
std::vector<float> mul_vectors(const std::vector<float> &a,
                               const std::vector<float> &b);
std::vector<float> div_vectors(const std::vector<float> &a,
                               const std::vector<float> &b);

// ─── CModel ─────────────────────────────────────────────────────────────────
//
// Réseau neuronal entièrement connecté, N couches.
//
// layers = { inputSize, hidden1, hidden2, ..., outputSize }
//    ex : { 784, 128, 64, 10 }
//         → couche d'entrée  : 784 neurones
//         → couche cachée 1  : 128 neurones  (ReLU)
//         → couche cachée 2  : 64 neurones   (ReLU)
//         → couche de sortie : 10 neurones   (Softmax)
//
enum class OutputMode {
  Classification, // Softmax + Cross-Entropy (classes discrètes)
  Regression      // Linéaire + MSE         (valeurs continues)
};

class CModel {
public:
  // Construit et initialise aléatoirement (He init)
  explicit CModel(const std::vector<int> &layers,
                  OutputMode mode = OutputMode::Classification);
  // Constructeur par défaut (pour désérialisation)
  CModel() = default;
  ~CModel() = default;

  // Forward pass : retourne { classe prédite, confiance %, activations par
  // couche } activations[0] = input, activations[1] = A1, ..., activations[L] =
  // softmax output
  struct ForwardResult {
    int predicted;                               // indice de la classe prédite
    float confidence;                            // pourcentage de confiance
    std::vector<std::vector<float>> activations; // taille = layers_.size()
    std::vector<std::vector<float>>
        preActivations; // Z de chaque couche (sauf input)
  };

  ForwardResult F_vPredict(const std::vector<float> &input) const;

  // Backprop + mise à jour des poids
  // Classification : trueLabel = indice de la classe correcte
  void F_vTrain(const std::vector<float> &input, int trueLabel,
                float learningRate);
  // Régression : target = vecteur de valeurs souhaitées (taille = couche
  // sortie)
  void F_vTrain(const std::vector<float> &input,
                const std::vector<float> &target, float learningRate);

  // Sérialisation binaire
  void F_vSave(const std::string &filename) const;

  // Accès à la topologie et au mode
  const std::vector<int> &F_vGetLayers() const { return layers_; }
  OutputMode F_vGetMode() const { return mode_; }

  // Poids et biais (accès direct pour le chargeur)
  std::vector<std::vector<float>>
      weights_; // weights_[l] : W de la couche l→l+1, taille [out x in]
  std::vector<std::vector<float>>
      biases_; // biases_[l]  : B de la couche l+1,   taille [out]

private:
  std::vector<int> layers_; // topologie complète, y compris input et output
  OutputMode mode_;         // mode de sortie

  void F_vInitWeights();
};

// ─── CModelLoader ────────────────────────────────────────────────────────────
class CModelLoader {
public:
  explicit CModelLoader(const std::string &filename);
  CModel F_vLoad();

private:
  std::string filename_;
};

#endif