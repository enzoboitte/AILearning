#include "include/CEntraineur.hpp"
#include "include/CModel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>

CEntraineur::CEntraineur(const std::string &l_sDatasetPath,
                         const std::vector<int> &layers)
    : g_vLayers(layers), g_genRandom(std::random_device{}()),
      g_distShuffle(0, 1000) {
  F_vChargerDataset(l_sDatasetPath);
  g_iNbEpochs = 500; // fixe, indépendant du dataset
  g_fTauxApprentissage = 0.01f;
  g_fLastMinLoss = 100.f;
}

CEntraineur::~CEntraineur() {}

void CEntraineur::F_vChargerImagePGM(const std::string &l_sChemin,
                                     std::vector<float> &l_lPixels) {
  std::ifstream l_fichier(l_sChemin);
  if (!l_fichier.is_open()) {
    throw std::runtime_error("Impossible d'ouvrir " + l_sChemin);
  }

  std::string ligne;
  std::getline(l_fichier, ligne); // P2
  std::getline(l_fichier, ligne); // largeur hauteur
  std::getline(l_fichier, ligne); // 255

  l_lPixels.clear();
  std::string mot;
  while (l_fichier >> mot) {
    l_lPixels.push_back(std::stof(mot) / 255.0f);
  }
}

void CEntraineur::F_vChargerDataset(const std::string &l_sJsonPath) {
  std::ifstream l_fichier(l_sJsonPath);
  if (!l_fichier.is_open())
    throw std::runtime_error("Dataset.json introuvable");

  Json::Value l_jDataset;
  l_fichier >> l_jDataset;

  std::unordered_set<std::string> l_sCheminsVus;
  std::vector<std::pair<std::vector<float>, int>> l_lOriginal;

  for (const auto &e : l_jDataset) {
    std::string l_sChemin = e["chemin"].asString();
    if (!l_sCheminsVus.insert(l_sChemin).second)
      continue;
    std::vector<float> l_lPixels;
    F_vChargerImagePGM(l_sChemin, l_lPixels);
    if (l_sCheminsVus.size() == 1)
      std::cout << "Taille image: " << l_lPixels.size() << " pixels\n";
    l_lOriginal.emplace_back(
        std::move(l_lPixels),
        e.isMember("label") ? static_cast<int>(e["label"].asFloat()) : 0);
  }

  g_lDonneesEntrainement.reserve(l_lOriginal.size() * 8);
  for (const auto &[l_lPixels, l_iLabel] : l_lOriginal) {
    g_lDonneesEntrainement.emplace_back(l_lPixels, l_iLabel);
    for (int i = 0; i < 7; ++i) {
      std::vector<float> l_lAug;
      F_vAugmenter(l_lPixels, l_lAug);
      g_lDonneesEntrainement.emplace_back(std::move(l_lAug), l_iLabel);
    }
  }

  // g_lDonneesEntrainement = l_lOriginal;

  std::cout << "Dataset chargé : " << l_lOriginal.size() << " images → "
            << g_lDonneesEntrainement.size() << " avec augmentation.\n";
}

void CEntraineur::F_vAugmenter(const std::vector<float> &l_lSrc,
                               std::vector<float> &l_lDst) {
  int l_iTotal = static_cast<int>(l_lSrc.size());
  int l_iW = static_cast<int>(std::sqrt(static_cast<float>(l_iTotal)));
  int l_iH = l_iTotal / l_iW;

  std::uniform_real_distribution<float> l_distNoise(-0.03f, 0.03f);
  std::uniform_int_distribution<int> l_distShift(-4, 4);
  std::uniform_real_distribution<float> l_distAngle(-15.f, 15.f);
  std::uniform_real_distribution<float> l_distZoom(0.85f, 1.15f);

  int l_iShiftX = l_distShift(g_genRandom);
  int l_iShiftY = l_distShift(g_genRandom);
  float l_fAngle = l_distAngle(g_genRandom) * 3.14159f / 180.f;
  float l_fZoom = l_distZoom(g_genRandom);
  float l_fCos = std::cos(l_fAngle) / l_fZoom;
  float l_fSin = std::sin(l_fAngle) / l_fZoom;
  float l_fCx = l_iW / 2.f;
  float l_fCy = l_iH / 2.f;

  l_lDst.resize(l_iTotal, 0.f);
  for (int l_iY = 0; l_iY < l_iH; ++l_iY) {
    for (int l_iX = 0; l_iX < l_iW; ++l_iX) {
      float l_fDx = l_iX - l_fCx - l_iShiftX;
      float l_fDy = l_iY - l_fCy - l_iShiftY;
      int l_iSrcX = static_cast<int>(l_fCx + l_fCos * l_fDx + l_fSin * l_fDy);
      int l_iSrcY = static_cast<int>(l_fCy - l_fSin * l_fDx + l_fCos * l_fDy);

      float l_fVal = 0.f;
      if (l_iSrcX >= 0 && l_iSrcX < l_iW && l_iSrcY >= 0 && l_iSrcY < l_iH)
        l_fVal = l_lSrc[l_iSrcY * l_iW + l_iSrcX];
      l_lDst[l_iY * l_iW + l_iX] =
          std::clamp(l_fVal + l_distNoise(g_genRandom), 0.f, 1.f);
    }
  }
}

void CEntraineur::F_vLancerEntrainement() {
  std::array<int, 10> l_aCount{};
  std::array<float, 10> l_aWeights{};
  for (const auto &[pixels, label] : g_lDonneesEntrainement)
    ++l_aCount[label];
  float l_fTotal = static_cast<float>(g_lDonneesEntrainement.size());
  for (int i = 0; i < 10; ++i)
    l_aWeights[i] = l_aCount[i] > 0 ? (l_fTotal / (10.f * l_aCount[i])) : 1.f;

  int l_iInputSize = static_cast<int>(g_lDonneesEntrainement[0].first.size());
  std::cout << "Taille input: " << l_iInputSize << "\n";

  // Compléter la topologie si l'utilisateur n'a pas précisé la taille d'entrée
  // (le vecteur donné en argument commence souvent à la première couche cachée)
  std::vector<int> l_vLayers = g_vLayers;
  if (l_vLayers.empty()) {
    // Topologie par défaut : [input, 128, 10]
    l_vLayers = {l_iInputSize, 128, 10};
    std::cout << "Topologie par défaut : [" << l_iInputSize << ", 128, 10]\n";
  } else if (l_vLayers.front() != l_iInputSize) {
    // L'utilisateur n'a pas inclus le layer d'entrée → on le préfixe
    l_vLayers.insert(l_vLayers.begin(), l_iInputSize);
  }

  // Affichage de la topologie
  std::cout << "Topologie réseau : [";
  for (size_t i = 0; i < l_vLayers.size(); ++i)
    std::cout << l_vLayers[i] << (i + 1 < l_vLayers.size() ? ", " : "]\n");

  CModel l_cModele(l_vLayers);

  try {
    CModelLoader l_cChargeur("model.txt");
    CModel l_cLoaded = l_cChargeur.F_vLoad();
    // On n'utilise le modèle chargé que si la topologie correspond
    if (l_cLoaded.F_vGetLayers() == l_vLayers) {
      l_cModele = l_cLoaded;
      std::cout
          << "Modèle chargé, reprise de l'entraînement.\n\nDémarrage...\n";
    } else {
      std::cout << "Topologie différente du modèle sauvegardé → nouveau "
                   "modèle.\n\nDémarrage...\n";
    }
  } catch (...) {
    std::cout << "Nouveau modèle initialisé.\n\nDémarrage...\n";
  }

  for (size_t l_iEpoch = 0; l_iEpoch < g_iNbEpochs; ++l_iEpoch) {
    std::shuffle(g_lDonneesEntrainement.begin(), g_lDonneesEntrainement.end(),
                 g_genRandom);

    size_t l_iErreurs = 0;
    double l_dLoss = 0.0;
    const bool l_bClassif =
        (l_cModele.F_vGetMode() == OutputMode::Classification);
    const int l_iNbSorties = l_vLayers.back();

    for (const auto &[l_lPixels, l_iVraiChiffre] : g_lDonneesEntrainement) {
      auto fwd = l_cModele.F_vPredict(l_lPixels);
      const auto &l_vOutput = fwd.activations.back();

      if (l_bClassif) {
        // ── Classification ──────────────────────────────────────────────────
        if (fwd.predicted != l_iVraiChiffre)
          ++l_iErreurs;

        // Loss : Cross-Entropy
        float l_fProb = std::clamp(l_vOutput[l_iVraiChiffre], 1e-7f, 1.f);
        l_dLoss -= std::log(l_fProb);

        l_cModele.F_vTrain(l_lPixels, l_iVraiChiffre,
                           F_fGetTauxApprentissage() *
                               l_aWeights[l_iVraiChiffre]);
      } else {
        // ── Régression ──────────────────────────────────────────────────────
        // Cible : vecteur de taille l_iNbSorties
        // Si 1 neurone → valeur normalisée du label [0,1]
        // Si N neurones → one-hot (même principe que classification mais MSE)
        std::vector<float> l_vTarget(l_iNbSorties, 0.f);
        if (l_iNbSorties == 1) {
          // Régression scalaire : normalise le label entre 0 et 1
          l_vTarget[0] = static_cast<float>(l_iVraiChiffre) / 9.f;
        } else {
          // One-hot sur N sorties
          if (l_iVraiChiffre < l_iNbSorties)
            l_vTarget[l_iVraiChiffre] = 1.f;
        }

        // Erreur de prédiction (classe la plus proche)
        if (fwd.predicted != l_iVraiChiffre)
          ++l_iErreurs;

        // Loss : MSE
        for (int k = 0; k < l_iNbSorties; ++k) {
          float d = l_vOutput[k] - l_vTarget[k];
          l_dLoss += d * d;
        }

        l_cModele.F_vTrain(l_lPixels, l_vTarget, F_fGetTauxApprentissage());
      }
    }

    float l_fErrorRate = (static_cast<float>(l_iErreurs) / l_fTotal) * 100.f;
    float l_fLoss = static_cast<float>(l_dLoss / l_fTotal);
    std::cout << "\rÉpoch " << (l_iEpoch + 1) << "/" << g_iNbEpochs
              << " | Loss : " << l_fLoss << " | Err : " << l_fErrorRate
              << "% | Taux : " << g_fTauxApprentissage << "    " << std::flush;

    F_vUpdateTauxApprentissage(l_fLoss);

    if ((l_iEpoch + 1) % g_iSaveStep == 0)
      l_cModele.F_vSave("model_epoch_" + std::to_string(l_iEpoch + 1) + ".txt");
  }

  std::cout << "\nEntraînement terminé.\n";
  l_cModele.F_vSave("model.txt");
}

float CEntraineur::F_fGetTauxApprentissage() { return g_fTauxApprentissage; }

void CEntraineur::F_vUpdateTauxApprentissage(float l_fLoss) {
  if (l_fLoss < g_fLastMinLoss) {
    g_fLastMinLoss = l_fLoss;
    g_iNbEpochsSinceLastMinLoss = 0;
  } else {
    ++g_iNbEpochsSinceLastMinLoss;
  }

  // Patience de 20 epochs avant de réduire
  if (g_iNbEpochsSinceLastMinLoss >= 20) {
    g_fTauxApprentissage *= 0.5f;
    g_iNbEpochsSinceLastMinLoss = 0;
    std::cout << "\n[LR réduit → " << g_fTauxApprentissage << "]\n";
  }

  // Floor
  g_fTauxApprentissage = std::max(g_fTauxApprentissage, 1e-4f);
  // Et reset si plateau trop long
  if (g_iNbEpochsSinceLastMinLoss >= 50) {
    g_fTauxApprentissage = 0.005f; // warm restart
    g_iNbEpochsSinceLastMinLoss = 0;
    g_fLastMinLoss = 100.f;
    std::cout << "\n[Warm restart LR → 0.005]\n";
  }
}
