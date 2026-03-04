#include "src/include/CEntraineur.hpp"
#include <iostream>

int main() {
  try {
    // Définir la structure du réseau neuronal
    // Format attendu : nombre de couches puis leur taille
    // Exemple : "3  128 64 10" → [128, 64, 10]  (l'entrée est déterminée
    // automatiquement) Ou      : "4  784 128 64 10" → topologie complète
    std::vector<int> neural;
    std::cout
        << "Structure du réseau neuronal (couches cachées + sortie):\n"
        << "  Exemple : 2  128 10\n"
        << "  (la taille d'entrée est déduite automatiquement du dataset)\n"
        << "Nombre de couches : ";
    int size;
    std::cin >> size;
    std::cout << "Taille de chaque couche : ";
    for (int i = 0; i < size; i++) {
      int neurons;
      std::cin >> neurons;
      neural.push_back(neurons);
    }

    CEntraineur entraineur("entrainement/dataset.json", neural);
    entraineur.F_vLancerEntrainement();
  } catch (const std::exception &e) {
    std::cerr << "Erreur : " << e.what() << std::endl;
    return 1;
  }
  return 0;
}