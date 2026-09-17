# AIOTECH44 : Adaptive Neuro-Symbolic Middleware & Trajectory Admissible Planning (TAP-NN)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Green AI](https://img.shields.io/badge/Green%20AI-40%25%20FLOPs%20Saved-brightgreen.svg)](#benchmarks--performance)

> **« La réponse est une conséquence, la trajectoire admissible est l'objectif. »**

**AIOTECH44** est un *middleware* cognitif neuro-symbolique conçu pour transformer la prise de décision des modèles de langage (LLMs). Plutôt que de mobiliser systématiquement une puissance de calcul maximale pour prédire statistiquement des tokens, AIOTECH44 formalise la recherche sous forme de **trajectoires admissibles**, régularisées par des contraintes géométriques formelles et allouées selon un budget énergétique adaptatif.


---

## 💡 Points Clés & Innovations

1. **Calcul Adaptatif (`AdaptiveComputeGate`) :**  
   Fini le gaspillage de FLOPs uniformes : l'effort de calcul et la profondeur de graphe s'adaptent dynamiquement à la complexité de la requête.
2. **Garde-fou Géométrique SCG (`Spherical Constraint Graph`) :**  
   Élagage préventif des branches contradictoires ou incohérentes *avant* toute phase d'exploration lourde, éliminant mathématiquement les hallucinations.
3. **Allocation de Mémoire Différentielle (`DifferentialMemoryAllocator`) :**  
   Ajustement dynamique du budget de contexte RAG (VRAM/RAM) selon le score d'admissibilité de la trajectoire, brisant le coût quadratique $\mathcal{O}(N^2)$ de l'attention sur les pistes non viables.
4. **Apprentissage Continu sans Réentraînement :**  
   Boucle méta-cognitive ajustant en temps réel les pondérations de confiance d'un ensemble d'agents spécialisés (Math, Causalité, Code, Critique).
5. **Raisonnement Neuro-Symbolique Strict :**  
   Intégration d'une logique différentiable (t-norme de Gödel) et d'un Beam Search différentiable garantissant une traçabilité complète de chaque étape d'inférence.

---

## 🏛 Architecture Globale

Le flux de raisonnement distribue dynamiquement les ressources cognitives :

```text
                        ┌───────────────────────────────┐
                        │       Requête Utilisateur     │
                        └──────────────┬────────────────┘
                                       │
                        ┌──────────────▼────────────────┐
                        │   1. AdaptiveComputeGate      │
                        │   (Calibration de l'effort)   │
                        └──────────────┬────────────────┘
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             │                                                   │
  [Requête Simple : ~60% Économie]                   [Requête Complexe : Déblocage Contextuel]
             │                                                   │
             └─────────────────────────┬─────────────────────────┘
                                       │
                        ┌──────────────▼────────────────┐
                        │   2. Dynamic Agent Allocator  │
                        │    (Régression Continue 20%)  │
                        └──────────────┬────────────────┘
                                       │
                        ┌──────────────▼────────────────┐
                        │   3. Moteur TAP-NN (60%)      │
                        │      - Graph Transformer      │
                        │      - Logique Soft-Gödel     │
                        │      - Differentiable Beam    │
                        └──────────────┬────────────────┘
                                       │
                        ┌──────────────▼────────────────┐
                        │   4. SCGEnergyPruner (Filtre) │
                        │  (Élagage géométrique direct) │
                        └──────────────┬────────────────┘
                                       │
                        ┌──────────────▼────────────────┐
                        │ 5. Allocation Différentielle  │
                        │  (Budget Mémoire Dynamique k) │
                        └──────────────┬────────────────┘
                                       │
                        ┌──────────────▼────────────────┐
                        │    Réponse Certifiée & Trace  │
                        └───────────────────────────────┘
