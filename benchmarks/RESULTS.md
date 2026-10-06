# Résultats mesurés – 2026-10-05

Environnement : CPython 3.12 (Pyodide 0.27.2), numpy. Commandes reproductibles :
`python -m benchmarks.bench_context` et `python -m benchmarks.bench_cache`.

Tests : `pytest` → **55 réussis, 2 ignorés** (`test_api.py` faute de FastAPI, `test_research.py`
faute de PyTorch dans cet environnement ; à lancer après `pip install -r requirements-full.txt`).

## Banc 1 – réduction du contexte (ARG)

Corpus de test : 53 documents (48 fiches produit quasi identiques + 5 politiques),
295 questions (247 à un document, 48 comparatives à deux documents). Référence : les 8
meilleurs documents d'une recherche TF-IDF, envoyés en entier (RAG standard).

| Groupe | Tokens référence | Tokens ARG | Réduction | Rappel référence | Rappel ARG | k moyen |
|---|---:|---:|---:|---:|---:|---:|
| Toutes | 841 | 123 | **-85,4 %** | 100 % | **100 %** | 1,17 |
| Un document | 841 | 106 | -87,4 % | 100 % | 100 % | 1,01 |
| Deux documents | 844 | 213 | -74,8 % | 100 % | 100 % | 2,00 |

Rappel = tous les documents nécessaires à la réponse sont présents dans le contexte envoyé.
τ n'influe pas sur la sélection (pilotée par le gain de couverture) mais sur le signal
`fallback` (contexte jugé insuffisant) : 0 % des requêtes à τ = 0,35 (défaut), 18 % à τ = 0,40,
33 % à τ = 0,50.

Historique :
- La première version de l'ARG jugeait chaque segment seul. Sur les questions comparatives,
  son rappel était de **0 %** (une seule fiche gardée). Ce banc l'a révélé ; la sélection est
  désormais jugée sur l'ensemble des segments retenus.
- Sans racinisation, « retourner » ne rejoignait pas « retournées » : la bonne politique était
  choisie mais signalée « contexte insuffisant » (2,4 % des requêtes à τ = 0,35). Une
  racinisation légère FR/EN (`aiotech/text.py:stem`) ramène ce taux à 0 %.

## Banc 2 – cache d'équivalence lexicale contextuelle

Cache *context-aware fuzzy / lexical* : similarité de Jaccard pondérée sur les traits lexicaux hachés
(mots racinisés et trigrammes de caractères), empreinte du contexte, garde sur les identifiants.
Aucun embedding neuronal.

Flux de 3 000 requêtes, loi de Zipf (s = 1,1) sur les 295 questions, 5 formulations par question.

| Garde identifiants | Taux de réponses servies par le cache | Réponses erronées servies |
|---|---:|---:|
| Oui (défaut) | 85,5 % | **0,00 %** |
| Non | 87,8 % | 4,23 % (4,8 % des hits) |

## Ce que ces chiffres ne disent pas

- Le corpus est synthétique et favorable : les 8 candidats sont des fiches quasi identiques
  dont une seule (ou deux) est utile. Sur un vrai corpus, la réduction sera plus faible ;
  elle doit être mesurée sur les documents et les questions du client.
- Le taux de cache dépend entièrement de la répétition du trafic réel (ici très répétitif).
  Les 85 % ne sont pas une prévision.
- Aucun LLM n'a été appelé : la qualité finale des réponses n'est pas mesurée ici. Il faut
  une évaluation avec le modèle et les données cibles (voir `aiotech/evaluation.py`).
