# AIOTECH 45

[![CI](https://github.com/albanfredon23/aiotech/actions/workflows/ci.yml/badge.svg)](https://github.com/albanfredon23/aiotech/actions/workflows/ci.yml)

Middleware placé devant n'importe quel LLM (Claude, GPT, Gemini, Mistral, Ollama… via LiteLLM)
pour **n'envoyer au modèle que ce qui est justifié** : le contexte documentaire utile, le
modèle adapté à la difficulté, et aucune requête déjà résolue.

AIOTECH 45 fusionne et recode :
- **aiotech-main (AIOTECH 44)** : le raisonnement contraint, sans la projection sphérique ;
- **aio-v3** : la passerelle de production (cache, routage, disjoncteur, garde-fous,
  multi-locataires, télémétrie), corrigée.

## Ce qui a changé : du raisonnement sphérique à l'ARG

AIOTECH 44 projetait chaque état sur l'hypersphère unité (normalisation L2) et mesurait les
violations par produit scalaire de vecteurs unitaires. L'échelle était jetée, les seuils
n'avaient pas d'unité, et les contraintes étaient des vecteurs aléatoires.

L'**ARG (Admissibility & Reachability Gate)** ne normalise rien. Chaque critère a une unité
explicite dans [0, 1] et l'admissibilité est leur **conjonction de Gödel** (un seul critère
défaillant suffit à rejeter) :

```
A(S) = T_G( reach(q, ∪S), coverage(q, ∪S), contraintes(S) ) = min(...)      admissible ⇔ A(S) ≥ τ

reach(q, x)    = 1 − ‖relu(q − x)‖₂ / ‖q‖₂     part de la masse euclidienne de la requête atteinte
coverage(q, x) = Σ IDF des mots de q présents dans x / Σ IDF des mots de q
```

- **Sélection de contexte** : ajout glouton du segment au plus grand gain de couverture,
  arrêt quand les segments restants sont redondants. Seuls les segments retenus partent au LLM.
- **Vérification de chaîne** : chaque étape de la réponse doit être atteignable depuis le
  contexte et les étapes précédentes ; T_G = min des supports, le maillon le plus faible est
  désigné (détecteur d'affirmations non fondées, heuristique lexicale).
- **Routage d'agents** : affinité `reach` + taux de succès appris en ligne, admissibilité de Gödel.

## Pipeline

```
requête → garde-fous → ARG (contexte) → cache → routage modèle → disjoncteur → LLM
        → validation JSON → vérification de chaîne → comptabilité (coût réel vs référence)
```

Chaque réponse renvoie `cost_usd`, `baseline_cost_usd` (modèle phare + tout le contexte) et
`saved_usd`. Les tarifs Claude par défaut datent du 2026-09-25 (`aiotech/gateway/pricing.py`) ;
les autres modèles se déclarent via `AIOTECH_PRICES`. Un modèle sans tarif a un coût `None` :
aucun montant n'est inventé.

## Résultats mesurés

Voir [benchmarks/RESULTS.md](benchmarks/RESULTS.md). Sur le corpus de test (295 questions) :
**-85 % de tokens de contexte à rappel identique (100 %)**, -75 % sur les questions à deux
documents ; cache sans aucune réponse erronée grâce à la garde sur les identifiants (4,2 %
d'erreurs sans elle). Corpus synthétique et favorable : ces chiffres valident la mécanique,
pas un gain chez un client donné.

## Installation

```bash
pip install -r requirements.txt          # cœur : numpy uniquement
pip install -r requirements-full.txt     # API, fournisseurs LLM, PyTorch, tests
cp .env.example .env
pytest                                    # 55 tests cœur + API (FastAPI) + recherche (PyTorch)
python -m benchmarks.bench_context
python -m benchmarks.bench_cache
uvicorn aiotech.api.main:app --port 8000
```

### Docker

```bash
cp .env.example .env          # clés fournisseurs + AIOTECH_ADMIN_TOKEN
docker compose up -d --build  # API sur :8000, site AIOTECH 46 sur :8080
docker build --target test .  # lance la suite de tests dans l'image
```

L'image tourne sous un utilisateur non root, persiste SQLite dans le volume `/data` et expose
un contrôle de santé sur `/health`. `--build-arg WITH_TORCH=true` ajoute PyTorch (CPU) pour
le module de recherche. La CI GitHub (`.github/workflows/ci.yml`) lance à chaque push tous les
tests (cœur, API, PyTorch), les bancs de mesure, puis construit et démarre l'image.

Le site statique est dans `site/index.html`.

Utilisation en bibliothèque :

```python
from aiotech.gateway import AiotechClient

client = AiotechClient()
r = client.completion_sync(
    [{"role": "user", "content": "Quel est le poids du module ORION-12 ?"}],
    documents=[{"text": "...", "source": "fiche-orion-12"}, ...],
    verify_chain=True,
)
print(r["content"], r["context"]["tokens_saved"], r["saved_usd"])
```

API : `POST /v1/query` (clé `X-API-Key`), `POST /v1/stream`, `POST /v1/economics`,
`/admin/*` (jeton `X-Admin-Token`), `GET /metrics` (Prometheus), `GET /dashboard`.

## Corrections apportées aux deux bases d'origine

| Problème | Origine | Correction |
|---|---|---|
| Le cœur plantait (3 sorties attendues, 4 renvoyées) | 44 | Recodé, testé |
| Les documents retenus n'étaient jamais envoyés au LLM : aucune économie réelle | 44 | Seuls les segments ARG partent au LLM, économie comptée |
| Embeddings via projection aléatoire, graphe et contraintes `torch.randn` | 44 | Embeddings lexicaux déterministes, contraintes explicites |
| FLOPs « simulés » par une formule inventée | 44 | Bancs mesurant tokens et rappel |
| Les k documents gardés étaient les k premiers de la liste | 44 | Les k plus proches (recherche) / gain de couverture (passerelle) |
| Cache « sémantique » = vecteur aléatoire d'un SHA-256 (exact déguisé) | v3 | Similarité lexicale réelle + garde identifiants + empreinte de contexte |
| Les erreurs étaient mises en cache | v3 | Jamais |
| `run_python` exécutait le code du LLM (`exec`) : exécution de code à distance | v3 | Remplacé par une calculatrice sur arbre syntaxique |
| `read_file` lisait n'importe quel fichier (dont `.env`) | v3 | Confiné à `AIOTECH_TOOLS_DIR` |
| Routes `/admin` sans authentification | v3 | Jeton d'administration obligatoire |
| Sans en-tête `X-API-Key`, aucun quota | v3 | Clé obligatoire par défaut |
| `/v1/stream` contournait garde-fous et quotas | v3 | Même contrôle que `/v1/query` |
| Modèles Claude classés « openai » par le disjoncteur | v3 | Détection du fournisseur corrigée |
| Recréer un locataire renvoyait une clé inutilisable | v3 | Noms uniques, erreur 409 |
| Mémoire « persistante » en `:memory:` | v3 | Chemin `AIOTECH_DB_PATH` |
| Chemin du tableau de bord codé en dur (`/home/claude/...`) | v3 | Relatif au paquet |
| Export Prometheus invalide (TYPE dupliqués) | v3 | Une ligne TYPE par famille, type `summary` |
| Fusion de couches linéaires à travers une activation (change la fonction) | v3 | Fusion seulement si strictement consécutives |
| Modèles par défaut retirés (gpt-3.5-turbo, claude-3-5-sonnet-20241022) | v3 | Configurables, défauts Claude actuels |

## Limites connues

- L'ARG utilise des traits lexicaux avec une racinisation légère : les flexions passent
  (retourner / retournées), une reformulation sans mot commun avec le document
  (synonymes purs) passe moins bien. `LiteLLMEmbedder` permet de brancher des embeddings
  neuronaux (payants), à intégrer au critère `reach`.
- Le routage par complexité est une heuristique ; le seuil doit être calibré sur le trafic réel.
- La vérification de chaîne détecte des affirmations sans appui lexical dans le contexte ;
  elle ne prouve pas qu'une réponse est vraie.
- `aiotech/research/` (PyTorch) est un module de recherche **non entraîné** : il a des tests de
  formes et de gradients, il n'a pas de résultats. Il n'a pas pu être exécuté dans
  l'environnement de livraison (PyTorch absent).
- `tests/test_api.py` n'a pas pu être exécuté dans l'environnement de livraison (FastAPI absent).
