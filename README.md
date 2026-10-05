# AIOTECH

[![CI](https://github.com/albanfredon23/aiotech/actions/workflows/ci.yml/badge.svg)](https://github.com/albanfredon23/aiotech/actions/workflows/ci.yml)
[![Site](https://github.com/albanfredon23/aiotech/actions/workflows/pages.yml/badge.svg)](https://github.com/albanfredon23/aiotech/actions/workflows/pages.yml)

**Site : [albanfredon23.github.io/aiotech](https://albanfredon23.github.io/aiotech/)** : le pipeline en 3D,
l'innovation ARG, une démonstration en direct et le calculateur d'économies d'échelle.

AIOTECH est un middleware placé devant n'importe quel LLM (Claude, GPT, Gemini, Mistral, Ollama…
via LiteLLM). Avant chaque appel, il retire le contexte inutile, réutilise les réponses déjà
calculées et choisit le modèle le moins cher capable de répondre. **N'est envoyé au modèle que
ce qui est justifié**, et chaque réponse indique son coût réel et l'économie réalisée.

Version du code : `aiotech` 45.0.0 (site : AIOTECH 46). Elle fusionne et recode
**AIOTECH 44**, le raisonnement contraint désormais sans projection sphérique, et
**aio-v3**, la passerelle de production corrigée.

## Résultats mesurés

Sur le corpus de test du dépôt (295 questions, 53 documents, aucun appel LLM) :

| Mesure | Résultat |
|---|---|
| Tokens de contexte envoyés au modèle | **−85 %** (−87 % à un document, −75 % à deux documents) |
| Documents nécessaires conservés (rappel) | **100 %** |
| Réponses erronées servies par le cache | **0 %** (4,2 % sans la garde sur les identifiants) |

Le corpus est synthétique et favorable : ces chiffres valident la mécanique, pas un gain chez un
client donné. Détails, protocole et historique des corrections : [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Démarrer

### Avec Docker

```bash
cp .env.example .env          # clés fournisseurs + AIOTECH_ADMIN_TOKEN
docker compose up -d --build  # API sur http://localhost:8000, site sur http://localhost:8080
docker build --target test .  # lance la suite de tests dans l'image
```

L'image tourne sous un utilisateur non root, garde SQLite dans le volume `/data` et expose un
contrôle de santé sur `/health`. `--build-arg WITH_TORCH=true` ajoute PyTorch (CPU) pour le
module de recherche.

### Avec Python

```bash
pip install -r requirements.txt          # cœur : numpy uniquement
pip install -r requirements-full.txt     # API, fournisseurs LLM, PyTorch, tests
pytest                                    # 55 tests cœur + tests API + tests PyTorch
python examples/quickstart.py             # démonstration hors ligne, sans clé API
uvicorn aiotech.api.main:app --port 8000
```

### En bibliothèque

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

## Le pipeline

```
requête → garde-fous → ARG (contexte) → cache → routage modèle → disjoncteur → LLM
        → validation JSON → vérification de chaîne → comptabilité (coût réel vs référence)
```

| Étape | Rôle | Module |
|---|---|---|
| Garde-fous | Bloque les injections de prompt (FR/EN), masque e-mails, cartes, IBAN, IP, téléphones | `aiotech/gateway/guardrails.py` |
| ARG | N'envoie que les segments de documents qui couvrent la question | `aiotech/reasoning/context_budget.py` |
| Cache | Réutilise une réponse équivalente, jamais pour un identifiant différent | `aiotech/gateway/cache.py` |
| Routage | Envoie les demandes simples au modèle économique | `aiotech/gateway/router.py` |
| Disjoncteur | Bascule vers un autre fournisseur en cas de panne | `aiotech/gateway/circuit_breaker.py` |
| Vérification | Valide le JSON, signale les phrases sans appui dans le contexte | `aiotech/reasoning/arg.py` |
| Comptabilité | Coût réel, coût de référence (modèle phare + tout le contexte), économie | `aiotech/gateway/client.py` |

Les tarifs Claude par défaut datent du 2026-09-25 (`aiotech/gateway/pricing.py`). Les autres
modèles se déclarent via `AIOTECH_PRICES`. Un modèle sans tarif a un coût `None` : aucun
montant n'est inventé.

## L'innovation : de la sphère à l'ARG

AIOTECH 44 projetait chaque état sur l'hypersphère unité (normalisation L2) et mesurait les
violations par produit scalaire de vecteurs unitaires. L'échelle était jetée, les seuils
n'avaient pas d'unité, et les contraintes étaient des vecteurs aléatoires.

L'**ARG (Admissibility & Reachability Gate)** ne normalise rien. Chaque critère a une unité
explicite dans [0, 1] et l'admissibilité est leur **conjonction de Gödel** : un seul critère
défaillant suffit à rejeter.

```
A(S) = min( reach(q, ∪S), coverage(q, ∪S), contraintes(S) )      admissible ⇔ A(S) ≥ τ

reach(q, x)    = 1 − ‖relu(q − x)‖₂ / ‖q‖₂     part de la masse euclidienne de la question atteinte
coverage(q, x) = Σ IDF des mots de q présents dans x / Σ IDF des mots de q
```

- **Sélection de contexte** : ajout glouton du segment au plus grand gain de couverture, arrêt
  quand les segments restants sont redondants. Le jugement porte sur l'ensemble S retenu : une
  question qui compare deux produits garde les deux fiches.
- **Vérification de chaîne** : chaque phrase de la réponse doit être atteignable depuis le
  contexte et les phrases précédentes ; le maillon le plus faible est désigné.
- **Routage d'agents** : affinité `reach` et taux de succès appris en ligne, admissibilité de Gödel.

## Économies d'échelle

Le middleware a un coût fixe mensuel et un coût par requête quasi nul ; l'économie qu'il produit
croît avec le volume. Le coût par requête baisse donc vers un plancher, et le seuil de rentabilité
est le volume où il passe sous le coût sans middleware. Formules : `aiotech/economics.py` (testées),
reprises à l'identique par le calculateur du [site](https://albanfredon23.github.io/aiotech/#economies).

Exemple (scénario par défaut, hypothèses à remplacer par les vôtres) : 1 M requêtes par mois,
3 000 tokens de contexte, Claude Opus 5.5 / Haiku 4.5 → **20 600 $ → 10 207 $ par mois**, seuil
de rentabilité à 126 000 requêtes par mois.

## Structure

```
aiotech/
  reasoning/     ARG, budget de contexte, agents
  gateway/       garde-fous, cache, routage, disjoncteur, client, tarifs, mémoire
  economics.py   modèle d'économies d'échelle
  tenancy.py     locataires, clés API, quotas
  telemetry.py   métriques Prometheus
  research/      version PyTorch différentiable (module de recherche, non entraîné)
  api/main.py    API FastAPI
benchmarks/      bancs de mesure et RESULTS.md
dashboard/       tableau de bord temps réel
site/            site AIOTECH 46, publié sur GitHub Pages
tests/           suite pytest
```

## Corrections apportées aux deux bases d'origine

| Problème | Origine | Correction |
|---|---|---|
| Le cœur plantait (3 sorties attendues, 4 renvoyées) | 44 | Recodé, testé |
| Les documents retenus n'étaient jamais envoyés au LLM : aucune économie réelle | 44 | Seuls les segments ARG partent au LLM, économie comptée |
| Embeddings via projection aléatoire, graphe et contraintes `torch.randn` | 44 | Embeddings lexicaux déterministes, contraintes explicites |
| FLOPs « simulés » par une formule inventée | 44 | Bancs mesurant tokens et rappel |
| Les k documents gardés étaient les k premiers de la liste | 44 | Les k plus proches / gain de couverture |
| Cache « sémantique » = vecteur aléatoire d'un SHA-256 | v3 | Similarité lexicale réelle, garde identifiants, empreinte de contexte |
| Les erreurs étaient mises en cache | v3 | Jamais |
| `run_python` exécutait le code produit par le LLM | v3 | Remplacé par une calculatrice sans exécution |
| `read_file` lisait n'importe quel fichier, dont `.env` | v3 | Confiné à `AIOTECH_TOOLS_DIR` |
| Routes `/admin` sans authentification | v3 | Jeton d'administration obligatoire |
| Sans en-tête `X-API-Key`, aucun quota | v3 | Clé obligatoire par défaut |
| `/v1/stream` contournait garde-fous et quotas | v3 | Même contrôle que `/v1/query` |
| Modèles Claude classés « openai » par le disjoncteur | v3 | Détection du fournisseur corrigée |
| Recréer un locataire renvoyait une clé inutilisable | v3 | Noms uniques, erreur 409 |
| Mémoire « persistante » en `:memory:` | v3 | Chemin `AIOTECH_DB_PATH` |
| Chemin du tableau de bord codé en dur | v3 | Relatif au paquet |
| Export Prometheus invalide | v3 | Une ligne TYPE par famille, type `summary` |
| Fusion de couches linéaires à travers une activation | v3 | Fusion seulement si strictement consécutives |

## Limites connues

- L'ARG utilise des traits lexicaux avec une racinisation légère : les flexions passent
  (retourner / retournées), les synonymes purs passent moins bien. `LiteLLMEmbedder` permet de
  brancher des embeddings neuronaux (payants).
- Le routage par complexité est une heuristique : le seuil se calibre sur le trafic réel.
- La vérification de chaîne détecte des affirmations sans appui lexical ; elle ne prouve pas
  qu'une réponse est vraie.
- `aiotech/research/` est un module de recherche non entraîné : il a des tests de formes et de
  gradients, pas de résultats.
- Les tests de l'API et de PyTorch n'ont pas pu être exécutés dans l'environnement de livraison ;
  la CI GitHub les lance à chaque push.

## Publication du site

Le workflow `.github/workflows/pages.yml` publie `site/` sur GitHub Pages à chaque push qui
modifie le site. À activer une seule fois : **Settings → Pages → Build and deployment → Source :
GitHub Actions**. Si le premier déploiement a échoué faute d'activation, le relancer depuis
l'onglet Actions (workflow « Site », bouton *Run workflow*).
