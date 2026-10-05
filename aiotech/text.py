"""
Outils texte : normalisation, découpage en mots, estimation de tokens, découpage en segments.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import List

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")

# Mots vides FR/EN : ignorés pour la couverture de requête (pas pour l'embedding).
STOPWORDS = frozenset(
    """
    a au aux avec ce ces cet cette dans de des du elle en et est etre il ils je la le les leur lui
    ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que quel quelle quels quelles
    qui sa se ses son sont sur ta te tes toi ton tu un une vos votre vous y d l s n c j m t
    comment combien quoi ou quand pourquoi sous peut peuvent lequel laquelle lesquels lesquelles
    faire fait dire avoir plus moins tres aussi entre ainsi alors donc cela ceci ca chaque quelque
    the a an and are as at be by for from has have in is it its of on or that the to was were what
    which who whom why how with this these those do does did can could should would will about
    into than then there their them they also more less very
    """.split()
)

# Suffixes retirés par la racinisation légère, du plus long au plus court.
_SUFFIXES = (
    "issements", "issement", "atrices", "ateurs", "ations", "ements", "ement", "atrice", "ateur",
    "ation", "ances", "ences", "ance", "ence", "euses", "euse", "ites", "ite", "ives", "ive",
    "ees", "ers", "ent", "ee", "er", "ez", "es", "e", "s", "x",
)


def stem(word: str) -> str:
    """Racinisation légère FR/EN : retourner, retournées -> retourn ; commandes -> command.

    Volontairement simple (un seul suffixe retiré, racine d'au moins 4 lettres) et
    identique au portage JavaScript du site. Les jetons contenant un chiffre sont gardés tels quels.
    """
    if any(ch.isdigit() for ch in word):
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize(text: str) -> str:
    return strip_accents(text.lower())


def words(text: str) -> List[str]:
    """Mots normalisés (minuscules, sans accents). Les nombres décimaux restent entiers."""
    return _WORD_RE.findall(normalize(text))


def content_words(text: str) -> List[str]:
    """Mots porteurs de sens, racinisés (mots vides et lettres isolées retirés)."""
    return [stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def identifiers(text: str) -> frozenset:
    """Jetons contenant un chiffre (références, montants, codes produit).

    Deux requêtes qui diffèrent sur ces jetons ne doivent jamais partager une réponse en cache.
    """
    return frozenset(w for w in words(text) if any(ch.isdigit() for ch in w))


def estimate_tokens(text: str) -> int:
    """Estimation du nombre de tokens (~4 caractères par token, règle usuelle des tokeniseurs BPE).

    C'est une estimation : pour une facturation exacte, utiliser le champ `usage` renvoyé
    par le fournisseur, que le client AIOTECH privilégie quand il est disponible.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(text: str, max_tokens: int = 120) -> List[str]:
    """Découpe un document en segments d'environ `max_tokens`, sans couper de phrase.

    Des segments de taille comparable rendent les distances euclidiennes comparables
    entre segments (c'est ce qui permet à l'ARG de se passer de normalisation sphérique).
    """
    sentences = split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for s in sentences:
        t = estimate_tokens(s)
        if current and current_tokens + t > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(s)
        current_tokens += t
    if current:
        chunks.append(" ".join(current))
    return chunks
