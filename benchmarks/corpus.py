"""
Corpus de test déterministe (fiches produit + documents de politique) avec vérité terrain.

Chaque question a un ou deux documents nécessaires ("gold") ; les questions comparatives
exigent deux fiches. Les distracteurs ont le
même gabarit et presque le même vocabulaire : seul l'identifiant change. C'est le cas
difficile d'un RAG réel (catalogues, contrats, tickets), où la recherche vectorielle
classique ramène beaucoup de voisins quasi identiques.

Limite assumée : corpus synthétique et lexical. Il mesure la mécanique de sélection
(tokens économisés, document utile conservé ou non), pas la qualité finale d'une réponse
de LLM, qui exige une évaluation avec le modèle et les données du client.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

FAMILIES = ["ORION", "VEGA", "ALTAIR", "LYRA", "SIRIUS", "DENEB"]
CITIES = ["Lyon", "Nantes", "Lille", "Grenoble", "Toulouse", "Rennes", "Bordeaux", "Dijon"]
SUPPLIERS = ["Delta Industries", "Méridien SA", "Atelier Brunel", "Nordtech", "Calypso Systèmes", "Groupe Arvel"]
MATERIALS = ["aluminium anodisé", "acier inoxydable", "polymère renforcé", "titane", "fonte d'aluminium"]
USES = ["la régulation thermique", "le pompage industriel", "la ventilation de salles serveurs",
        "le contrôle de pression", "la filtration d'air", "le refroidissement de batteries"]

POLICIES = [
    ("politique-retours", "Retours et remboursements",
     "Les commandes peuvent être retournées sous 30 jours après livraison. Le remboursement est émis sous 10 jours ouvrés "
     "après réception du colis. Les produits sur mesure ne sont ni repris ni échangés. Les frais de retour sont à la charge "
     "du client sauf en cas de produit défectueux."),
    ("politique-livraison", "Livraison",
     "La livraison standard prend 3 à 5 jours ouvrés en France métropolitaine. La livraison express en 24 heures coûte 19 euros. "
     "Les commandes passées avant 14 heures sont expédiées le jour même. Les envois vers la Corse prennent 2 jours supplémentaires."),
    ("politique-garantie", "Garantie commerciale",
     "La garantie commerciale couvre les défauts de fabrication. Elle ne couvre pas l'usure normale ni une installation non conforme. "
     "Une extension de garantie de 2 ans peut être souscrite au moment de l'achat pour 8 pour cent du prix catalogue."),
    ("politique-donnees", "Données personnelles",
     "Les données des clients sont hébergées en France et conservées 3 ans après le dernier achat. Toute demande d'effacement "
     "est traitée sous 30 jours. Les données ne sont jamais revendues à des tiers."),
    ("politique-support", "Support technique",
     "Le support technique est joignable du lundi au vendredi de 8 heures à 18 heures. Les clients sous contrat premium "
     "disposent d'une astreinte 24 heures sur 24 avec un délai d'intervention de 4 heures."),
]

POLICY_QUESTIONS = [
    ("politique-retours", "Sous combien de jours peut-on retourner une commande ?", "30 jours"),
    ("politique-retours", "Les produits sur mesure sont-ils repris ?", "ni repris"),
    ("politique-livraison", "Combien coûte la livraison express ?", "19 euros"),
    ("politique-livraison", "Quel est le délai de livraison standard ?", "3 à 5 jours"),
    ("politique-garantie", "Combien coûte l'extension de garantie ?", "8 pour cent"),
    ("politique-donnees", "Combien de temps les données des clients sont-elles conservées ?", "3 ans"),
    ("politique-support", "Quel est le délai d'intervention du contrat premium ?", "4 heures"),
]


@dataclass
class Question:
    text: str
    gold: Tuple[str, ...]  # sources des documents nécessaires à la réponse (1 ou 2)
    answer: str            # fragment de réponse présent dans un document gold

    @property
    def multi_doc(self) -> bool:
        return len(self.gold) > 1


def build_corpus(n_products: int = 48, seed: int = 7) -> Dict[str, object]:
    rng = random.Random(seed)
    docs: List[Dict[str, object]] = []
    questions: List[Question] = []
    facts: List[Dict[str, object]] = []
    for i in range(n_products):
        fam = FAMILIES[i % len(FAMILIES)]
        code = f"{fam}-{10 + i}"
        weight = f"{rng.randint(12, 95) / 10:.1f}".replace(".", ",")
        city = rng.choice(CITIES)
        supplier = rng.choice(SUPPLIERS)
        price = f"{rng.randint(4, 60) * 50:,}".replace(",", " ")
        warranty = rng.choice([1, 2, 3, 5])
        material = rng.choice(MATERIALS)
        use = rng.choice(USES)
        power = rng.randint(2, 40) * 25
        text = (
            f"Fiche technique du module {code}. Le module {code} est conçu pour {use}. "
            f"Poids : {weight} kg. Matériau du boîtier : {material}. Puissance nominale : {power} watts. "
            f"Le module {code} est fabriqué à {city} et fourni par {supplier}. "
            f"Prix catalogue : {price} euros hors taxes. Garantie constructeur : {warranty} ans. "
            f"Le module {code} est compatible avec les armoires de la gamme {fam} et se fixe sur rail standard."
        )
        src = f"fiche-{code.lower()}"
        docs.append({"text": text, "source": src, "metadata": {"type": "fiche", "gamme": fam}})
        facts.append({"code": code, "src": src, "weight": weight, "city": city})
        g = (src,)
        questions += [
            Question(f"Quel est le poids du module {code} ?", g, f"{weight} kg"),
            Question(f"Où est fabriqué le module {code} ?", g, city),
            Question(f"Quel est le prix catalogue du {code} ?", g, f"{price} euros"),
            Question(f"Quelle est la durée de garantie du module {code} ?", g, f"{warranty} ans"),
            Question(f"Qui fournit le module {code} ?", g, supplier),
        ]
    # Questions comparatives : la réponse exige DEUX fiches.
    for _ in range(n_products):
        a, b = rng.sample(facts, 2)
        pair = (a["src"], b["src"])
        if rng.random() < 0.5:
            questions.append(Question(f"Lequel est le plus lourd : le module {a['code']} ou le module {b['code']} ?",
                                      pair, f"{a['weight']} kg"))
        else:
            questions.append(Question(f"Les modules {a['code']} et {b['code']} sont-ils fabriqués dans la même ville ?",
                                      pair, str(a["city"])))
    for src, title, body in POLICIES:
        docs.append({"text": f"{title}. {body}", "source": src, "metadata": {"type": "politique"}})
    questions += [Question(q, (src,), a) for src, q, a in POLICY_QUESTIONS]
    rng.shuffle(questions)
    return {"documents": docs, "questions": questions}


def paraphrases(q: Question) -> List[str]:
    """Variantes de surface d'une même question (pour le banc du cache)."""
    t = q.text
    base = t.rstrip(" ?")
    return [t, t.lower(), base + " ?", base.replace("Quel est le ", "").replace("Quelle est la ", "") + " ?",
            "Peux-tu me dire : " + t[0].lower() + t[1:]]
