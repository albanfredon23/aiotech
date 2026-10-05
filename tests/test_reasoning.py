import numpy as np
import pytest

from aiotech.embeddings import HashingEmbedder
from aiotech.reasoning.agents import AgentModulator, OnlineSuccessEstimator
from aiotech.reasoning.arg import ArgConstraints, ReachabilityGate, godel_tnorm, soft_godel
from aiotech.reasoning.context_budget import ContextBudgeter
from aiotech.text import chunk_text, estimate_tokens

DOCS = [
    {"text": "Fiche du module ORION-12. Poids : 4,2 kg. Fabriqué à Lyon. Garantie : 3 ans.", "source": "orion-12"},
    {"text": "Fiche du module ORION-13. Poids : 7,9 kg. Fabriqué à Lille. Garantie : 2 ans.", "source": "orion-13"},
    {"text": "Fiche du module ORION-14. Poids : 5,5 kg. Fabriqué à Rennes. Garantie : 5 ans.", "source": "orion-14"},
    {"text": "Politique de retours : les commandes sont retournables sous 30 jours.", "source": "retours"},
    {"text": "Livraison standard en 3 à 5 jours ouvrés, express en 24 heures.", "source": "livraison"},
]


# ── t-normes ──────────────────────────────────────────────────────────────────
def test_godel_is_min_and_empty_is_true():
    assert godel_tnorm([0.9, 0.2, 0.7]) == pytest.approx(0.2)
    assert godel_tnorm([]) == 1.0


def test_soft_godel_close_to_min_and_below():
    vals = [0.9, 0.3, 0.8]
    s = soft_godel(vals, temperature=0.01)
    assert s == pytest.approx(0.3, abs=0.02)
    assert s <= 0.3 + 1e-9


# ── pas de projection sphérique ───────────────────────────────────────────────
def test_embeddings_are_not_projected_on_unit_sphere():
    e = HashingEmbedder()
    norms = [float(np.linalg.norm(e.embed_one(t))) for t in ["court", "un texte nettement plus long que le premier"]]
    assert all(abs(n - 1.0) > 0.1 for n in norms)
    assert norms[1] > norms[0]  # l'échelle (quantité d'information) est conservée


def test_reach_is_bounded_and_monotone():
    q = np.array([1.0, 1.0, 0.0])
    assert ReachabilityGate.reach(q, np.array([1.0, 1.0, 5.0])) == pytest.approx(1.0)
    assert ReachabilityGate.reach(q, np.zeros(3)) == pytest.approx(0.0)
    half = ReachabilityGate.reach(q, np.array([1.0, 0.0, 0.0]))
    assert 0.0 < half < 1.0


# ── sélection de contexte ─────────────────────────────────────────────────────
def test_arg_keeps_gold_and_drops_distractors():
    b = ContextBudgeter(gate=ReachabilityGate(tau=0.35))
    sel = b.select("Quel est le poids du module ORION-12 ?", DOCS)
    kept = [s.source for s in sel.selected]
    assert kept == ["orion-12"]
    assert sel.tokens_selected < sel.tokens_candidates
    assert 0.0 < sel.reduction < 1.0
    assert not sel.fallback


def test_arg_policy_question():
    sel = ContextBudgeter().select("Sous combien de jours peut-on retourner une commande ?", DOCS)
    assert [s.source for s in sel.selected] == ["retours"]


def test_two_document_question_keeps_both_documents():
    sel = ContextBudgeter().select("Lequel est le plus lourd : le module ORION-12 ou le module ORION-14 ?", DOCS)
    assert {s.source for s in sel.selected} == {"orion-12", "orion-14"}
    assert sel.tokens_saved > 0


def test_forbidden_term_rejects_segment():
    cons = ArgConstraints(forbidden_terms=["Lyon"])
    sel = ContextBudgeter().select("poids du module ORION-12", DOCS, cons)
    assert "orion-12" not in [s.source for s in sel.selected]


def test_required_terms_and_metadata_degree():
    c = ArgConstraints(required_terms=["poids", "garantie"], metadata={"langue": "fr"})
    assert c.degree("Poids 4 kg, garantie 2 ans", {"langue": "fr"}) == 1.0
    assert c.degree("Poids 4 kg", {"langue": "fr"}) == 0.5
    assert c.degree("Poids 4 kg, garantie 2 ans", {"langue": "en"}) == 0.0


def test_unrelated_question_gets_no_context_and_is_flagged():
    sel = ContextBudgeter().select("Quel temps fera-t-il demain à Paris ?", DOCS)
    assert sel.fallback is True
    assert sel.selected == [] and sel.tokens_selected == 0


def test_weak_but_related_context_is_sent_and_flagged():
    b = ContextBudgeter(gate=ReachabilityGate(tau=0.99))
    sel = b.select("Quelle est la garantie du module ORION-12 en Antarctique ?", DOCS)
    assert sel.fallback is True
    assert [s.source for s in sel.selected] == ["orion-12"]


def test_light_stemming_matches_inflections():
    from aiotech.text import content_words, stem
    assert stem("retourner") == stem("retournées".replace("é", "e")) == "retourn"
    assert stem("commandes") == stem("commande") == "command"
    assert stem("ORION-12".lower()) == "orion-12"
    assert content_words("Sous combien de jours peut-on retourner une commande ?") == ["jour", "retourn", "command"]
    sel = ContextBudgeter().select("Sous combien de jours peut-on retourner une commande ?", DOCS)
    assert [s.source for s in sel.selected] == ["retours"] and not sel.fallback


def test_token_budget_is_respected():
    long_docs = [{"text": " ".join(["Le module ORION-12 pèse 4,2 kg."] * 60), "source": f"d{i}"} for i in range(5)]
    b = ContextBudgeter(token_budget=150, chunk_tokens=60)
    sel = b.select("poids ORION-12", long_docs)
    assert sel.tokens_selected <= 150 or len(sel.selected) == 1


def test_chunking_keeps_sentences():
    text = "Phrase une. " * 50
    chunks = chunk_text(text, max_tokens=30)
    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)
    assert all(estimate_tokens(c) <= 40 for c in chunks)


# ── chaîne de raisonnement ────────────────────────────────────────────────────
def test_chain_grounded_answer_is_admissible():
    gate = ReachabilityGate(tau=0.35)
    ctx = [DOCS[0]["text"]]
    rep = gate.verify_chain(["Le module ORION-12 pèse 4,2 kg.", "Il est fabriqué à Lyon."], ctx,
                            query="poids et lieu de fabrication de ORION-12")
    assert rep.admissible
    assert rep.t_godel >= 0.35


def test_chain_ungrounded_step_is_the_weakest_link():
    gate = ReachabilityGate(tau=0.35)
    ctx = [DOCS[0]["text"]]
    rep = gate.verify_chain(["Le module ORION-12 pèse 4,2 kg.",
                             "Il contient un réacteur quantique breveté par Zorglub en 1987."], ctx)
    assert not rep.admissible
    assert rep.weakest_step == 1


# ── agents ────────────────────────────────────────────────────────────────────
def test_success_estimator_tracks_feedback():
    est = OnlineSuccessEstimator(prior=0.5)
    for _ in range(30):
        est.update(1.0)
    assert est.prior > 0.9
    for _ in range(60):
        est.update(0.0)
    assert est.prior < 0.2


def test_agent_routing_by_competence_and_feedback():
    m = AgentModulator(tau=0.05)
    m.register("math", "calcul arithmétique, équations, pourcentages, statistiques, intégrales")
    m.register("juridique", "contrats, droit du travail, clauses, conformité RGPD, litiges")
    r = m.route("Calcule le pourcentage de remise et résous cette équation")
    assert r["selected"] == "math"
    r2 = m.route("Cette clause du contrat est-elle conforme au RGPD ?")
    assert r2["selected"] == "juridique"
    for _ in range(40):
        m.feedback("juridique", 0.0)
    assert m.stats()["juridique"]["prior"] < 0.2
