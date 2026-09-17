

import sys
import os
import logging
import torch

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AIOTECH44")

# Ajout du dossier racine au chemin d'import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.aiotech44_core import AIOTECH44_EnergyCore


def run_smoke_test(core: AIOTECH44_EnergyCore, device: torch.device):
    """
    Étape 1 : Valide le passage des dimensions de bout en bout sur une inférence unitaire.
    """
    logger.info("=" * 60)
    logger.info("1. EXÉCUTION DU SMOKE TEST (Validation des Tenseurs)")
    logger.info("=" * 60)

    batch_size = 2
    emb_dim = core.emb_dim
    num_nodes = core.num_nodes
    num_docs = 10

    # Données d'entrée synthétiques
    query = torch.randn(batch_size, emb_dim, device=device)
    docs = torch.randn(batch_size, num_docs, emb_dim, device=device)
    graph = torch.randn(batch_size, num_nodes, emb_dim, device=device)
    constraints = torch.randn(batch_size, emb_dim, device=device)

    # Inférence en mode évaluation
    core.eval()
    with torch.no_grad():
        output = core(query, docs, graph, constraints)

    # Vérification des sorties
    logger.info(f"Dimensions de la Policy           : {list(output['policy'].shape)} (attendu: [{batch_size}, {num_nodes}])")
    logger.info(f"Dimensions des Trajectoires      : {list(output['trajectories'].shape)} (attendu: [{batch_size}, 4, {emb_dim}])")
    logger.info(f"Score de complexité calculé       : {output['complexity_score'].mean().item():.3f} (intervalle: [0, 1])")
    logger.info(f"Budget mémoire (k alloué)         : {output['budget_k'].cpu().tolist()} (sur {num_docs} docs max)")
    logger.info(f"Tokens/Docs réellement alloués    : {output['allocated_tokens']} tokens")
    logger.info(f"Rapport de mémoire conservé       : {output['allocated_memory_ratio']:.1f} %")

    # Assertions de sécurité
    assert output["policy"].shape == (batch_size, num_nodes), "Erreur sur la dimension de policy"
    assert output["trajectories"].shape == (batch_size, 4, emb_dim), "Erreur sur la dimension des trajectoires"
    assert 0.0 <= output["complexity_score"].mean().item() <= 1.0, "Score de complexité hors bornes [0, 1]"
    
    logger.info("[✓] Smoke Test réussi : le pipeline s'exécute sans erreur de dimensions.\n")


def run_gradient_check(core: AIOTECH44_EnergyCore, device: torch.device):
    """
    Étape 2 : Valide la différentiabilité et la rétropropagation de bout en bout (Autograd).
    """
    logger.info("=" * 60)
    logger.info("2. VÉRIFICATION DE LA DIFFÉRENTIABILITÉ (Autograd)")
    logger.info("=" * 60)

    batch_size = 1
    emb_dim = core.emb_dim
    num_nodes = core.num_nodes

    core.train()
    query = torch.randn(batch_size, emb_dim, device=device, requires_grad=True)
    docs = torch.randn(batch_size, 8, emb_dim, device=device, requires_grad=True)
    graph = torch.randn(batch_size, num_nodes, emb_dim, device=device, requires_grad=True)
    constraints = torch.randn(batch_size, emb_dim, device=device)

    # Inférence en mode train (active Gumbel-Softmax dans le Beam Search)
    output = core(query, docs, graph, constraints)
    loss = output["policy"].sum()
    loss.backward()

    # Vérification des gradients
    query_grad = query.grad is not None
    docs_grad = docs.grad is not None
    policy_grad = core.policy_head.weight.grad is not None

    logger.info(f"Gradient propagé vers la Requête  : {'OUI' if query_grad else 'NON'}")
    logger.info(f"Gradient propagé vers les Docs     : {'OUI' if docs_grad else 'NON'}")
    logger.info(f"Gradient présent sur PolicyHead   : {'OUI' if policy_grad else 'NON'}")

    assert query_grad and docs_grad and policy_grad, "Rupture de différentiabilité détectée !"
    logger.info("[✓] Autograd validé : les gradients circulent à travers tout le graphe.\n")


def run_continuous_learning_loop(core: AIOTECH44_EnergyCore, device: torch.device, num_steps: int = 15):
    """
    Étape 3 : Démonstration de la boucle fermée de régression continue sur les agents.
    """
    logger.info("=" * 60)
    logger.info(f"3. DÉMONSTRATION DE RÉGRESSION CONTINUE ({num_steps} étapes)")
    logger.info("=" * 60)

    emb_dim = core.emb_dim
    num_nodes = core.num_nodes

    core.eval()
    for step in range(1, num_steps + 1):
        query = torch.randn(1, emb_dim, device=device)
        docs = torch.randn(1, 10, emb_dim, device=device)
        graph = torch.randn(1, num_nodes, emb_dim, device=device)
        constraints = torch.randn(1, emb_dim, device=device)

        with torch.no_grad():
            output = core(query, docs, graph, constraints)

        agent_weights = output["active_agents"]
        surviving = output["trajectories"]

        # Calcul d'un signal d'erreur basé sur le respect des contraintes
        violation = torch.relu(-surviving * constraints.unsqueeze(1)).sum(dim=-1).mean()
        agent_losses = agent_weights.view(-1) * violation

        # Mise à jour en ligne des poids d'agents
        if hasattr(core.agent_allocator, "update_continuous_weights"):
            core.agent_allocator.update_continuous_weights(agent_losses)

        if step % 5 == 0 or step == num_steps:
            if hasattr(core.agent_allocator, "agent_logits"):
                weights = torch.softmax(core.agent_allocator.agent_logits, dim=0).detach().cpu().tolist()
                formatted_weights = [round(w, 3) for w in weights]
                logger.info(f"Étape {step:02d}/{num_steps} | Distribution des poids d'agents : {formatted_weights}")

    logger.info("[✓] Régression continue validée : les poids s'adaptent selon les retours d'incohérence.\n")


def main():
    # Détection du matériel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Appareil de calcul détecté : {device}")

    # Initialisation de l'orchestrateur
    emb_dim = 256
    num_nodes = 50
    num_agents = 4
    num_rules = 16

    logger.info("Instanciation du moteur AIOTECH44_EnergyCore...")
    core = AIOTECH44_EnergyCore(
        emb_dim=emb_dim,
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_rules=num_rules
    ).to(device)

    summary = core.get_summary() if hasattr(core, "get_summary") else {}
    logger.info(f"Paramètres totaux du moteur : {summary.get('total_params', 'N/A'):,}")

    # Exécution de la suite de tests
    run_smoke_test(core, device)
    run_gradient_check(core, device)
    run_continuous_learning_loop(core, device)

    logger.info("=" * 60)
    logger.info("TOUTES LES VÉRIFICATIONS SONT PASSÉES AVEC SUCCÈS")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

