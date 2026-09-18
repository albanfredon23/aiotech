import torch
from core.aiotech44_core import AIOTECH44_EnergyCore


def main():
    print("=" * 60)
    print("       AIOTECH44 : PIPELINE & VALIDATION D'INTÉGRATION")
    print("=" * 60)

    emb_dim = 256
    num_nodes = 50
    num_agents = 4
    batch_size = 2
    seq_len = 10

    # 1. Instanciation du modèle
    core = AIOTECH44_EnergyCore(
        emb_dim=emb_dim,
        num_nodes=num_nodes,
        num_agents=num_agents
    )

    # 2. Entrées de test synthétiques
    query_emb = torch.randn(batch_size, emb_dim, requires_grad=True)
    retrieved_docs = torch.randn(batch_size, seq_len, emb_dim, requires_grad=True)
    graph_nodes = torch.randn(batch_size, num_nodes, emb_dim, requires_grad=True)
    constraints = torch.randn(batch_size, emb_dim)

    # -------------------------------------------------------------
    # Étape A : SMOKE TEST
    # -------------------------------------------------------------
    print("\n[1/3] Exécution du Smoke Test...")
    core.eval()
    with torch.no_grad():
        output = core(query_emb, retrieved_docs, graph_nodes, constraints)

    print("✓ Inférence exécutée avec succès !")
    print(f" - Complexité évaluée     : {output['complexity_score'].mean().item():.3f}")
    print(f" - Forme de la politique   : {output['policy'].shape}")
    print(f" - Trajectoires admises   : {output['trajectories'].shape}")
    print(f" - Budget k alloué        : {output['budget_k'].tolist()}")
    print(f" - Tokens alloués (somme) : {output['allocated_tokens']}")
    print(f" - Ratio mémoire utilisé  : {output['allocated_memory_ratio']:.1%}")
    print(f" - Poids des agents       : {output['active_agents'].numpy().round(3)}")

    # -------------------------------------------------------------
    # Étape B : GRADIENT CHECK (Vérification autograd de bout en bout)
    # -------------------------------------------------------------
    print("\n[2/3] Exécution du Gradient Check...")
    core.train()
    train_out = core(query_emb, retrieved_docs, graph_nodes, constraints)
    
    # Perte combinée : décision + régularisation géométrique SCG
    loss = train_out["policy"].sum() + train_out["scg_loss"]
    loss.backward()

    assert query_emb.grad is not None, "Gradient absent sur query_emb"
    assert retrieved_docs.grad is not None, "Gradient absent sur retrieved_docs"
    assert core.policy_head.weight.grad is not None, "Gradient absent sur policy_head"
    print("✓ Propagation du gradient vérifiée de bout en bout sans rupture.")

    # -------------------------------------------------------------
    # Étape C : CONTINUOUS LEARNING (Boucle de 15 étapes)
    # -------------------------------------------------------------
    print("\n[3/3] Exécution de la boucle d'apprentissage continu (15 étapes)...")
    initial_weights = core.agent_allocator.agent_logits.clone().detach()

    for step in range(15):
        # Simulation d'un signal de coût d'erreur asymétrique sur les agents
        # (ex. l'agent 1 et 2 accumulent des violations SCG)
        synthetic_agent_losses = torch.tensor([0.8, 0.6, 0.1, 0.05])
        core.agent_allocator.update_continuous_weights(synthetic_agent_losses)

    final_weights = core.agent_allocator.agent_logits.clone().detach()
    poids_modifies = not torch.equal(initial_weights, final_weights)
    assert poids_modifies, "Les poids des agents n'ont pas été actualisés !"

    print(f" - Logits initiaux : {initial_weights.numpy().round(3)}")
    print(f" - Logits finaux   : {final_weights.numpy().round(3)}")
    print("✓ Mise à jour continue des poids validée sur 15 étapes.")

    print("\n" + "=" * 60)
    print("   TOUTES LES VÉRIFICATIONS D'INTÉGRATION SONT PASSÉES")
    print("=" * 60)


if __name__ == "__main__":
    main()
