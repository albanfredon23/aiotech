import torch
from core.aiotech44_core import AIOTECH44_EnergyCore

def main():
    print("=" * 55)
    print("  INITIALISATION DU MIDDLEWARE COGNITIF AIOTECH44")
    print("=" * 55)

    emb_dim = 256
    num_nodes = 50
    num_agents = 4

    # Instanciation de l'orchestrateur central
    engine = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=num_nodes, num_agents=num_agents)

    # Simulation d'un contexte de test
    query_emb = torch.randn(1, emb_dim)
    retrieved_docs = torch.randn(1, 10, emb_dim)
    graph_nodes = torch.randn(1, num_nodes, emb_dim)
    constraints = torch.randn(1, emb_dim)

    # Exécution de la passe avant
    output = engine(query_emb, retrieved_docs, graph_nodes, constraints)

    print("\n✓ Inférence exécutée avec succès !")
    print(f" - Complexité évaluée     : {output['complexity_score'].item():.3f}")
    print(f" - Trajectoires admises   : {output['trajectories'].shape}")
    print(f" - Budget mémoire alloué  : {output['budget_k'].item()} nœuds")
    print(f" - Poids des agents (20%) : {output['active_agents'].detach().numpy().round(3)}")
    print("=" * 55)

if __name__ == "__main__":
    main()
