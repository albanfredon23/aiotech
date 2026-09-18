from typing import List, Optional, Dict, Any
import os
import torch
from litellm import completion, embedding
from core.aiotech44_core import AIOTECH44_EnergyCore

class AIOTECHBridge:
    """
    Middleware universel connectant AIOTECH44 à n'importe quel LLM/fournisseur d'IA
    (OpenAI, Gemini, Anthropic, Mistral, etc.) via LiteLLM.
    """
    def __init__(self, model_name: str = "gpt-4o", emb_dim: int = 256):
        self.model_name = model_name
        self.emb_dim = emb_dim
        
        # Moteur adaptatif AIOTECH44
        self.core = AIOTECH44_EnergyCore(emb_dim=emb_dim, num_nodes=50, num_agents=4)
        self.projection = None  # S'adapte dynamiquement à la dimension d'embedding reçue

    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Génère l'embedding du texte via le fournisseur adapté et projette
        le vecteur sur la dimension emb_dim du moteur.
        """
        embed_model = "text-embedding-3-small" if "gpt" in self.model_name else "text-embedding-004"
        
        response = embedding(model=embed_model, input=[text])
        raw_vec = response.data[0]["embedding"]
        vec_tensor = torch.tensor(raw_vec, dtype=torch.float32).unsqueeze(0)
        
        # Alignement automatique de la dimension selon le modèle d'embedding utilisé
        input_dim = vec_tensor.size(-1)
        if self.projection is None or self.projection.in_features != input_dim:
            self.projection = torch.nn.Linear(input_dim, self.emb_dim)
            
        return self.projection(vec_tensor)

    def run_inference(
        self, 
        prompt: str, 
        reference_docs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Exécute le filtrage préventif SCG et l'inférence adaptative avant d'appeler le LLM.
        """
        if reference_docs is None:
            reference_docs = ["Document de référence par défaut."]

        # 1. Encodage sémantique de la requête et des documents
        query_emb = self.get_embedding(prompt)
        doc_embs = [self.get_embedding(doc) for doc in reference_docs]
        docs_tensor = torch.stack(doc_embs, dim=1)

        # 2. Initialisation des états latents et des contraintes SCG
        graph_nodes = torch.randn(1, 50, self.emb_dim)
        constraints = torch.randn(1, self.emb_dim)

        # 3. Filtrage SCG et calcul adaptatif AIOTECH44
        with torch.no_grad():
            decision = self.core(query_emb, docs_tensor, graph_nodes, constraints)

        complexity = decision["complexity_score"].item()
        active_budget = decision["budget_k"].mean().item()

        # 4. Inférence conditionnée vers le LLM sélectionné
        system_instruction = (
            f"Tu agis via le middleware AIOTECH44 (Complexité: {complexity:.2f}, "
            f"Budget alloué: {active_budget:.1f}). Sois concis, logique et sans hallucination."
        )

        response = completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 if complexity < 0.5 else 0.7
        )

        return {
            "response": response.choices[0].message.content,
            "complexity_score": complexity,
            "allocated_budget": active_budget,
            "model_used": self.model_name
        }
