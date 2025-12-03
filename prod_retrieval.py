# backend/app/services/retrieval/retriever.py
from typing import List, Dict, Optional
import numpy as np
from app.services.embeddings.embedder import LocalEmbedder
from app.services.vectorstore.faiss_manager import FaissManager
from app.services.retrieval.reranker import Reranker
from app.services.retrieval.query_analyzer import QueryAnalyzer
import logging

logger = logging.getLogger(__name__)

class AdvancedRetriever:
    """
    Multi-stage retrieval pipeline:
    1. Query analysis & expansion
    2. Vector similarity search (high recall)
    3. Reranking (high precision)
    4. Diversity filtering
    """
    
    def __init__(
        self,
        embedder: LocalEmbedder,
        vector_store: FaissManager,
        reranker: Optional[Reranker] = None,
        query_analyzer: Optional[QueryAnalyzer] = None
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker or Reranker()
        self.query_analyzer = query_analyzer or QueryAnalyzer()
    
    async def retrieve(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
        enable_reranking: bool = True,
        enable_diversity: bool = True,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve most relevant documents
        
        Args:
            query: User query
            tenant_id: Multi-tenant isolation
            top_k: Final number of results
            enable_reranking: Use reranker for precision
            enable_diversity: Apply MMR for diversity
            filters: Metadata filters (e.g., {'source': 'doc1.pdf'})
        
        Returns:
            List of results with scores and metadata
        """
        try:
            # Step 1: Query Analysis
            analyzed_query = await self.query_analyzer.analyze(query)
            logger.info(f"Query type: {analyzed_query['type']}, complexity: {analyzed_query['complexity']}")
            
            # Step 2: Query Expansion (for complex queries)
            queries_to_search = [query]
            if analyzed_query['complexity'] == 'complex':
                expanded_queries = await self.query_analyzer.expand_query(query)
                queries_to_search.extend(expanded_queries[:2])  # Add 2 expansions
            
            # Step 3: Vector Search (high recall)
            all_candidates = []
            for q in queries_to_search:
                query_embedding = self.embedder.embed([q])[0]
                
                # Retrieve top-20 candidates (high recall)
                candidates = await self.vector_store.search(
                    query_embedding,
                    k=20,
                    tenant_id=tenant_id,
                    filters=filters
                )
                all_candidates.extend(candidates)
            
            # Deduplicate
            seen_ids = set()
            unique_candidates = []
            for candidate in all_candidates:
                if candidate['id'] not in seen_ids:
                    unique_candidates.append(candidate)
                    seen_ids.add(candidate['id'])
            
            logger.info(f"Retrieved {len(unique_candidates)} unique candidates")
            
            # Step 4: Reranking (high precision)
            if enable_reranking and len(unique_candidates) > top_k:
                reranked = await self.reranker.rerank(
                    query=query,
                    documents=[c['text'] for c in unique_candidates],
                    top_k=min(top_k * 2, len(unique_candidates))
                )
                
                # Merge reranking scores
                for i, candidate in enumerate(unique_candidates):
                    if i < len(reranked):
                        candidate['rerank_score'] = reranked[i]['score']
                    else:
                        candidate['rerank_score'] = 0.0
                
                # Sort by rerank score
                unique_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
                unique_candidates = unique_candidates[:top_k * 2]
            
            # Step 5: Diversity (MMR - Maximal Marginal Relevance)
            if enable_diversity:
                final_results = self._apply_mmr(
                    candidates=unique_candidates,
                    query_embedding=query_embedding,
                    lambda_param=0.7,  # Balance relevance vs diversity
                    top_k=top_k
                )
            else:
                final_results = unique_candidates[:top_k]
            
            # Step 6: Add metadata
            for result in final_results:
                result['query_type'] = analyzed_query['type']
                result['retrieval_confidence'] = self._calculate_confidence(result)
            
            logger.info(f"Returning {len(final_results)} final results")
            return final_results
        
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            raise
    
    def _apply_mmr(
        self,
        candidates: List[Dict],
        query_embedding: np.ndarray,
        lambda_param: float = 0.7,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance for diversity
        
        MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected_docs))
        """
        if len(candidates) <= top_k:
            return candidates
        
        selected = []
        candidate_embeddings = np.array([c['embedding'] for c in candidates])
        
        while len(selected) < top_k and len(candidates) > 0:
            if not selected:
                # First document: highest relevance
                best_idx = 0
                best_score = candidates[0].get('rerank_score', 0.0)
                for i, c in enumerate(candidates):
                    score = c.get('rerank_score', 0.0)
                    if score > best_score:
                        best_idx = i
                        best_score = score
            else:
                # MMR selection
                best_idx = 0
                best_mmr = -float('inf')
                
                selected_embeddings = np.array([s['embedding'] for s in selected])
                
                for i, candidate in enumerate(candidates):
                    # Relevance score
                    relevance = candidate.get('rerank_score', 0.0)
                    
                    # Diversity score (max similarity to selected)
                    similarities = np.dot(selected_embeddings, candidate_embeddings[i])
                    max_similarity = np.max(similarities)
                    
                    # MMR
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                    
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i
            
            selected.append(candidates.pop(best_idx))
            candidate_embeddings = np.delete(candidate_embeddings, best_idx, axis=0)
        
        return selected
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate retrieval confidence score"""
        vector_score = result.get('similarity_score', 0.0)
        rerank_score = result.get('rerank_score', 0.0)
        
        # Weighted average
        confidence = 0.4 * vector_score + 0.6 * rerank_score
        return min(max(confidence, 0.0), 1.0)

# backend/app/services/retrieval/reranker.py
from typing import List, Dict
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    """Cross-encoder reranker for high precision"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name, max_length=512)
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder
        
        Returns:
            Sorted list of {'text': str, 'score': float, 'index': int}
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Compute scores
        scores = self.model.predict(pairs, convert_to_numpy=True)
        
        # Sort by score
        results = [
            {'text': doc, 'score': float(score), 'index': i}
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]

# backend/app/services/retrieval/query_analyzer.py
from typing import Dict, List
import re
from transformers import pipeline

class QueryAnalyzer:
    """Analyze and expand queries for better retrieval"""
    
    def __init__(self):
        # Lazy load T5 for query expansion
        self.expansion_model = None
    
    async def analyze(self, query: str) -> Dict:
        """
        Analyze query characteristics
        
        Returns:
            {
                'type': 'factoid' | 'analytical' | 'comparative' | 'procedural',
                'complexity': 'simple' | 'medium' | 'complex',
                'entities': List[str],
                'keywords': List[str]
            }
        """
        query_lower = query.lower()
        
        # Detect query type
        if any(word in query_lower for word in ['how to', 'steps', 'guide', 'tutorial']):
            query_type = 'procedural'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            query_type = 'comparative'
        elif any(word in query_lower for word in ['why', 'explain', 'analyze', 'discuss']):
            query_type = 'analytical'
        else:
            query_type = 'factoid'
        
        # Assess complexity
        word_count = len(query.split())
        has_multiple_clauses = ',' in query or ' and ' in query_lower
        
        if word_count > 15 or has_multiple_clauses:
            complexity = 'complex'
        elif word_count > 8:
            complexity = 'medium'
        else:
            complexity = 'simple'
        
        # Extract keywords (simple heuristic)
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an'}
        keywords = [word for word in query_lower.split() if word not in stop_words and len(word) > 3]
        
        return {
            'type': query_type,
            'complexity': complexity,
            'keywords': keywords[:5],
            'entities': []  # TODO: Add NER
        }
    
    async def expand_query(self, query: str) -> List[str]:
        """Generate alternative query formulations"""
        # Simple expansion (in production, use T5 or similar)
        expansions = []
        
        # Synonym replacement (simplified)
        synonyms = {
            'how to': ['steps to', 'guide to', 'way to'],
            'what is': ['define', 'explain', 'description of'],
            'why': ['reason for', 'cause of'],
        }
        
        for key, values in synonyms.items():
            if key in query.lower():
                for value in values:
                    expansions.append(query.lower().replace(key, value))
        
        return expansions[:2]