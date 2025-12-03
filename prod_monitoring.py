# backend/app/services/evaluation/metrics.py
from typing import Dict, List, Optional
from prometheus_client import Counter, Histogram, Gauge
import asyncio
from datetime import datetime, timedelta
import numpy as np

class RAGMetrics:
    """Prometheus metrics for RAG system"""
    
    # Counters
    queries_total = Counter('rag_queries_total', 'Total queries processed', ['status'])
    documents_ingested = Counter('rag_documents_ingested', 'Total documents ingested', ['status'])
    
    # Histograms
    retrieval_latency = Histogram('rag_retrieval_latency_seconds', 'Retrieval latency', buckets=[0.1, 0.5, 1.0, 2.0, 5.0])
    generation_latency = Histogram('rag_generation_latency_seconds', 'Generation latency', buckets=[1, 5, 10, 20, 30])
    total_latency = Histogram('rag_total_latency_seconds', 'Total query latency', buckets=[1, 5, 10, 20, 30, 60])
    
    # Gauges
    active_queries = Gauge('rag_active_queries', 'Currently processing queries')
    index_size = Gauge('rag_index_size', 'Total vectors in index')
    cache_hit_rate = Gauge('rag_cache_hit_rate', 'Cache hit rate')
    
    def __init__(self):
        self.query_log = []
    
    async def log_query(
        self,
        query_id: str,
        query: str,
        num_sources: int,
        retrieval_time: float,
        generation_time: float,
        total_time: float,
        status: str = 'success'
    ):
        """Log query metrics"""
        self.queries_total.labels(status=status).inc()
        
        self.retrieval_latency.observe(retrieval_time / 1000)
        self.generation_latency.observe(generation_time / 1000)
        self.total_latency.observe(total_time / 1000)
        
        self.query_log.append({
            'query_id': query_id,
            'query': query,
            'num_sources': num_sources,
            'retrieval_time_ms': retrieval_time,
            'generation_time_ms': generation_time,
            'total_time_ms': total_time,
            'timestamp': datetime.utcnow(),
            'status': status
        })
    
    def get_recent_stats(self, hours: int = 24) -> Dict:
        """Get statistics for recent queries"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [q for q in self.query_log if q['timestamp'] > cutoff]
        
        if not recent:
            return {}
        
        retrieval_times = [q['retrieval_time_ms'] for q in recent]
        generation_times = [q['generation_time_ms'] for q in recent]
        total_times = [q['total_time_ms'] for q in recent]
        
        return {
            'total_queries': len(recent),
            'avg_retrieval_time_ms': np.mean(retrieval_times),
            'p95_retrieval_time_ms': np.percentile(retrieval_times, 95),
            'avg_generation_time_ms': np.mean(generation_times),
            'p95_generation_time_ms': np.percentile(generation_times, 95),
            'avg_total_time_ms': np.mean(total_times),
            'p95_total_time_ms': np.percentile(total_times, 95)
        }

# backend/app/services/evaluation/evaluator.py
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGEvaluator:
    """Evaluate RAG quality"""
    
    @staticmethod
    def evaluate_retrieval(
        query: str,
        retrieved_docs: List[Dict],
        ground_truth_docs: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate retrieval quality
        
        Metrics:
        - Recall@K: % of relevant docs retrieved
        - Precision@K: % of retrieved docs that are relevant
        - MRR: Mean Reciprocal Rank
        - NDCG: Normalized Discounted Cumulative Gain
        """
        if not ground_truth_docs:
            return {'warning': 'No ground truth provided'}
        
        retrieved_ids = [doc.get('id') for doc in retrieved_docs]
        
        # Recall@K
        relevant_retrieved = len(set(retrieved_ids) & set(ground_truth_docs))
        recall = relevant_retrieved / len(ground_truth_docs) if ground_truth_docs else 0
        
        # Precision@K
        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
        
        # MRR
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_docs:
                mrr = 1.0 / (i + 1)
                break
        
        return {
            'recall_at_k': round(recall, 3),
            'precision_at_k': round(precision, 3),
            'f1_score': round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0, 3),
            'mrr': round(mrr, 3)
        }
    
    @staticmethod
    def evaluate_generation(
        generated_answer: str,
        reference_answer: Optional[str] = None,
        retrieved_docs: List[Dict] = None
    ) -> Dict:
        """
        Evaluate generation quality
        
        Metrics:
        - Faithfulness: Answer supported by retrieved docs
        - Answer Relevancy: Answer addresses the query
        - Context Precision: Retrieved docs are relevant
        """
        metrics = {}
        
        # Simple faithfulness check (overlap with sources)
        if retrieved_docs:
            source_texts = " ".join([doc['text'] for doc in retrieved_docs])
            answer_words = set(generated_answer.lower().split())
            source_words = set(source_texts.lower().split())
            
            overlap = len(answer_words & source_words)
            faithfulness = overlap / len(answer_words) if answer_words else 0
            metrics['faithfulness_score'] = round(faithfulness, 3)
        
        # Length checks
        metrics['answer_length'] = len(generated_answer.split())
        metrics['is_too_short'] = len(generated_answer.split()) < 10
        metrics['is_too_long'] = len(generated_answer.split()) > 500
        
        # Hallucination indicators
        hallucination_phrases = [
            "i don't know",
            "cannot find",
            "not mentioned",
            "no information"
        ]
        metrics['contains_uncertainty'] = any(phrase in generated_answer.lower() for phrase in hallucination_phrases)
        
        return metrics

# backend/app/services/evaluation/ab_testing.py
import random
from typing import Dict, List
from datetime import datetime

class ABTestingManager:
    """A/B testing for RAG configurations"""
    
    def __init__(self):
        self.experiments = {}
        self.results = []
    
    def create_experiment(
        self,
        name: str,
        variant_a: Dict,
        variant_b: Dict,
        traffic_split: float = 0.5
    ):
        """
        Create A/B test
        
        Args:
            name: Experiment name
            variant_a: Config for variant A (e.g., {'reranking': True})
            variant_b: Config for variant B
            traffic_split: % traffic to variant B
        """
        self.experiments[name] = {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'traffic_split': traffic_split,
            'created_at': datetime.utcnow()
        }
    
    def get_variant(self, experiment_name: str, user_id: str) -> Dict:
        """Get variant for user"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        experiment = self.experiments[experiment_name]
        
        # Consistent assignment based on user_id
        random.seed(hash(user_id))
        use_variant_b = random.random() < experiment['traffic_split']
        
        variant = experiment['variant_b'] if use_variant_b else experiment['variant_a']
        variant_name = 'B' if use_variant_b else 'A'
        
        return {
            'variant_name': variant_name,
            'config': variant
        }
    
    def log_result(
        self,
        experiment_name: str,
        variant_name: str,
        user_id: str,
        metrics: Dict,
        user_rating: Optional[int] = None
    ):
        """Log experiment result"""
        self.results.append({
            'experiment': experiment_name,
            'variant': variant_name,
            'user_id': user_id,
            'metrics': metrics,
            'user_rating': user_rating,
            'timestamp': datetime.utcnow()
        })
    
    def get_experiment_results(self, experiment_name: str) -> Dict:
        """Analyze experiment results"""
        results_a = [r for r in self.results if r['experiment'] == experiment_name and r['variant'] == 'A']
        results_b = [r for r in self.results if r['experiment'] == experiment_name and r['variant'] == 'B']
        
        if not results_a or not results_b:
            return {'error': 'Insufficient data'}
        
        def avg_metric(results, metric_key):
            values = [r['metrics'].get(metric_key, 0) for r in results]
            return np.mean(values) if values else 0
        
        return {
            'variant_a': {
                'sample_size': len(results_a),
                'avg_latency_ms': avg_metric(results_a, 'total_time_ms'),
                'avg_rating': np.mean([r['user_rating'] for r in results_a if r['user_rating']])
            },
            'variant_b': {
                'sample_size': len(results_b),
                'avg_latency_ms': avg_metric(results_b, 'total_time_ms'),
                'avg_rating': np.mean([r['user_rating'] for r in results_b if r['user_rating']])
            }
        }