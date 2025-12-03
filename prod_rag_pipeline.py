# backend/app/services/generation/prompt_manager.py
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    system: str
    user: str
    few_shot: List[Dict] = None

class PromptManager:
    """Manage prompts for different query types"""
    
    TEMPLATES = {
        'factoid': PromptTemplate(
            system="""You are a precise AI assistant. Answer questions using ONLY the provided context.
Rules:
- If the answer is in the context, provide it clearly and concisely
- If unsure or information is missing, say "I cannot find this information in the provided documents"
- Cite specific document sources when possible
- Do not make assumptions or use external knowledge""",
            user="""Context:
{context}

Question: {query}

Answer:"""
        ),
        
        'analytical': PromptTemplate(
            system="""You are an analytical AI assistant. Provide well-reasoned answers based on the context.
Rules:
- Synthesize information from multiple sources
- Explain reasoning and connections between concepts
- Structure your answer clearly (use sections if needed)
- Always cite sources: [Source: document_name.pdf]
- Acknowledge limitations in the available information""",
            user="""Context:
{context}

Question: {query}

Provide a detailed analytical answer:"""
        ),
        
        'comparative': PromptTemplate(
            system="""You are a comparison specialist AI. Compare and contrast based on the context.
Rules:
- Structure comparison clearly (similarities/differences)
- Be objective and balanced
- Cite specific sources for each point
- Use tables or lists for clarity when appropriate
- Note if information is incomplete""",
            user="""Context:
{context}

Question: {query}

Provide a structured comparison:"""
        ),
        
        'procedural': PromptTemplate(
            system="""You are a tutorial AI assistant. Provide step-by-step guidance based on the context.
Rules:
- Break down into clear, numbered steps
- Include prerequisites if mentioned
- Add warnings or important notes
- Cite the source document
- Note if steps are incomplete or unclear""",
            user="""Context:
{context}

Question: {query}

Provide step-by-step instructions:"""
        )
    }
    
    @classmethod
    def build_prompt(
        cls,
        query: str,
        context: List[Dict],
        query_type: str = 'factoid',
        max_context_tokens: int = 8000
    ) -> str:
        """Build optimized prompt"""
        template = cls.TEMPLATES.get(query_type, cls.TEMPLATES['factoid'])
        
        # Format context with sources
        context_parts = []
        current_tokens = 0
        
        for doc in context:
            source = doc.get('metadata', {}).get('source', 'unknown')
            text = doc['text']
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            tokens = len(text) // 4
            
            if current_tokens + tokens > max_context_tokens:
                break
            
            context_parts.append(f"[Source: {source}]\n{text}\n")
            current_tokens += tokens
        
        formatted_context = "\n---\n".join(context_parts)
        
        # Build final prompt
        user_prompt = template.user.format(
            context=formatted_context,
            query=query
        )
        
        return f"{template.system}\n\n{user_prompt}"

# backend/app/services/generation/llm.py
from llama_cpp import Llama
from typing import Dict, Optional, Generator
import time
import logging

logger = logging.getLogger(__name__)

class ProductionLLM:
    """Production-ready LLM with error handling and streaming"""
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 32768,
        n_threads: int = 8,
        n_gpu_layers: int = 0  # Set > 0 for GPU
    ):
        logger.info(f"Loading LLM from {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        logger.info("LLM loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict:
        """
        Generate response with performance metrics
        
        Returns:
            {
                'text': str,
                'tokens_generated': int,
                'generation_time_ms': int,
                'tokens_per_second': float
            }
        """
        start_time = time.time()
        
        stop_sequences = stop or ["</s>", "[/INST]", "Question:", "Context:"]
        
        try:
            if stream:
                return self._generate_stream(prompt, max_tokens, temperature, top_p, stop_sequences)
            
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
                echo=False
            )
            
            generation_time = (time.time() - start_time) * 1000  # ms
            
            text = output["choices"][0]["text"].strip()
            tokens_generated = output["usage"]["completion_tokens"]
            tokens_per_second = tokens_generated / (generation_time / 1000) if generation_time > 0 else 0
            
            logger.info(f"Generated {tokens_generated} tokens in {generation_time:.0f}ms ({tokens_per_second:.1f} tok/s)")
            
            return {
                'text': text,
                'tokens_generated': tokens_generated,
                'generation_time_ms': int(generation_time),
                'tokens_per_second': round(tokens_per_second, 2)
            }
        
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: List[str]
    ) -> Generator:
        """Stream tokens for real-time display"""
        for token in self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True
        ):
            yield token["choices"][0]["text"]

# backend/app/services/rag/pipeline.py
from typing import Dict, List, Optional
import time
import logging
from app.services.retrieval.retriever import AdvancedRetriever
from app.services.generation.llm import ProductionLLM
from app.services.generation.prompt_manager import PromptManager
from app.services.evaluation.metrics import RAGMetrics

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Production-ready RAG pipeline with monitoring"""
    
    def __init__(
        self,
        retriever: AdvancedRetriever,
        llm: ProductionLLM,
        metrics: Optional[RAGMetrics] = None
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_manager = PromptManager()
        self.metrics = metrics or RAGMetrics()
    
    async def run(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        top_k: int = 5,
        stream: bool = False,
        enable_reranking: bool = True
    ) -> Dict:
        """
        Execute full RAG pipeline
        
        Returns:
            {
                'answer': str,
                'sources': List[Dict],
                'confidence': float,
                'metrics': Dict,
                'query_id': str
            }
        """
        query_id = f"q_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            # Step 1: Retrieval
            logger.info(f"[{query_id}] Starting retrieval for: {query[:50]}...")
            retrieval_start = time.time()
            
            retrieved_docs = await self.retriever.retrieve(
                query=query,
                tenant_id=tenant_id,
                top_k=top_k,
                enable_reranking=enable_reranking
            )
            
            retrieval_time = (time.time() - retrieval_start) * 1000
            logger.info(f"[{query_id}] Retrieved {len(retrieved_docs)} documents in {retrieval_time:.0f}ms")
            
            if not retrieved_docs:
                return {
                    'answer': "I couldn't find any relevant information in the knowledge base to answer your question.",
                    'sources': [],
                    'confidence': 0.0,
                    'metrics': {
                        'retrieval_time_ms': int(retrieval_time),
                        'generation_time_ms': 0,
                        'total_time_ms': int((time.time() - start_time) * 1000)
                    },
                    'query_id': query_id
                }
            
            # Step 2: Prompt Construction
            query_type = retrieved_docs[0].get('query_type', 'factoid')
            prompt = self.prompt_manager.build_prompt(
                query=query,
                context=retrieved_docs,
                query_type=query_type
            )
            
            # Step 3: Generation
            logger.info(f"[{query_id}] Generating answer...")
            generation_result = self.llm.generate(
                prompt=prompt,
                stream=stream,
                temperature=0.1,
                max_tokens=1024
            )
            
            if stream:
                # Return streaming generator
                return self._stream_response(
                    query_id=query_id,
                    generation_result=generation_result,
                    sources=retrieved_docs,
                    retrieval_time=retrieval_time,
                    start_time=start_time
                )
            
            # Step 4: Calculate confidence
            avg_retrieval_score = sum(doc.get('retrieval_confidence', 0) for doc in retrieved_docs) / len(retrieved_docs)
            confidence = min(avg_retrieval_score * 0.7 + 0.3, 1.0)  # Boost slightly
            
            # Step 5: Format sources
            sources = [
                {
                    'text': doc['text'][:300] + '...' if len(doc['text']) > 300 else doc['text'],
                    'source': doc['metadata'].get('source', 'unknown'),
                    'score': doc.get('retrieval_confidence', 0.0)
                }
                for doc in retrieved_docs
            ]
            
            total_time = (time.time() - start_time) * 1000
            
            # Step 6: Log metrics
            await self.metrics.log_query(
                query_id=query_id,
                query=query,
                num_sources=len(retrieved_docs),
                retrieval_time=retrieval_time,
                generation_time=generation_result['generation_time_ms'],
                total_time=total_time
            )
            
            logger.info(f"[{query_id}] Completed in {total_time:.0f}ms")
            
            return {
                'answer': generation_result['text'],
                'sources': sources,
                'confidence': round(confidence, 2),
                'metrics': {
                    'retrieval_time_ms': int(retrieval_time),
                    'generation_time_ms': generation_result['generation_time_ms'],
                    'total_time_ms': int(total_time),
                    'tokens_generated': generation_result['tokens_generated'],
                    'tokens_per_second': generation_result['tokens_per_second']
                },
                'query_id': query_id
            }
        
        except Exception as e:
            logger.error(f"[{query_id}] Pipeline error: {str(e)}")
            raise
    
    def _stream_response(self, query_id, generation_result, sources, retrieval_time, start_time):
        """Stream response tokens"""
        for token in generation_result:
            yield {
                'token': token,
                'query_id': query_id
            }
        
        # Final message with metadata
        yield {
            'done': True,
            'sources': sources,
            'metrics': {
                'retrieval_time_ms': int(retrieval_time),
                'total_time_ms': int((time.time() - start_time) * 1000)
            }
        }