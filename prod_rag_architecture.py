"""
ARCHITECTURE PRODUCTION-READY - RAG SYSTEM
============================================

Structure des dossiers:
-----------------------
rag-system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app avec middleware
│   │   ├── config.py                  # Configuration centralisée
│   │   ├── dependencies.py            # Injection de dépendances
│   │   │
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── auth.py
│   │   │   │   │   ├── documents.py
│   │   │   │   │   ├── query.py
│   │   │   │   │   ├── admin.py
│   │   │   │   │   └── health.py
│   │   │   │   └── router.py
│   │   │   └── middleware/
│   │   │       ├── auth.py
│   │   │       ├── rate_limit.py
│   │   │       ├── logging.py
│   │   │       └── error_handler.py
│   │   │
│   │   ├── core/
│   │   │   ├── security.py            # JWT, encryption
│   │   │   ├── cache.py               # Redis cache
│   │   │   ├── database.py            # PostgreSQL
│   │   │   └── monitoring.py          # Prometheus metrics
│   │   │
│   │   ├── models/
│   │   │   ├── user.py
│   │   │   ├── document.py
│   │   │   ├── query.py
│   │   │   └── tenant.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── auth.py
│   │   │   ├── document.py
│   │   │   └── query.py
│   │   │
│   │   ├── services/
│   │   │   ├── ingestion/
│   │   │   │   ├── parser_factory.py
│   │   │   │   ├── parsers/
│   │   │   │   │   ├── pdf.py
│   │   │   │   │   ├── docx.py
│   │   │   │   │   ├── html.py
│   │   │   │   │   └── ocr.py
│   │   │   │   ├── chunker.py
│   │   │   │   └── preprocessor.py
│   │   │   │
│   │   │   ├── embeddings/
│   │   │   │   ├── embedder.py
│   │   │   │   └── batch_processor.py
│   │   │   │
│   │   │   ├── vectorstore/
│   │   │   │   ├── faiss_manager.py
│   │   │   │   ├── index_manager.py
│   │   │   │   └── backup_manager.py
│   │   │   │
│   │   │   ├── retrieval/
│   │   │   │   ├── retriever.py
│   │   │   │   ├── reranker.py
│   │   │   │   └── query_analyzer.py
│   │   │   │
│   │   │   ├── generation/
│   │   │   │   ├── llm.py
│   │   │   │   ├── prompt_manager.py
│   │   │   │   └── output_parser.py
│   │   │   │
│   │   │   ├── evaluation/
│   │   │   │   ├── metrics.py
│   │   │   │   └── ab_testing.py
│   │   │   │
│   │   │   └── finetune/
│   │   │       ├── trainer.py
│   │   │       ├── dataset_builder.py
│   │   │       └── evaluator.py
│   │   │
│   │   ├── tasks/
│   │   │   ├── celery_app.py
│   │   │   ├── ingestion_tasks.py
│   │   │   └── training_tasks.py
│   │   │
│   │   └── utils/
│   │       ├── logger.py
│   │       ├── validators.py
│   │       └── helpers.py
│   │
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── e2e/
│   │
│   ├── alembic/                       # Migrations DB
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   │
│   ├── requirements/
│   │   ├── base.txt
│   │   ├── dev.txt
│   │   └── prod.txt
│   │
│   ├── pyproject.toml
│   └── pytest.ini
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/
│   ├── package.json
│   └── vite.config.ts
│
├── infrastructure/
│   ├── kubernetes/
│   │   ├── deployments/
│   │   ├── services/
│   │   └── ingress/
│   ├── terraform/
│   └── helm/
│
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   └── alerts/
│
├── scripts/
│   ├── deploy.sh
│   ├── backup.sh
│   └── migrate.sh
│
├── docs/
│   ├── api/
│   ├── architecture/
│   └── deployment/
│
├── .env.example
├── .gitignore
├── docker-compose.prod.yml
├── Makefile
└── README.md

TECHNOLOGIES STACK:
-------------------
Backend:
  - FastAPI + Uvicorn (async)
  - PostgreSQL (metadata + users)
  - Redis (cache + rate limiting)
  - Celery + RabbitMQ (async tasks)
  - Prometheus + Grafana (monitoring)

ML/AI:
  - Mistral-7B-Instruct (GGUF 4-bit)
  - BGE-M3 embeddings (multilingual)
  - Faiss (vectorstore)
  - PEFT + LoRA (fine-tuning)
  - Sentence-Transformers (reranking)

Sécurité:
  - JWT + OAuth2
  - AES-256 encryption
  - Rate limiting (Redis)
  - Input validation (Pydantic)
  - RBAC (Role-Based Access Control)

Infra:
  - Docker + Kubernetes
  - Nginx (reverse proxy)
  - Let's Encrypt (SSL)
  - AWS S3 (backup)
  - CloudWatch (logs)
"""