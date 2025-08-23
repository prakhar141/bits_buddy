# Bits Buddy - Deployment Kit

This folder contains your original `cleaned_buddy.py` and deployment support files for:
- Docker
- Kubernetes (manifests)
- Optional Redis and Pinecone service drop-ins

**Important:** Your LLM prompts, multi-step RAG pipeline, and model calls are unchanged. The optional services are integrated via environment flags:
- USE_REDIS=true
- USE_PINECONE=true

Follow the roadmap in the project root or run the steps from the Streamlit app folder.

