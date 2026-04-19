**AUTHOR: FRANCISCO BROISSIN**

**⚓ MED RAG Agent: Regulatory Expert System**

This repository hosts a Retrieval-Augmented Generation (RAG) application designed to navigate complex European Union Marine Equipment Directive EU/2014/90  (MED). It allows users to query technical documentation using natural language, backed by the power of Google Gemini and ChromaDB.
🚀 Development Journey & Architecture

The project followed a rigorous 4-stage development lifecycle to ensure scalability and cloud compatibility:

**1. Proof of Concept (Google Colab)**

    Objective: Validate the core RAG logic.

    Process: Initial testing of PDF parsing and vectorization using LangChain.

    Outcome: Successfully demonstrated that a Large Language Model (LLM) could accurately retrieve context from specific regulatory PDFs.

**2. Local Environment & Virtualization (Linux/Ubuntu)**

    Objective: Transition from notebook to a structured application.

    Environment: Developed in a Python Virtual Environment within a Linux (Ubuntu) terminal.

    Milestone: Implemented a persistent ChromaDB vector store and integrated the Streamlit UI. Testing was performed locally to ensure robust document processing.

**3. Containerization Logic & Cloud Readiness**

    Objective: Prepare the application for universal deployment.

    Logic: The app is designed to run within a Docker-style container environment on the cloud. This ensures that the specific versions of libraries (like pysqlite3 for database compatibility) are consistent across all environments.

    Infrastructure: Refined the code to handle cloud-specific constraints, such as port binding and environment variable management for API keys.

**4. Continuous Deployment (GitHub & Render)**

    Objective: Live operation and public access.

    Workflow: Automated deployment via GitHub. Any changes pushed to the main branch are automatically built and deployed to Render.

    Live App: https://med-rag-web.onrender.com

**🛠️ Tech Stack**

    LLM: Google Gemini 1.5 Flash (via API)

    Orchestration: LangChain

    Vector Store: ChromaDB

    Interface: Streamlit

    Environment: Linux / Python 3.10+

    Cloud Hosting: Render

**📋 How to Use**

    Query: Ask a question regarding EU Marine Equipment Directive.

    Retrieve: The system searches the vectorized database for relevant excerpts.

    Generate: Gemini synthesizes a precise answer based only on the provided documentation.
