This is a comprehensive "Technical Cheat Sheet" for your defense. I’ve organized it into the **Idea $\rightarrow$ Problem $\rightarrow$ Solution $\rightarrow$ Result** framework you requested, with every technical detail from your codebase.

---

### **1. Data Acquisition & Pipeline (The Foundation)**
*   **Source:** Scopus Metadata (2020–2025).
*   **Scope:** 10 Top Indonesian Research Institutions (BRIN, UI, ITB, etc.).
*   **Cleaning Logic:**
    *   **Noise Removal:** Regex-based removal of copyright headers (e.g., `© 2025 Elsevier...`) from abstracts to prevent vector pollution.
    *   **De-duplication:** Authors mapped via **Unique Scopus IDs**, not names (handles name ambiguity).
*   **Implementation:** Python `Pandas` for CSV parsing + `execute_values` for high-performance PostgreSQL bulk inserts.

### **2. Vectorization & Storage (The "AI" Engine)**
*   **Embedding Model:** `nomic-embed-text-v1.5` (Matryoshka-capable).
*   **Vector Engine:** PostgreSQL 17 with `pgvector` extension.
*   **Technical Detail - MRL (Matryoshka Representation Learning):** 
    *   The model is trained to store the most important information in the **first few dimensions**.
    *   **Dimensions Used:** 64, 128, 256, 512, 768.
*   **Optimization (Functional HNSW Indexes):** 
    *   Created **HNSW (Hierarchical Navigable Small World)** indexes on *slices* of the vectors.
    *   **The Pro Move:** You didn't just index the column; you indexed the result of the `sub_vector` function. This allows O(log n) search speed even on truncated 64-dim slices.

### **3. The Search Logic (Expertise Modeling)**
*   **Strategy:** **Two-Stage Retrieval.**
    *   **Stage 1 (Shortlisting):** Fast Approximate Nearest Neighbor (ANN) search using a truncated dimension (e.g., 128) to find the top 100 candidate articles.
    *   **Stage 2 (Reranking):** Precise Cosine Similarity calculation using the **full 768-dim** vector on the 100 candidates to get the final "True Score."
*   **Scoring (Weighted CombSUM):**
    *   Expert Score = $\sum (\text{Similarity Score} \times \text{Author Position Weight})$.
    *   **Author Weights:** 1st Author (1.0), 2nd (0.8), 3rd (0.6), 4th+ (0.4).
    *   **Logic:** Recognizes that authorship order is a proxy for contribution level.

### **4. Evaluation Framework (Blind Test)**
*   **Methodology:** **Blind Pairwise Comparison.**
*   **Process:**
    1.  Pick two random dimension configurations (e.g., 64 vs 768).
    2.  Anonymize results as "Result A" and "Result B."
    3.  User votes (Win/Loss/Draw).
*   **Metric:** **Elo Rating System.** 
    *   Treats each dimension length as a "player" in a tournament.
    *   Matches the standard used by "Chatbot Arena" (LMSYS) for evaluating LLMs.

### **5. Deployment & Infrastructure (The Operations)**
*   **Containerization:** Docker Compose (Microservice architecture).
*   **Internal Services:** 
    *   `web` (Flask + Gunicorn/4 workers).
    *   `db` (Postgres + pgvector).
    *   `tei-embedding` (Rust-based Text-Embeddings-Inference for high-speed local inference).
*   **Networking Isolation:**
    *   `bthesis_gustav`: Private backend network (Hidden DB).
    *   `proxy`: Public edge network (Cloudflared/NPM).
*   **Reverse Tunneling:** 
    *   **Cloudflare Tunnel (`cloudflared`):** Outbound-only tunnel. No port forwarding required on the router.
    *   **Nginx Proxy Manager (NPM):** Handles SSL termination and domain routing to the Flask container.

---

### **Summary Table for "Examiner Questions"**

| Feature | The Problem | Your Solution |
| :--- | :--- | :--- |
| **Search Speed** | Full 768 vectors are computationally expensive. | **Matryoshka Truncation** (Shortlisting on 128-dim). |
| **Accuracy** | Low dimensions lose semantic detail. | **Two-Stage Reranking** (Final check on 768-dim). |
| **Expert Authority** | A researcher with 100 irrelevant papers shouldn't rank high. | **Semantic Similarity x Author Weighting**. |
| **Access** | Local hosting is hard to share for evaluation. | **Cloudflare Tunnel** (Secure public URL). |
| **Database Performance** | Sequential scanning of vectors is slow. | **Functional HNSW Indexes** on sliced sub-vectors. |

---

### **The "10-Minute Pitch" Script:**
1.  **Idea:** "I built an expert search system that finds researchers based on the semantic meaning of their work, not just keywords."
2.  **Problem:** "High-dimensional AI models are slow and expensive to scale in production."
3.  **Solution:** "I implemented **Matryoshka Embeddings**. This allowed me to use a 'Two-Stage' search: a very fast 'shortlist' phase using small vector slices (e.g., 128 dimensions), followed by a high-precision rerank."
4.  **Result:** "I deployed this via a secure Docker-based microservice stack and evaluated the quality using a Blind Pairwise Test, calculating Elo ratings to prove that lower dimensions can achieve near-perfect accuracy with significantly lower latency."

**Print this out or keep it on your phone during the defense—you now have the answer to every 'How' and 'Why' they can throw at you.**