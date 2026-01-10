#### Materials
Embeddings are numerical representations of text (or other data) in a high-dimensional vector space where semantic similarities are preserved as geometric relationships.
In simpler terms: embeddings convert text into lists of numbers that capture meaning. Words or phrases with similar meanings will have similar number patterns.
Example: [0.243, 0.895803, -0.328932, 0.533, 0.1, ..., 0.7748] Dimensions is the number of numerical values in each embedding vector.
Vector: [0.243, 0.895803, -0.328932, 0.533, 0.1, ..., 0.7748] 
Value at the 1st dimension: 0.243
Value at the 2nd dimension:  0.895803
Value at the 3rd dimension: -0.328932**How Embeddings are created within the Embedding Model:****Note: **This information provides a basic understanding of how embedding models work internally during vector generation. This foundational knowledge will help you grasp key concepts covered in the Additional task and avoid common misunderstandings when working with Embedding models.&#xFEFF;
Input text -&gt;
Tokenizer -&gt;
Embedding lookup -&gt;
Encoder (Transformer) -&gt;
Pooling -&gt;
Normalization -&gt;
Use
We start generating embeddings with some text:
"*I love machine learning.*"
Converts text to subword/byte tokens, often something like:
Input: "**I**** ****love**** ****machine**** ****learning****.**"
-&gt;
Tokens: ["**I**", "**love**", "**machine**", "**learning**", "**.**"]
-&gt;
Token IDs: [40, **4682**, **903**, **723**, **11**].[&#xFEFF;](https://rahullokurte.com/understanding-token-and-positional-embeddings-in-transformers)&#xFEFF;
Each token ID maps to a learnable vector from the model's embedding matrix E of size [V×d]
"I love machine learning."
**After Transformer:**
Token 40 ("I") → [0.23, -0.45, 0.67, ..., 0.12] (384 dimensions)
Token 4682 ("love") → [-0.12, 0.78, -0.34, ..., 0.89] (384 dimensions) 
Token 903 ("machine") → [0.45, -0.23, 0.91, ..., -0.56] (384 dimensions)
Token 723 ("learning") → [0.78, 0.34, -0.67, ..., 0.23] (384 dimensions)
Token 11 (".") → [-0.09, 0.12, 0.45, ..., -0.78] (384 dimensions)
Read more:
[Article 1](https://transformers-goto-guide.hashnode.dev/how-transformers-work-tokenization-embeddings-and-positional-encoding-explained-part-1)
[Article 2](https://rahullokurte.com/understanding-token-and-positional-embeddings-in-transformers)&#xFEFF;
The sequence of token vectors passes through multiple self-attention layers (typically 12 layers in BERT-base) to produce contextualized representations.
**What happens:**
- **Initial state:** Each token has its basic embedding
- **After Layer 1:** "love" embedding is influenced by "I", "machine", "learning" context
- **After Layer 6:** "machine" now strongly associates with "learning"
- **After Layer 12:** Final contextualized embeddings where: "machine" embedding is now specialized for ML context (not physical machines)
- "learning" embedding is contextualized for educational/AI meaning
- Each token's final embedding captures its meaning within this specific sentence
**Result:** 5 contextualized token embeddings, still 768-dim each, but now context-aware
After the transformer layers, you have multiple token embeddings, one for each token in your sentence:
After Transformer:
Token "I" → [0.23, -0.45, 0.67, ..., 0.12] 
Token "love" → [-0.12, 0.78, -0.34, ..., 0.89] 
Token "machine" → [0.45, -0.23, 0.91, ..., -0.56] 
Token "learning" → [0.78, 0.34, -0.67, ..., 0.23]
Token "." → [-0.09, 0.12, 0.45, ..., -0.78]
**Pooling combines multiple token vectors into one sentence vector.**
**Why do you need one vector?**
Vector databases and similarity search systems expect:
- Each document/sentence = exactly 1 vector
- All vectors have the same dimensions
- You can compare "sentence A" vs "sentence B" with a single similarity calculation
Without pooling, you'd have to answer: "How do I compare sentence A (5 vectors) with sentence B (7 vectors)?" It's not clear how to do this efficiently.
Normalization ensures that similarity scores aren't skewed by vector magnitude, it focuses on the angle between vectors (semantic similarity) rather than their length.
If you want to know more:
[Article 1](https://medium.com/@tellmetiger/embedding-normalization-dummy-learn-2ac8d816e776)
[Article 2](https://milvus.io/ai-quick-reference/how-do-i-know-if-i-need-to-normalize-the-sentence-embeddings-for-example-applying-l2-normalization-and-what-happens-if-i-dont-do-it-when-computing-similarities)&#xFEFF;
In this step, we get the embeddings for "*I love machine learning.*", and can save it to the Vector DB or make a search in Vector DB (we are ready to use it).
**Embedding models:**
- **OpenAI Models**: `text-embedding-3-small` (up to 1536 dimensions) - cost-effective, good performance
- `text-embedding-3-large` (up to 3072 dimensions, can be reduced to lower dimensions) - highest accuracy
**Open Source Models**: 
- `sentence-transformers/all-MiniLM-L6-v2 ` (up to 384 dimensions)
- Various Hugging Face transformer models
&#xFEFF;
## Vector Databases and Storage
A vector database is a specialized storage system optimized for:
 - Storing high-dimensional vectors (embeddings)
 - Performing fast similarity searches
 - Scaling to millions or billions of vectors
Popular Vector Databases: FAISS, ChromaDB, Milvus, PGVector (PostgreSQL extension)&#xFEFF;
## Data Transformation Pipeline
Raw Data (source) → Load → Transform → Embed → Store → Retrieve&#xFEFF;
1. Source: Documents (.PDF, .TXT, .HTML), APIs, databases
1. Load: Extract and clean text content from the Source
1. Transform: Split into chunks, preprocess
1. Embed: Generate vector representations
1. Store: Save in vector database with metadata
1. Retrieve: Perform similarity searches
### Document Chunking Strategy
**Documents are typically split into smaller chunks because:**
- LLMs have context length limits
- Smaller chunks provide more precise retrieval
- Better performance for similarity matching
**Common chunking approaches:**
- Fixed size (100-500 words)
- Sentence-based boundaries
- Semantic chunking (maintaining topic coherence)
### Data Sample from Vector DB:
&#xFEFF;
In the sample above, we can see the sample with text chunks and their embeddings in Vector DB:
document_name
text
embedding
In this column is stored  metadata (and usually this column has the name 'metadata')
In this column are stored   the text chunks
In this column we can find all the embeddings (vectors) of the text chunks
&#xFEFF;
## RAG Pipeline
**User Input → Retrieval [Embed → Vector Search] → Augmentation → Generation → Response**&#xFEFF;
User Input -&gt; 
**Embed →**
**Vector Search →**
**Augmentation →**
**Generation →**
**Response**
Original user input
Convert User Input into embeddings
Search the *top-k* most relevant text chunks by comparing the embedding from the User Input with the embeddings that represent chunks. 
Algorithms for Vector search will be provided below
Concat Retrieved context after the Retrieval step with the original User Input
Provide LLM with the Augmented prompt for response generation
Response from LLM
&#xFEFF;
## Search Algorithms in RAG
**1. Euclidean Distance (L2 Distance):**&#xFEFF;**2. Cosine Similarity**&#xFEFF;**Note:** **You rarely need to understand all these algorithms in detail. The two mentioned above are included because you will use them in the practice tasks.** 
Other popular algorithms include Manhattan Distance (L1), Minkowski Distance, Dot Product, and Jaccard Similarity, among others.&#xFEFF;
## Tuning RAG Performance
Many people see vector search in Vector DBs as mysterious, but let's examine how PgVector (a PostgreSQL extension) actually works. In reality, there's no magic involved. It's simply a SELECT query that compares vectors using the algorithms described above.`microwave_data` table in SQL DB (PgVector)&#xFEFF;
**Query sample:**
SELECT text, embedding &lt;-&gt;  '[0.23, -0.45, 0.67, ..., 0.12]'::vector AS distance
FROM microwave_data
WHERE embedding &lt;-&gt;  '[0.23, -0.45, 0.67, ..., 0.12]'::vector &lt;= {score}
ORDER BY distance
LIMIT {top_k};
**Key Parameters:**
`top_k`: Number of Retrieved Documents
- **Low values (1-3)**: Focused, specific context
- **High values (5-10)**: Broader context, higher computational cost
- **Optimal range**: Usually 3-5 for most applications
`score`: Quality Filter (for this particular SQL query)
- **0.8-0.9**: Include more documents, some may be less relevant
- **0.4-0.7**: Balanced relevance
- **0.1-0.3**: Only highly relevant documents
`score_threshold`: Quality filter for search results (commonly used in code and standard across many libs)
- **Low threshold (0.1-0.3)**: Include more documents, some may be less relevant
- **Medium threshold (0.4-0.7)**: Balanced relevance
- **High threshold (0.8-0.9)**: Only highly relevant documents
**Note:** While you'll likely never write vector search queries manually, we're demonstrating this to show there's no magic, it's just a simple SQL query.&#xFEFF;
### Useful links
- Watch the video: [Transformers with 3Blue1Brown](https://www.youtube.com/watch?v=wjZofJX0v4M) and [What are Word Embeddings?](https://www.youtube.com/watch?v=wgfSDrqYMJ4)&#xFEFF;
- &#xFEFF;[Embeddings, explanation from Google](https://developers.google.com/machine-learning/crash-course/embeddings/embedding-space)&#xFEFF;
- &#xFEFF;[Dimensionality of Word Embeddings](https://www.baeldung.com/cs/dimensionality-word-embeddings)&#xFEFF;
- &#xFEFF;[Transformers with interactions](https://poloclub.github.io/transformer-explainer/)&#xFEFF;
- Watch the video: [Deep Dive into LLMs with Andrej Karpathy](https://www.youtube.com/watch?v=7xTGNNLPyMI)&#xFEFF;
&#xFEFF;
### Key Takeaways
- **RAG is Not Magic:** It's standard technology: SQL queries with similarity operators, REST APIs, and proven information retrieval techniques combined with LLMs.
- **Solves Knowledge Gaps:** Enables LLMs to access current, domain-specific information beyond training data instead of hallucinating outdated responses.
- ** Simple Implementation:** Standard ETL pipeline: chunk text, generate embeddings via APIs, store in vector databases, retrieve using similarity search.
- ** Basic Parameter Tuning**: Performance optimization through adjusting `top_k`, similarity thresholds, and chunking strategies - no complex algorithms required.