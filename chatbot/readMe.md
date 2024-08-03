## Sentiment Analyzer with Memory

### Documentation

#### Design Choices:

1. **Memory Size**: 
   - **Basic System**: A fixed memory size (e.g., 5 interactions) was chosen to balance simplicity and effectiveness.
   - **Advanced System**: Uses a vector database to dynamically manage memory, providing more flexibility and robustness.

2. **Device Handling**: 
   - Both systems support GPU acceleration for efficient model inference, ensuring scalability.

3. **Label Mapping**: 
   - A customizable label map allows for easy adaptation to different classification tasks.

#### Implementation Details:

1. **Basic Memory System**:
   - Utilizes a `deque` to manage a fixed number of recent interactions.
   - Concatenates recent interactions for context-aware sentiment prediction.

2. **Advanced Memory System**:
   - Employs `SentenceTransformers` to generate embeddings for interactions.
   - Uses `Faiss` to index and retrieve relevant embeddings based on semantic similarity.
   - Combines retrieved interactions with the current prompt for enhanced context-awareness.

#### Observations:

- **Basic System**: 
  - Simple and effective for short-term interactions.
  - May struggle with long-term dependencies.

- **Advanced System**: 
  - Offers superior performance in coherence, relevance, and context-awareness.
  - Particularly effective for complex or extended conversations.