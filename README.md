# Arabic RAG System

## Setup

1. Clone the repository
2. All dependencies are included in the `requirements.txt` file, and you can install them using using `pip install -r requirements.txt`
3. Create a `.env` file and add two API keys `OPENAI_API_KEY=XXXXX` and `TOGETHER_API_KEY=XXXXX`.
2. Once installed, you can run a simple Gradio interface using `python src/main.py` to interact with the RAG system.

## Architecture Overview
### Ingestion Pipeline
The ingestion process begins with PDFPlumber, chosen for its ability to read PDF files line by line and parse tables effectively. Key steps include:

- Loading PDFs into pages.
- Flipping each line to accommodate the right-to-left reading order of Arabic while preserving numbers and non-English words.
- Stripping Arabic text of diacritics and other non-essential characters.
- Removing specific punctuation that affects text integrity post-flip.
- Cleaning footer lines and excess newline characters to reduce noise.

### Splitting Mechanism
To preserve document layout and integrity:
A recursive text splitting method is employed, with a chunk size of 1200 characters and a 400-character overlap, ensuring continuity and coherence, particularly within table structures.

### Indexing
The system uses dual indexing for enhanced query matching:

A factual index focuses on supporting queries requiring precise context, using smaller text chunks and pre-formulated questions.
A summary index caters to broader contextual needs using summarized document representations.

### Storing
Data storage is managed using:

- Chroma DB for storing processed text representations.
- A key-value docstore for maintaining original documents.

### Querying
Query responses are tailored based on the nature of the query:

- Fact-based queries retrieve from the factual index using a keyword matching approach.
- Summary-based queries employ a tree summarization method, summarizing responses generated from multiple document chunks for comprehensive answers.

### Models Used
- Embedding Models: Multilingual variations and text-embedding models were evaluated for their efficiency in capturing multilingual contexts.
- Query Answering Models: Llama-3 and Mixtral models were tested to optimize response accuracy and relevance.