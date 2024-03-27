### 📚 Chat with PDF using AWS Bedrock

This application leverages AWS Bedrock's powerful language models and FAISS for vector storage to allow users to interactively ask questions and receive answers from a corpus of PDF documents. Utilizing Streamlit for the frontend, it provides an intuitive UI for querying documents processed and ingested into a vector store.

### Features

- **Data Ingestion**: Automatically loads and processes PDF documents from a specified directory.
- **Vector Embedding**: Converts documents into vector embeddings using AWS Bedrock for efficient similarity searches.
- **Interactive Q&A**: Allows users to ask questions and receive answers based on the content of the ingested documents.
- **Two Language Models**: Offers the choice between Mistral and Amazon's Titan embeddings for answering questions.
- **Vector Store Management**: Enables updating the vector store with new or modified documents through the UI.

### Installation

1. **Clone the repository**:

```
git clone <repository-url>
cd <repository-folder>
```

2. **Install dependencies**:

```
pip install -r requirements.txt
```

3. **Run the Streamlit app**:

```
streamlit run app.py
```

### Usage

1. **Start the application** as described in the Installation section.
2. **Navigate** to the Streamlit UI in your web browser.
3. **Upload PDF documents** or use the provided sample documents.
4. **Ask a question** in the text input field to receive an answer based on the documents' content.
5. **Manage the vector store** through the sidebar to update or create new embeddings as needed.

### Requirements

- Python 3.x
- Streamlit
- AWS Account and AWS Bedrock access
- FAISS for vector storage


