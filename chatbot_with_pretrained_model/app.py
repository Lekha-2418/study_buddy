from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google.colab import files
from PyPDF2 import PdfReader


# Upload the PDF
uploaded = files.upload()

# Get the uploaded file path
pdf_path = list(uploaded.keys())[0]
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    if page.extract_text():
        text += page.extract_text() + " "

# 2. Split into chunks
def chunk_text(text, chunk_size=300):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

chunks = list(chunk_text(text))

# 3. Embed chunks with sentence-transformers
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# 4. Create FAISS index for retrieval
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Function to answer questions
def ask(question, top_k=3):
    q_embedding = model.encode([question])
    distances, indices = index.search(np.array(q_embedding), top_k)
    answers = [chunks[i] for i in indices[0]]
    return "\n---\n".join(answers)

# 5. Try asking a question
while True:
    q = input("Ask a question (or type 'exit'): ")
    if q.lower() == "exit":
        break
    print("\nAnswer:\n", ask(q))
