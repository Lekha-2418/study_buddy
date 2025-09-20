# 📘 Study Buddy

An AI-powered chatbot that lets you **upload a PDF** (e.g., research papers, books, notes, reports) and **ask questions** about its content.  
The system extracts the text from the PDF, processes it, and uses a Large Language Model (LLM) to provide **context-aware answers**.

---

## 🚀 Features

- 📤 **Upload PDF** – Support for multi-page PDFs.
- 🔎 **Text Extraction** – Extracts and processes text for analysis.
- 💬 **Ask Questions** – Query the document in natural language.
- 🤖 **AI-Powered Answers** – Uses a transformer-based LLM for accurate responses.
- ⚡ **Streamlit UI** – Simple and interactive web interface.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – UI for file upload & chat
- **PyPDF2 / pdfplumber** – PDF text extraction
- **Hugging Face Transformers** – Model loading (e.g., Granite / LLaMA / other LLMs)
- **PyTorch** (with CUDA for GPU acceleration)
- **LangChain** (optional, for embeddings & retrieval)

---

## 📂 Project Structure

```
project/
│── app.py              # Main Streamlit app
│── requirements.txt    # Python dependencies
│── README.md           # Documentation
│── models/             # (Optional) Local model weights
│── utils/              # Helper functions (PDF parsing, text processing)
```

---

## ⚡ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pdf-chatbot.git
   cd pdf-chatbot
   ```

2. Create a virtual environment & activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Install PyTorch with GPU (CUDA 12.1 for RTX 40 series):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

## ▶️ Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser at `http://localhost:8501`

3. Upload a **PDF file** and start chatting with your document!

---

## 📌 Example

- Upload: *research_paper.pdf*
- Ask:  
  > "What are the key findings of this paper?"  
  > "Summarize section 3 in 2 lines."  
  > "Who are the authors?"

The chatbot responds with answers extracted and reasoned from the PDF content.

---

## 🤝 Contributing

Pull requests are welcome!  
If you’d like to add new features (like support for multiple PDFs, vector database storage, or advanced summarization), feel free to fork and improve.

---

## 📜 License

This project is licensed under the MIT License – feel free to use and modify.

---

## 🙌 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/)
- [LangChain](https://www.langchain.com/)
