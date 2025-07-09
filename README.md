# Research Assistant

A powerful, modular, and extensible AI research assistant for processing, summarizing, and analyzing academic papers (PDFs or arXiv IDs) using state-of-the-art language models.

---

## 🚀 Features

- **PDF & arXiv Support:** Upload your own PDFs or fetch papers by arXiv ID.
- **Automated Document Processing:** Extracts, chunks, and embeds document content for efficient retrieval.
- **Flexible Query Types:**  
  - Summarize papers (general, methods, results, background)
  - Extract key claims, methodologies, or citations
  - Answer specific questions about the document(s)
  - Generate literature reviews across multiple papers
  - Compare methodologies and results between papers
- **Interactive Streamlit UI:**  
  - Upload, process, and query documents with ease
  - View loaded documents and debug information
- **Modular Workflow:**  
  - Each step (processing, retrieval, summarization, etc.) is a separate function for easy customization and extension.
- **LLM-Driven:** Uses large language models (LLMs) for summarization, extraction, and synthesis.

---

## 🛠️ How It Works

1. **Document Input:**  
   - Upload a PDF or enter an arXiv ID.
   - The assistant processes the document, splits it into chunks, and creates embeddings for retrieval.

2. **Query Submission:**  
   - Choose a query type (summarize, question, extract claims, etc.) and enter your question or request.
   - The assistant retrieves relevant chunks from the document(s) and runs the appropriate workflow steps.

3. **Information Extraction:**  
   - Each workflow step (e.g., `generate_summary`, `extract_claims`, `generate_literature_review`) adds its results to a central `extracted_info` dictionary in the session state.

4. **Final Answer Synthesis:**  
   - The assistant combines all extracted information and generates a comprehensive, well-structured answer using the LLM.

5. **Results Display:**  
   - The answer, along with debug information and document metadata, is displayed in the UI.

---

## 🧩 Project Structure

```
research_assistant/
  ├── app.py                # Main workflow logic (direct function calls, no workflow engine)
  ├── models/
  │   └── agent.py          # AgentState and related data models
  ├── processors/
  │   └── document_processor.py
  ├── tools/
  │   └── agent_tools.py    # LLM and document tools
  ├── graph/
  │   └── nodes.py          # All workflow step functions (process, retrieve, summarize, etc.)
  └── ...
streamlit_app.py            # Streamlit UI
requirements.txt            # Python dependencies
```

---

## 🖥️ Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** to the provided URL (usually http://localhost:8501).

4. **Upload a PDF or enter an arXiv ID, process it, and ask your research question!**

---

## ⚙️ Customization

- **Add new workflow steps:**  
  Implement a new function in `nodes.py` and call it in the workflow in `app.py`.
- **Change LLM or embedding model:**  
  Edit `agent_tools.py` to use your preferred model or API.
- **Extend the UI:**  
  Modify `streamlit_app.py` to add new controls, display more info, or improve the user experience.

---

## 📝 Example Queries

- “Summarize the main contributions of this paper.”
- “What methodology was used in this research?”
- “Extract the key claims and supporting evidence.”
- “Compare the results of these two papers.”
- “Generate a literature review on self-attention mechanisms.”

---

## 🧠 How State and Information Extraction Works

- The assistant maintains a central `extracted_info` dictionary in the session state.
- Each workflow step (summarization, claim extraction, etc.) adds its results to this dictionary.
- The final answer is synthesized from all available extracted information, ensuring a comprehensive and document-grounded response.

---

## 🤝 Contributing

Pull requests and suggestions are welcome!  
Feel free to open issues for bugs, feature requests, or questions.

---

## 📄 License

This project is licensed under the MIT License.

---

**Enjoy your AI-powered research assistant!**  
If you have questions or want to extend the project, just ask.