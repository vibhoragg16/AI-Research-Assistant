# AI Research Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, multi-agent AI research assistant that automates the entire literature review process. This tool uses a sophisticated graph-based workflow to intelligently find, process, analyze, and synthesize information from academic papers, delivering comprehensive research summaries on demand.

---

## 🚀 Features

-   **Automated Literature Review:** Go from a simple query to a full literature review in minutes.
-   **Multi-Agent Workflow:** A team of specialized AI agents (Supervisor, Researcher, Editor, Writer) collaborates to produce high-quality, reliable results.
-   **Dynamic & Intelligent:** Uses LangGraph to create a flexible, stateful workflow that can make decisions and route tasks based on intermediate results.
-   **Powerful Search:** Leverages both the Semantic Scholar API for academic papers and the Tavily AI search engine for broader context.
-   **High-Speed LLM Inference:** Powered by Groq for near-instantaneous language model processing.
-   **Interactive UI:** A user-friendly Streamlit interface allows for easy query submission and progress monitoring.

---

## 🛠️ How It Works

The assistant operates like a highly efficient research team, managed by an AI Supervisor.

1.  **Query Submission:** You provide a research topic (e.g., "The role of AI in climate change mitigation").
2.  **Supervisor Takes Charge:** The Supervisor agent receives the query and initiates the workflow, starting with the Researcher agent.
3.  **Information Gathering:** The **Researcher Agent** uses its tools (`semantic_scholar_search`, `tavily_search`) to find a list of potentially relevant academic papers.
4.  **Critical Filtering:** The Supervisor passes the list to the **Editor Agent**, which carefully reviews each paper's title and abstract to select only the most relevant documents.
5.  **Content Processing:** The system downloads and processes the content of the filtered papers.
6.  **Synthesis & Summary:** The **Writer Agent** receives the processed content and synthesizes it into a single, coherent, and well-structured research summary, identifying key themes, findings, and future directions.
7.  **Results Display:** The final summary and a list of the source papers are displayed in the Streamlit UI.

---

## 🧠 Core Technology: The LangGraph Engine

This project's intelligence is powered by **LangGraph**, a library for building stateful, multi-agent applications. Instead of a simple, linear chain of commands, LangGraph allows us to define the workflow as a **graph**, enabling complex, human-like reasoning.

-   **Stateful Execution:** The graph maintains a central `ResearchState` object that is passed between steps. This allows agents to have a shared understanding of the progress and access all previously gathered information.
-   **Nodes as Steps:** Each function in the research process (searching, filtering, writing) is a "node" in the graph.
-   **Conditional Edges:** The true power comes from "conditional edges." After a node completes, the graph can make a decision. For example, the `router` function checks if the initial paper search was successful. If not, it can end the process early. If it was, it routes the state to the Editor agent. This allows for dynamic, intelligent loops and branches in the workflow.

This graph-based architecture is what allows the Supervisor to effectively manage the team and produce a result that is more than just the sum of its parts.

---

## 🧩 Project Structure

The project is organized for clarity and extensibility. The core logic resides in the `research_assistant` directory, where the agentic graph workflow is defined.

```
research_assistant/
  ├── graph/
  │   ├── workflow.py         # Defines the main LangGraph workflow and state
  │   ├── nodes.py            # Functions for each step (search, filter, process, summarize)
  │   └── router.py           # Logic for conditional routing within the graph
  ├── tools/
  │   └── agent_tools.py      # Creates the Supervisor and Worker Agents (Researcher, Editor, Writer)
  ├── models/
  │   └── ...                 # Pydantic data models for state management
  └── processors/
      └── document_processor.py # Handles PDF downloading and text extraction
streamlit_app.py              # The Streamlit User Interface
requirements.txt              # Python dependencies
.env                          # For API keys (GROQ_API_KEY, TAVILY_API_KEY)
```

---

## 🤖 Meet the Agent Team

The research process is managed by a **Supervisor** and carried out by a team of specialized **Worker Agents**. This architecture allows for a robust and intelligent workflow.

### 1. The Supervisor

The Supervisor is the brain of the operation. It doesn't perform research tasks itself; instead, it manages the team.

-   **Role:** To direct the workflow by choosing the next worker agent based on the user's request and the current progress.
-   **Process:** It examines the state of the research after each worker completes a task and routes the process to the next appropriate worker. When the goal is finally achieved, the Supervisor concludes the workflow.

### 2. The Worker Agents

These are the specialists who get the work done. Each is an LLM-powered agent with a specific role and a set of tools.

#### The Researcher Agent

This agent is the information gatherer.

-   **Prompt:** "You are a research assistant responsible for finding relevant academic papers."
-   **Tools:** `semantic_scholar_search`, `tavily_search`.

#### The Editor Agent

This agent acts as a critical reviewer.

-   **Prompt:** "You are an editor responsible for filtering and selecting the most relevant papers."
-   **Task:** Carefully examines each paper to determine its true relevance, ensuring the final summary is high-quality and on-topic.

#### The Writer Agent

The final specialist in the team.

-   **Prompt:** "You are a research writer responsible for generating a comprehensive summary."
-   **Task:** Synthesizes all information into a single, coherent research summary.

---

## 🖥️ Getting Started

### Prerequisites

You will need Python 3.8+ and API keys from:

-   **Groq:** For the LLM.
-   **Tavily AI:** For the search tool.

### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vibhoragg16/AI-Research-Assistant.git](https://github.com/vibhoragg16/AI-Research-Assistant.git)
    cd AI-Research-Assistant
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    Create a file named `.env` in the root directory and add your API keys:
    ```env
    GROQ_API_KEY="your_groq_api_key"
    TAVILY_API_KEY="your_tavily_api_key"
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

5.  **Open your browser** to the provided URL (usually `http://localhost:8501`) and start your research!

---

## 📝 Example Queries

-   “Generate a literature review on self-attention mechanisms in natural language processing.”
-   “What are the latest advancements in battery technology for electric vehicles?”
-   “Summarize the key findings on the impact of remote work on employee productivity.”
-   "Compare the methodologies used in recent studies on reinforcement learning for robotics."

---

## 🤝 Contributing

Pull requests and suggestions are welcome! Feel free to open issues for bugs, feature requests, or questions.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more information.
