# PathFinder Visualizer: Bellman-Ford vs A*

An interactive web application built with **Streamlit** to visualize and compare the **Bellman-Ford** and **A*** shortest path algorithms side-by-side.

---

## Live Demo
[Click here to view the live application!](https://pathfinder-visualiser.streamlit.app/)



---

## Key Features
- **Interactive Graph Creation:** Manually add nodes and weighted edges, or instantly generate complex random graphs.  
- **Dual Algorithm Visualization:** Watch Bellman-Ford and A* run simultaneously on the same graph in a side-by-side view.  
- **Step-by-Step Animation:** Use advanced controls (play, pause, step-through, speed control) to analyze the entire execution of each algorithm.  
- **Detailed Analytics:** Compare final path costs, execution times, and the full paths found by each algorithm in a clear summary table.  
- **JSON Export:** Download the complete graph structure and run results for further analysis or record-keeping.  

---

## Tech Stack
| Component | Technology |
|------------|-------------|
| Framework | Streamlit |
| Visualization | Plotly |
| Graph Engine | NetworkX |
| Language | Python |

---

## Setup and Run Locally

To run this project on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/sumanthd032/PathFinder-Visualiser
cd PathFinder-Visualiser
```

### 2. Create and activate a virtual environment
```bash
# Create the environment
python -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate

# Or on Windows
# venv\Scripts\activate
```

### 3. Install the required dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

The application should now be open and running in your web browser! 
