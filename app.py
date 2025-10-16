import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import random
import time
import heapq

st.set_page_config(
    page_title="PathFinder Visualizer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'graph' not in st.session_state:
    st.session_state.graph = nx.Graph()
if 'snapshots' not in st.session_state:
    st.session_state.snapshots = {'bf': [], 'a_star': []}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'step' not in st.session_state:
    st.session_state.step = 0

COLORS = {
    'default_node': '#ADD8E6',   # Light Blue
    'start_node': '#39FF14',     # Neon Green
    'goal_node': '#FF3131',      # Neon Red
    'current_node': '#FFFF00',   # Yellow
    'neighbor_node': '#FFA500',  # Orange
    'frontier_node': '#FF00FF',  # Magenta
    'default_edge': '#888',
    'highlighted_edge': '#FFFF00', # Yellow
    'path_edge': '#39FF14'        # Neon Green
}

def bellman_ford(graph, start_node):
    g = graph.to_directed()
    distance = {node: float('inf') for node in g.nodes()}
    predecessor = {node: None for node in g.nodes()}
    distance[start_node] = 0
    snapshots = []
    
    snapshots.append({'distances': distance.copy(), 'highlighted_edge': None, 'message': f"Init: Start '{start_node}' dist=0"})

    for i in range(len(g.nodes()) - 1):
        for u, v, data in g.edges(data=True):
            weight = data['weight']
            snapshots.append({'distances': distance.copy(), 'highlighted_edge': (u, v), 'message': f"Iter {i+1}: Check ({u}→{v})"})
            if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessor[v] = u
                snapshots.append({'distances': distance.copy(), 'highlighted_edge': (u, v), 'message': f"Relaxed '{v}', dist={distance[v]:.1f}"})
                
    for u, v, data in g.edges(data=True):
        if distance[u] != float('inf') and distance[u] + data['weight'] < distance[v]:
            st.error("Graph contains a negative-weight cycle!")
            return None, None, None
            
    return distance, predecessor, snapshots

def a_star(graph, start_node, goal_node, heuristic='euclidean'):
    pos = nx.spring_layout(graph, seed=42)
    def get_heuristic(node1, node2):
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        return abs(x1 - x2) + abs(y1 - y2) if heuristic == 'manhattan' else ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    frontier = [(0, start_node)]
    came_from = {node: None for node in graph.nodes()}
    cost_so_far = {node: float('inf') for node in graph.nodes()}
    cost_so_far[start_node] = 0
    snapshots = []
    visited = set()

    snapshots.append({'frontier': [n for _, n in frontier], 'current': None, 'neighbor': None, 'visited': visited.copy(), 'message': f"Init: Add '{start_node}' to frontier"})

    while frontier:
        _, current_node = heapq.heappop(frontier)

        if current_node in visited:
            continue
        visited.add(current_node)

        snapshots.append({'frontier': [n for _, n in frontier], 'current': current_node, 'neighbor': None, 'visited': visited.copy(), 'message': f"Pop '{current_node}' from frontier"})
        
        if current_node == goal_node:
            snapshots.append({'frontier': [], 'current': current_node, 'neighbor': None, 'visited': visited.copy(), 'message': "Goal reached!"})
            break

        for neighbor in graph.neighbors(current_node):
            if neighbor in visited:
                continue
            
            new_cost = cost_so_far[current_node] + graph[current_node][neighbor]['weight']
            snapshots.append({'frontier': [n for _, n in frontier], 'current': current_node, 'neighbor': neighbor, 'visited': visited.copy(), 'message': f"Check neighbor '{neighbor}'"})
            
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + get_heuristic(neighbor, goal_node)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current_node
                snapshots.append({'frontier': [n for _, n in frontier], 'current': current_node, 'neighbor': neighbor, 'visited': visited.copy(), 'message': f"Add '{neighbor}' to frontier"})
    
    return came_from, cost_so_far, snapshots

def reconstruct_path(predecessors, start, goal):
    path = []
    current = goal
    while current is not None and current != start:
        path.append(current)
        current = predecessors.get(current)
    if current == start:
        path.append(start)
    return path[::-1] if path and path[-1] == start else []

def draw_graph(graph, pos, node_colors, edge_colors, edge_widths):
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    
    colored_edge_traces = []
    for color, edges in edge_colors.items():
        for u, v in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            colored_edge_traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode='lines',
                line=dict(color=color, width=edge_widths.get((u, v), 5)),
                hoverinfo='none'
            ))

    node_x, node_y, node_c, node_t = [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_c.append(node_colors.get(node, COLORS['default_node']))
        node_t.append(node)
        
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_t, textposition="top center",
        marker=dict(size=25, color=node_c, line_width=2))

    fig = go.Figure(data=[edge_trace, *colored_edge_traces, node_trace],
                 layout=go.Layout(showlegend=False, hovermode='closest',
                                  margin=dict(b=5, l=5, r=5, t=5),
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

st.sidebar.header("Graph Controls 🕸️")
st.sidebar.subheader("Manual Creation")
col1, col2 = st.sidebar.columns(2)
with col1:
    node_to_add = st.text_input("Add Node", placeholder="e.g., A")
if st.sidebar.button("Add Node", use_container_width=True):
    if node_to_add:
        st.session_state.graph.add_node(node_to_add)
        st.rerun()
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    edge_from = st.text_input("From", placeholder="A")
with col2:
    edge_to = st.text_input("To", placeholder="B")
with col3:
    edge_weight = st.number_input("Weight", min_value=-10, value=1)
if st.sidebar.button("Add Edge", use_container_width=True):
    if edge_from and edge_to and edge_from in st.session_state.graph and edge_to in st.session_state.graph:
        st.session_state.graph.add_edge(edge_from, edge_to, weight=edge_weight)
        st.rerun()
    else:
        st.sidebar.error("Both nodes must exist in the graph.")
st.sidebar.divider()
st.sidebar.subheader("Automatic Generation")
num_nodes = st.sidebar.slider("Number of Nodes", 2, 20, 5)
if st.sidebar.button("Generate Random Graph", use_container_width=True):
    G = nx.gnp_random_graph(num_nodes, 0.5, directed=False)
    mapping = {i: chr(65 + i) for i in range(num_nodes)}
    G = nx.relabel_nodes(G, mapping)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.randint(1, 10)
    st.session_state.graph = G
    st.session_state.snapshots = {'bf': [], 'a_star': []}
    st.rerun()
if st.sidebar.button("Clear Graph", use_container_width=True, type="primary"):
    st.session_state.graph = nx.Graph()
    st.session_state.snapshots = {'bf': [], 'a_star': []}
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Pathfinding Controls 🎯")
nodes_list = sorted(list(st.session_state.graph.nodes()))
col1, col2 = st.sidebar.columns(2)
with col1:
    start_node = st.selectbox("Start Node", nodes_list)
with col2:
    goal_node = st.selectbox("Goal Node", nodes_list, index=min(1, len(nodes_list)-1) if nodes_list else 0)

heuristic = st.sidebar.selectbox("A* Heuristic", ['euclidean', 'manhattan'])

if st.sidebar.button("Run Pathfinding", use_container_width=True, type="primary"):
    if start_node and goal_node and st.session_state.graph.number_of_nodes() > 0:
        bf_start_time = time.time()
        bf_dist, bf_pred, bf_snapshots = bellman_ford(st.session_state.graph, start_node)
        bf_end_time = time.time()
        bf_path = reconstruct_path(bf_pred, start_node, goal_node)
        
        a_star_start_time = time.time()
        a_came_from, a_cost, a_star_snapshots = a_star(st.session_state.graph, start_node, goal_node, heuristic)
        a_star_end_time = time.time()
        a_star_path = reconstruct_path(a_came_from, start_node, goal_node)
        
        st.session_state.snapshots['bf'] = bf_snapshots
        st.session_state.snapshots['a_star'] = a_star_snapshots
        st.session_state.results = {
            'bf_time': bf_end_time - bf_start_time,
            'bf_cost': bf_dist[goal_node] if goal_node in bf_dist else 'inf',
            'bf_path': bf_path,
            'a_star_time': a_star_end_time - a_star_start_time,
            'a_star_cost': a_cost[goal_node] if goal_node in a_cost else 'inf',
            'a_star_path': a_star_path,
        }
        st.session_state.step = 0
    else:
        st.sidebar.warning("Graph is empty or nodes not selected.")

st.title("PathFinder Visualizer: Bellman-Ford vs A* 🚀")

if not st.session_state.graph.nodes():
    st.info("Build a graph using the sidebar controls to begin.")
elif not st.session_state.snapshots['bf']:
    st.info("Select Start/Goal nodes and click 'Run Pathfinding' to visualize.")
    pos = nx.spring_layout(st.session_state.graph, seed=42)
    node_colors = {node: COLORS['default_node'] for node in st.session_state.graph.nodes()}
    if start_node: node_colors[start_node] = COLORS['start_node']
    if goal_node: node_colors[goal_node] = COLORS['goal_node']
    fig = draw_graph(st.session_state.graph, pos, node_colors, {}, {})
    st.plotly_chart(fig, use_container_width=True)
else:
    max_steps = max(len(st.session_state.snapshots['bf']), len(st.session_state.snapshots['a_star']))
    st.session_state.step = st.slider("Animation Step", 0, max_steps - 1, st.session_state.step)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bellman-Ford")
        bf_step = min(st.session_state.step, len(st.session_state.snapshots['bf']) - 1)
        snapshot = st.session_state.snapshots['bf'][bf_step]
        node_colors = {node: COLORS['default_node'] for node in st.session_state.graph.nodes()}
        node_colors[start_node] = COLORS['start_node']
        edge_colors = {COLORS['default_edge']: []}
        edge_widths = {}
        if snapshot['highlighted_edge']:
            u, v = snapshot['highlighted_edge']
            edge_colors[COLORS['highlighted_edge']] = [(u, v)]
            edge_widths[(u, v)] = 5
        
        if st.session_state.step >= max_steps -1 and st.session_state.results['bf_path']:
             path_edges = list(zip(st.session_state.results['bf_path'][:-1], st.session_state.results['bf_path'][1:]))
             edge_colors[COLORS['path_edge']] = path_edges
        
        pos = nx.spring_layout(st.session_state.graph, seed=42)
        fig_bf = draw_graph(st.session_state.graph, pos, node_colors, edge_colors, edge_widths)
        st.plotly_chart(fig_bf, use_container_width=True)
        st.info(f"Step {bf_step+1}: {snapshot['message']}")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Path Cost", f"{st.session_state.results['bf_cost']:.2f}" if isinstance(st.session_state.results['bf_cost'], (int, float)) else "N/A")
        m_col2.metric("Time", f"{st.session_state.results['bf_time']:.4f}s") # <-- CORRECTED
        
    with col2:
        st.subheader("A* Search")
        a_star_step = min(st.session_state.step, len(st.session_state.snapshots['a_star']) - 1)
        snapshot = st.session_state.snapshots['a_star'][a_star_step]
        node_colors = {node: COLORS['default_node'] for node in st.session_state.graph.nodes()}
        node_colors[start_node] = COLORS['start_node']
        node_colors[goal_node] = COLORS['goal_node']
        for node in snapshot.get('visited', []): node_colors[node] = 'grey'
        for node in snapshot.get('frontier', []): node_colors[node] = COLORS['frontier_node']
        if snapshot.get('neighbor'): node_colors[snapshot['neighbor']] = COLORS['neighbor_node']
        if snapshot.get('current'): node_colors[snapshot['current']] = COLORS['current_node']
        edge_colors = {}
        edge_widths = {}
        if st.session_state.step >= max_steps -1 and st.session_state.results['a_star_path']:
            path_edges = list(zip(st.session_state.results['a_star_path'][:-1], st.session_state.results['a_star_path'][1:]))
            edge_colors[COLORS['path_edge']] = path_edges

        pos = nx.spring_layout(st.session_state.graph, seed=42)
        fig_a_star = draw_graph(st.session_state.graph, pos, node_colors, edge_colors, edge_widths)
        st.plotly_chart(fig_a_star, use_container_width=True)
        st.info(f"Step {a_star_step+1}: {snapshot['message']}")
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Path Cost", f"{st.session_state.results['a_star_cost']:.2f}" if isinstance(st.session_state.results['a_star_cost'], (int, float)) else "N/A")
        m_col2.metric("Time", f"{st.session_state.results['a_star_time']:.4f}s")