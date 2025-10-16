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


def bellman_ford(graph, start_node):
    """
    Bellman-Ford algorithm implementation.
    Returns distance, predecessors, and a list of snapshots for visualization.
    """
    g = graph.to_directed()
    
    distance = {node: float('inf') for node in g.nodes()}
    predecessor = {node: None for node in g.nodes()}
    distance[start_node] = 0
    
    snapshots = []
    
    snapshots.append({
        'type': 'init',
        'distances': distance.copy(),
        'highlighted_edge': None,
        'message': f"Initialization: Distances set to infinity, start node '{start_node}' to 0."
    })
    
    for i in range(len(g.nodes()) - 1):
        for u, v, data in g.edges(data=True):
            weight = data['weight']
            
            snapshots.append({
                'type': 'relaxation',
                'distances': distance.copy(),
                'highlighted_edge': (u, v),
                'message': f"Iteration {i+1}: Checking edge ({u}, {v}) with weight {weight}..."
            })
            
            if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessor[v] = u
                snapshots.append({
                    'type': 'update',
                    'distances': distance.copy(),
                    'highlighted_edge': (u, v),
                    'message': f"Relaxation successful! Distance to '{v}' updated to {distance[v]}."
                })
                
    for u, v, data in g.edges(data=True):
        if distance[u] != float('inf') and distance[u] + data['weight'] < distance[v]:
            st.error("Graph contains a negative-weight cycle!")
            return None, None, None
            
    return distance, predecessor, snapshots

def a_star(graph, start_node, goal_node, heuristic='euclidean'):
    """
    A* search algorithm implementation.
    Returns came_from, cost_so_far, and a list of snapshots for visualization.
    """
    pos = nx.spring_layout(graph, seed=42)

    def get_heuristic(node1, node2):
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        if heuristic == 'manhattan':
            return abs(x1 - x2) + abs(y1 - y2)
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    frontier = [(0, start_node)] 
    came_from = {node: None for node in graph.nodes()}
    cost_so_far = {node: float('inf') for node in graph.nodes()}
    cost_so_far[start_node] = 0
    
    snapshots = []
    
    snapshots.append({
        'type': 'init',
        'frontier': [start_node],
        'came_from': came_from.copy(),
        'message': f"Initialization: Start node '{start_node}' added to frontier."
    })

    while frontier:
        current_priority, current_node = heapq.heappop(frontier)
        
        if current_node == goal_node:
            snapshots.append({'type': 'goal', 'current': current_node, 'message': "Goal reached!"})
            break

        for neighbor in graph.neighbors(current_node):
            new_cost = cost_so_far[current_node] + graph[current_node][neighbor]['weight']
            
            snapshots.append({
                'type': 'explore',
                'current': current_node,
                'neighbor': neighbor,
                'frontier': [n for _, n in frontier],
                'message': f"Exploring from '{current_node}' to '{neighbor}'..."
            })
            
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + get_heuristic(neighbor, goal_node)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current_node
                snapshots.append({
                    'type': 'update',
                    'frontier': [n for _, n in frontier],
                    'came_from': came_from.copy(),
                    'message': f"Path to '{neighbor}' updated with new cost {new_cost}."
                })
    
    return came_from, cost_so_far, snapshots

def draw_graph(graph, **kwargs):
    if not graph.nodes():
        return go.Figure(layout=go.Layout(
                    title=dict(text='<br>Interactive Graph Canvas'),
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    pos = nx.spring_layout(graph, seed=42)
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y = [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=list(graph.nodes()),
        textposition="top center", marker=dict(showscale=False, colorscale='YlGnBu', reversescale=True,
        color=[], size=25, colorbar=dict(thickness=15, title=dict(text='Node Connections', side='right'),
        xanchor='left'), line_width=2))
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title=dict(text='<br>Interactive Graph Canvas', font=dict(size=16)), showlegend=False, hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2].get('weight', 1)
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=str(weight), showarrow=False,
            font=dict(size=14, color="white"), bgcolor="#888", opacity=0.8)
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
    st.rerun()
if st.sidebar.button("Clear Graph", use_container_width=True, type="primary"):
    st.session_state.graph = nx.Graph()
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Pathfinding Controls 🎯")
nodes_list = list(st.session_state.graph.nodes())
col1, col2 = st.sidebar.columns(2)
with col1:
    start_node = st.selectbox("Start Node", nodes_list)
with col2:
    goal_node = st.selectbox("Goal Node", nodes_list)

heuristic = st.sidebar.selectbox("A* Heuristic", ['euclidean', 'manhattan'])

if st.sidebar.button("Run Pathfinding", use_container_width=True, type="primary"):
    if start_node and goal_node:
        bf_start_time = time.time()
        bf_dist, bf_pred, bf_snapshots = bellman_ford(st.session_state.graph, start_node)
        bf_end_time = time.time()
        
        a_star_start_time = time.time()
        a_star_came_from, a_star_cost, a_star_snapshots = a_star(st.session_state.graph, start_node, goal_node, heuristic)
        a_star_end_time = time.time()
        
 
        st.success(f"Algorithms executed! Bellman-Ford took {bf_end_time - bf_start_time:.4f}s, A* took {a_star_end_time - a_star_start_time:.4f}s.")
    else:
        st.sidebar.warning("Please select a start and goal node.")

st.title("PathFinder Visualizer: Bellman-Ford vs A* 🚀")
st.markdown("Build or generate a graph, then select start/goal nodes and run the pathfinding simulation.")
fig = draw_graph(st.session_state.graph)
st.plotly_chart(fig, use_container_width=True)