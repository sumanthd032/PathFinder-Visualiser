import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import random
import time
import heapq
import json
import asyncio

st.set_page_config(
    page_title="PathFinder Visualizer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

default_state = {
    'graph': nx.Graph(),
    'snapshots': {'bf': [], 'a_star': []},
    'results': {},
    'step': 0,
    'is_playing': False,
    'speed': 0.25  
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

COLORS = {
    'default_node': 'skyblue',
    'start_node': '#39FF14',
    'goal_node': '#FF3131',
    'current_node': 'yellow',
    'neighbor_node': 'orange',
    'frontier_node': 'magenta',
    'visited_node': '#636E72',
    'default_edge': '#BDC3C7',
    'highlighted_edge': 'yellow',
    'path_edge': '#39FF14'
}

def bellman_ford(graph, start_node):
    g = graph.to_directed()
    distance = {node: float('inf') for node in g.nodes()}
    predecessor = {node: None for node in g.nodes()}
    distance[start_node] = 0
    snapshots = [{'type': 'init', 'distances': distance.copy(), 'message': f"Init: Start '{start_node}' dist=0"}]
    
    for i in range(len(g.nodes()) - 1):
        for u, v, data in g.edges(data=True):
            weight = data['weight']
            snapshots.append({'type': 'check', 'distances': distance.copy(), 'highlighted_edge': (u, v), 'message': f"Iter {i+1}: Check ({u}→{v})"})
            if distance[u] != float('inf') and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessor[v] = u
                snapshots.append({'type': 'relax', 'distances': distance.copy(), 'highlighted_edge': (u, v), 'message': f"Relaxed '{v}', dist={distance[v]:.1f}"})
                
    for u, v, data in g.edges(data=True):
        if distance[u] != float('inf') and distance[u] + data['weight'] < distance[v]:
            st.error("Graph contains a negative-weight cycle!"); return None, None, None
            
    return distance, predecessor, snapshots

def a_star(graph, start_node, goal_node, heuristic='euclidean'):
    pos = nx.spring_layout(graph, seed=42)
    def get_heuristic(n1, n2):
        x1, y1 = pos[n1]; x2, y2 = pos[n2]
        return abs(x1 - x2) + abs(y1 - y2) if heuristic == 'manhattan' else ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    frontier = [(0, start_node)]
    came_from, cost_so_far = {}, {node: float('inf') for node in graph.nodes()}
    cost_so_far[start_node] = 0
    snapshots, visited = [], set()
    snapshots.append({'type': 'init', 'frontier': [n for _, n in frontier], 'visited': visited.copy(), 'message': f"Init: Add '{start_node}' to frontier"})

    while frontier:
        _, current_node = heapq.heappop(frontier)
        if current_node in visited: continue
        
        snapshots.append({'type': 'pop', 'frontier': [n for _,n in frontier], 'current': current_node, 'visited': visited.copy(), 'message': f"Pop '{current_node}' from frontier"})
        visited.add(current_node)
        
        if current_node == goal_node:
            snapshots.append({'type': 'goal', 'frontier': [], 'current': current_node, 'visited': visited.copy(), 'message': "Goal reached!"})
            break

        for neighbor in graph.neighbors(current_node):
            if neighbor in visited: continue
            new_cost = cost_so_far[current_node] + graph[current_node][neighbor]['weight']
            snapshots.append({'type': 'check', 'frontier': [n for _,n in frontier], 'current': current_node, 'neighbor': neighbor, 'visited': visited.copy(), 'message': f"Check neighbor '{neighbor}'"})
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + get_heuristic(neighbor, goal_node)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current_node
                snapshots.append({'type': 'update', 'frontier': [n for _,n in frontier], 'current': current_node, 'neighbor': neighbor, 'visited': visited.copy(), 'message': f"Add '{neighbor}' to frontier"})
    return came_from, cost_so_far, snapshots

def reconstruct_path(predecessors, start, goal):
    path, current = [], goal
    while current is not None and current in predecessors:
        path.append(current)
        prev = predecessors.get(current)
        if prev == current: break 
        current = prev
    if current == start: path.append(start)
    return path[::-1] if path and path[-1] == start else []

def draw_graph(graph, pos, node_colors, edge_colors, edge_widths):
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color=COLORS['default_edge']), hoverinfo='none', mode='lines')
    
    colored_edge_traces = []
    for color, edges in edge_colors.items():
        for u, v in edges:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            colored_edge_traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(color=color, width=edge_widths.get((u, v), 8)), hoverinfo='none'))
    
    node_x, node_y, node_c, node_t = [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_c.append(node_colors.get(node, COLORS['default_node'])); node_t.append(node)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_t, textfont=dict(color='white', size=14), textposition="middle center", marker=dict(size=35, color=node_c, line=dict(color='white', width=2)))
    
    fig = go.Figure(data=[edge_trace, *colored_edge_traces, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=5,l=5,r=5,t=5), plot_bgcolor='#1E1E1E', paper_bgcolor='#1E1E1E', xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=str(edge[2].get('weight', 1)), showarrow=False, font=dict(size=12, color="white"), bgcolor="#333", opacity=0.8)
    return fig

with st.sidebar:
    st.title("PathFinder Controls")
    st.divider()
    st.header("Graph Controls 🕸️")
    with st.expander("Manual Creation", expanded=False):
        c1, c2 = st.columns(2)
        node_to_add = c1.text_input("Add Node", placeholder="e.g., A")
        if c2.button("Add", use_container_width=True):
            if node_to_add: st.session_state.graph.add_node(node_to_add); st.rerun()
        c1, c2, c3 = st.columns(3)
        edge_from = c1.text_input("From", placeholder="A"); edge_to = c2.text_input("To", placeholder="B"); edge_weight = c3.number_input("Weight", min_value=-10, value=1, label_visibility="collapsed")
        if st.button("Add Edge", use_container_width=True):
            if edge_from and edge_to and edge_from in st.session_state.graph and edge_to in st.session_state.graph:
                st.session_state.graph.add_edge(edge_from, edge_to, weight=edge_weight); st.rerun()
            else: st.error("Both nodes must exist.")
    st.subheader("Automatic Generation")
    num_nodes = st.slider("Number of Nodes", 2, 20, 5)
    if st.button("Generate Random Graph", use_container_width=True):
        G = nx.gnp_random_graph(num_nodes, 0.5, directed=False)
        mapping = {i: chr(65 + i) for i in range(num_nodes)}
        G = nx.relabel_nodes(G, mapping)
        for (u, v) in G.edges(): G.edges[u,v]['weight'] = random.randint(1, 10)
        st.session_state.graph = G; st.session_state.snapshots = {'bf': [], 'a_star': []}; st.rerun()
    if st.button("Clear Graph", use_container_width=True, type="primary"):
        st.session_state.graph = nx.Graph(); st.session_state.snapshots = {'bf': [], 'a_star': []}; st.rerun()
    st.divider()
    st.header("Pathfinding Controls 🎯")
    nodes_list = sorted(list(st.session_state.graph.nodes()))
    c1, c2 = st.columns(2)
    start_node = c1.selectbox("Start Node", nodes_list)
    goal_node = c2.selectbox("Goal Node", nodes_list, index=min(1, len(nodes_list)-1) if len(nodes_list)>1 else 0)
    heuristic = st.selectbox("A* Heuristic", ['euclidean', 'manhattan'])
    if st.button("Run Pathfinding", use_container_width=True, type="primary"):
        if start_node and goal_node and st.session_state.graph.number_of_nodes() > 0:
            bf_start_time = time.perf_counter()
            bf_dist, bf_pred, bf_snapshots = bellman_ford(st.session_state.graph, start_node)
            bf_end_time = time.perf_counter()
            a_star_start_time = time.perf_counter()
            a_came_from, a_cost, a_star_snapshots = a_star(st.session_state.graph, start_node, goal_node, heuristic)
            a_star_end_time = time.perf_counter()
            st.session_state.snapshots = {'bf': bf_snapshots, 'a_star': a_star_snapshots}
            st.session_state.results = {
                'bf_time': bf_end_time - bf_start_time, 'bf_cost': bf_dist.get(goal_node, float('inf')), 'bf_path': reconstruct_path(bf_pred, start_node, goal_node),
                'a_star_time': a_star_end_time - a_star_start_time, 'a_star_cost': a_cost.get(goal_node, float('inf')), 'a_star_path': reconstruct_path(a_came_from, start_node, goal_node)
            }
            st.session_state.step = 0; st.session_state.is_playing = False
        else: st.warning("Graph is empty or nodes not selected.")

st.title("PathFinder Visualizer 🧭")
st.markdown("An interactive tool to see the Bellman-Ford and A* shortest path algorithms in action.")

if not st.session_state.graph.nodes():
    st.info("Build a graph using the sidebar controls to begin.")
elif not st.session_state.snapshots.get('bf'):
    st.info("Select Start/Goal nodes and click 'Run Pathfinding' to visualize.")
    pos = nx.spring_layout(st.session_state.graph, seed=42)
    node_colors = {n: COLORS['default_node'] for n in st.session_state.graph.nodes()}
    if start_node: node_colors[start_node] = COLORS['start_node']
    if goal_node: node_colors[goal_node] = COLORS['goal_node']
    fig = draw_graph(st.session_state.graph, pos, node_colors, {}, {})
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
else:
    max_steps = max(len(st.session_state.snapshots['bf']), len(st.session_state.snapshots['a_star']))
    
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 6])
        if c1.button("⏮️ Step Back"): st.session_state.step = max(0, st.session_state.step - 1)
        if c2.button("Step Fwd ⏭️"): st.session_state.step = min(max_steps - 1, st.session_state.step + 1)
        play_pause_text = "Pause ⏸️" if st.session_state.is_playing else "Play ▶️"
        if c3.button(play_pause_text, type="primary"): st.session_state.is_playing = not st.session_state.is_playing
        st.session_state.step = c4.slider("Animation Progress", 0, max_steps - 1, st.session_state.step, label_visibility="collapsed")
        st.session_state.speed = st.select_slider("Animation Speed", options=[1, 0.5, 0.25, 0.1, 0.05], value=st.session_state.speed, format_func=lambda x: f"{1/x:.0f}x")
    
    st.write("") 

    col1, col2 = st.columns(2)
    pos = nx.spring_layout(st.session_state.graph, seed=42)
    is_final_step = st.session_state.step >= max_steps - 1
    
    with col1:
        with st.container(border=True):
            st.subheader("Bellman-Ford")
            bf_step = min(st.session_state.step, len(st.session_state.snapshots['bf']) - 1)
            snapshot = st.session_state.snapshots['bf'][bf_step]
            node_colors = {n: COLORS['default_node'] for n in st.session_state.graph.nodes()}
            node_colors[start_node] = COLORS['start_node']
            edge_colors, edge_widths = {}, {}
            if he := snapshot.get('highlighted_edge'): edge_colors[COLORS['highlighted_edge']] = [he]; edge_widths[he] = 6
            if is_final_step and (path := st.session_state.results['bf_path']): edge_colors[COLORS['path_edge']] = list(zip(path[:-1], path[1:]))
            fig_bf = draw_graph(st.session_state.graph, pos, node_colors, edge_colors, edge_widths)
            st.plotly_chart(fig_bf, use_container_width=True, config={'displayModeBar': False})
            st.info(f"Step {bf_step+1}: {snapshot['message']}")
    
    with col2:
        with st.container(border=True):
            st.subheader("A* Search")
            a_star_step = min(st.session_state.step, len(st.session_state.snapshots['a_star']) - 1)
            snapshot = st.session_state.snapshots['a_star'][a_star_step]
            node_colors = {n: COLORS['default_node'] for n in st.session_state.graph.nodes()}
            if goal_node: node_colors[goal_node] = COLORS['goal_node']
            for n in snapshot.get('visited', []): node_colors[n] = COLORS['visited_node']
            for n in snapshot.get('frontier', []): node_colors[n] = COLORS['frontier_node']
            if n := snapshot.get('neighbor'): node_colors[n] = COLORS['neighbor_node']
            if c := snapshot.get('current'): node_colors[c] = COLORS['current_node']
            node_colors[start_node] = COLORS['start_node']
            edge_colors = {}
            if is_final_step and (path := st.session_state.results['a_star_path']): edge_colors[COLORS['path_edge']] = list(zip(path[:-1], path[1:]))
            fig_a_star = draw_graph(st.session_state.graph, pos, node_colors, edge_colors, {})
            st.plotly_chart(fig_a_star, use_container_width=True, config={'displayModeBar': False})
            st.info(f"Step {a_star_step+1}: {snapshot['message']}")

    st.divider()

    with st.expander("📊 View Final Analysis & Export Results"):
        results = st.session_state.results
        bf_path_str = " → ".join(results['bf_path']) if results['bf_path'] else "No path found"
        a_star_path_str = " → ".join(results['a_star_path']) if results['a_star_path'] else "No path found"
        analysis_data = {
            "Metric": ["Path Cost", "Execution Time (s)", "Path Found"],
            "Bellman-Ford": [f"{results['bf_cost']:.2f}" if isinstance(results['bf_cost'], (int, float)) else "N/A", f"{results['bf_time']:.5f}", bf_path_str],
            "A* Search": [f"{results['a_star_cost']:.2f}" if isinstance(results['a_star_cost'], (int, float)) else "N/A", f"{results['a_star_time']:.5f}", a_star_path_str]
        }
        st.table(analysis_data)
        export_data = {"graph": nx.node_link_data(st.session_state.graph), "parameters": {"start_node": start_node, "goal_node": goal_node, "a_star_heuristic": heuristic}, "results": {"bellman_ford": {"cost": results['bf_cost'], "time_seconds": results['bf_time'], "path": results['bf_path']}, "a_star": {"cost": results['a_star_cost'], "time_seconds": results['a_star_time'], "path": results['a_star_path']}}}
        st.download_button(label="💾 Export Results as JSON", data=json.dumps(export_data, indent=4), file_name=f"pathfinder_results_{start_node}_to_{goal_node}.json", mime="application/json", use_container_width=True)

    if st.session_state.is_playing:
        if st.session_state.step < max_steps - 1:
            st.session_state.step += 1
            time.sleep(st.session_state.speed)
            st.rerun()
        else:
            st.session_state.is_playing = False
            st.rerun()