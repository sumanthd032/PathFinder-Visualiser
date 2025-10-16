import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import random

st.set_page_config(
    page_title="PathFinder Visualizer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'graph' not in st.session_state:
    st.session_state.graph = nx.Graph()

def draw_graph(graph):
    """
    Takes a NetworkX graph and returns a Plotly figure for visualization.
    """
    if not graph.nodes():
        return go.Figure(layout=go.Layout(
                    title=dict(text='<br>Interactive Graph Canvas'),
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    pos = nx.spring_layout(graph, seed=42)

    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(graph.nodes()),
        textposition="top center",
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=25,
            colorbar=dict(
                thickness=15,
                title=dict(text='Node Connections', side='right'),
                xanchor='left',
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(
                        text='<br>Interactive Graph Canvas',
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2].get('weight', 1)
        fig.add_annotation(
            x=(x0+x1)/2, y=(y0+y1)/2,
            text=str(weight),
            showarrow=False,
            font=dict(size=14, color="white"),
            bgcolor="#888",
            opacity=0.8
        )
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

st.title("PathFinder Visualizer: Bellman-Ford vs A* 🚀")
st.markdown("Build a graph using the controls in the sidebar or generate a random one to get started.")

fig = draw_graph(st.session_state.graph)
st.plotly_chart(fig, use_container_width=True)