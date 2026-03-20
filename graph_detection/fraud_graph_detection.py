import networkx as nx


def build_transaction_graph(df):

    G = nx.Graph()

    for idx, row in df.iterrows():

        card_node = f"card_{idx}"

        merchant_node = f"merchant_{int(row['Amount_original'])}"

        G.add_edge(card_node, merchant_node)

    return G


def compute_graph_risk(df):

    G = build_transaction_graph(df)

    centrality = nx.degree_centrality(G)

    graph_scores = []

    for idx in df.index:

        card_node = f"card_{idx}"

        score = centrality.get(card_node, 0)

        graph_scores.append(score)

    df["graph_risk_score"] = graph_scores

    return df