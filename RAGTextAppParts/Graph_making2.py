import os
import json
import pandas as pd
from pyvis.network import Network
'''
So this thing generates an html file that shows off the graph.
'''

if __name__ == '__main__':
    with open("pair_stuff.json", "r") as f:
        data = json.load(f)
    
    net = Network(notebook=True, directed = True)
    
    agg_data = {'edge':[], 'x0':[], 'x1':[], 'weight':[]}
    for key, value in data.items():
        x0,x1 = key.split(' : ')
        agg_data['edge'].append(key)
        agg_data['x0'].append(x0)
        agg_data['x1'].append(x1)
        agg_data['weight'].append(len(''.join(data[key]['sub_text'])))

    agg_data = pd.DataFrame.from_dict(agg_data)
    print(agg_data)

    # --- Normalize weights ---
    max_w = agg_data["weight"].max()
    agg_data["norm_weight"] = agg_data["weight"] / max_w

    # --- Threshold filter (keep only stronger edges) ---
    # threshold = 
    # agg_data = agg_data[agg_data["norm_weight"] > threshold]

    # --- Build network ---
    # Add nodes with size scaled by max edge weight they're involved in
    for node in pd.concat([agg_data["x0"], agg_data["x1"]]).unique():
        # Node weight = max norm_weight of any edge touching it
        node_weight = agg_data.loc[
            (agg_data["x0"] == node) | (agg_data["x1"] == node), "norm_weight"
        ].max()
        size = 10 + node_weight * 40
        font_size = 20 + node_weight * 30
        net.add_node(node, label=node, size = size, font={"size": font_size}, title=f"{node} (score={node_weight:.2f})")

    # Add edges
    for _, row in agg_data.iterrows():
        net.add_edge(
            row["x0"],
            row["x1"],
            value=row["norm_weight"],  # controls edge thickness
            title=f"weight={row['weight']}",
        )

    net.force_atlas_2based()  # nice force-directed layout
    net.show("graph.html")

   