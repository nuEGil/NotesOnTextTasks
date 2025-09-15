import os
import pandas as pd
from pyvis.network import Network
'''
So this thing generates an html file that shows off the graph.
'''

if __name__ == '__main__':
    data_ = pd.read_csv('pair counts 10 sentence.csv')
    data_['counts'] = data_['counts']/data_['counts'].max()
    
    net = Network(notebook=True, directed = True)
    for _, row in data_.iterrows():
        net.add_node(row['x0'], label=row['x0'], font={"size": 30})
        net.add_node(row['x1'], label=row['x1'], font={"size": 30})
        net.add_edge(row['x0'], row['x1'], value=row['counts'])  # 'value' controls edge thickness

    net.force_atlas_2based()  # nice force-directed layout
    net.show("graph.html")