import pandas as pd
from pyvis.network import Network
from IPython.display import HTML

def graphmaker(data_):
  
    net = Network(notebook=True, directed = True, cdn_resources='in_line')
    thresh = 0.5 # the lower the threshold, the more nodes connect, but the more the thing looks like a ball.
    for _, row in data_.iterrows():
        if row['counts']>thresh:
            net.add_node(row['x0'], label=row['x0'], font={"size": 30})
            net.add_node(row['x1'], label=row['x1'], font={"size": 30})
            net.add_edge(row['x0'], row['x1'], value=row['counts'])  # 'value' controls edge thickness

    net.force_atlas_2based()  # nice force-directed layout
    net.show("graph.html")

if  __name__ == '__main__':
    data_ = pd.read_csv('pair counts 10 sentence.csv')
    graphmaker(data_)

    HTML(filename="graph.html")