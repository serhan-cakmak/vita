import pandas as pd
from causallearn.search.Granger.Granger import Granger
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

openface_features = [
    # " AU20_r",
    # " AU45_r",
    # " AU01_r", " AU02_r", " AU04_r",
    # " AU05_r",
    # " AU09_r",
    # " AU17_r",
    # " AU23_r",
    # " AU25_r", " AU26_r",  # openface features that deducts the models performance
    " AU15_r",
    " AU12_r",
    " AU07_r",
    " AU10_r",
    " AU14_r",
    " AU06_r",  # the actual features that are used for openface

]

broader = openface_features.copy()
[broader.append(f"{feat}_new") for feat in openface_features]
print(broader)

data = pd.read_csv("data/openface/P06_W1.csv")[openface_features].values
G = Granger()
coeff = G.granger_lasso(data)



num_vars = data.shape[1]
G = nx.DiGraph()
threshold = 0.005

print(coeff.shape)
for i in range(num_vars):
    print(openface_features[i])

    for j in range(num_vars):
        total_effect = coeff[i, j] + coeff[i, j + num_vars]
        print(
            f"  From {broader[j]}: Lag1 = {coeff[i, j]:.4f}, Lag2 = {coeff[i, j + num_vars]:.4f}, Total = {total_effect:.4f}")

        if i != j and abs(total_effect) > threshold:
            G.add_edge(broader[j], broader[i], weight=total_effect)

print(coeff)



plt.figure(figsize=(15, 12))
pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjusted layout for better spacing

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=3000, font_size=12, font_weight='bold')

edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})

plt.title("Causal Graph based on Granger Lasso")
plt.axis('off')
plt.tight_layout()
plt.show()

