import numpy as np
import pandas as pd

from causal_grapher import *


apply_ensemble = False
show_sepset = True

params = {
    "method": "pc",
    "features": "difsa",  # "all", "difsa, test

    "modify_actions": False,
    "normalize": False,  # has no effect on the performance
    "shuffle_df": False,
    # "pca_ncomponents": 0.85,        # gives some abstraction to the features


    "all": [" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r", " AU07_r", " AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r", "alphaRatio_sma3", "Loudness_sma3", "spectralFlux_sma3", "hammarbergIndex_sma3", "rewards", "speech_duration", "silence_duration"],
    "difsa": [" AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU06_r", " AU09_r", " AU12_r", " AU15_r", " AU17_r", " AU20_r", " AU25_r", " AU26_r", "alphaRatio_sma3", "Loudness_sma3", "spectralFlux_sma3", "hammarbergIndex_sma3", "rewards", "speech_duration", "silence_duration"],
    "test": [],

    # "prior_knowledge": [(" AU06_r"," AU12_r" )],
    "pc": {
        "alpha": 0.01,
        "indep_test": "kci",  # ["fisherz", "chisq", "gsq", "kci"]
        "uc_rule": 0,  # 0(default), 1, 2. But 2 performs the worst
    },
    "fci": {
        "alpha": 0.01,
        "independence_test_method": "kci",
        "uc_rule": 0,  # 0(default), 1, 2. But 2 performs the worst
        # "cache_path": "tmp/fci_cache_all_feats_1.json",     # just to test uc_rule

    },
    "ges": {
        "score_func": "local_score_CV_general",
        # local_score_CV_general, local_score_BIC, local_score_BDeu, local_score_CV_multi,
    },
    "grasp": {
        "score_func": "local_score_CV_general",  # local_score_CV_general, local_score_BIC, local_score_BDeu
        "maxP": None,
        "depth": 3,  # d <= 3
    },
    "lingam": {
        "name": "ica",  # "ica", "direct", "rcd"
        "specific_model_params": {
            "direct": {
                "apply_prior_knowledge_softly": True,
            },
            "ica": {
                "max_iter": 10000,  # max tested was 300000

            },
            "rcd": {
                "cor_alpha": 0.01,
                "max_explanatory_num": 7,
            }
        },
    },
    "gin": {

    },
    "camuv": {
        "alpha": 0.01,
        "num_explanatory_vals": 3
    },
    "exact": {
        "search_method": "astar"  # "astar", dp
    }
}
cg = CausalGrapher("data/results/extended_data.xlsx", params=params)
base_df = cg.df.copy()
# cg.get_scatter_plot(best_line=True, degree=2)

graphs = []
feats = cg.df.columns
num_ensemble = 1

for _ in range(num_ensemble):
    G = cg.get_causal_graph()
    if show_sepset:
        num = len(G.graph)
        for i in range (num):
            for j in range (num):
                if i != j and G.graph[i][j] == 0 and G.graph[j][i] == 0:
                    sepset = [feats[int(node.get_name()[1:]) - 1] for node in G.get_sepset(G.nodes[i], G.nodes[j])]
                    if len(sepset) > 0:
                        print(feats[i], feats[j], sepset)
    graphs.append(G)
    print("Graph ", _ + 1)
    if apply_ensemble:
        cg.df = base_df.sample(frac=0.8, replace=False).reset_index(drop=True)
        # cg.sample_df(1, replace=True) # because the data is too small and noicy maybe not the best idea

    else:
        cg.visualize_graph(G)
        break



if apply_ensemble:
    majored_g = ensemble(graphs, cg.method, feats)
    cg.visualize_graph(majored_g)


# cg.learn_direction("rewards", " AU07_r+ AU14_r", method= "anm") # "pnl", "anm

#
# for i in [" AU12_r", " AU15_r"]:
#     cg.check_independency("rewards", i)
#     cg.learn_direction("rewards", i, method="anm")  # "pnl", "anm
#     cg.learn_direction("rewards", i, method="pnl")  # "pnl", "anm
