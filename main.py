from causal_grapher import CausalGrapher

params = {
    "method": "pc",

    "modify_actions": False,
    "normalize": False,  # has no effect on the performance
    "shuffle_df": True,
    "pca": True,        # gives some abstraction to the features

    "features_to_drop": ["interrupt", "actions", "action1", "action2", "action3", "action4", "ex1", "ex2", "ex3", "ex4",
                        " AU01_r", " AU02_r", " AU04_r", " AU05_r", " AU09_r"," AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r", " AU45_r", # openface features that deducts the models performance
                        # " AU15_r",
                        #  " AU07_r", " AU10_r", " AU14_r", " AU06_r", " AU12_r",  # the actual features that are used for openface
                        "alphaRatio_sma3", "Loudness_sma3", "spectralFlux_sma3", "hammarbergIndex_sma3",   #opensmile features
                        # "rewards",
                        #  "speech_duration", "silence_duration"
                         ],

    # "prior_knowledge": [("speech_duration", "rewards"), (" AU12_r", "rewards")],
    "pc": {
        "alpha": 0.01,
        "indep_test": "kci",  # ["fisherz", "chisq", "gsq", "kci"]
        "uc_rule": 1,  # 0(default), 1, 2. But 2 performs the worst
    },
    "fci": {
        "alpha": 0.01,
        "independence_test_method": "kci",
    },
    "ges": {
        "score_func": "local_score_CV_general",
        # local_score_CV_general, local_score_BIC, local_score_BDeu, local_score_CV_multi,
    },
    "grasp": {
        "score_func": "local_score_CV_general",  # local_score_CV_general, local_score_BIC, local_score_BDeu
        "maxP": None,
        "depth": 5,  # d <= 3
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
        "num_explanatory_vals": 7
    },
    "exact": {
        "search_method": "astar"  # "astar", dp
    }
}

cg = CausalGrapher("data/results/extended_data.xlsx", params=params)

# cg.get_scatter_plot(best_line=False, degree=1)
#
for _ in range(1):
    G = cg.get_causal_graph()
    if G is not None:
        cg.visualize_graph(G)
    cg.shuffle_df()


# cg.learn_direction("rewards", "wai", method= "anm") # "pnl", "anm


# for i in ["spectralFlux_sma3", "rewards", " AU10_r", " AU06_r", " AU14_r"]:
#     cg.check_independency("rewards", i)
    # cg.learn_direction(" AU12_r", i, method="anm")  # "pnl", "anm
