import os
import random
import warnings

import numpy as np

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.search.HiddenCausal.GIN.GIN import GIN
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam import CAMUV
from causallearn.search.FCMBased.lingam.utils import make_dot
from causallearn.search.FCMBased.PNL.PNL import PNL
from causallearn.search.FCMBased.ANM.ANM import ANM
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.utils.cit import *

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import urllib.parse

from pathlib import Path


class CausalGrapher:
    def __init__(self, data_file, params):
        self.df = self.read_excel(data_file, params["features_to_drop"])
        if params["shuffle_df"]:
            self.shuffle_df()
        if params["modify_actions"]:
            self.create_past_actions()
            self.categorize_column("actions")
        if "prior_knowledge" in params:
            name = params["method"]
            if name == "lingam":
                params["lingam"]["specific_model_params"]["direct"][
                    "prior_knowledge"] = self.create_prior_knowledge_matrix(
                    params["prior_knowledge"])
            elif name in ["pc", "fci"]:
                params[name]["background_knowledge"] = self.get_background_knowledge_object(params["prior_knowledge"])

        if params["normalize"]:
            self.normalize_df()
        if "pca_ncomponents" in params:
            self.apply_pca(params["pca_ncomponents"])

        self.method = params["method"]
        self.params = params[self.method]

    def apply_pca(self, n_components=0.85):

        # The code below applies pca to each group of features. In a paper they tested this and worked well.
        # Cannot be said the same for our case.

        # info = [["rewards"], [" AU07_r", " AU10_r", " AU14_r", " AU06_r", " AU12_r", " AU15_r"],
        #         ["speech_duration", "silence_duration"],
        #         ["alphaRatio_sma3", "Loudness_sma3", "spectralFlux_sma3", "hammarbergIndex_sma3"]]

        info = [["rewards"], self.df.columns.drop("rewards")]
        dfs = pd.DataFrame()
        for group in info:
            matrix = self.df[group]
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(matrix)
            pca = PCA(n_components=n_components)
            tmp = pd.DataFrame(pca.fit_transform(scaled_data))

            loadings = pca.components_

            threshold = 0.45
            loadings = pd.DataFrame(loadings.T, columns=tmp.columns,
                                    index=matrix.columns)
            loadings.columns = ["+".join(loadings.index[loadings[i].abs() > threshold]) for i in
                                range(loadings.shape[1])]
            loadings.columns = [
                loadings.columns[i] if loadings.columns[i] != "" else loadings.iloc[:, i].abs().idxmax() + "*" for i in
                range(loadings.columns.__len__())]

            print(loadings.to_string())
            tmp.columns = loadings.columns
            dfs = pd.concat([dfs, tmp], axis=1)

        # dfs = pd.concat([dfs, self.df.drop(columns=["rewards"])], axis=1)
        self.df = dfs

    def get_background_knowledge_object(self, prior_knowledge):
        features = {self.df.columns[i]: i + 1 for i in range(len(self.df.columns))}
        bk = BackgroundKnowledge()
        for cause, effect in prior_knowledge:
            node1 = GraphNode("X" + str(features[cause]))
            node2 = GraphNode("X" + str(features[effect]))
            bk.add_required_by_node(node1, node2)
            # bk.add_forbidden_by_node(node1, node2)
        return bk

    def shuffle_df(self):
        cols = self.df.columns.tolist()
        random.shuffle(cols)
        print(self.df.columns)
        self.df = self.df[cols]

    def create_prior_knowledge_matrix(self, prior_knowledge=None):
        res = []
        n = self.df.columns.__len__()
        for i in range(n):
            res.append([-1] * n)

        for cause, effect in prior_knowledge:
            if "AU" in cause:
                cause = " " + cause + "_r"
            if "AU" in effect:
                effect = " " + effect + "_r"

            features = {self.df.columns[i]: i for i in range(len(self.df.columns))}
            # todo I think it should be other way around but in the result this seems more true. Maybe because of the
            # implementation
            res[features[effect]][features[cause]] = 1
        # [print(res[i],self.df.columns[i] ) for i in range(n)]
        return res

    def create_past_actions(self):
        self.df["actions"] = self.df["actions"].shift(1)
        self.df["actions"][0] = random.randint(0, 2)
        self.df["actions"] = self.df["actions"].astype(int)
        # print(self.df.head())

    def normalize_df(self):
        scaler = MinMaxScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)
        # print(self.df.head())

    def categorize_column(self, column):
        self.df = pd.concat([self.df, pd.get_dummies(self.df[column], prefix=column)], axis=1)
        self.df = self.df.drop(columns=[column])
        # print(self.df.head())

    def apply_pc(self):
        return pc(self.df.values, **self.params)

    def apply_fci(self):
        return fci(self.df.values, **self.params)

    def apply_gin(self):
        return GIN(self.df.values)

    def apply_ges(self):
        return ges(self.df.values, **self.params)

    def apply_grasp(self):
        return grasp(self.df.values, **self.params)

    def apply_camuv(self):
        P, U = CAMUV.execute(self.df.values, **self.params)
        print([f"{i} - {self.df.columns[i]}" for i in range(len(self.df.columns))])
        for i, result in enumerate(P):
            if not len(result) == 0:
                print("child: " + str(i) + ",  parents: " + str(result))

        for result in U:
            print(result)

    def apply_lingam(self):
        if self.params["name"] == "ica":
            model = lingam.ICALiNGAM(42, **self.params["specific_model_params"][self.params["name"]])
        elif self.params["name"] == "direct":
            model = lingam.DirectLiNGAM(42, **self.params["specific_model_params"][self.params["name"]])
        elif self.params["name"] == "rcd":
            model = lingam.RCD(**self.params["specific_model_params"][self.params["name"]])
        else:
            print("No name provided for lingam model.")
        model.fit(self.df.values)
        self.params = {"name": self.params["name"]}
        return model.adjacency_matrix_

    def read_excel(self, file_name, features_to_drop=[]):
        df = pd.read_excel(file_name)
        try:
            df = df.drop(
                columns=features_to_drop)
        except:
            warnings.warn(
                "If you do not test with rosas/reward file, there is a problem with the columns while dropping.")
        # columns=["interrupt", "action1", "action2", "action3", "action4"])
        return df

    def get_causal_graph(self):
        if self.method == "pc":
            cg = self.apply_pc()
            return cg.G
        elif self.method == "fci":
            G, _ = self.apply_fci()
            return G
        elif self.method == "ges":
            return self.apply_ges()["G"]
        elif self.method == "grasp":
            return self.apply_grasp()
        elif self.method == "lingam":
            return self.apply_lingam()
        elif self.method == "gin":
            G, list_of_parents = self.apply_gin()
            print(list_of_parents)
            return G
        elif self.method == "camuv":
            self.apply_camuv()
            return None
        elif self.method == "exact":
            dag_est, search_stats = bic_exact_search(self.df.values)
            print(dag_est)
            print(search_stats)
            return dag_est

    def get_target(self):

        folder_name = "images1/" + self.method + "/"
        self.create_folder(folder_name)

        num = os.listdir(folder_name).__len__().__str__()
        params_str = ""
        if isinstance(self.params, dict):
            params_str = "_".join(f"{key}={urllib.parse.quote(str(value))}" for key, value in self.params.items())
        file_name = params_str + "-" + num
        return folder_name, file_name

    def visualize_graph(self, cg):
        folder_name, file_name = self.get_target()

        if self.method == "lingam" or self.method == "exact":
            dot = make_dot(cg, labels=self.df.columns)
            dot.render(filename=file_name, directory=folder_name, format="png", cleanup=True)
            return

        pyd = None
        print(cg)

        if self.method == "gin":
            pyd = GraphUtils.to_pydot(cg)
        else:
            pyd = GraphUtils.to_pydot(cg, labels=self.df.columns)

        pyd.write_png(folder_name + "/" + file_name + ".png")

        # img = Image.open(name)
        # img.show()

    def create_folder(self, name):
        Path(name).mkdir(parents=True, exist_ok=True)

    def get_scatter_plot(self, best_line=False, degree=2):
        folder_name = "images1/bivariate_plots/"
        self.create_folder(folder_name)

        for i in range(len(self.df.columns)):
            for j in range(i + 1, len(self.df.columns)):
                x = self.df[self.df.columns[i]]
                y = self.df[self.df.columns[j]]

                # Create scatter plot
                ax = self.df.plot.scatter(x=self.df.columns[i], y=self.df.columns[j])

                # Fit a line
                # m, b = np.polyfit(x, y, 1)
                # ax.plot(x, m * x + b, color='red', label='Best fit line')
                if best_line:
                    p = np.polyfit(x, y, degree)
                    f = np.poly1d(p)

                    # Plot the polynomial fit
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit = f(x_fit)
                    ax.plot(x_fit, y_fit, color='red', label='Polynomial fit')

                # Add labels and title
                ax.set_xlabel(self.df.columns[i])
                ax.set_ylabel(self.df.columns[j])
                ax.set_title(f'Scatter plot of {self.df.columns[i]} vs {self.df.columns[j]}')
                ax.legend()

                plt.savefig(folder_name + self.df.columns[i] + "_" + self.df.columns[j] + ".png")
                plt.close()

    def learn_direction(self, x, y, method="pnl"):
        if method == "pnl":
            model = PNL()
        elif method == "anm":
            model = ANM()

        row1 = self.df[x].values.reshape(-1, 1)
        row2 = self.df[y].values.reshape(-1, 1)

        p_forward, p_backward = model.cause_or_effect(row1, row2)
        print(f"{x} -> {y}: {p_forward}")
        print(f"{y} -> {x}: {p_backward}")

    def check_independency(self, x, y, conditional_set=None):
        dct = {self.df.columns[i]: i for i in range(len(self.df.columns))}
        conditional_set = [dct[i] for i in conditional_set] if conditional_set is not None else None
        print(x, "-", y)
        for i in [fisherz, chisq, gsq, kci]:
            obj = CIT(self.df.values, i)
            p_val = obj(dct[x], dct[y], )
            print("\t", i, p_val)

    def sample_df(self, frac=1, replace=False):
        if replace:
            warnings.warn("Be careful about the frac value since it can converge to 0.")
        self.df = self.df.sample(frac=frac, replace=replace).reset_index(drop=True)


def average_graphs(graphs):
    # Initialize a zero matrix
    avg_matrix = np.zeros_like(graphs[0].graph)

    # Sum all adjacency matrices
    for graph in graphs:
        avg_matrix += graph.graph

    # Divide by the number of graphs to get the average
    avg_matrix = np.round(np.divide(avg_matrix, len(graphs))).astype(int)
    print(avg_matrix)

    # avg_graph = GeneralGraph(graphs[0].nodes)
    # avg_graph.graph = avg_matrix
    graphs[0].graph = avg_matrix

    return graphs[0]


def ensemble(graphs, method):
    n = graphs[0].graph.shape[0]
    num = len(graphs)
    majority = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            edge_type = get_types_edges(graphs, i, j, method, num)
            print(f"{i} -> {j}: {edge_type}")
            redge_type = get_types_edges(graphs, j, i, method, num)
            print(f"{j} -> {i}: {redge_type}")
            if redge_type == "directed" or redge_type == "notancestor":
                majority = modify_majority_matrix(majority, j, i, redge_type, method)
            else:
                majority = modify_majority_matrix(majority, i, j, edge_type, method)

            print(f"Majority: {int(majority[i][j])}, {int(majority[j][i])}")

    print(majority)

    graphs[0].graph = majority

    return graphs[0]


def get_types_edges(graphs, i, j, type, n):
    directed = 0
    undirected = 0
    confounder = 0
    notancestor = 0

    if type == "fci":
        for graph in graphs:
            if graph.graph[i][j] == -1 and graph.graph[j][i] == 1:
                directed += 1
            elif graph.graph[i][j] == 2 and graph.graph[j][i] == 1:
                notancestor += 1
            elif graph.graph[i][j] == 1 and graph.graph[j][i] == 1:
                confounder += 1
            elif graph.graph[i][j] == 2 and graph.graph[j][i] == 2:
                undirected += 1
    if type == "pc":
        for graph in graphs:
            if graph.graph[i][j] == -1 and graph.graph[j][i] == 1:
                directed += 1
            elif graph.graph[i][j] == -1 and graph.graph[j][i] == -1:
                undirected += 1

    dicti = {"directed": directed, "undirected": undirected, "confounder": confounder, "notancestor": notancestor,
             "independent": n - directed - undirected - confounder - notancestor}
    print(dicti)

    if dicti["independent"] < n/2:
        dicti.pop("independent")

    key_order = ["directed", "notancestor", "confounder", "undirected", "independent"]
    return max(dicti.items(), key=lambda x: (x[1], key_order.index(x[0])))[0]


def modify_majority_matrix(majority, i, j, edge_type, method):
    if method == "fci":
        if edge_type == "directed":
            majority[i][j] = -1
            majority[j][i] = 1
        elif edge_type == "undirected":
            majority[i][j] = 2
            majority[j][i] = 2
        elif edge_type == "confounder":
            majority[i][j] = 1
            majority[j][i] = 1
        elif edge_type == "notancestor":
            majority[i][j] = 2
            majority[j][i] = 1
    else:
        if edge_type == "directed":
            majority[i][j] = -1
            majority[j][i] = 1
        elif edge_type == "undirected":
            majority[i][j] = -1
            majority[j][i] = -1

    return majority
