import pandas as pd
import os
import ast

data_columns = ["observations", "actions", "rewards"]
res = []

observation_features = ["ex1", "ex2", "ex3", "ex4", "silence_duration", "speech_duration", "action1", "action2",
                        "action3", "action4", "interrupt"]
# complete the features
openface_features = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r',
                     ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']

opensmile_features = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3", "spectralFlux_sma3"]

openface_results = []
opensmile_results = []


def get_filenames(folder_name):
    # Get all the files in the current directory
    files = os.listdir("data/" + folder_name + "/")
    return files


def read_logs(file_name, flag=False):
    # Read the excel file
    print("Reading file: ", file_name)
    df = pd.read_excel("data/logs/" + file_name)

    tmp = []

    for data_column in data_columns:
        filled_indexes = df[data_column][df[data_column].notnull()]
        if data_column == "actions":
            filled_indexes = filled_indexes.astype(int)
            tmp.append(filled_indexes.values.tolist())

        elif data_column == "observations":
            filled_indexes = filled_indexes.apply(ast.literal_eval)
            for i in range(len(observation_features)):
                col_name = observation_features[i]

                a = filled_indexes.apply(lambda x: x[i])

                if "ex" in col_name:
                    if col_name.endswith(file_name[-6]):  # X.xlsx
                        a = a.apply(lambda x: 1)
                    else:
                        a = a.apply(lambda x: 0)

                tmp.append(a.values.tolist())

        elif data_column == "rewards":
            if flag:
                return filled_indexes.values.mean()
            tmp.append(filled_indexes.values.tolist())

    for j in range(len(tmp[0])):
        # res.append([tmp[0][j], tmp[1][j], tmp[2][j]])
        res.append([tmp[i][j] for i in range(len(tmp))])

    return df.loc[filled_indexes.index, "timestamp_sec"].tolist()


def calculate_indexes(face, timesteps):
    alpha = 0  # timestep difference between the current and the previous timestep, in openface every observation is
    # made in 0.033 seconds, in opensmile every observation is made in 0.01 seconds
    indexes = []
    if face:
        alpha = 0.03333
    else:
        alpha = 0.01
    for timestep in timesteps:
        indexes.append(int(timestep / alpha))
    return indexes


def read_openface(file_name, timesteps):
    df = pd.read_csv("data/openface/" + file_name)
    indexes = calculate_indexes(True, timesteps)
    openface_results.append(df[openface_features].loc[indexes])


def read_opensmile(filename, timesteps):
    df = pd.read_csv("data/opensmile/" + filename)
    indexes = calculate_indexes(False, timesteps)
    opensmile_results.append(df[opensmile_features].loc[indexes])


def get_all_data(log_files, openfiles):
    # Get all the data from the excel files
    timesteps = []
    for file_name in log_files:
        timesteps.append(read_logs(file_name))
    for i in range(len(openfiles)):
        read_openface(openfiles[i], timesteps[i])
        read_opensmile(openfiles[i], timesteps[i])

    merge_results()


def merge_results():
    # Convert the data to excel
    data_columns.remove("observations")
    df = pd.DataFrame(res, columns=observation_features + data_columns)

    openface_df = pd.concat(openface_results).reset_index(drop=True)
    opensmile_df = pd.concat(opensmile_results).reset_index(drop=True)

    df = pd.concat([df, openface_df, opensmile_df], axis=1)

    df.to_excel("data/results/result.xlsx", index=False)
    return df


def read_rosas_sav(filename="data/evaluations/ROSAS_PANAS_WAI.sav"):
    df = pd.read_spss(filename)
    return df


def get_reward_rosas(logs):
    df = read_rosas_sav()

    tmp= []
    for log in logs:
        tmp.append(read_logs(log,True))

    rosas_list = df[["ROSAS_1", "ROSAS_2", "ROSAS_3", "ROSAS_4"]].values.flatten().tolist()
    panas_list = df[["PANAS_1", "PANAS_2", "PANAS_3", "PANAS_4"]].values.flatten().tolist()
    wai_list = df[["WAI_1", "WAI_2", "WAI_3", "WAI_4"]].values.flatten().tolist()

    df = pd.DataFrame({
        "rewards": tmp,
        "rosas":rosas_list,
        "panas":panas_list,
        "wai":wai_list
    })
    df.to_excel("data/results/reward_rosas.xlsx", index=False)

# logs = get_filenames("logs")
# openfiles = get_filenames("openface")


# get_all_data(logs, openfiles)
# read_excel(file_names[1])

# ----------------------------

logs = get_filenames("logs")
get_reward_rosas(logs)
