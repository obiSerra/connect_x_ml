import csv
import glob
import os
import re

import pandas
from stable_baselines3 import PPO

from lookahead import multistep_agent, multistep_agent_factory
from utils import (ConnectFourGym, TqdmCallback, agent_factory,
                   get_win_percentages)

models = ["PPO_custom_p1", "PPO_custom_p2",
          "PPO", "PPO_custom_p3", "PPO_custom_p4", "PPO_custom_p5",
          "PPO_custom_p6", "PPO_custom_p7", "PPO_custom_p8", "PPO_custom_p9"]


header = ['model', 'step', 'vs random', 'vs random invalid', 'vs lookahead',
          'vs lookahead invalid', 'vs negamax', 'vs negamax invalid']
data = []


def save_data(data_list):
    with open('data/models_benchmark.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        for data in data_list:
            # write the data
            writer.writerow(data)


df = pandas.read_csv('data/models_benchmark.csv')

adv_agent = multistep_agent_factory()
models_dir = "models"
for model_name in models:
    print(f"[+] Testing model {model_name}")

    search = re.compile(f"{models_dir}/{model_name}/([0-9]*)\.zip")

    saved_versions = glob.glob(f"{models_dir}/{model_name}/*.zip")
    versions = [search.match(version)[1]
                for version in saved_versions if search.match(version)]

    versions.sort(key=lambda x: int(x))

    model = None
    for version in versions:
        print(f"    Version {version}")
        model_exists = len(df[df['model'] + "_" + df['step'].astype(str)
                              == model_name + "_" + version]) > 0
        if not model_exists:
            model = PPO.load(f"{models_dir}/{model_name}/{version}.zip")
            agent = agent_factory(model)
            print("     vs random")
            vs_random = get_win_percentages(agent1=agent, agent2="random")
            print(f"     win: {vs_random[0][0]} | invalid: {vs_random[0][1]}")
            print("     vs adv_agent")
            vs_lookahead = get_win_percentages(agent1=agent, agent2=adv_agent)
            print(
                f"     win: {vs_lookahead[0][0]} | invalid: {vs_lookahead[0][1]}")
            print("     vs negamax")
            vs_negamax = get_win_percentages(agent1=agent, agent2="negamax")
            print(
                f"     win: {vs_negamax[0][0]} | invalid: {vs_negamax[0][1]}")

            row = [model_name, version, vs_random[0][0], vs_random[0][1],
                   vs_lookahead[0][0], vs_lookahead[0][1], vs_negamax[0][0],
                   vs_negamax[0][1]]

            data.append(row)

save_data(data)
