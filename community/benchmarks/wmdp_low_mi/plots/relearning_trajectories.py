# %%
import pickle
import time

import matplotlib.pyplot as plt

import wandb

plt.style.use("default")
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 10


# %%
# === CELL 1: Load runs from wandb ===

api = wandb.Api(timeout=3600)
unlearn_project_name = "filyp/open-unlearning"
relearn_project_name = "filyp/rel-open-unlearning"

# %%
method_names = [
    "GradDiff",
    "NPO",
    "RMU",
    "SimNPO",
    "UNDIAL",
    "CIR",
    "NPOstrict",
    "CIRstrict",
]
version = "v3"

split = "bio"
# split = "cyber"

metrics_names = [
    "train/recall_loss",
    "train/forget_acc_t0",
    "train/forget_acc_t1",
    "train/wikitext_loss",
    "train/wikitext_kl",
]
file_name = f"{version}_3B__{split}"

# %% load from wandb

method_histories = {}
for method_name in method_names:
    runs = api.runs(
        relearn_project_name,
        filters={
            "display_name": {"$regex": f"^{version}_3B_{method_name}_{split}_(\d+)$"}
        },
    )

    method_histories[method_name] = []
    for relearn_run in runs:
        print(relearn_run.name)
        
        # get this same run from the unlearn project
        unlearn_runs = api.runs(
            unlearn_project_name,
            filters={"display_name": relearn_run.name}
        )
        assert len(unlearn_runs) == 1
        unlearn_run = unlearn_runs[0]

        for i in range(10):
            try:
                relearn_history = relearn_run.history(keys=metrics_names)
                unlearn_history = unlearn_run.history(keys=metrics_names)
                break
            except Exception as e:
                print(f"{i}: Error loading history for {relearn_run.name}: {e}")
                time.sleep(2**i)
        method_histories[method_name].append((unlearn_history, relearn_history))


with open(f"{file_name}.pkl", "wb") as f:
    pickle.dump(method_histories, f)

# %% simply load instead from a file
with open(f"{file_name}.pkl", "rb") as f:
    method_histories = pickle.load(f)

# %%
# Create a color map for each method
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_map = {method: colors[i % len(colors)] for i, method in enumerate(method_names)}

# metric_name = "train/forget_acc_t1"
metric_name = "train/recall_loss"
# metric_name = "train/wikitext_kl"
# metric_name = "train/wikitext_loss"

# # ! all trajectories
# for method_name in method_names:
#     for i, (unl_hist, rel_hist) in enumerate(method_histories[method_name]):
#         args = {"label": method_name} if i == 0 else dict()
#         plt.plot(rel_hist[metric_name], color=color_map[method_name], alpha=1, **args)

# # # ! only the average trajectory
# for method_name in method_names:
#     for i, (unl_hist, rel_hist) in enumerate(method_histories[method_name]):
#         if i == 0:
#             avg = np.array(rel_hist[metric_name])
#         else:
#             avg += np.array(rel_hist[metric_name])
#     args = {"label": method_name}
#     plt.plot(avg / len(method_histories[method_name]), color=color_map[method_name], alpha=1, **args)
    
# # ! top n
n = 1
for method_name in method_names:
    score_and_hist = []
    for i, (unl_hist, rel_hist) in enumerate(method_histories[method_name]):
        # score = history["train/forget_acc_t1"].iloc[-1]
        score = max(rel_hist["train/forget_acc_t1"])
        score_and_hist.append((score, unl_hist, rel_hist))
    score_and_hist.sort(key=lambda x: x[0])
    for i, (score, unl_hist, rel_hist) in enumerate(score_and_hist[:n]):
        args = {"label": method_name} if i == 0 else dict()
        plt.plot(rel_hist[metric_name], color=color_map[method_name], alpha=1, **args)

plt.legend()

# %%
