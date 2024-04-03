import os

os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.facecolor"] = "white"
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data

from dotenv import load_dotenv

load_dotenv(override=True)


def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict["median"]
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label="Truth", color="black")
    plt.plot(pred, label=model_name, color="purple")
    plt.axes().set_facecolor("white")
    # shade 90% confidence interval
    samples = pred_dict["samples"]
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color="purple")
    if show_samples:
        samples = pred_dict["samples"]
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color="purple", alpha=0.3, linewidth=1)
    plt.legend(loc="upper left")
    if "NLL/D" in pred_dict:
        nll = pred_dict["NLL/D"]
        if nll is not None:
            plt.text(
                0.03,
                0.85,
                f"NLL/D: {nll:.2f}",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.5),
            )
    plt.show()


arima_hypers = dict(p=[12, 30], d=[1, 2], q=[0])

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
)

model_hypers = {
    "LLMTime GPT-3.5": {"model": "gpt-3.5-turbo-instruct", **gpt3_hypers},
    #'LLMTime GPT-4': {'model': 'gpt-4', **gpt4_hypers},
    #'LLMTime GPT-3': {'model': 'text-davinci-003', **gpt3_hypers},
    #'PromptCast GPT-3': {'model': 'text-davinci-003', **promptcast_hypers},
    #'PromptCast GPT-3': {'model': 'gpt-3.5-turbo-instruct', **promptcast_hypers},
    "ARIMA": arima_hypers,
}

model_predict_fns = {
    "LLMTime GPT-3.5": get_llmtime_predictions_data,
    # 'LLMTime GPT-4': get_llmtime_predictions_data,
    # 'PromptCast GPT-3': get_promptcast_predictions_data,
    "ARIMA": get_arima_predictions_data,
}

model_names = list(model_predict_fns.keys())

datasets = get_datasets()
ds_name = "AirPassengersDataset"

data = datasets[ds_name]
train, test = data  # or change to your own data

# fit into a straight line
new_train_values = list(
    range(int(train.values.min()), int(train.values.min()) + len(train))
)
new_train = pd.Series(new_train_values, index=train.index)
new_test_values = list(
    range(int(new_train.values.max()), int(new_train.values.max()) + len(test))
)
new_test = pd.Series(new_test_values, index=test.index)

out = {}
for model in model_names:  # GPT-4 takes a about a minute to run
    model_hypers[model].update({"dataset_name": ds_name})  # for promptcast
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 10
    pred_dict = get_autotuned_predictions_data(
        new_train,
        new_test,
        hypers,
        num_samples,
        model_predict_fns[model],
        verbose=False,
        parallel=False,
    )
    out[model] = pred_dict
    plot_preds(new_train, new_test, pred_dict, model, show_samples=True)

# fit into a curve
quad_new_train_values = [
    v * v for v in range(int(train.values.min()), int(train.values.min()) + len(train))
]
new_train = pd.Series(quad_new_train_values, index=train.index)
quad_new_test_values = [
    v * v
    for v in range(int(new_train.values.max()), int(new_train.values.max()) + len(test))
]
new_test = pd.Series(quad_new_test_values, index=test.index)

for model in model_names:  # GPT-4 takes a about a minute to run
    model_hypers[model].update({"dataset_name": ds_name})  # for promptcast
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 10
    pred_dict = get_autotuned_predictions_data(
        new_train,
        new_test,
        hypers,
        num_samples,
        model_predict_fns[model],
        verbose=False,
        parallel=False,
    )
    out[model] = pred_dict
    plot_preds(new_train, new_test, pred_dict, model, show_samples=True)
