import os
import torch

os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from models.llmtime import get_llmtime_predictions_data

plt.style.use("ggplot")


def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict["median"]
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label="Truth", color="black")
    plt.plot(pred, label=model_name, color="purple")
    # shade 90% confidence interval
    samples = pred_dict["samples"]
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color="purple")
    # plot median prediction line
    plt.plot(pred.index, np.quantile(samples, 0.5, axis=0), color="purple")
    # plot 10 random samples
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


print(torch.cuda.max_memory_allocated())
print()

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(
        base=10, prec=3, signed=True, time_sep=", ", bit_sep="", minus_sign="-"
    ),
)

mistral_api_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(
        base=10, prec=3, signed=True, time_sep=", ", bit_sep="", minus_sign="-"
    ),
)

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
)


llma2_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True),
)


promptcast_hypers = dict(
    temp=0.7,
    settings=SerializerSettings(
        base=10,
        prec=0,
        signed=True,
        time_sep=", ",
        bit_sep="",
        plus_sign="",
        minus_sign="-",
        half_bin_correction=False,
        decimal_point="",
    ),
)

arima_hypers = dict(p=[12, 30], d=[1, 2], q=[0])

model_hypers = {
    "LLMTime GPT-3.5": {"model": "gpt-3.5-turbo-instruct", **gpt3_hypers},
    "LLMTime GPT-4": {"model": "gpt-4", **gpt4_hypers},
    "LLMTime GPT-3": {"model": "text-davinci-003", **gpt3_hypers},
    "PromptCast GPT-3": {"model": "text-davinci-003", **promptcast_hypers},
    "LLMA2": {"model": "llama-7b", **llma2_hypers},
    "mistral": {"model": "mistral", **llma2_hypers},
    "mistral-api-tiny": {"model": "mistral-api-tiny", **mistral_api_hypers},
    "mistral-api-small": {"model": "mistral-api-tiny", **mistral_api_hypers},
    "mistral-api-medium": {"model": "mistral-api-tiny", **mistral_api_hypers},
    "ARIMA": arima_hypers,
}


model_predict_fns = {
    #'LLMA2': get_llmtime_predictions_data, ## had an issue with loading tokenizer
    #'mistral': get_llmtime_predictions_data, ## was too slow and didn't work
    "LLMTime GPT-4": get_llmtime_predictions_data,  ## exceeded current quota
    #'mistral-api-tiny': get_llmtime_predictions_data
    "LLMTime GPT-3.5": get_llmtime_predictions_data,  ## exceeded current quota
    #'PromptCast GPT-3': get_promptcast_predictions_data, ## davinci-003 was deprecated
    "ARIMA": get_arima_predictions_data,
}


model_names = list(model_predict_fns.keys())

## Load air passenger data
# datasets = get_datasets()
# ds_name = 'AirPassengersDataset'
# data = datasets[ds_name]
# train, test = data # or change to your own data

# Load COVID data
full_data = pd.read_csv(
    "https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/truth-Incident%20Hospitalizations.csv"
)
ds_name = "CovidHospDataset"
us_data = full_data.query("location == 'US'")
us_data.set_index("date", inplace=True)
pd.to_datetime(us_data.index)

## TODO: add start of for loop over forecast dates
# train = us_data["value"][:-60]  # beginning to last 60 rows
# test = us_data["value"][-60:]  # last 60 rows to end

train = us_data.iloc[: int(len(us_data) * 0.8)]  # 80% of data
train = train["value"]
test = us_data.iloc[int(len(us_data) * 0.8) :]  # 20% of data
test = test["value"]

## transform data using fourth root
train = np.power(train, 1 / 4)
test = np.power(test, 1 / 4)

## running more samples
## get other models running
## LLAMA didn't work
## mistral took a long time to load (will it do that every time?)
## gpt-4 worked out of the box
## gpt-3 was deprecated
## work on data pipeline, standardize ways to extract/filter data
## relabeling the index/Date column for the series
## create for loop over different times
## save output to files
## create a hub directory set-up within the project
## code up output file to match "hub" output
## run more samples


out = {}
for model in model_names:  # GPT-4 takes a about a minute to run
    model_hypers[model].update({"dataset_name": ds_name})  # for promptcast
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 6
    pred_dict = get_autotuned_predictions_data(
        train,
        test,
        hypers,
        num_samples,
        model_predict_fns[model],
        verbose=False,
        parallel=False,
    )
    pred_dict["samples"] = np.power(pred_dict["samples"], 4)
    out[model] = pred_dict
    plot_preds(
        np.power(train, 4), np.power(test, 4), pred_dict, model, show_samples=True
    )
