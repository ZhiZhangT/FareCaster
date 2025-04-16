import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import (
    LSTM,
    TimesNet,
    Autoformer,
    NLinear,
    DLinear,
    NBEATS,
    NHITS,
)
from neuralforecast.losses.pytorch import MAE, MSE

import torch
import random
import constants
import os
import sys
import matplotlib.pyplot as plt
import json
from utils.logging import create_run_directory, Logger, plot_losses

if __name__ == "__main__":
    # hyperparameters
    # number of steps to predict
    horizon = 15
    batch_size = 32
    num_epochs = 50
    # number of batches to run before updating trainable parameters
    val_check_steps = 5103
    # max_steps == number of batches per epoch * number of epochs
    max_steps = val_check_steps * num_epochs
    # based on empirical findings, use either 0.0005 or 0.0010
    lr = 0.0005
    # based on empirical findings
    encoder_n_layers = 1
    decoder_layers = 1
    encoder_hidden_size = 16  # memory vector size
    decoder_hidden_size = 64  # fully connected layer size
    # see here for more information: https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html#3-training-with-exogenous-variables
    stat_exog_list = [
        "startingAirport_0",
        "startingAirport_1",
        "startingAirport_2",
        "startingAirport_3",
        "startingAirport_4",
        "startingAirport_5",
        "startingAirport_6",
        "startingAirport_7",
        "startingAirport_8",
        "startingAirport_9",
        "startingAirport_10",
        "startingAirport_11",
        "startingAirport_12",
        "startingAirport_13",
        "startingAirport_14",
        "startingAirport_15",
        "destinationAirport_0",
        "destinationAirport_1",
        "destinationAirport_2",
        "destinationAirport_3",
        "destinationAirport_4",
        "destinationAirport_5",
        "destinationAirport_6",
        "destinationAirport_7",
        "destinationAirport_8",
        "destinationAirport_9",
        "destinationAirport_10",
        "destinationAirport_11",
        "destinationAirport_12",
        "destinationAirport_13",
        "destinationAirport_14",
        "destinationAirport_15",
    ]
    hist_exog_list = [
        "searchDayOfWeek",
        "flightDayOfWeek",
        "daysBetweenSearchAndFlight",
        "searchMonth",
        "flightMonth",
        "departureHourUTC",
        "seatsRemaining",
    ]
    train_loss_criteria = MSE()
    val_loss_criteria = MSE()

    train_loss_criteria_name = train_loss_criteria.__class__.__name__
    val_loss_criteria_name = val_loss_criteria.__class__.__name__

    run_name = f"SOTA_SlidingWindow_{train_loss_criteria_name}LossTrain_{val_loss_criteria_name}LossVal"

    # Create run directory
    run_dir = create_run_directory(run_name)

    print(f"Starting training run in directory: {run_dir}")

    # save common hyperparameters
    summary = {
        "hyperparameters": {
            "common": {
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "sequence_length": horizon,
                "encoder_n_layers": encoder_n_layers,
                "decoder_layers": decoder_layers,
                "encoder_hidden_size": encoder_hidden_size,
                "decoder_hidden_size": decoder_hidden_size,
                "use_sliding_window": True,  # only sliding window data is available
                "stat_exog_list": stat_exog_list,
                "hist_exog_list": hist_exog_list,
                "train_loss_criteria": train_loss_criteria_name,
                "val_loss_criteria": val_loss_criteria_name,
            },
        }
    }

    # Set up logging to capture all print statements
    sys.stdout = Logger(os.path.join(run_dir, "training_log.txt"))

    random.seed(constants.RANDOM_STATE)

    df = pd.read_csv("saved/train.csv")
    static_df = pd.read_csv("saved/static.csv")
    df_val_first = pd.read_csv("saved/validation_first.csv")
    df_val_second = pd.read_csv("saved/validation_second.csv")

    df["ds"] = pd.to_datetime(df["ds"])
    df_val_first["ds"] = pd.to_datetime(df_val_first["ds"])
    df_val_second["ds"] = pd.to_datetime(df_val_second["ds"])

    print(f"Shape of df: {df.shape}")
    print(f"Shape of df_val: {df_val_first.shape}")

    models = [
        NBEATS(
            h=horizon,
            input_size=horizon,
            batch_size=batch_size,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            learning_rate=lr,
            loss=train_loss_criteria,
            random_seed=constants.RANDOM_STATE,
        ),
        NHITS(
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
            h=horizon,
            input_size=horizon,
            batch_size=batch_size,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            learning_rate=lr,
            loss=train_loss_criteria,
            random_seed=constants.RANDOM_STATE,
        ),
        LSTM(
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            h=horizon,
            input_size=horizon,
            batch_size=batch_size,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            learning_rate=lr,
            loss=train_loss_criteria,
            random_seed=constants.RANDOM_STATE,
            encoder_n_layers=encoder_n_layers,
            decoder_layers=decoder_layers,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
        ),
        # NLinear does not actually support future/static exogenous variables
        NLinear(
            h=horizon,
            input_size=horizon,
            batch_size=batch_size,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            learning_rate=lr,
            loss=train_loss_criteria,
            random_seed=constants.RANDOM_STATE,
        ),
        # DLinear does not actually support future/static exogenous variables
        DLinear(
            h=horizon,
            input_size=horizon,
            batch_size=batch_size,
            max_steps=max_steps,
            val_check_steps=val_check_steps,
            learning_rate=lr,
            loss=train_loss_criteria,
            random_seed=constants.RANDOM_STATE,
        ),
        # Autoformer(
        #     stat_exog_list=stat_exog_list,
        #     hist_exog_list=hist_exog_list,
        #     h=horizon,
        #     input_size=horizon,
        #     batch_size=batch_size,
        #     max_steps=max_steps,
        #     val_check_steps=val_check_steps,
        #     learning_rate=lr,
        #     loss=train_loss_criteria,
        #     random_seed=constants.RANDOM_STATE,
        # ),
        # TimesNet(
        #     stat_exog_list=stat_exog_list,
        #     hist_exog_list=hist_exog_list,
        #     h=horizon,  # number of steps to predict
        #     input_size=horizon,  # length of input sequence
        #     batch_size=batch_size,
        #     max_steps=max_steps,
        #     val_check_steps=val_check_steps,
        #     learning_rate=lr,
        #     loss=train_loss_criteria,
        #     hidden_size=32,
        #     conv_hidden_size=32,
        #     random_seed=constants.RANDOM_STATE,
        # ),
    ]

    # TRAIN THE MODEL
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df=df, static_df=static_df)
    nf.save(path=f"{run_dir}/nixtla")

    # TEST MODEL ON VALIDATION SET
    Y_hat_df_val = nf.predict(df=df_val_first, static_df=static_df)
    print(f"Shape of Y_hat_df_val: {Y_hat_df_val.shape}")
    # save the predictions to a CSV file
    Y_hat_df_val.to_csv(f"{run_dir}/validation_predictions.csv", index=False)
    ground_truth_tensor = torch.tensor(df_val_second["y"].values, dtype=torch.float32)

    results = dict()
    for model in nf.models:
        model_name = model.__class__.__name__
        train_losses = model.train_trajectories
        # train_losses shape is [(epoch, loss)]
        # get only the loss values
        train_losses = [loss[1] for loss in train_losses]
        valid_losses = nf.models[0].valid_trajectories
        valid_losses = [loss[1] for loss in valid_losses]
        plot_losses(
            losses={
                "train_losses": train_losses,
                "val_losses": valid_losses,
            },
            save_dir=run_dir,
            model_type=model_name,
        )

        predicted_tensor = torch.tensor(
            Y_hat_df_val[model_name].values, dtype=torch.float32
        )

        # NOTE: both val_mae and val_loss are 1D tensors
        val_mae = MAE()(y=ground_truth_tensor, y_hat=predicted_tensor)

        val_loss = val_loss_criteria(
            y=ground_truth_tensor,
            y_hat=predicted_tensor,
            y_insample=None,  # the y_insample seems to be a bug in the library because it is not used
        )
        print(
            f"BEST {model_name} Model Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
        )
        results[model_name] = {
            "best_val_loss": val_loss.item(),
            "best_val_mae": val_mae.item(),
        }

    # save experiment results on validation set
    summary["results"] = results

    with open(os.path.join(run_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Plot comparative bar chart for val MAE
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    mae_values = [results[m]["best_val_mae"] for m in model_names]
    # Use a subset of colors based on the number of models
    colors = ["blue", "green", "purple", "red", "yellow"][: len(model_names)]
    plt.bar(model_names, mae_values, color=colors)
    plt.title("Model Comparison: Val MAE")
    plt.ylabel("Mean Absolute Error")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(os.path.join(run_dir, "model_comparison.png"))
    plt.close()

    # Close the logger
    sys.stdout.close()
    sys.stdout = sys.__stdout__
