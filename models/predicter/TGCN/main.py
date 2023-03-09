import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import utils.callbacks
import models
import tasks
import utils.data
from pytorch_lightning.plugins import DDPPlugin

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "PeMS08": {"feat": "data/PeMS08.npy","feat2": "data/pems08_cm_random_mat_0.10%25.npy", "adj": "data/Adj.npy"},
}


def get_model(args, dm):
    model = None
    if args.model_name == "TGCN":
        model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val,feat_max_val2=dm.feat_max_val2, **vars(args)
    )
    return task

def get_callbacks(args):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
    ]
    return callbacks


def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"],feat_path2=DATA_PATHS[args.data]["feat2"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    callbacks = get_callbacks(args)
    task = get_task(args, model, dm)
    trainer=pl.Trainer.from_argparse_args(args,check_val_every_n_epoch=10,strategy=DDPPlugin(find_unused_parameters=False), gpus=2,callbacks=callbacks,max_epochs = 200)#speed up!
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm,ckpt_path='best')
    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    #parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", default="PeMS08"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN"),
        default="TGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    temp_args, _ = parser.parse_known_args()
    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)
    args = parser.parse_args()
    results = main(args)


