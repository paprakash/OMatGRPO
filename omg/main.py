from omg import __version__
from omg.omg_cli import OMGCLI
from omg.omg_lightning import OMGLightning
from omg.omg_trainer import OMGTrainer
from omg.datamodule.dataloader import OMGDataModule


def main():
    """Main function to run the Open Materials Generation (OMatG) command line interface."""
    OMGCLI(model_class=OMGLightning, datamodule_class=OMGDataModule, trainer_class=OMGTrainer,
           parser_kwargs={"description": f"Open Materials Generation (OMatG) Version {__version__}"})


if __name__ == "__main__":
    main()
