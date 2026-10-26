import warnings

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from src.data.utils import in_channels_from_data_type, map_data_dir_to_dataset
from src.model.lightning_model import UNetLightningModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.init_args.data_dir",
            "model.dataset",
            compute_fn=map_data_dir_to_dataset,
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.init_args.data_type",
            "model.in_channels",
            compute_fn=in_channels_from_data_type,
            apply_on="instantiate",
        )


def main():
    warnings.filterwarnings("ignore", module="rasterio")
    cli = MyLightningCLI(UNetLightningModule)


if __name__ == "__main__":
    main()
