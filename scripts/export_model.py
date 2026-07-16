import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.constants import LABEL_COLS
from src.models import build_model


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def export_torchscript(config_path: str):
    cfg = load_config(config_path)

    checkpoint_path = cfg["outputs"]["best_checkpoint"]
    output_path = cfg["outputs"]["torchscript_model"]
    final_size = tuple(cfg["preprocessing"]["final_size"])

    model = build_model(
        model_name=cfg["training"]["model_name"],
        n_outputs=len(LABEL_COLS),
        backbone=cfg["training"].get("backbone", "resnet34"),
        dropout=cfg["training"].get("dropout", 0.3),
        pretrained=False
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    example_input = torch.randn(1, 1, *final_size)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(output_path)

    print(f"Exported TorchScript model to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet3d.yaml",
    )
    args = parser.parse_args()

    export_torchscript(args.config)


if __name__ == "__main__":
    main()
