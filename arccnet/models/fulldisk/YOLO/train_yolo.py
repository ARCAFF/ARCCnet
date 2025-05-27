import comet_ml
from ultralytics import YOLO

comet_ml.login(project_name="arcaff-v2-fulldisk-detection-classification", workspace="arcaff")


def main():
    """
    Train YOLO model for solar active region detection.

    This function loads a pretrained YOLO model and trains it on
    the ARCCnet fulldisk dataset for solar active region detection.
    """
    model = YOLO("yolo11x.pt")  # load a pretrained model

    # Define training arguments
    train_args = {
        "data": "config.yaml",
        "imgsz": 1024,  # Image size
        "batch": 32,
        "epochs": 10000,
        "device": [0, 1, 2, 3],
        "patience": 500,
        "dropout": 0.25,
        "fliplr": 0.5,
    }

    model.train(**train_args)


if __name__ == "__main__":
    main()
