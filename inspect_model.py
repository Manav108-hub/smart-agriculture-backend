import torch

def inspect_checkpoint(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    print("State dict keys:")
    for key, value in state_dict.items():
        print(f" - {key}: {value.shape}")

# Run on your trained model
if __name__ == "__main__":
    inspect_checkpoint("saved_models/Crop_Yield_best.pth")