import pickle
import torch

pkl_path = "weight/senet50_ft_weight.pkl"
pth_path = "weight/senet50_vggface2.pth"

with open(pkl_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

if "state_dict" in data:
    state_dict = data["state_dict"]
else:
    state_dict = data

torch.save(state_dict, pth_path)

print(f"Converted and saved as {pth_path}")