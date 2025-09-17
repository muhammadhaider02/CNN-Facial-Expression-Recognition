import torch

sd = torch.load(
    "weight/senet50_vggface2.pth",
    map_location="cpu",
    weights_only=False
)

print(type(sd), len(sd))
print(list(sd.keys())[:10])