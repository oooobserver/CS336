import torch

# with mean 0 and variance 1 (also called the standard normal distribution).
x = torch.randn(2000, 3000)

mean = sum(x.flatten()) / len(x.flatten())
print("Mean:", mean)

variance = sum((x.flatten() - mean) ** 2) / len(x.flatten())
print("Variance:", variance)

# for val in x.flatten():
#     print(val.item())

x = torch.tensor([[1.0, 2, 3], [4, 5, 6]])  # @inspect x
x = torch.zeros(4, 8)  # 4x8 matrix of all zeros @inspect x
x = torch.ones(4, 8)  # 4x8 matrix of all ones @inspect x
x = torch.randn(4, 8)  # 4x8 matrix of iid Normal(0, 1) samples @inspect x

print(x.device)
print(torch.cuda.is_available())


linear = torch.nn.Linear(8, 16)
linear.load_state_dict({"weight": torch.randn(16, 8)})
print(linear)
