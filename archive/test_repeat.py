"""Test tensor repeat operation."""

import torch

# Simulate what happens in training
grade_targets = torch.tensor([1])  # Shape: (1,)
num_slices = 5
batch_size = 1

print("Original grade_targets:", grade_targets)
print("Shape:", grade_targets.shape)
print("Value:", grade_targets.item())

# This is what we do in training
grade_targets_expanded = grade_targets.repeat(num_slices * batch_size)

print("\nAfter repeat:")
print("Expanded:", grade_targets_expanded)
print("Shape:", grade_targets_expanded.shape)
print("Values:", grade_targets_expanded.tolist())

# Check if any value is out of bounds
if (grade_targets_expanded > 1).any():
    print("\n⚠️ WARNING: Values > 1 detected!")
else:
    print("\n✅ All values are valid (0 or 1)")
