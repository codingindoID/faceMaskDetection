import os

with_mask = len(os.listdir("dataset/with_mask"))
without_mask = len(os.listdir("dataset/without_mask"))

print("With Mask:", with_mask)
print("Without Mask:", without_mask)