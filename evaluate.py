import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import EmotionCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_ds = datasets.ImageFolder("data/test", transform=transform)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = EmotionCNN(num_classes=len(test_ds.classes)).to(DEVICE)
model.load_state_dict(torch.load("emotion_cnn.pth", map_location=DEVICE))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total:.2%}")
