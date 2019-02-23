from torchvision import transforms
from PIL import Image


train_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224), interpolation=Image.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
