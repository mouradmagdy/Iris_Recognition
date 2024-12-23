import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from PIL import Image
import random
import os
import cv2
from PIL import Image
import pickle

def read_img(path):
    img = cv2.imread(path,0)
    tensor = torch.tensor(img, dtype=torch.float32)
    return tensor

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Use mean/std for grayscale
])

def extract_embeddings(image,model,transform):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0)
        image = image.cuda()
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        embeddings = model.forward_one(image)
    return embeddings  

def init_verification_database():
    if os.path.exists("reg_users.txt"):
        print("Database loaded successfully!")
    global registered_users
    with open("reg_users.txt", "rb") as file:
            registered_users = pickle.load(file)
    print(registered_users)

def init_recognition_database():
    if os.path.exists("reg_users2.txt"):
        print("Database loaded successfully!")
    global registered_users
    with open("reg_users2.txt", "rb") as file:
            registered_users = pickle.load(file)
    print(registered_users)

def register_user_verification(user_id, img1path, transform):
    registered_users[user_id] = img1path
    print(f"User '{user_id}' registered successfully!")
    with open("reg_users.txt", "wb") as file:
      pickle.dump(registered_users, file)

def register_user_recognition(user_id, img1path, transform):
    img = Image.open(img1path).convert('L') 
    features = extract_embeddings(img,model,transform)
    registered_users[user_id] = features
    print(f"User '{user_id}' registered successfully!")
    with open("reg_users2.txt", "wb") as file:
      pickle.dump(registered_users, file)


def authenticate_user_recognition(test_img_pth, model, transform, threshold=2.6):
    ## open image, extract_embeddings , iterate overall database , minimzie 3al distance
    target_img = Image.open(test_img_pth).convert('L')  
    target_features = extract_embeddings(target_img,model,transform)
    
    best_match = None
    best_score = float('inf')

    for user_id, reg_img_embedding in registered_users.items():
        score = F.pairwise_distance(target_features,reg_img_embedding).item()
        if score < best_score:
            best_match = user_id
            best_score = score
            print("score",score)
    
    print("Best score",best_score)
    if best_score < threshold:
        return best_match,best_score
    else:
        return None,best_score
        

def authenticate_user_verification(user_id,test_img_pth, model, transform, threshold=0.5):
    test_img = Image.open(test_img_pth).convert('L')  
    test_img = transform(test_img).unsqueeze(0).to(device="cuda")
    model.eval()
    if user_id not in registered_users:
         return "Access Denied: User not registered in the database!"
    reg_img = Image.open(registered_users[user_id]).convert('L')  
    reg_img = transform(reg_img).unsqueeze(0).to(device="cuda")
    with torch.no_grad():
        output = model(test_img, reg_img).squeeze()
        predictions = torch.sigmoid(output) > 0.5  # Threshold at 0.5
        # print(torch.sigmoid(output))
    if predictions > threshold:
        return f"Access Granted: Welcome {user_id}!"

    return "Access Denied: User not recognized!"



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Using a pretrained model (ResNet18) as the feature extractor
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove the classification head

        # Add a fully connected layer for similarity computation
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward_one(self, x):
        return self.feature_extractor(x)

    def forward(self, img1, img2):
        if img1.shape[1] == 1:  # Check if the input is grayscale
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        similarity = self.fc(torch.abs(feat1 - feat2))
        return similarity


if __name__ == '__main__':
#   init_verification_database()
#   init_recognition_database()

  model = SiameseNetwork().cuda()
  model.load_state_dict(torch.load("siamese_model.pth"))
  model.eval()
#   print(authenticate_user_recognition("G:\\4th year biomedical\\Biometrics\\code\\CASIA-Iris-Thousand\\CASIA-Iris-Thousand\\406\\L\\S5406L07.jpg",model,transform))
#   register_user("S5630L09","G:\\4th year biomedical\\Biometrics\\code\\CASIA-Iris-Thousand\\CASIA-Iris-Thousand\\630\\L\\S5630L09.jpg",transform)
#   register_user("mariam","S3_R1.jpeg",transform)
#   print(authenticate_user(test_img_pth="G:\\4th year biomedical\\Biometrics\\code\\CASIA-Iris-Thousand\\CASIA-Iris-Thousand\\157\\L\\S5157L01.jpg",model=model,transform=transform))
#   print(authenticate_user(user_id="S5961",test_img_pth="G:\\4th year biomedical\\Biometrics\\code\\CASIA-Iris-Thousand\\CASIA-Iris-Thousand\\962\\L\\S5962L01.jpg",model=model,transform=transform))
 