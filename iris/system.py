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
    transforms.Grayscale(num_output_channels=1),  
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
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
        print(f"Welcome {best_match} ")
        return best_match,best_score
    else:
        print("User is not recognized")
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
        predictions = torch.sigmoid(output) > 0.5  
    if predictions > threshold:
        return f"Access Granted!"

    return "Access Denied: User not recognized!"



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity() 

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward_one(self, x):
        return self.feature_extractor(x)

    def forward(self, img1, img2):
        if img1.shape[1] == 1:  
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        similarity = self.fc(torch.abs(feat1 - feat2))
        return similarity


if __name__ == '__main__':
  init_verification_database()
#   init_recognition_database()

  model = SiameseNetwork().cuda()
  model.load_state_dict(torch.load("iris/siamese_model.pth"))
  model.eval()
#   print(authenticate_user_verification(user_id="962",test_img_pth="C:/Users/user/Downloads/CASIA-Iris-Thousand/CASIA-Iris-Thousand/962/L/S5962L05.jpg",model=model,transform=transform))
  print(authenticate_user_verification(user_id="963",test_img_pth="C:/Users/user/Downloads/CASIA-Iris-Thousand/CASIA-Iris-Thousand/963/L/S5963L05.jpg",model=model,transform=transform))

 