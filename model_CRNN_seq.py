import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
abc = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25
}

inverted_abc = {v: k for k, v in abc.items()}

class CRNN(nn.Module):
    def __init__(self, cnn_model, rnn_hidden_size, num_classes):
        super(CRNN, self).__init__()
        self.cnn = cnn_model
        self.rnn = nn.GRU(input_size=72, hidden_size=rnn_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden_size*2, num_classes)

    def forward(self, x):
        # Extract features with the CNN
        batch_size, _,  c, h, w = x.size()
        x = x.view(batch_size, c, h, w)
        x = self.cnn(x)
        # Reshape the output from the CNN
        seq_length = batch_size
        batch_size = 1
        x = x.view(batch_size, seq_length, -1)
        # Pass the features to the RNN
        x, _ = self.rnn(x)
        # Use the output from the RNN for classification
        x = self.fc(x)
        #softmax
        x = F.log_softmax(x, dim=2)
        return x

# Define your CNN model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 3 * 3, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 8 * 3 * 3)
        #x = self.fc(x)
        return x

# Initialize the model
cnn_model = SimpleCNN()

# Load the state dictionary
state_dict = torch.load('/home/xnmaster/XNAPproject-grup_08/model_CNN/results/dataset2/trained_model.pth', map_location=torch.device('cpu'))
#state_dict = torch.load('/home/xnmaster/XNAPproject-grup_08/model_CNN/results/dataset2/trained_model.pth')
cnn_model.load_state_dict(state_dict)

# Create the CRNN model
crnn_model = CRNN(cnn_model, rnn_hidden_size=256, num_classes=27)  # 26 letters + 1 for blank


# Define the transformations
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to 32x32
    transforms.ToTensor(),  # Convert to tensor
])

class ImageFolderWithLengths(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform = None, loader=datasets.folder.default_loader, is_valid_file=None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.abc = abc  # Your class dictionary

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # The length of the word is the number of characters in the class name
        length = len(self.classes[target])
        return img, target, length

    def target_transform(self, target):
        class_name = self.classes[target]
        return self.abc[class_name]

import os
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=ToTensor(), target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.imgs = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for dirname in dirnames:
                filenames = sorted(os.listdir(os.path.join(dirpath, dirname)))
                for filename in filenames:
                    path = os.path.join(dirpath, dirname, filename)
                    value_before_dot = filename.rsplit('.', 1)[0][-1]
                    target = self.get_target(dirname, value_before_dot)
                    length = len(filename)
                    self.imgs.append((path, target, length))


    def get_target(self, path, index):
        # Extract the specific character from the path and convert it to your target
        # This is just an example, adjust this to fit your needs
        target_char = path[int(index)]  
        if target_char in "- '":
            target= 26
        else:
            target = abc[target_char]  # Convert the character to an integer
        return target

    def __getitem__(self, index):
        path, target, length = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, length

    def __len__(self):
        return len(self.imgs)

# Create the datasets
train_dataset = CustomDataset(root_dir='/home/xnmaster/XNAPproject-grup_08/words_segmented_test', transform=transform)
val_dataset = CustomDataset(root_dir='/home/xnmaster/XNAPproject-grup_08/words_segmented_test', transform=transform)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Define the loss function and the optimizer
loss_function = CTCLoss(blank=26, zero_infinity=False)  # Assuming blank label is 26
optimizer = optim.Adam(crnn_model.parameters())

def train():
    # Training loop
    lib_images = []
    lib_labels = []
    lib_lengths = []

    word_images = []
    word_labels = []
    word_lengths = []
    lns = []
    for i, (images, labels, lengths) in enumerate(train_loader):
        lns.append(lengths)
        # If the current word is finished, process the batch and start a new one
        if i > 0 and lns[i-1] != lns[i]:
            word_images = torch.stack(word_images)
            word_labels = torch.stack(word_labels)
            word_lengths = torch.stack(word_lengths)
            # Move the images and labels to the GPU if available
            lib_images.append(word_images)
            lib_labels.append(word_labels) 
            lib_lengths.append(word_lengths)
            
            word_images = []
            word_labels = []
            word_lengths = []
        word_images.append(images)
        word_labels.append(labels)
        word_lengths.append(lengths)

        # process the batch
    lib_images = lib_images[1:]
    lib_labels = lib_labels[1:]
    lib_lengths = lib_lengths[1:]
    for epoch in range (10):
        for word_images, word_labels, word_lengths in zip(lib_images, lib_labels, lib_lengths):
            word_images = word_images.to(device)
            word_labels = word_labels.to(device)
            word_lengths = word_lengths.to(device)
            outputs = crnn_model(word_images)
            outputs = outputs.transpose(0, 1)
            input_lengths = torch.tensor([outputs.size(0)], dtype=torch.long)        
            target_lengths = torch.tensor([word_labels.size(0)], dtype=torch.long)
            word_labels = word_labels.unsqueeze(0)
            word_labels = word_labels.view(-1)
            loss = loss_function(outputs, word_labels, input_lengths, target_lengths)

        # Backward pass and optimization
        print("Logits min:", outputs.min().item(), "max:", outputs.max().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f'Epoch [{epoch + 1}/10], Step [{epoch + 1}/{len(train_loader)}], Loss: {loss.item()}')

        

    # Save the model
    torch.save(crnn_model.state_dict(), 'crnn_model.pth')

train()

