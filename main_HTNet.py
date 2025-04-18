from os import path
import os
import numpy as np
import cv2
import dlib
import time
import pandas
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
from facenet_pytorch import MTCNN
from Model import HTNet  # Import your model from Model.py
from Model import Fusionmodel

# ------------------------------
# Utility functions and modules
# ------------------------------

def reset_weights(m):
    """Reset the weights of a network layer to avoid weight leakage."""
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # Uncomment the print statement for debugging if needed
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def preprocess_frame(frame):
    """
    Preprocess a single frame by converting to grayscale, applying CLAHE,
    detecting the face using dlib, and cropping to a 224x224 image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    detector = dlib.get_frontal_face_detector()
    rects = detector(enhanced, 1)
    if rects:
        rect = rects[0]
        face = enhanced[rect.top():rect.bottom(), rect.left():rect.right()]
        return cv2.resize(face, (224, 224))
    return cv2.resize(enhanced, (224, 224))

def temporal_interpolate(sequence, target_length=16):
    """
    Standardize a sequence (e.g., video frames) to a fixed target length.
    Useful when working with video samples where micro-expression durations vary.
    """
    original_length = len(sequence)
    indices = np.linspace(0, original_length - 1, target_length).astype(np.float32)
    interpolated_sequence = []
    for idx in indices:
        lower = int(np.floor(idx))
        upper = int(np.ceil(idx))
        if upper >= original_length:
            interpolated_sequence.append(sequence[lower])
        else:
            weight = idx - lower
            # Linear interpolation between frames (or features)
            frame = (1 - weight) * sequence[lower] + weight * sequence[upper]
            interpolated_sequence.append(frame)
    return np.array(interpolated_sequence)

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Compute focal loss for multi-class classification to address class imbalance.
    This function wraps around the standard cross-entropy loss.
    """
    # Compute cross entropy loss for each sample
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # probability for the true class
    loss = alpha * (1 - pt) ** gamma * ce_loss
    return loss.mean()

def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall

def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''

# ------------------------------
# Domain-specific functions
# ------------------------------

def whole_face_block_coordinates():
    df = pandas.read_csv('combined_3_class2_for_optical_flow.csv')
    m, n = df.shape
    base_data_src = './datasets/combined_datasets_whole'
    image_size_u_v = 28
    face_block_coordinates = {}
    for i in range(0, m):
        image_name = str(df['sub'][i]) + '_' + str(df['filename_o'][i]) + ' .png'
        img_path_apex = base_data_src + '/' + df['imagename'][i]
        train_face_image_apex = cv2.imread(img_path_apex)
        face_apex = cv2.resize(train_face_image_apex, (28, 28), interpolation=cv2.INTER_AREA)
        mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
        batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)
        if batch_landmarks is None:
            batch_landmarks = np.array([[[9.528073, 11.062551],
                                          [21.396168, 10.919773],
                                          [15.380184, 17.380562],
                                          [10.255435, 22.121233],
                                          [20.583706, 22.25584]]])
        row_n, col_n = np.shape(batch_landmarks[0])
        for i in range(0, row_n):
            for j in range(0, col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21
        batch_landmarks = batch_landmarks.astype(int)
        face_block_coordinates[image_name] = batch_landmarks[0]
    return face_block_coordinates

def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()
    whole_optical_flow_path = './datasets/STSNet_whole_norm_u_v_os'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    four_parts_optical_flow_imgs = {}
    for n_img in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[n_img] = []
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        four_part_coordinates = face_block_coordinates_dict[n_img]
        l_eye = flow_image[four_part_coordinates[0][0]-7:four_part_coordinates[0][0]+7,
                             four_part_coordinates[0][1]-7:four_part_coordinates[0][1]+7]
        l_lips = flow_image[four_part_coordinates[1][0]-7:four_part_coordinates[1][0]+7,
                              four_part_coordinates[1][1]-7:four_part_coordinates[1][1]+7]
        nose = flow_image[four_part_coordinates[2][0]-7:four_part_coordinates[2][0]+7,
                          four_part_coordinates[2][1]-7:four_part_coordinates[2][1]+7]
        r_eye = flow_image[four_part_coordinates[3][0]-7:four_part_coordinates[3][0]+7,
                           four_part_coordinates[3][1]-7:four_part_coordinates[3][1]+7]
        r_lips = flow_image[four_part_coordinates[4][0]-7:four_part_coordinates[4][0]+7,
                            four_part_coordinates[4][1]-7:four_part_coordinates[4][1]+7]
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)
    print(len(four_parts_optical_flow_imgs))
    return four_parts_optical_flow_imgs

class Fusionmodel(nn.Module):
    def __init__(self):
        super(Fusionmodel, self).__init__()
        self.fc1 = nn.Linear(15, 3)  # 15 -> 3
        self.bn1 = nn.BatchNorm1d(3)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(6, 3)  # 6 -> 2
        self.relu = nn.ReLU()

    def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
        fuse_five_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), dim=1)
        fuse_out = self.fc1(fuse_five_features)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)
        fuse_whole_five_parts = torch.cat((whole_feature, fuse_out), dim=1)
        fuse_whole_five_parts = self.relu(fuse_whole_five_parts)
        fuse_whole_five_parts = self.d1(fuse_whole_five_parts)
        out = self.fc_2(fuse_whole_five_parts)
        return out

# ------------------------------
# Main training and evaluation
# ------------------------------

def main(config):
    learning_rate = 0.00005
    batch_size = 256
    epochs = 800
    all_accuracy_dict = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use focal loss in place of the original CrossEntropyLoss
    # (For multi-class, we use our focal_loss function defined above.)
    loss_fn = focal_loss  

    if config.train:
        if not path.exists('ourmodel_threedatasets_weights'):
            os.mkdir('ourmodel_threedatasets_weights')

    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    total_gt = []
    total_pred = []
    best_total_pred = []
    t = time.time()

    main_path = './datasets/three_norm_u_v_os'
    subName = os.listdir(main_path)
    all_five_parts_optical_flow = crop_optical_flow_block()
    data_augment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor()
    ])

    print(subName)

    for n_subName in subName:
        print('Subject:', n_subName)
        y_train = []
        y_test = []
        four_parts_train = []
        four_parts_test = []
        
        # Build training dataset
        expression = os.listdir(os.path.join(main_path, n_subName, 'u_train'))
        for n_expression in expression:
            img_list = os.listdir(os.path.join(main_path, n_subName, 'u_train', n_expression))
            for n_img in img_list:
                y_train.append(int(n_expression))
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                # (Optional:) If desired, you could call preprocess_frame or temporal_interpolate here on lr_eye_lips.
                pil_image = Image.fromarray(cv2.cvtColor(lr_eye_lips, cv2.COLOR_BGR2RGB))
                augment = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.Resize((28, 28)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
                ])
                augmented_tensor = augment(pil_image)
                # Convert from [C, H, W] to [H, W, C] to match the test data shape
                four_parts_train.append(augmented_tensor.permute(1, 2, 0).numpy())

        # Build test dataset
        expression = os.listdir(os.path.join(main_path, n_subName, 'u_test'))
        for n_expression in expression:
            img_list = os.listdir(os.path.join(main_path, n_subName, 'u_test', n_expression))
            for n_img in img_list:
                y_test.append(int(n_expression))
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_test.append(lr_eye_lips)
                
        weight_path = path.join('ourmodel_threedatasets_weights', n_subName + '.pth')

        # Initialize model
        model = HTNet(
            image_size=28,
            patch_size=7,
            dim=128,
            heads=4,
            num_hierarchies=3,
            block_repeats=(2, 2, 10),
            num_classes=3
        )
        model = model.to(device)

        fusion_model = Fusionmodel().to(device)

        if not config.train and os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path))
            print(f"✅ Loaded pretrained weights from {weight_path}")
        elif not config.train:
            print(f"⚠️ Pretrained weights not found for {n_subName} — training from scratch.")
            
        # Set up optimizer
        # For further improvements, you can adjust parameter groups for different model parts.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Prepare DataLoader for train and test sets
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        four_parts_train = torch.Tensor(np.array(four_parts_train)).permute(0, 3, 1, 2)
        dataset_train = TensorDataset(four_parts_train, y_train)
        train_dl = DataLoader(dataset_train, batch_size=batch_size)

        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test = torch.Tensor(np.array(four_parts_test)).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(four_parts_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, epochs + 1):
            if config.train:
                model.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0

                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    features = model(x, motion_input=None, return_features=True)
                    logits = fusion_model(
                        features[:, :3],   # whole
                        features[:, 3:6],  # l_eye
                        features[:, 6:9],  # l_lips
                        features[:, 9:12], # nose
                        features[:, 12:15],# r_eye
                        features[:, 15:18] # r_lips
                    )
                    yhat = logits
                    loss = focal_loss(yhat, y)  # Using focal loss
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * x.size(0)
                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(train_dl.dataset)

            model.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            for batch in test_dl:
                x = batch[0].to(device)
                y = batch[1].to(device)
                features = model(x, motion_input=None, return_features=True)
                # Assuming feature split: whole + 5 parts (3 + 12 dims = 15)
                logits = fusion_model(
                    features[:, :3],   # whole
                    features[:, 3:6],  # l_eye
                    features[:, 6:9],  # l_lips
                    features[:, 9:12], # nose
                    features[:, 12:15],# r_eye
                    features[:, 15:18] # r_lips
                )
                yhat = logits
                loss = focal_loss(yhat, y)  # Using focal loss in evaluation too
                val_loss += loss.item() * x.size(0)
                num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(test_dl.dataset)

            # Save best model weights for the subject
            temp_best_each_subject_pred = []
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())
                best_each_subject_pred = temp_best_each_subject_pred
                if config.train:
                    torch.save(model.state_dict(), weight_path)

        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {'pred': best_each_subject_pred, 'truth': y.tolist()}
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')
        total_pred.extend(torch.max(yhat, 1)[1].tolist())
        total_gt.extend(y.tolist())
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default=False, help='Train the model if set to True, else use pre-trained weights')
    config = parser.parse_args()
    main(config)
