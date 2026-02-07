import torch
import clip
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
from PIL import Image as PImage
from collections import defaultdict as dd
from tqdm import tqdm
import pickle
from refer import REFER
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import random


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize REFER
data_root = 'scratch/coco'
dataset = 'refcoco' 
splitBy = 'unc'
refer = REFER(data_root, dataset, splitBy)

xs, ys = 224, 224

# Function to extract image features using CLIP
def get_img_features(model, img):
    if img is None:
        return None
    img = preprocess(PImage.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img)
    return features.cpu().numpy()

# Function to extract bounding box features
def compute_posfeats(img_id, ann_id):
    img = refer.Imgs[img_id]
    bb = refer.Anns[ann_id]['bbox']
    fname = os.path.join(refer.IMAGE_DIR, img['file_name'])
    if not os.path.isfile(fname): return None
    img = io.imread(fname)
    if len(img.shape) < 3: return None
    ih, iw, _ = img.shape
    x,y,w,h = bb
    return np.array([x/iw, y/ih, (x+w)/iw, (y+h)/ih, (w*h)/(iw*ih), iw/ih, 
                     np.sqrt(((x+w/2)-iw/2)**2 + ((y+h/2)-ih/2)**2) / np.sqrt((iw/2)**2 + (ih/2)**2)]).reshape(1,7)

# Function to extract bounded subimage
def get_bounded_subimage(img_id, ann_id, xs=224, ys=224):
    bbox = refer.Anns[ann_id]['bbox']
    img = refer.Imgs[img_id]
    I = io.imread(os.path.join(refer.IMAGE_DIR, img['file_name']))
    sub = I[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    if len(sub) == 0: return None
    pim = PImage.fromarray(sub)
    pim2 = pim.resize((xs,ys), PImage.Resampling.LANCZOS)
    img = np.array(pim2)
    if len(img.shape) < 3: return None
    return img

# Extract features for words_as_classifiers
words_as_classifiers = dd(list)
train_ids = refer.getRefIds(split='train')

for ref_id in train_ids:
    ref = refer.Refs[ref_id]
    img_id = ref['image_id']
    ann_id = ref['ann_id']
    sub_image = get_bounded_subimage(img_id, ann_id, xs, ys)
    if sub_image is None: continue
    sub_image = get_img_features(model, sub_image)
    if sub_image is None: continue
    pos_feats = compute_posfeats(img_id, ann_id)
    if pos_feats is None: continue
    feature_vector = np.concatenate((sub_image, pos_feats), axis=1)
    for sent in ref['sentences']:
        for word in sent['tokens']:
            words_as_classifiers[word].append(feature_vector)

pickle.dump(words_as_classifiers, open("clip_pred.pkl", "wb"))

# Function to get negative samples
def get_negative_samples(wac, word, num_negatives=3):
    negs = []
    words = list(wac.keys())
    num_pos = len(wac[word])
    words.remove(word)
    if not words:
        return negs
    while len(negs) < num_negatives * num_pos:
        neg_word = random.choice(words)
        if wac[neg_word]:  
            neg = random.choice(wac[neg_word])  
            negs.append(neg)
    return negs

# Train classifiers for words
wac_train = {}
for word in tqdm(words_as_classifiers):
    pos = words_as_classifiers[word]
    if len(pos) < 1: continue
    neg = get_negative_samples(words_as_classifiers, word)
    classifier = LogisticRegression('l2', C=10, max_iter=1000)
    X_train = np.array(pos + neg).reshape(len(pos) + len(neg), -1)
    y_train = np.array([0] * len(pos) + [1] * len(neg))
    classifier.fit(X_train, y_train)
    wac_train[word] = classifier

pickle.dump(wac_train, open("wac_clip.pkl", "wb"))

# Evaluation functions
def extract_object_features(image_id, ann_ids):
    object_features = {}
    for ann_id in ann_ids:
        pos_feats = compute_posfeats(image_id, ann_id)
        cropped_img = get_bounded_subimage(image_id, ann_id, xs, ys)
        if pos_feats is None or cropped_img is None:
            continue  
        clip_features = get_img_features(model, cropped_img)
        if clip_features is not None:
            combined_features = np.concatenate([clip_features, pos_feats], axis=1)
            object_features[ann_id] = combined_features  
    return object_features

def compute_object_probabilities(sentence_tokens, object_features):
    object_probabilities = {}
    smoother = 0.001 
    for word in sentence_tokens:
        for obj_id, features in object_features.items():
            if obj_id not in object_probabilities:
                object_probabilities[obj_id] = 1.0  
            if word in wac_train:
                classifier = wac_train[word]
                probability = classifier.predict_proba(features.reshape(1, -1))[0][0]  
                object_probabilities[obj_id] *= probability
            else:
                object_probabilities[obj_id] *= smoother
    return object_probabilities

def evaluate_instance(reference_id, true_labels, predicted_labels):
    reference = refer.Refs[reference_id]
    image_id = reference['image_id']
    ann_id = reference['ann_id']
    ann_ids = refer.getAnnIds(image_ids=[image_id])
    object_features = extract_object_features(image_id, ann_ids)
    for sentence in reference['sentences']:
        object_probabilities = compute_object_probabilities(sentence['tokens'], object_features)
        if object_probabilities:
            best_object = max(object_probabilities, key=object_probabilities.get)
            predicted_labels.append(best_object)
            true_labels.append(ann_id)

def evaluate_model(eval_ids):
    true_labels, predicted_labels = [], []
    for reference_id in tqdm(eval_ids):
        evaluate_instance(reference_id, true_labels, predicted_labels)
    accuracy = metrics.accuracy_score(predicted_labels, true_labels)
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy

eval_ids = refer.getRefIds(split='val')
evaluate_model(eval_ids)

def evaluate_on_test_set():
    eval_ids = refer.getRefIds(split='test')
    evaluate_model(eval_ids)

evaluate_on_test_set()
