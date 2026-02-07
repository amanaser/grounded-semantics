from refer import REFER
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os
from PIL import Image as PImage # pillow

data_root = 'scratch/coco'  # contains refclef, refcoco, refcoco+, refcocog and images
dataset = 'refcoco' 
splitBy = 'unc'
refer = REFER(data_root, dataset, splitBy)


# print stats about the given dataset
print ('dataset [%s_%s] contains: ' % (dataset, splitBy))
ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print ('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

print ('\nAmong them:')
if dataset == 'refclef':
    if splitBy == 'unc':
        splits = ['train', 'val', 'testA', 'testB', 'testC']
    else:
        splits = ['train', 'val', 'test']
elif dataset == 'refcoco':
    splits = ['train', 'val', 'test']
elif dataset == 'refcoco+':
    splits = ['train', 'val', 'test']
elif dataset == 'refcocog':
    splits = ['train', 'val']  # we don't have test split for refcocog right now.
    
for split in splits:
    ref_ids = refer.getRefIds(split=split)
    print ('%s refs are in split [%s].' % (len(ref_ids), split))

ref_ids = refer.getRefIds()
print(len(ref_ids))
ref_id = 35254 # pick a random ref_id
ref = refer.Refs[ref_id]

ref # a dictionary with all of the needed info for a referring expression+image

print ('ref_id [%s] (ann_id [%s])' % (ref_id, refer.refToAnn[ref_id]['id']))
# show the segmentation of the referred object
# plt.figure()
# refer.showRef(ref, seg_box='box')
# plt.show()

def get_bounded_subimage(img_id, ann_id, xs=224,ys=224, show=False):
    bbox = refer.Anns[ann_id]['bbox']
    bbox = [int(b) for b in bbox]
    img = refer.Imgs[img_id]
    I = io.imread(os.path.join(refer.IMAGE_DIR, img['file_name']))
    sub = I[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    if show:
        plt.figure()
        ax = plt.gca()
        ax.imshow(sub)
        plt.show()
    if len(sub) == 0: return None
    pim = PImage.fromarray(sub)
    pim2 = pim.resize((xs,ys), PImage.Resampling.LANCZOS)
    img = np.array(pim2)
    if len(img.shape) < 3: return None
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    return img

get_bounded_subimage(ref['image_id'], ref['ann_id'], show=False)

def compute_posfeats(img_id, ann_id,):
    img = refer.Imgs[img_id]
    bb = refer.Anns[ann_id]['bbox']
    fname = os.path.join(refer.IMAGE_DIR, img['file_name'])
    if not os.path.isfile(fname): return None
    img = io.imread(fname)
    
    if len(img.shape) < 3: return None
    ih, iw, _ = img.shape
    x,y,w,h = bb
    # x1, relative
    x1r = x / iw
    # y1, relative
    y1r = y / ih
    # x2, relative
    x2r = (x+w) / iw
    # y2, relative
    y2r = (y+h) / ih
    # area
    area = (w*h) / (iw*ih)
    # ratio image sides (= orientation)
    ratio = iw / ih
    # distance from center (normalised)
    cx = iw / 2
    cy = ih / 2
    bcx = x + w / 2
    bcy = y + h / 2
    distance = np.sqrt((bcx-cx)**2 + (bcy-cy)**2) / np.sqrt(cx**2+cy**2)
    # done!
    return np.array([x1r,y1r,x2r,y2r,area,ratio,distance]).reshape(1,7)


# example using the sheep example on the compute_posfeats function
compute_posfeats(ref['image_id'], ref['ann_id']).flatten()

import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from collections import defaultdict as dd
from tqdm import tqdm
import pickle

base_model = VGG19(weights='imagenet', include_top=True)
xs, ys = 224, 224

for layer in base_model.layers:
    print(layer.name)

layer = 'predictions'
model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)

def get_img_features(model, img):
    if img is None: return None
    img = preprocess_input(img)
    yhat = model.predict(img)
    return yhat

words_as_classifiers = dd(list) # use something like this dictionary to store positive examples

# first, get all of the training data
train_ids = refer.getRefIds(split='train')

# for a single train_id, you can get its image_id and the ann_id (i.e., the referring expression)
for ref_id in train_ids:
    ref = refer.Refs[ref_id]
    img_id = ref['image_id']
    ann_id = ref['ann_id']

    #then you'll need to get the bounded subimage by calling the get_bounded_subimage function.
    sub_image = get_bounded_subimage(img_id, ann_id, xs, ys)
    if sub_image is None: 
        continue

    #then, you'll need to pass that image through a convnet like you did for A5
    sub_image = get_img_features(model, sub_image)
    if sub_image is None:
        continue

    #optionally, you can call the compute_posfeats function to get some additional features
    pos_feats = compute_posfeats(img_id, ann_id)
    if pos_feats is None:
        continue

    #concatenate these to the convnet output to form a single vector for this image
    feature_vector = np.concatenate((sub_image, pos_feats), axis=1)

    #add this feature vector to a list of positive examples for each word in the referring expression
    # you may need to flatten() the feature vector
    for sent in ref['sentences']:
            for word in sent['tokens']:
                words_as_classifiers[word].append(feature_vector)

pickle.dump(words_as_classifiers, open( "vgg19_pred.pkl", "wb" ))

# once you have all of the positive examples for all of the words, you'll need to find negative examples for each word
# the number of negative examples should be a function of how many positive examples there are
import random

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

# finally, train a binary classifier for each word
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


wac_train = {}
for word in tqdm(words_as_classifiers):
    pos = words_as_classifiers[word]

    if len(pos) < 1: 
        continue 

    neg = get_negative_samples(words_as_classifiers, word)
    classifier = LogisticRegression('l2', C=10, max_iter=1000)

    X_train = np.array(pos + neg).reshape(len(pos) + len(neg), -1)
    y_train = np.array([0] * len(pos) + [1] * len(neg))

    # X_train = np.vstack((pos, neg))  
    # y_train = np.array([1] * len(pos) + [0] * len(neg)) 

    classifier.fit(X_train, y_train)
    wac_train[word] = classifier

pickle.dump(wac_train, open( "wac.pkl", "wb" ) )

# step through the eval ids
# get all of the objects as feature vectors for an image using your convnet
# for each object,
# for each referring expression / sentence
# apply all of the feature vectors to your trained classifiers for each word in the sentence
# multiply the classifier probabilities together for each word
# e.g., for the last sentence above: Pblue(object1) * Pshirt(object1)
# find the object with the highest resulting multiplied probability, compare to gold, compute accuracy

eval_ids = refer.getRefIds(split='val')

true_labels = []
predicted_labels = []

def extract_object_features(image_id, ann_ids):

    object_features = {}

    for ann_id in ann_ids:
        pos_feats = compute_posfeats(image_id, ann_id)
        cropped_img = get_bounded_subimage(image_id, ann_id, xs, ys)

        if pos_feats is None or cropped_img is None:
            continue  

        cnn_features = get_img_features(model, cropped_img)
        if cnn_features is not None:
            combined_features = np.concatenate([cnn_features, pos_feats], axis=1)
            object_features[ann_id] = combined_features  # store feature vector
    
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
    for reference_id in tqdm(eval_ids):
        evaluate_instance(reference_id, true_labels, predicted_labels)

    accuracy = metrics.accuracy_score(predicted_labels, true_labels)
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy

evaluate_model(eval_ids)

def evaluate_on_test_set():
    eval_ids = refer.getRefIds(split='test')
    true_labels = []  
    predicted_labels = [] 

    for ref_id in tqdm(eval_ids):
        evaluate_instance(ref_id, true_labels, predicted_labels)

    final_accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")

evaluate_on_test_set()



