import os
import _pickle as pickle
import cv2
import re
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score,r2_score, mean_squared_error,average_precision_score,accuracy_score,hamming_loss,classification_report,roc_curve,precision_recall_curve,precision_score ,recall_score,confusion_matrix

#from skimage.measure import compare_ssim
#from PIL import Image
def nothing():
    pass
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)
    else:
        nothing()

def load_training_data(input_folder):
    training_data = []

    if not os.path.isdir(input_folder):
        raise IOError("The folder " + input_folder + " doesn't exist")
        
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith('.jpg')):
            filepath = os.path.join(root, filename)
            #print(root)
            #print(filename)
            object_class = filepath.split('\\')[-2]#取出文件路径名称的倒数第二个，即类
            #print(object_class)
            training_data.append({'object_class': object_class, 
                'image_path': filepath})
                    
    return training_data

class FeatureBuilder(object):
    def extract_features(self, img):
        keypoints = StarFeatureDetector().detect(img)#特征提取
        keypoints, feature_vectors = compute_sift_features(img, keypoints)
        return feature_vectors

    def get_codewords(self, input_map, scaling_size, max_samples=8):
        keypoints_all = []
        count = 0
        cur_class = ''
        for item in input_map:
            if count >= max_samples:
                if cur_class != item['object_class']:
                    count = 0
                else:
                    continue

            count += 1

            if count == max_samples:
                print ("Built centroids for", item['object_class'])

            cur_class = item['object_class']
            img = cv2.imread(item['image_path'])
            img = resize_image(img, scaling_size)

            #num_dims = 128
            feature_vectors = self.extract_features(img)
            keypoints_all.extend(feature_vectors) 

        kmeans, centroids = BagOfWords().cluster(keypoints_all)
        return kmeans, centroids

class BagOfWords(object):
    def __init__(self, num_clusters=64):
        self.num_dims = 128
        self.num_clusters = num_clusters
        self.num_retries = 10

    def cluster(self, datapoints):
        kmeans = KMeans(self.num_clusters, 
                        n_init=max(self.num_retries, 1),
                        max_iter=20, tol=1.0)

        res = kmeans.fit(datapoints)
        centroids = res.cluster_centers_
        return kmeans, centroids

    def normalize(self, input_data):
        sum_input = np.sum(input_data)

        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

    def construct_feature(self, img, kmeans, centroids):
        keypoints = StarFeatureDetector().detect(img)
        keypoints, feature_vectors = compute_sift_features(img, keypoints)
        labels = kmeans.predict(feature_vectors)
        feature_vector = np.zeros(self.num_clusters)#数组

        for i, item in enumerate(feature_vectors):
            feature_vector[labels[i]] += 1

        feature_vector_img = np.reshape(feature_vector, 
                ((1, feature_vector.shape[0])))#转化为矩阵
        #print(feature_vector_img)
        #print(self.normalize(feature_vector_img))
        return self.normalize(feature_vector_img)

# Extract features from the input images and 
# map them to the corresponding object classes
def get_feature_map(input_map, kmeans, centroids, scaling_size):
    feature_map = []
     
    for item in input_map:
        temp_dict = {}
        temp_dict['object_class'] = item['object_class']
    
        #print ("Extracting features for", item['image_path'])
        img = cv2.imread(item['image_path'])
        img = resize_image(img, scaling_size)

        temp_dict['feature_vector'] = BagOfWords().construct_feature(
                    img, kmeans, centroids)

        if temp_dict['feature_vector'] is not None:
            feature_map.append(temp_dict)
        #print(feature_map)

    return feature_map

class StarFeatureDetector(object):
    def __init__(self):
        self.detector = cv2.xfeatures2d.StarDetector_create()

    def detect(self, img):
        return self.detector.detect(img)

# Extract SIFT features
def compute_sift_features(img, keypoints):
    if img is None:
        raise TypeError('Invalid input image')

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = cv2.xfeatures2d.SIFT_create().compute(img_gray, keypoints)
    return keypoints, descriptors

# Resize the shorter dimension to 'new_size' 
# while maintaining the aspect ratio
def resize_image(input_img, new_size):
    h, w = input_img.shape[:2]
    scaling_factor = new_size / float(h)

    if w < h:
        scaling_factor = new_size / float(w)

    new_shape = (int(w * scaling_factor), int(h * scaling_factor))
    return cv2.resize(input_img, new_shape) 

class ERFTrainer(object):
    def __init__(self, X, label_words):
        self.le = preprocessing.LabelEncoder()  
        self.clf = ExtraTreesClassifier(n_estimators=400,
                max_depth=None, random_state=1)
        #self.clf = svm.SVC()
        y = self.encode_labels(label_words)
        self.clf.fit(np.asarray(X), y)
        
        self.rg=ExtraTreesRegressor(n_estimators=400,
                                    max_depth=None,random_state=1,)
        #self.rg = svm.SVR()
        self.rg.fit(np.asarray(X), y)
        #用编码过的XY训练
    def encode_labels(self, label_words):
        self.le.fit(label_words) 
        return np.array(self.le.transform(label_words), dtype=np.float32)

    def classify(self, X):
        label_nums = self.clf.predict(np.asarray(X))
        proba = self.clf.predict_proba(np.asarray(X))
        #proba = self.clf.decision_function(np.asarray(X))
        label_words = self.le.inverse_transform([int(x) for x in label_nums]) 
        return label_words,label_nums,proba
    
    def regressor(self,X):##
        value = self.rg.predict(np.asarray(X))
        return value
        
class ImageTagExtractor(object):
    def __init__(self, model_file, codebook_file):
        with open(model_file, 'rb') as f:
            self.erf = pickle.load(f)

        with open(codebook_file, 'rb') as f:
            self.kmeans, self.centroids = pickle.load(f)

    def predict(self, img, scaling_size):
        img = resize_image(img, scaling_size)
        feature_vector = BagOfWords().construct_feature(
                img, self.kmeans, self.centroids)
        image_tag ,image_label,proba = self.erf.classify(feature_vector)[:3]
        return image_tag,image_label,proba
    #测试函数中用到
    def feature_vector(self, img, scaling_size):
        img = resize_image(img, scaling_size)
        feature_vector = BagOfWords().construct_feature(
                img, self.kmeans, self.centroids)
        return feature_vector
    # 测试函数中用到  
    
    def get_value(self,img,scaling_size):##
        img = resize_image(img, scaling_size)
        feature_vector = BagOfWords().construct_feature(
                img, self.kmeans, self.centroids)
        value =  self.erf.regressor(feature_vector)
        return value
     

    #print('\033[1;32;40m{0} is: {1}\033[0m'.format(name,result))
def build_visual_codebook(training_data):
    print ("====== Building visual codebook ======")
    kmeans, centroids = FeatureBuilder().get_codewords(training_data, scaling_size)
    codebook_file='TP_codebook_file-ERT999.pkl'###############################
    with open(codebook_file, 'wb') as f:
        pickle.dump((kmeans, centroids), f)
    
    # Extract features from input images
    print ("\n====== Building the feature map ======")
    feature_map = get_feature_map(training_data, kmeans, centroids, scaling_size)
    feature_map_file="TP_feature_map-ERT999.pkl"#################################
    with open(feature_map_file, 'wb') as f:
        pickle.dump(feature_map, f)

def model(feature_map_file,model_file):
    with open(feature_map_file, 'rb') as f:
        feature_map = pickle.load(f)

    # Extract feature vectors and the labels
    label_words = [x['object_class'] for x in feature_map]#特征映射中所有的类标签
    print(label_words)
    dim_size = feature_map[0]['feature_vector'].shape[1]#列表第一字典中特征向量取列的维度32
    #print(dim_size)
    X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map]
    print(X[0])           #特征向量             32列
    # Train the Extremely Random Forests classifier
    erf = ERFTrainer(X, label_words) 
    with open(model_file, 'wb') as f:
        pickle.dump(erf, f)
