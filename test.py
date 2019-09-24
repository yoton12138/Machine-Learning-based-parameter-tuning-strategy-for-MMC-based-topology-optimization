from ET import *
from operator import itemgetter
 #FeatureBuilder,BagOfWords,StarFeatureDetector,ERFTrainer,ImageTagExtractor

feature_map_file = 'TP_feature_map-ET-l.pkl'###########################
model_file = 'TP_model_file-ET-l.pkl'#######################
codebook_file = 'TP_codebook_file-ET-l.pkl'#################################
scaling_size = 656
test_img = "L"

def test(test_img):
        i=0
        temp = []
        name = []
        results = []

        for root, dirs, files in os.walk(test_img):
            for filename in (x for x in files if x.endswith('.jpg')):
                filepath = os.path.join(root, filename)
                temp.append(filepath)
        
        for path in temp:
            test_path = temp[i]
            test_image = cv2.imread(test_path)
            name.append(int(re.sub(r'\D','',test_path.split('\\')[-1])))
            result, y_predlabel,proba = ImageTagExtractor(model_file, 
                codebook_file).predict(test_image, scaling_size)#分类值

            if result == 'feasible':
                result = 1
            else:
                result = 0
          
            results.append(result)
            i=i+1
        temp = np.array([name,results])
        temp = temp.transpose()
        name_list = sorted(temp, key=itemgetter(0))#按第一列排个序
        #name
        with open ('class_results-ET-l.txt',"w",encoding = 'gbk') as f:###########
            f.write('name\tresult\n')
            for i in range(len(name_list)):
                for j in range(len(name_list[i])):
                    f.write(str(name_list[i][j]))
                    f.write('\t')
                f.write('\n')
test(test_img)
    