# Import the required libraries 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
import json

# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')

# Text Data Features
print ("Prepare text data of Train and Test ... ")
def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]

# Feature Engineering 
print ("TF-IDF on text data ... ")
tfidf = TfidfVectorizer(binary=True, norm='l2')
def tfidf_features(txt, flag):
    if flag == "train":
    	x = tfidf.fit_transform(txt)
    else:
	    x = tfidf.transform(txt)
    x = x.astype('float16')
    return x 

X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")


# Label Encoding - Target 
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)
# X_scaled = preprocessing.scale(X.toarray())
# test_scaled = preprocessing.scale(X_test.toarray())

# Model Training 
print ("Train the model ... ")
classifier = SVC(C=100, # penalty parameter, setting it to a larger value 
                 kernel='rbf', # kernel type, rbf working fine here
                 degree=3, # default value, not tuned yet
                 gamma=2, # kernel coefficient, not tuned yet
                 coef0=1, # change to 1 from default value of 0.0
                 shrinking=True, # using shrinking heuristics
                 tol=0.00001, # stopping criterion tolerance 
                 probability=False, # no need to enable probability estimates
                 class_weight=None, # all classes are treated equally 
                 verbose=1, # print the logs 
                 max_iter=-1, # no limit, let it run
                 decision_function_shape=None, # will use one vs rest explicitly 
                 random_state=None)
# model=VotingClassifier(estimators=[('clf1',LogisticRegression(C=10,dual=False)),
#                                   ('clf2',SVC(C=100,gamma=2,kernel='rbf',probability=True)),
#                                   ('clf',RandomForestClassifier(n_jobs=-1, n_estimators=100,min_samples_leaf=3,max_features=0.7, oob_score=True))],voting='soft',weights=[1,1,1])

model = OneVsRestClassifier(classifier, n_jobs=-1)
# model = OneVsOneClassifier(classifier, n_jobs=4)
model.fit(X, y)



# Predictions 
print ("Predict on test data ... ")
y_test = model.predict(X_test)
y_pred = lb.inverse_transform(y_test)

# Submission
print ("Generate Submission File ... ")
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('svm_output_ensemble.csv', index=False)