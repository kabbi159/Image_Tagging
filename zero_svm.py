import numpy as np
import h5py
from sklearn.linear_model import LinearRegression
from ranking import *

n_all = 1000
def MSE(a,b):
    mse =0.0
    print(a.shape, b.shape)
    if a.shape != b.shape:
        print("Size of vector mixmatch - cannot calculate Mean Squared error")
    for i in range(0,len(a)):
        mse += np.linalg.norm(b[i] - a[i])**2
    mse = 1.0/len(a)*mse
    return mse

f = h5py.File('l2_normalized_semantic_SVM_full_data_with_val_291labels_no_zero.mat' )

def F1_score(tags_pred, tags_actual,k1=3, k2=5):
    k1_count=0.0
    k2_count=0.0
    for i in range(0,k1):
        if tags_actual[tags_pred[i]] == 1:
            k1_count += 1

    for i in range(0,k2):
        if tags_actual[tags_pred[i]] == 1:
            k2_count += 1

    num_tags = 1.0*(len(tag_word_vectors) + sum(tags_actual ))/2
    
    k1_recall = 1.0*k1_count/num_tags
    k2_recall = 1.0*k2_count/num_tags 
    k1_precision = k1_count/k1
    k2_precision = k2_count/k2 
    print(int(num_tags), int(k1_count), int(k2_count))
    return 2.0*k1_precision*k1_recall/(k1_precision + k1_recall), 2.0*k2_precision*k2_recall/(k2_precision+k2_recall)


print("-----------------------------------------\nLoading Data")

training_data = np.transpose(f["prepared_training_data"])[0:n_all]
training_label = np.transpose(f["prepared_training_label"])[0:n_all]
valid_data = np.transpose(f["prepared_val_data"])[0:n_all]
valid_label = np.transpose(f["prepared_val_label"])[0:n_all]
testing_data = np.transpose(f["prepared_testing_data"])[0:n_all]
testing_label = np.transpose(f["prepared_testing_label"])[0:n_all]

n_training = len(training_data)
n_valid = len(valid_data)
n_testing = len(testing_data)

tag_word_vectors = np.transpose(h5py.File('291labels.mat')["semantic_mat"])

print("Done")

print("Ranking SVM for Training Data")
r=RankSVM()
w_list=np.zeros([n_training,300])
for i in range(0,len(training_data)):
    r.fit(tag_word_vectors,training_label[i])
    w_list[i] = r.coef_
print("Done")

print("Fitting Linear Regression model")
lin_reg = LinearRegression()
lin_reg.fit(training_data, w_list)
print(lin_reg.score(training_data, w_list))
A = lin_reg.coef_
print(w_list.shape," = ", A.shape, training_data.shape)
print("Done")

print("Ranking SVM for Testing Data")
r=RankSVM()

# for j in range(0,n_testing):
    # w = np.dot(testing_data[j], np.transpose(A))
    # tags_pred_score = np.dot(w,np.transpose(tag_word_vectors)) 
    # tag_pred_ranked = [i[0] for i in sorted(enumerate(tags_pred_score), key=lambda x:x[1])]
    # F1_score(tag_pred_ranked, testing_label[j],100,200)

for j in range(0,n_training):
    w = np.dot(training_data[j], np.transpose(A))
    tags_pred_score = np.dot(w,np.transpose(tag_word_vectors))

    tag_pred_ranked = [i[0] for i in sorted(enumerate(tags_pred_score), key=lambda x:x[1])]
    tag_pred_ranked.reverse()
    F1_score(tag_pred_ranked, training_label[j],15,30)

# w_list_t=np.zeros([n_testing,300])
# for i in range(0,n_testing):
    # r.fit(tag_word_vectors,testing_label[i])
    # w_list_t[i] = r.coef_
# print("Done")





# print("MSE Error" + str(MSE(w_list_t, np.dot(testing_data,np.transpose(A)))))
# print(w_list_t - np.dot(testing_data,np.transpose(A)))
