# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:39:12 2015

@author: yang
"""
import h5py,numpy as np
import scipy.io
import os
import theano.tensor as T
output_folder='./predict_results/'


"""
Load features, split and tag annotation.
You can replace with your own dataset
"""
train_test_path='l2_normalized_semantic_SVM_full_data_with_val_291labels_no_zero.mat'
train_test_data_package=h5py.File(train_test_path,'r')

training_data_target=train_test_data_package.get('prepared_training_label')
training_data_target=np.array(training_data_target).transpose().astype("float32")
normalized_training_data_source=train_test_data_package.get('prepared_training_data')
normalized_training_data_source=np.array(normalized_training_data_source).transpose().astype("float32")

testing_data_target=train_test_data_package.get('prepared_testing_label')
testing_data_target=np.array(testing_data_target).transpose().astype("float32")

normalized_test_data_source=train_test_data_package.get('prepared_testing_data')
normalized_test_data_source=np.array(normalized_test_data_source).transpose().astype("float32")

val_data_target=train_test_data_package.get('prepared_val_label')
val_data_target=np.array(val_data_target).transpose().astype("float32")

normalized_val_data_source=train_test_data_package.get('prepared_val_data')
normalized_val_data_source=np.array(normalized_val_data_source).transpose().astype("float32")
print 'Loading finish'

"""
Build and compile the model
"""

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Lambda
from tagging_objective import semantic_RankNet_mean,rank_layer

highest_val_F1=0
model = Sequential()


model.add(Dense(input_dim=4096, output_dim=8192, init="uniform"))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(input_dim=8192, output_dim=2048, init="uniform"))
model.add(Activation("relu"))
model.add(Dense(input_dim=2048, output_dim=300, init="uniform"))
model.add(Activation("relu"))
model.add(Dense(input_dim=300, output_dim=300, init="uniform"))
model.add(Activation("linear"))
model.add(Lambda(rank_layer,output_shape=[training_data_target.shape[1]]))

print 'Start compile'
optimizer_init=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss=semantic_RankNet_mean, optimizer=optimizer_init)
print 'Compile finish'

#train and save predict values
if not os.path.exists(output_folder): os.makedirs(output_folder)

GT=np.sum(val_data_target[val_data_target==1.])
instance_num=normalized_val_data_source.shape
instance_num=instance_num[0]
prediction_num=instance_num*5.

"""
Train 200 epochs
"""
for i_epoch in range(0,200):
    #train the model for 1 epoch
    model.fit(normalized_training_data_source,training_data_target, nb_epoch=1, batch_size=1000, verbose=1,shuffle=1)
    
    #get F1 score on validation data
    confidence = model.predict( normalized_val_data_source, batch_size=5000)
    sort_indices = np.argsort(confidence)
    sort_indices=sort_indices[:,::-1]
    static_indices = np.indices(sort_indices.shape)
    sorted_annotation= val_data_target[static_indices[0],sort_indices]
    top_5_annotation=sorted_annotation[:,0:5]
    TP=np.sum(top_5_annotation[top_5_annotation==1.])
    val_recall=TP/GT
    val_precision=TP/prediction_num
    current_val_F1=2.*val_recall*val_precision/(val_recall+val_precision)
    #save model weights if the validation F1 improved
    if current_val_F1>highest_val_F1:
        highest_val_F1=current_val_F1
        model.save_weights('best_model_Weights.h5',overwrite=True)
    print 'Current val F1 is {}. Current epoch is {}. Best val F1 is {}.'.format(current_val_F1,i_epoch,highest_val_F1)
#predict ranking direction on testing data with the best model weights
model.load_weights('best_model_Weights.h5')
predicted_label_confidence = model.predict( normalized_test_data_source, batch_size=5000)
  
#save predicted label confidence
scipy.io.savemat(output_folder+ 'predict.mat',mdict={'predicted_label_confidence':predicted_label_confidence})

