The purpose of this competition is to create machine learning model that will detect FOG episodes.
Our model was based on a recurrent neural network LSTM (Long Short-Term Memory) 
The model in some cases was able to classify a FOG event. The model was able to classify mainly a FOG event of the type: StartHesitation


How the algorithm works:
The algorithm loaded the training files, performed preprocessing for the information and then performed the training for the data.
The preprocessing step:
The training files contained a column that testified whether it was information substantiated by the experts or whether it was information that had no basis at all. Therefore the first step was to filter out all the information that was not based in order not to train the network with this information.
Since this is a challenge, we did not receive viewable test files, but only the possibility to upload our code to the cloud and receive the final score.
In order to be able to present the results of this project, we took 10 percent of the training information and converted it to test files by omitting columns.
Training phase:
We used all the information from defog and tdcsfog. We did not use the rest of the information that came uncatalogued
Since there is a connection between successive times and whether a FOG event took place, we divided the information into bulks of 8, so that each timestamp is attached to the timestamp that was before it.
Our model was: LSTM with a loss function of cross entropy loss and the optimizer was Adam. 
Since each sample can belong to only a particular class ,we used Cross entropy because it measures the difference between the predicted probability and the true probability.
Predication phase:
We ran the model on about 90 training files (that were not trained) that were converted to test files.
To evaluate our results and compare them to the ground truth, we used a measure of the competition: Mean Average Precision. So we calculated the average precision of each event type and then averaged it.


The files:
detecting_algorithm.py - Our Neural Network Algorithm detecting FOG
create_submission_file.py- code that convert some of the train data to test data
evaluating_score.py- code that evaluate the score for each event and the mean for all events (mAP)
