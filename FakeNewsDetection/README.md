# Fake News Detection


![image](https://user-images.githubusercontent.com/77210430/120220863-0f556800-c25b-11eb-9008-ce73d5d37cf6.png)

## Explanation

In the FakeNewsCountVectorizer&MultinomialNB : I've used countvectorizer to extract features from the dataset, first with the help of stop words i've stopped the common words in english language which are not useful in understanding the news, making news small and easily readable. Then converted the dataset into a DataFrame by giving input a dataset X_train which is splitted and fit transformed with the help of countvectorizer i.e count_train. After that I've initialised our machine learning algorithm which is Multinomial Naive Bayes (MultinomialNB) and trained that model for count_train and y_train. And done the prediction on count_test and calculated the accuracy with the help of y_test and count_test. At the end I've printed the confusion matrix.


In the FakeNewsTfidVectorizer&PassiveAggressiveClassifier : Whole code is same but instead of countvectorizer I've used TfidVectorizer to extract features and to stop words. Also, the machine learning algorithm used in this code is PassiveAggressiveClassifier. Rest everything is same in both codes, workflow is also same.

NOTE: I got more accuracy in PassiveAggressiveClassifer maybe beacuse I've gave the max iterations as 50 in it and I didn't gave that parameter in MultinomialNB.
