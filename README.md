# My_graduation_project
Introduction to this project
-----------------
In order to study which resampling method performs better on CMS dataset, it helps the machine learning model to obtain better prediction ability. We use ten resampling methods and five machine learning models. The experimental results show that the combination of XGBoost model and SMOTEENN resampling method has the best performance, with an AUC of 0.85. In addition, the hybrid resampling method consisting of oversampling and undersampling is the best among the three categories. A brief description of the part of the code used in the experiment follows.

Code Function Description
----------------
*process_data.ipynb:

1. Introduce the data sets in the folder '_used', perform the following operations on each data set, calculate the number of NaN in each data set, and generate a comparison graph of the number of NaN and the number of non-nan, and save it in the folder 'Pictures'.

2. Replace NaN with the median in each dataset.

3. Replace the commas in the value section with Spaces.

4. Change all 'string' values to 'double' for descriptive analysis and model training.

5. Merge the data sets of year 17 and year 18 in each data set, and finally get the descriptive statistics table, and save the table in the 'description_used' folder.

*process_crime_data.ipynb:

1. The fraud data set in the 'LEIE' folder is introduced, and the data set is extracted according to the years, and the fraud data of 2017,2018 and 2019 are respectively generated and stored in the 'crime_data' folder.

*give_crime_label.ipynb:

1. Label the three data sets of Part B, Part D and DMEPOS with fraud labels, and save the labeled data sets in the folder 'crime_data'.
2. Calculate the number of fraud labels 1 in several Part B datasets after labeling.
3. All data sets were coded with One-hot Encoder.
4 Merge the data sets of Part B, Part D and DMEPOS for each year and save them in 'every_year_combined_Dataset'.

*combine_dataset.ipynb:

1. Remove the feature NPI from the dataset in the 'every_year_combined_dataset' folder, because it has no effect on model training.
2. Zeroes are added to the merged data set.
3. Save the merged dataset in the 'train_and_test_dataset' folder.

*data_sampling.ipynb:

1. Because the dataset in the 'train_and_test_dataset' folder contains a very small number of letters, remove this part of data and save it to the 'train_and_test_dataset' folder.
2. Apply the 10 resampling methods to the training set, and save the newly generated resampled training set in the 'resampled_datasets' folder.
3. Train Random Forest, Naive Bayes and Logistic Regressoin with the newly generated resampling training set, and save the trained models in the 'trained_models' folder.

*trains_model.ipynb:

1. The GBM and XGBoost models were trained with ten kinds of resampled data, and the trained models were saved in the 'trained_models' folder.

*evaluation.ipynb:

1. The ROC curve, AUC and F1-score of all mechanical learning models under all resampling data were generated, and the generated pictures were saved in the 'metrics_pictures' folder.
