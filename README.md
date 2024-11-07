# mir_finalProj

Table of Contents:
1. [Goals](#Goals)
2. [Proposal](#Proposal)
3. [Datasets](#Datasets)
4. [Software](#Software)
5. [Methodology](#Methodology)


## Goals: <a name="Goals">
- 11/7/24:
  - pick ml algorithms

    - JC
      - Data Analysis & Visualization
      - K-Neighbors Classifier :  KNeighborsClassifier looks for topmost n_neighbors using different distance methods like Euclidean distance.
      - Decision Tree Classifier : In Decision tree each node is trained by splitting the data is continuously according to a certain parameter.
    - Ciara
      - Random Forest : Random Forest Classifier fits a number of decision tree classifiers on many sub-samples of the dataset and then use the average to improve the results.
      - Logistics Regression : Logistic Regression is a regression model that predicts the probability of a given data belongs to the particular category or not.
    - Matt
      - Cat Boost : CatBoost implements decision trees and restricts the features split per level to one, which help in decreasing prediction time. It also handles categorical features effectively.
      - Gradient Boost : In Gradient Boost an decision trees are implemented in a sequential manner which enhance the performance.
  
- 11/14/24:
  - 
- 11/21/24:

## Proposal <a name="Proposal">
We propose to use the GTZAN genre classification dataset to compare the efficacy of traditional machine learning MIR techniques against deep learning approaches. Specifically, we hope to use some of the music data extraction techniques from Librosa, such as pairing MFCC analysis and spectral centroid features with random forest or k-nearest neighbor classifier algorithms. These results will then be compared with approaches from convolutional networks in Tensorflow via the Keras API, which are provided with the spectrograms generated by Librosa from the raw audio. In making this comparison we hope to display the strengths and weaknesses of each approach, and provide further analysis on why and how these results were reached.

## Datasets <a name="Datasets">
GTZAN will provide the source data for this genre classification project. With 1,000 audio examples from ten diverse genres including classical, blues, country, disco, hiphop, and more, we are certain this can provide our software with ample training and testing data. GTZAN is among the most popular datasets to work with in the MIR field, and has been relied upon by scholars around the world. Additionally, GTZAN provides correct metadata that we can compare our results to for accuracy measurements.
## Software <a name="Software">
**Traditional Machine Learning:**
- Librosa for feature extraction, specifically MFCC, spectral centroid, and another
analysis to be paired with different ML techniques like k-nearest neighbor,
random forest, and Support Vector Machines provided by the SciKit framework Deep Learning
- Librosa for conversion of audio to spectrograms, to be paired with TensorFlow via Keras API to implement convolutional neural network and recurrent neural network training
**Comparison:**
- SciKit Learn, along with native Python data visualization
- TensorBoard for deep learning results visualization

## Methodlogy <a name="Methodology">

Exploratory Data Analysis 
Data Preprocessing
modle training
Evaluation


