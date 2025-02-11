# Audio Genre Classification Using CNN  

## Project Overview  
This project focuses on classifying audio files based on their Mel-Frequency Cepstral Coefficients (MFCC) features using a Convolutional Neural Network (CNN). The goal is to categorize 115 audio files into six predefined genres:  

- National Anthem  
- Marathi Bhavgeet  
- Marathi Lavni  
- Songs by Asha Bhosale  
- Songs by Kishore Kumar  
- Songs by Michael Jackson  

Additionally, an optional classification task was explored to distinguish between:  
- Male singers  
- Female singers  
- Duets (both male and female singers)  

## Approach & Methodology  

1. **Data Preprocessing & Feature Extraction**  
   - MFCC features were extracted from the audio files.  
   - Mean-max pooling was applied for dimensionality reduction.  

2. **Exploratory Data Analysis (EDA)**  
   - Distribution of MFCC values was analyzed.  
   - No missing values were found in the dataset.  

3. **Model Development**  
   - A 1D CNN model was trained to classify the audio files.  
   - Data augmentation techniques (scaling, shifting, and removing frames) were used to improve generalization.  
   - Hyperparameter tuning was performed for optimization.  

4. **Training & Evaluation**  
   - 750 augmented audio files were used for training.  
   - The dataset was split into training, validation, and test sets.  
   - Techniques like batch normalization, dropout layers, and early stopping were implemented to prevent overfitting.  
   - Model performance was evaluated using accuracy, precision, recall, and confusion matrices.  

## Results & Insights  
- The CNN model achieved an accuracy of 85% for genre classification.  
- The model struggled with classifying English songs (Michael Jackson) due to a dataset imbalance favoring Hindi/Marathi songs.  
- The male vs. female singer classification model achieved 93% accuracy.  
- The ROC curve and confusion matrix analysis indicate strong classification performance across most categories.  

## Dataset & Code  
- **MFCC features dataset** (processed) can be accessed [here](https://drive.google.com/file/d/1elbAdFgFlls1yj0GB3qTKHRajY2qKkrI/view?usp=sharing).  
- The trained model and dataset for male-female classification can be accessed [here](https://drive.google.com/drive/folders/1Bgpu7LMjyHB55nPdvm-5wa1ueXjcJLJ0?usp=sharing).  

## Learnings & Challenges  
- Initially explored unsupervised learning, but CNN-based supervised learning provided better results.  
- Feature engineering played a crucial role in improving classification accuracy.  
- Data augmentation was essential for overcoming dataset limitations.  
- Overfitting mitigation techniques like batch normalization and dropout significantly improved model generalization.  

## Team Members  
- Kanika Sharma (23B2287)  
- Mahima Sahu (23B2231)  
- Dnyaneshwari Kate (23B2201)  
- Hima Varsha (23B2236)  

## Acknowledgment  
This project was conducted as part of DS203 - Exercise 7 (Project), evaluated based on problem-solving accuracy, feature engineering quality, and overall methodology.  
