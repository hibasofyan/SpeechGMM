
# Language Recognition System using GMMs

> **High-precision language identification system achieving 98.4% accuracy**  
> A supervised language classification system implementing Gaussian Mixture Models (GMMs) for identifying  five languages: **English**, **Classical Arabic**, **Spanish**, **French**, and **Russian**.  

GMMs offer an advantage as they do not require **huge amounts** of training data to perform well. Their statistical modeling approach makes them efficient for cases where data is **limited**, while still effectively capturing the acoustic features.


## 1. Technical Overview
This system employs a statistical approach to language recognition by training Gaussian Mixture Models (GMMs) on **Mel-Frequency Cepstral Coefficients (MFCCs)**.  
The methodology adapts unsupervised learning techniques for a supervised classification task, where each language is modeled by a dedicated GMM.  
Prediction is performed using **maximum likelihood estimation**.



## 2. **Core Mathematical Model**

The system utilizes a **one-versus-all classification strategy**. Each language $L_i$ is modeled by a dedicated GMM $\lambda_i$.  
For a given test observation sequence:

$$
X = \{x_1, x_2, ..., x_T\}
$$

The classification decision is made by finding the language model that maximizes the likelihood:

$$
L^* = \arg\max_i P(X \mid \lambda_i)
$$



### 2.1.  GMM Formulation
Each GMM $\lambda_i$ is a probabilistic model characterized by a weighted sum of $M$ Gaussian components, defined by:

- $\pi_m$: mixture weights  
- $\mu_m$: mean vectors  
- $\Sigma_m$: covariance matrices  

The probability density function for a feature vector $x_t$ is:

$$
p(x_t \mid \lambda_i) = \sum_{m=1}^M \pi_m \cdot N(x_t; \mu_m, \Sigma_m)
$$



### 2.2. Feature Extraction
- **MFCC Computation**: 13 coefficients per audio frame.  
- **Audio Pre-processing**: Silence removed using energy-based VAD:  
  - Minimum silence duration: **700 ms**  
  - Detection threshold: **-40 dBFS**



## 3. Dataset Specifications
```
Total Dataset: 2,125 Audio Samples
├── Training Set: 2,000 samples (400 per language)
├── Test Set: 125 samples (25 per language)  
└── Source: Mozilla Common Voice 
```


### 3.1. Dataset Curation and Diversity
Dataset curated for **diverse speaker profiles**:  
- Variation in **age, gender, and accent**  
- Manual pre-processing and balanced partitioning into train/test sets  


### 3.2. Dataset Distribution

| Language         | Training Samples | Test Samples | Training Duration (s) | Test Duration (s) |
|------------------|------------------|--------------|-----------------------|-------------------|
| English          | 400              | 25           | 2316.73               | 141.21            |
| Classical Arabic | 400              | 25           | 1766.97               | 135.55            |
| French           | 400              | 25           | 1885.42               | 133.50            |
| Russian          | 400              | 25           | 2038.15               | 138.30            |
| Spanish          | 400              | 25           | 1930.50               | 135.55            |

**Statistical Properties**:
- Avg. clip duration: 4–6 seconds  
- Max. duration: ~10 seconds  
- Consistent durations ensure stable GMM performance



## 4. Hyperparameter Optimization
GMMs were trained with **8, 16, 32, and 64 components** for each language.  
The evaluation demonstrated that **for all 5 languages, the best-performing model was a GMM with 64 components**.



## 5. Performance Analysis

### 5.1. Classification Results (64-component GMM)

| Language         | Precision | Recall  | F1-Score |
|------------------|-----------|---------|----------|
| English          | 1.0000    | 1.0000  | 1.0000   |
| Classical Arabic | 0.9615    | 1.0000  | 0.9804   |
| Spanish          | 1.0000    | 1.0000  | 1.0000   |
| French           | 0.9615    | 1.0000  | 0.9804   |
| Russian          | 1.0000    | 0.9200  | 0.9583   |

> **OVERALL SYSTEM ACCURACY: 98.40%** 



## 6. Application Interface

The trained models are integrated into a desktop application with **two modes**:

### 6.1. Interpret Mode
1. **Language Detection** using GMMs  
2. **Speech-to-Text** using `speech_recognition`
3. **Translation** with Gemini-1.5-Flash  
4. **Text-to-Speech** synthesis via `pyttsx3`

#### Processing Pipeline
```
Audio Input → Language Detection → Speech Recognition → Translation → Speech Synthesis
     ↓              ↓                    ↓                 ↓              ↓
   GMM Classifier  98.4% Accuracy   speech_recognition   Gemini-1.5-Flash   pyttsx3
```


### 6.2. Conversation Mode
- Builds on Interpret Mode  
- Maintains **conversation history** for context-aware, multi-turn dialogue.

#### Processing Pipeline
```
Interpret Pipeline + ConversationBufferMemory(LangChain)
```




## 7. Installation
### 7.1. Clone the repository

### 7.2. Environment configuration

Create a `.env` file at the project root and add your Gemini API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 7.3. Install dependencies

```bash
pip install -r requirements.txt
```

### 7.4. Launch the application

```bash
python frame.py
```
## Contact

For questions, or  issues, please:
- Contact: [hibasofyan3@gmail.com]
- Contact: [aryadacademie@gmail.com]
