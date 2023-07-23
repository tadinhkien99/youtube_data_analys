# Youtube Comment Sentiment Analysis and Topic Classification

## 0. Overview

The YouTube API provides developers with access to various data related to YouTube videos, including comments posted by
users. This project utilizes this API to gather a significant amount of comment data for analysis.

The first step of the project involves retrieving comments from YouTube videos using the YouTube API. The API allows
access to comments along with their timestamps, ratings, and user information. By leveraging the API, the project
obtains a diverse dataset of comments from different videos and users.

Once the comment data is collected, the project applies sentiment analysis techniques to determine the sentiment
conveyed in each comment. Sentiment analysis involves using natural language processing (NLP) algorithms to identify and
categorize the sentiment as positive, negative, or neutral. This analysis provides insights into the overall sentiment
of the comments associated with a particular video or topic.

In addition to sentiment analysis, the project also focuses on emotion classification. Emotion classification aims to
identify and classify the specific emotions expressed in the comments. This step involves training machine learning
models on labeled emotion data to predict the emotions associated with new comments. The emotions could include
happiness, sadness, anger, excitement, and more.

To achieve accurate sentiment analysis and emotion classification, the project utilizes various NLP techniques and deep
learning. Include tokenization, deep learning models (transformers), and supervised learning techniques.

The ultimate goal of the project is to provide insights into the sentiment and emotions expressed by users in the
comments section of YouTube videos. The analysis can be valuable for content creators, marketers, and researchers to
understand audience reactions, engagement, and overall sentiment towards specific videos or topics.

By utilizing the YouTube API data and employing advanced NLP and machine learning techniques, this project contributes
to the field of sentiment analysis and emotion classification, enabling a better understanding of user sentiments and
emotions in the context of YouTube comments.

- [Youtube Comment Sentiment Analysis and Topic Classification](#project-title)
    - [0. Overview](#0-overview)
    - [1. Dataset](#1-dataset)
    - [2. Run code](#2-run-code)
    - [3. Roadmap](#3-roadmap)
    - [4. Feature Processed](#4-feature-processed)
    - [5. Models](#5-models)
    - [6. Evaluate](#6-evaluate)
    - [7. Test](#7-test)
    - [8. Deployment](#7-deployment)
    - [9. License](#8-license)
    

## 1. Dataset

The project relies on the TMDb API to retrieve movie data, including information such as title, release year, genre,
runtime, budget, and popularity. This data serves as the feature set used for training and evaluating the machine
learning models. The API provides a vast collection of movies, ensuring a diverse dataset for model training.

| Parameter         | Type     | Sample Value                                          |
|:------------------|:---------|:------------------------------------------------------|
| `Video ID`        | `Object` | `ekre0aU6CiI`                                         |
| `Comment`         | `Object` | `Speaking straight facts Zero`                        |
| `Keyword`         | `Object` | `Game reviews`                                        |
| `Topic`           | `Object` | `https://en.wikipedia.org/wiki/Action-adventure_game` |
| `Sentiment`       | `Float`  | `0.2`                                                 |
| `Sentiment_Label` | `Object` | `Positive`                                            |

## 2. Run code

Clone the project

```bash

  git clone https://link-to-project
```

Go to the project directory

```bash
  cd project_folder
```

Install libraries

```bash
  pip install -r requirements.txt
```

Prepare dataset

```bash
  jupyter notebook youtube_topic_preparation.ipynb
```

Run notebook EDA and Model on dataset

```bash
  jupyter notebook youtube_sentiment_topic.ipynb
```

## 3. Roadmap

- Prepare `api_key` from [YouTube API](https://developers.google.com/youtube/v3).
- Collect data from [YouTube API](https://developers.google.com/youtube/v3).
- YouTube's data will be preprocessed, EDA before feed to model NLP.
- After training model, will calculate accuracy score to compare them.
- Make pipeline to predict new data.

## 4. Feature Processed

To prepare the data for the machine learning models, feature engineering techniques were applied. This involved
transforming and encoding categorical variables, drop non english comment, remove special characters, handling missing data, and scaling numerical features to ensure
compatibility with the models. The engineered features were then used to train and evaluate the models.

## 5. Models

- [Bert](https://huggingface.co/bert-base-uncased)

  The BERT (Bidirectional Encoder Representations from Transformers) model, developed by Google, is a state-of-the-art transformer-based model for natural language processing tasks. It uses a bidirectional approach to capture the context of words in a sentence, resulting in rich contextual representations. The Hugging Face library provides pre-trained BERT models and tools for easy implementation, fine-tuning, and transfer learning for a wide range of NLP tasks.

- [DistilRoBerta](https://huggingface.co/distilroberta-base)

  DistilRoBERTa is a distilled version of the RoBERTa model. It is a transformer-based model that uses a bidirectional approach to capture the context of words in a sentence. The model is pre-trained on a large corpus of data and fine-tuned on the emotion classification task.

- [SentimentIntensityAnalyzer](https://www.nltk.org/api/nltk.sentiment.vader.html)

  The SentimentIntensityAnalyzer is a rule-based sentiment analysis tool that uses a lexicon of words to determine the sentiment conveyed in a text. The lexicon contains a list of positive and negative words, along with their intensity scores. The tool calculates the sentiment score of a text by summing the intensity scores of the words in the text. The sentiment score can range from -1 to 1, with -1 indicating negative sentiment, 0 indicating neutral sentiment, and 1 indicating positive sentiment.

## 6. Evaluate

- Accuracy score

## 7. Test

- Model Sentiment return: `{'negative': '17.0%', 'neutral': '36.0%', 'positive': '47.0%'}`

- Model Topic return: `{'anger': '8.0%', 'disgust': '3.0%', 'fear': '2.0%', 'joy': '33.0%', 'neutral': '28.0%', 'sadness': '14.0%', 'surprise': '12.0%'}`

## 8. Deployment

For deploy, we can use cloud service: **AWS**, **Azure**, **GCP**,...

## 9. License

[MIT](https://choosealicense.com/licenses/mit/)

## 🚀 About Me

- [Andreas](your-link)

## Support

For support, email yourmail@.com.
