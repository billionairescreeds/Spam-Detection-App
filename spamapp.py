import numpy as np
import streamlit as st
import pandas as pd

import seaborn as sns


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


sms = pd.read_excel('SMSSpamCollection.xlsx',header=None)
sms.drop([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],axis=1, inplace=True)
sms [['Label', 'Message']] = sms[0].str.split('\t',expand=True)

sms.drop(0,axis=1,inplace= True)
import re  #it is a library for dealing with numbers
import string

def clean_mess(message):
    text = message.lower()
    text = re.sub(r'\d+','',text)
    text = text.translate(str.maketrans("", "",string.punctuation))
    text= text.strip()
    return text

sms['Message'].apply(clean_mess)
sms['Clean Message'] = sms['Message'].apply(clean_mess)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
vc = TfidfVectorizer(stop_words="english",max_features=5000)
x = vc.fit_transform(sms['Clean Message'])
y = sms['Label']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
log = LogisticRegression()
log.fit(x_train, y_train)
predictions = log.predict(x_test)
from sklearn import metrics
# print(metrics.confusion_matrix(y_test,predictions))
# print(metrics.classification_report(y_test,predictions))

def predict(message):
    clean_message = clean_mess(message)
    vcm = vc.transform([clean_message])
    predictions = log.predict(vcm)
    return predictions

st.title('Spam Detection App')
st.subheader('This app detects spam or ham of your mails')

mail = st.text_area('Please Enter mail')

if st.button('Predict'):
    prediction = predict(mail)
    st.write(prediction[0])



# import numpy as np
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import joblib
# import matplotlib.pyplot as plt
# import re
# import string

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import metrics

# # App Title
# st.title('Spam Detection App')

# # Load and clean data
# sms = pd.read_excel('SMSSpamCollection.xlsx', header=None)
# sms.drop(columns=list(range(1, 14)), inplace=True)  # Drop unwanted columns
# sms[['Label', 'Message']] = sms[0].str.split('\t', expand=True)
# sms.drop(columns=[0], inplace=True)

# # Text cleaning function
# def clean_mess(message):
#     text = message.lower()
#     text = re.sub(r'\d+', '', text)
#     text = text.strip()
#     return text

# # Apply cleaning
# sms['Clean Message'] = sms['Message'].apply(clean_mess)

# # Features and target
# x = sms['Clean Message']
# y = sms['Label']

# # Vectorization
# vc = TfidfVectorizer(stop_words="english", max_features=5000)
# x = vc.fit_transform(x)

# # Split data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # Train model
# log = LogisticRegression()
# log.fit(x_train, y_train)

# # Predictions
# predictions = log.predict(x_test)

# # Display results in Streamlit
# st.subheader('Evaluation Metrics')
# st.write("Confusion Matrix:")
# st.write(metrics.confusion_matrix(y_test, predictions))

# st.write("Classification Report:")
# st.text(metrics.classification_report(y_test, predictions))
