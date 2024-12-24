#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().system(' pip install --upgrade category_encoders')


# In[ ]:


#get_ipython().system('python -m nltk.downloader stopwords')


# In[ ]:


#get_ipython().system('python -m nltk.downloader punkt')


# In[1]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re


# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders 
from category_encoders.binary import BinaryEncoder


# In[3]:


import json
from os import listdir
from os.path import isfile, join


# In[4]:


import pickle
# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[6]:


from datetime import datetime
import calendar

# In[25]:


def clean_text(text):
      text = text.lower()
      pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
      text = pattern.sub('', text)
      text = " ".join(filter(lambda x:x[0]!='@', text.split()))
      emoji = re.compile("["
                            u"\U0001F600-\U0001FFFF"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
      
      text = emoji.sub(r'', text)
      text = text.lower()
      text = re.sub(r"i'm", "i am", text)
      text = re.sub(r"he's", "he is", text)
      text = re.sub(r"she's", "she is", text)
      text = re.sub(r"that's", "that is", text)        
      text = re.sub(r"what's", "what is", text)
      text = re.sub(r"where's", "where is", text) 
      text = re.sub(r"\'ll", " will", text)  
      text = re.sub(r"\'ve", " have", text)  
      text = re.sub(r"\'re", " are", text)
      text = re.sub(r"\'d", " would", text)
      text = re.sub(r"\'ve", " have", text)
      text = re.sub(r"won't", "will not", text)
      text = re.sub(r"don't", "do not", text)
      text = re.sub(r"did't", "did not", text)
      text = re.sub(r"can't", "can not", text)
      text = re.sub(r"it's", "it is", text)
      text = re.sub(r"couldn't", "could not", text)
      text = re.sub(r"have't", "have not", text)
      text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
      return text
# In[ ]:


def predictAccidentLevel():


# In[7]:


#user_dict =  json.loads(open('/content/drive/MyDrive/NLP_Chatbot_Capstone_Project/user_data.json').read())
    user_dict =  json.loads(open('user_data.json').read())

# In[8]:


# temp_test_entry={
#     'Data':'2016-01-01 00:00:00',
#     'Countries':'Country_01',
#     'Local':'Local_01',
#     'Industry Sector':'Mining',
#     'Accident Level': 'I',
#     'Potential Accident Level':'IV',
#     'Genre':'Male',
#     'Employee or Third Party':'Third Party',
#     'Critical Risk':'Pressed',
#     'Description':'While removing the drill rod of the Jumbo 08 for maintenance, the supervisor proceeds to loosen the support of the intermediate centralizer to facilitate the removal, seeing this the mechanic supports one end on the drill of the equipment to pull with both hands the bar and accelerate the removal from this, at this moment the bar slides from its point of support and tightens the fingers of the mechanic between the drilling bar and the beam of the jumbo.',
# }


# In[9]:


    temp_test_entry={}
    temp_test_entry['Description'] = user_dict['Description']
    temp_test_entry['Critical Risk'] = user_dict['CriticalRisk']
    temp_test_entry['Data'] = user_dict['IncidentDate']
    temp_test_entry['Countries'] = user_dict['CountryName']
    temp_test_entry['Local'] = user_dict['Location']
    temp_test_entry['Industry Sector'] = user_dict['IndustrialSector']
    temp_test_entry['Genre'] = user_dict['Gender']
    temp_test_entry['Employee or Third Party'] = user_dict['Employment Type']
    temp_test_entry['Potential Accident Level'] = user_dict['Potential Accident Level']
#temp_test_entry['Accident Level'] = user_dict['Description']


# In[10]:


    date_event=datetime.strptime(temp_test_entry['Data'], '%Y-%m-%d %H:%M:%S')
    temp_test_entry['Date of Incidents']=date_event
    temp_test_entry['Year of Incident'],temp_test_entry['Month of Incident']=date_event.year,date_event.month
    temp_test_entry['Date of Incident'],temp_test_entry['Day of Incident']=date_event.day,calendar.day_name[date_event.weekday()]


# In[11]:


#path_initial_encoders='/content/drive/MyDrive/NLP_Chatbot_Capstone_Project/Initial Encoders'
#path_cleaning_encoders='/content/drive/MyDrive/NLP_Chatbot_Capstone_Project/Cleaning Encoders'
#path_tf_idf_vectorizer='/content/drive/MyDrive/NLP_Chatbot_Capstone_Project/Trained TF-IDF Vectorizer'
#path_default_classifiers='/content/drive/MyDrive/NLP_Chatbot_Capstone_Project/Trained Default Machine Learning Classifiers'


# In[12]:


    path_initial_encoders='Initial Encoders'
    path_cleaning_encoders='Cleaning Encoders'
    path_tf_idf_vectorizer='Trained TF-IDF Vectorizer'
    path_default_classifiers='Trained Default Machine Learning Classifiers'


# In[13]:


    encoded_test_entry=temp_test_entry


# ### **Initial Encoders**

# In[14]:


    files=[f for f in listdir(path_initial_encoders) if isfile(join(path_initial_encoders, f))]
#files


# In[15]:


#files.remove('Ordinal Encoder_Accident Level')


# In[16]:


    ordinal_encoded_features=[]
    label_encoded_features=[]
    binary_encoded_features=[]


# In[17]:


    for i in range(0,len(files)):
      with open(path_initial_encoders+'/'+files[i] ,'rb') as f:
        encoder=pickle.load(f)
      if files[i]=='Ordinal Encoder_Accident Level':
        final_encoder=encoder
      feature_name=files[i].split('_')[1].strip()
      if files[i].split('_')[0].strip().__contains__('Label'):
        encoded_test_entry[feature_name]=encoder.transform(np.array(temp_test_entry[feature_name]).reshape(-1,1))[0]
        label_encoded_features.append(feature_name)
        # print(encoder.classes_)
      elif files[i].split('_')[0].strip().__contains__('Ordinal'):
        if feature_name=='Accident Level':
          continue
        else:
          encoded_test_entry[feature_name]=encoder.transform(np.array(temp_test_entry[feature_name]).reshape(-1,1))[0][0]
          ordinal_encoded_features.append(feature_name)
          # print(encoder.categories_)
      else:
        continue
#encoded_test_entry


# ### **Cleaning Encoders**

# In[18]:


    files=[f for f in listdir(path_cleaning_encoders) if isfile(join(path_cleaning_encoders, f))]
#files


# In[19]:


    for i in range(0,len(files)):
      #print(files[i])
      with open(path_cleaning_encoders+'/'+files[i] ,'rb') as f:
        encoder=pickle.load(f)
      feature_name=files[i].split('_')[1].strip()
      if files[i].split('_')[0].strip().__contains__('Label'):
        # print(feature_name,temp_test_entry[feature_name])
        # print(encoder.transform(np.array(temp_test_entry[feature_name]).reshape(-1,1)))    
        encoded_test_entry[feature_name]=encoder.transform(np.array(temp_test_entry[feature_name]).reshape(-1,1))[0]
        label_encoded_features.append(feature_name)
        # print(encoder.classes_)
      elif files[i].split('_')[0].strip().__contains__('Ordinal'):
        # print(feature_name,temp_test_entry[feature_name])
        # print(encoder.transform(np.array(temp_test_entry[feature_name]).reshape(-1,1)))
        encoded_test_entry[feature_name]=encoder.transform(np.array(temp_test_entry[feature_name]).reshape(-1,1))[0][0]
        ordinal_encoded_features.append(feature_name)
        # print(encoder.categories_)
      elif files[i].split('_')[0].strip().__contains__('Binary'):
        binary_encoder=encoder
      elif files[i].split('_')[0].strip().__contains__('Scaler'):
        # print(feature_name)
        value=(np.array([encoded_test_entry['Potential Accident Level']]).reshape(-1,1))
        encoded_test_entry['Potential Accident Level']=encoder.transform(value)[0][0]
      else:
        continue


# In[20]:


#y=encoded_test_entry['Accident Level']
#del encoded_test_entry['Accident Level']


# In[21]:


    encoded_test_entry['index']=0
    temp=[]
    for i in list(encoded_test_entry.keys()):
      temp.append(encoded_test_entry[i])
    temp=pd.DataFrame(np.array([temp]),columns=list(encoded_test_entry.keys()))
#temp.columns


# In[22]:


    encodings=binary_encoder.transform(temp)


# ### **TF-IDF Vectorizer**

# In[23]:


    files=[f for f in listdir(path_tf_idf_vectorizer) if isfile(join(path_tf_idf_vectorizer, f))]


# In[24]:


    stop_words = list(stopwords.words('english'))
    punctuations = list(string.punctuation)
    stop_words_list = stop_words+punctuations





# In[26]:


    encodings['Merged_Description']=encodings['Critical Risk']+encodings['Description']


# In[27]:


    with open(path_tf_idf_vectorizer+'/'+files[0],'rb') as f:
      tf_idf_vec=pickle.load(f)
    encodings['Clean Words']=(clean_text(" ".join([w for w in word_tokenize(encodings['Description'][0]) if not w in stop_words_list])))
    words_transformed_test=tf_idf_vec.transform(encodings['Clean Words'])
    encoded_bag_of_words_model=pd.DataFrame(words_transformed_test.todense(),columns=['word_'+str(i)+'_'+tf_idf_vec.get_feature_names()[i] for i in range(0,words_transformed_test.todense().shape[1])])


# In[28]:


    final_dataframe=pd.concat([encodings,encoded_bag_of_words_model],axis=1)


# In[29]:


    columns_to_delete=['Critical Risk','Description','Data','Date of Incidents','Merged_Description','index','Clean Words']
    for j in columns_to_delete:
      del final_dataframe[j]


# ### **Modelling**

# In[30]:


    files=[f for f in listdir(path_default_classifiers) if isfile(join(path_default_classifiers, f))]


# In[31]:


    with open(path_default_classifiers+'/'+files[0],'rb') as f:
      random_forest=pickle.load(f)


# In[32]:


    prediction=random_forest.predict(final_dataframe.iloc[0].values.reshape(1,-1))


# In[33]:


    predicted_class=final_encoder.inverse_transform([prediction])[0][0]
#predicted_class,final_encoder.inverse_transform([[y]])[0][0]
    return predicted_class


# In[ ]:


