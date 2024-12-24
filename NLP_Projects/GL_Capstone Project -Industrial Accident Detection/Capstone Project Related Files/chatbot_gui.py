#!/usr/bin/env python
# coding: utf-8

# In[21]:


import random
import json


# In[22]:


def loadIntents():
    intents = json.loads(open('my_intents.json').read())
    return intents


# In[23]:


def saveMessage(tag,msg):
    try:
        user_dict =  json.loads(open('user_data.json').read())
    except:
        user_dict = {}
        
    user_dict[tag] = msg
    with open('user_data.json', 'w') as json_file:
      json.dump(user_dict, json_file)


# In[24]:


next_tag = None
def chatbot_response(tag, msg = ''):
    try:
        columns = ['Description','CriticalRisk','IncidentDate','CountryName', 'Location', 'IndustrialSector', 'Gender', 'Employment Type', 'Potential Accident Level']
        # loadIntents()
        intents = loadIntents()
        for intent in intents['intents']:
            if intent['tag'] == tag:
                res = intent['responses'][random.randrange(0,len(intent['responses']))]
                global next_tag
                next_tag = intent['nextTag']

                # if user intent is in column list of the dataset:
                if intent['tag'] in columns:
                    saveMessage(intent['tag'],msg)

                # if current intent = Employment Type/Predict accident level:
                if intent['tag'] == 'Potential Accident Level':
                    import NLP_Capstone_Server_Loading_File as model_Loader
                    prediction = model_Loader.predictAccidentLevel()
                    prediction_description = {'I':'Not Severe','II':'Minor','III':'Moderate','IV':'Severe','V':'Very Severe'}
                    res = res + prediction + '(' + prediction_description[prediction] + ')'
                    
                    # Convert the json to pandas dataframe
                    # userData = pandas DataFrame(json)
                    # res = res + predictAccidentLevel(userData)
        return res
    except:
        if next_tag == 'End of the conversation':
            raise Exception('Conversation Completed')
        else:
            raise Exception('Internal Error occured due to Invalid User Input')


# In[25]:


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    tag = next_tag
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(tag,msg)
        ChatLog.insert(END, "SafetyBot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


# In[26]:


from tkinter import *

window = Tk()
try:
    window.title("Safety chatbot")

    window.geometry("450x500")
    window.resizable(width=FALSE, height=FALSE)

    #Create Chat window
    ChatLog = Text(window, bd=0, bg="white", height="8", width="50", font="Arial",)

    ChatLog.config(state=DISABLED)

    #Bind scrollbar to Chat window
    scrollbar = Scrollbar(window, command=ChatLog.yview, cursor="heart")
    ChatLog['yscrollcommand'] = scrollbar.set

    # Send a welcome message
    ChatLog.config(state=NORMAL)
    ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    ChatLog.insert(END, "SafetyBot: " + chatbot_response('greeting') + '\n\n')
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)

    #Create Button to send message
    SendButton = Button(window, font=("Verdana",12,'bold'), text="Send", width="5", height=5,
                        bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                        command= send )

    #Create the box to enter message
    EntryBox = Text(window, bd=0, bg="white",width="29", height="5", font="Arial")
    #EntryBox.bind("<Return>", send)
    b1 = Button(window, text="Quit", command=window.destroy)

    #Place all components on the screen
    scrollbar.place(x=423,y=6, height=386)
    ChatLog.place(x=6,y=6, height=386, width=420)
    EntryBox.place(x=75, y=401, height=90, width=300)
    SendButton.place(x=6, y=401, height=90)
    b1.place(x = 394, y = 401, height=90)

    window.mainloop()
except Exception as error:
    print(repr(error))
    window.destroy()


# In[ ]:




