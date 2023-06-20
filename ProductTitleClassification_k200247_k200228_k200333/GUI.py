import tkinter as tk
import Utilities
import pickle
from collections import defaultdict
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np

with open('unique_labels.json', 'r') as f:
    unique_labels = json.load(f)

unique_label_c1 = np.array(unique_labels['c1'])
unique_label_c2 = np.array(unique_labels['c2'])
unique_label_c3 = np.array(unique_labels['c3'])



def decode_cat01(number):
    le = LabelEncoder()
    le.fit(unique_label_c1)
    LabelEncoder()
    le.transform(unique_label_c1)
    return str(le.inverse_transform([number]))

def decode_cat02(number):
    le = LabelEncoder()
    le.fit(unique_label_c2)
    LabelEncoder()
    le.transform(unique_label_c2)
    return str(le.inverse_transform([number]))

def decode_cat03(number):
    le = LabelEncoder()
    le.fit(unique_label_c3)
    LabelEncoder()
    le.transform(unique_label_c3)
    return str(le.inverse_transform([number]))

def Load_Svm_Models():
    model_c1_svm = pickle.load(open('model1.pickle', 'rb'))
    model_c2_svm = pickle.load(open('model2.pickle', 'rb'))
    model_c3_svm = pickle.load(open('model3.pickle', 'rb'))
    return model_c1_svm, model_c2_svm, model_c3_svm

def Predict_Query_SVM(query):
    tfidf_vectorizer = pickle.load(open('vectorizer.pickle','rb'))
    corpus_vocabulary = defaultdict(None, copy.deepcopy(tfidf_vectorizer.vocabulary_))
    corpus_vocabulary.default_factory = corpus_vocabulary.__len__
    m1,m2,m3=Load_Svm_Models()
    tfidf_transformer_query = TfidfVectorizer()
    tfidf_transformer_query.fit_transform(query)
    for word in tfidf_transformer_query.vocabulary_.keys():
        if word in tfidf_vectorizer.vocabulary_:
            corpus_vocabulary[word]

    tfidf_transformer_query_sec = TfidfVectorizer(vocabulary=corpus_vocabulary)
    query_tfidf_matrix = tfidf_transformer_query_sec.fit_transform(query)
    return m1.predict(query_tfidf_matrix), m2.predict(query_tfidf_matrix),m3.predict(query_tfidf_matrix)

def codeRunner(query,output_label,output_label1,output_label2):
    query=Utilities.PreProcessing(query)
    query=[query]
    m1,m2,m3=Predict_Query_SVM(query)
    Ctg1=decode_cat01(int(m1))
    Ctg2=decode_cat02(int(m2))
    Ctg3=decode_cat03(int(m3))
    output_label.configure(text=Ctg3)
    output_label1.configure(text=Ctg1)
    output_label2.configure(text=Ctg2)


root = tk.Tk()
root.attributes('-fullscreen', True)
image_path = r'03.png'
image = tk.PhotoImage(file=image_path)
label = tk.Label(root, image=image)
label.pack(fill=tk.BOTH, expand=tk.YES)
def exit_fullscreen(event):
    root.attributes('-fullscreen', False)
    root.quit()
root.bind('<Escape>', exit_fullscreen)
font = ("Comic Sans MS", 18, "italic")
string_input = tk.Entry(root,justify="center",font=font)
string_input.place(relx=0.25, rely=0.35, relwidth=0.4, relheight=0.15, anchor="center")
string_input.insert(0, "Provide Title's Description...")
button1 = tk.Button(root, text="CLASSIFY", bg='red', fg='white', font=font, borderwidth=0, highlightthickness=0, command=lambda: codeRunner(string_input.get(),output_label,output_label1,output_label2))
button1.place(relx=0.416, rely=0.7, relheight=0.05, anchor="center")
font1 = ("Comic Sans MS", 10, "italic")
output_label = tk.Label(root, font=font1,bg='yellow', fg='red', borderwidth=0, highlightthickness=0)
output_label.place(relx=0.76, rely=0.65, anchor="center")
output_label1 = tk.Label(root, font=font1,bg='yellow', fg='red', borderwidth=0, highlightthickness=0)
output_label1.place(relx=0.63, rely=0.33, anchor="center")
output_label2 = tk.Label(root, font=font1,bg='yellow', fg='red', borderwidth=0, highlightthickness=0)
output_label2.place(relx=0.876, rely=0.33, anchor="center")
root.mainloop()