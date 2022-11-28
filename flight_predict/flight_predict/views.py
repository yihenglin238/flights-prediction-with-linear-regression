import code
from django.shortcuts import render
import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder


# our home page view
def index(request):    
    return render(request, 'index.html')


 #custom method for generating predictions
def getPredictions(arline, flight, city, dptime, stops, artime, dest, classes, duration,days):
    import pickle
    here = os.path.dirname(os.path.abspath(__file__))

    model_name = os.path.join(here, 'flight_ticket_price_predict_model.sav')
    scaled_name = os.path.join(here,'scaler.sav')
    #le_name = os.path.join(here,'le.sav')
    model = pickle.load(open(model_name, "rb"))
    scaled = pickle.load(open(scaled_name, "rb"))
    le = LabelEncoder()
    test = [[arline,flight,city,dptime,stops,artime,dest,classes,duration,days]]
    test = pd.DataFrame(test)
    for col in test.columns:
        if test[col].dtype=='object':
            test[col] = le.fit_transform(test[col])
    prediction = model.predict(scaled.transform(test))
    
    return prediction
        

# our result page view
def result(request):
    arline = request.GET['arline']
    flight = request.GET['flight']
    city = request.GET['city']
    dptime = request.GET['dptime']
    stops = request.GET['stops']
    artime = request.GET['artime']
    dest = request.GET['dest']
    classes = request.GET['class']
    duration = float(request.GET['duration'])
    days = int(request.GET['days'])
    result = getPredictions(arline, flight, city, dptime, stops, artime, dest, classes, duration,days)

    return render(request, 'index.html', {'result':result})