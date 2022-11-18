import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from waitress import serve
import pickle
import plotly
import plotly.express as px
import json

app = Flask(__name__,template_folder="templates")

LRModel = pickle.load(open('logReg.pkl', 'rb')) # load model1 to the server, 'rb' - read binary
LRAccuracy = pickle.load(open('LRAcc.pkl', 'rb')) 
#df_time_series = pd.read_pickle('m2.pkl') 

def predictLogReg(arr):
    hello = int(LRModel.predict(arr))
    msg = ""
    if hello == 0:
        msg = "Prediction based on SepalLength = {0}, SepalWidth = {1}, PetalLength = {2} , PetalWidth =  {3} - Setosa Class".format(arr[0][0],arr[0][1],arr[0][2],arr[0][3])
    elif hello == 1:
        msg = "Prediction based on SepalLength = {0}, SepalWidth = {1}, PetalLength = {2} , PetalWidth =  {3} - Versicolor Class".format(arr[0][0],arr[0][1],arr[0][2],arr[0][3])
    elif hello == 2:
        msg = "Prediction based on SepalLength = {0}, SepalWidth = {1}, PetalLength = {2} , PetalWidth =  {3} - Virginica Class".format(arr[0][0],arr[0][1],arr[0][2],arr[0][3])
    return hello,msg

#default accuracy working 
@app.route('/LRAcc/', methods=['GET'])
def ComputeAccuracy():
    acc1 = str(LRAccuracy)
    return render_template("testAcc.html", acc = acc1)

@app.route('/panda/', methods=['GET'])
def Compute():
    acc1 = str(LRAccuracy)
    # return jsonify(acc1)
    return jsonify(
        {"predicted": 123,
         "SepalLength":126,
         "SepalWidth":53,
         "PetalLength":23,
         "PetalWidth":25
        })


@app.route('/logRegForm/', methods=['GET'])
def getForm():
        #return string
    return render_template("form.html")

#Prediction based(0,1) 4 input features
# POST via JSON from .NET
#for query string copy paste
# http://localhost:8080/result/?SepalLength=4.9&SepalWidth=3.0&PetalLength=1.5&PetalWidth=1.5
@app.route('/result/', methods=['GET','POST'])
def PredictWithInput():
    Msg = ""
    pred_get = 0
    content = ""
    if request.method == 'POST':
        content = request.get_json()
        if len(content) != 0 :
            SepalLength = float(content['SepalLength'])
            SepalWidth = float(content['SepalWidth'])
            PetalLength = float(content['PetalLength'])
            PetalWidth = float(content['PetalWidth'])
            print(SepalLength,SepalWidth,PetalLength,PetalWidth)
        else:
            SepalLength = float(request.form.get('SepalLength')) 
            SepalWidth = float(request.form.get('SepalWidth'))  
            PetalLength = float(request.form.get('PetalLength')) 
            PetalWidth = float(request.form.get('PetalWidth')) 
    else:
        SepalLength = request.args.get('SepalLength', type= float)
        SepalWidth = request.args.get('SepalWidth', type= float)
        PetalLength = request.args.get('PetalLength', type= float)
        PetalWidth = request.args.get('PetalWidth', type= float)

    if SepalLength < 0.0 or SepalWidth < 0.0 or PetalLength < 0.0 or PetalWidth < 0.0:
        Msg = "Negative values are not allowed"
        return jsonify({
        "Msg":Msg
        })
        
    arr = np.array([[SepalLength,SepalWidth,PetalLength,PetalWidth]])
    pred_get,Msg = predictLogReg(arr)
    return jsonify({
        "predict":pred_get,
        "Msg":Msg,
    })
    
# @app.route('/model2', methods=['GET'])
# def callModelTwo():
#    xValue = request.args.get('x', type= int)
#    print(df_time_series[xValue])
#    return str(df_time_series[xValue])



@app.route('/graph',methods=['GET'])
def bar_with_plotly():
    
   # Students data available in a list of list
    students = [['Akash', 34, 'Sydney', 'Australia'],
                ['Rithika', 30, 'Coimbatore', 'India'],
                ['Priya', 31, 'Coimbatore', 'India'],
                ['Sandy', 32, 'Tokyo', 'Japan'],
                ['Praneeth', 16, 'New York', 'US'],
                ['Praveen', 17, 'Toronto', 'Canada']]
      
    # Convert list to dataframe and assign column values
    df = pd.DataFrame(students,
                      columns=['Name', 'Age', 'City', 'Country'],
                      index=['a', 'b', 'c', 'd', 'e', 'f'])
      
    # Create Bar chart
    fig = px.bar(df, x='Name', y='Age', color='City', barmode='group')
      
    # Create graphJSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
      
    # Use render_template to pass graphJSON to html
    return render_template('bar.html', graphJSON=graphJSON)
  

if __name__ == '__main__':
    print("Starting the server.....")
    serve(app, host="0.0.0.0", port=8080)

#http://localhost:8080/model1?x=25
#http://localhost:8080/model2?x=30
