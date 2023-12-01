from flask import Flask,jsonify,render_template,request
from preprocessing import rem_skew,scale
import pandas as pd
import joblib
sr_model_svc=joblib.load("ml_model_svc.pkl")
sr_model_knc=joblib.load("ml_model_knc.pkl")
app=Flask(__name__)

@app.route("/home",methods=["GET","POST"])
def func1():
    return render_template("index.html")

@app.route("/predict_output",methods=["GET","POST"])
def func2():
    try:
        sepal_length=request.form.get("s_l")
        sepal_width=request.form.get("s_w")
        petal_length=request.form.get("p_l")
        petal_width=request.form.get("p_w")
        '''
        Whenever we are getting any data from UI it comes in the form of string. If you wnt integer or float you need to type caste it.
        '''
        sepal_length=float(sepal_length)
        sepal_width=float(sepal_width)
        petal_length=float(petal_length)
        petal_width=float(petal_width)
        feature_df=pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                                columns=["sepal_length","sepal_width","petal_length","petal_width"])
        
        feature_df=rem_skew(feature_df)
        feature_df=scale(feature_df)
        array_input=feature_df.values
        result1=sr_model_svc.predict(array_input)# output will look like [0] or [1] or [2]
        if result1[0]==0:
            result1="Setosa"
        elif result1[0]==1:
            result1="Versicolor"
        elif result1[0]==2:
            result1="Virginica"
        else:
            return jsonify({"output":"The model is not able to predict the output"})
        result2=sr_model_knc.predict(array_input)# output will look like [0] or [1] or [2]
        if result2[0]==0:
            result2="Setosa"
        elif result2[0]==1:
            result2="Versicolor"
        elif result2[0]==2:
            result2="Virginica"
        else:
            return jsonify({"output":"The model is not able to predict the output"})
        return f"The output of the flower using support vector classifier is {result1} and the output using Kneighbour classifier is {result2}"
    except Exception as e:
        print(e)
        return jsonify({"output": "Please fill all the required input data"})
if __name__=="__main__":
    app.run(debug=True,port=1010)
    