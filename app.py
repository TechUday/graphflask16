from flask import Flask, render_template, request
from model import search, stockpredict
import sqlite3
from flask import Flask, render_template, request
from model import search, stockpredict
import sqlite3
from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle
#importing libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.models import load_model
from flask import Flask,render_template,request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename


def find_top_confirmed(n = 15):

  import pandas as pd

  corona_df=pd.read_csv("covid-19-dataset-3.csv")
  by_country = corona_df.groupby('Province_State').sum()[['Confirmed', 'Deaths', 'Recovered', 'Active']]
  cdf = by_country.nlargest(n, 'Confirmed')[['Confirmed']]
  return cdf


cdf=find_top_confirmed()
pairs=[(province_state,confirmed) for province_state,confirmed in zip(cdf.index,cdf['Confirmed'])]


import folium
import pandas as pd
corona_df = pd.read_csv("covid-19-dataset-3.csv")
corona_df=corona_df[['Lat','Long_','Confirmed']]
corona_df=corona_df.dropna()

m=folium.Map(location=[34.223334,-82.461707],
            tiles='Stamen toner',
            zoom_start=8)

def circle_maker(x):
    folium.Circle(location=[x[0],x[1]],
                 radius=float(x[2]),
                 color="red",
                 popup='confirmed cases:{}'.format(x[2])).add_to(m)
corona_df.apply(lambda x:circle_maker(x),axis=1)

html_map=m._repr_html_()



## graph2

def find_top_confirmed1(n = 15):

  import pandas as pd
  corona_df1 = pd.read_csv('covid-19-dataset-1.csv')
  by_country1 = corona_df1.groupby('Country_Region').sum()[['Confirmed', 'Deaths', 'Recovered', 'Active']]
  cdf1 = by_country1.nlargest(n, 'Confirmed')[['Confirmed']]
  return cdf1

cdf1=find_top_confirmed1()
pairs1=[(province_state1,confirmed1) for province_state1,confirmed1 in zip(cdf1.index,cdf1['Confirmed'])]

import folium
import pandas as pd
corona_df1 = pd.read_csv('covid-19-dataset-1.csv')

corona_df1=corona_df1.dropna()

m1=folium.Map(location=[34.223334,-82.461707],
            tiles='Stamen toner',
            zoom_start=8)

def circle_maker1(x1):
    folium.Circle(location=[x1[0],x1[1]],
                 radius=float(x1[2])*10,
                 color="blue",
                 popup='{}\n confirmed cases:{}'.format(x1[3],x1[2])).add_to(m1)
corona_df1[['Lat','Long_','Confirmed','Combined_Key']].apply(lambda x1:circle_maker1(x1),axis=1)

html_map1=m1._repr_html_()



## graph3


def find_top_confirmed2(n = 15):

    import pandas as pd
    corona_df2=pd.read_csv("covid-19-dataset-2.csv")
    by_country2 = corona_df2.groupby('Country_Region').sum()[['Confirmed', 'Deaths', 'Recovered', 'Active']]
    cdf2 = by_country2.nlargest(n, 'Confirmed')[['Confirmed']]
    return cdf2

cdf2=find_top_confirmed2()
pairs2=[(country2,confirmed2) for country2,confirmed2 in zip(cdf2.index,cdf2['Confirmed'])]


import folium
import pandas as pd
corona_df2 = pd.read_csv("covid-19-dataset-2.csv")
corona_df2=corona_df2[['Lat','Long_','Confirmed']]
corona_df2=corona_df2.dropna()

m2=folium.Map(location=[34.223334,-82.461707],
            tiles='Stamen toner',
            zoom_start=8)

def circle_maker2(x2):
    folium.Circle(location=[x2[0],x2[1]],
                 radius=float(x2[2]),
                 color="green",
                 popup='confirmed cases:{}'.format(x2[2])).add_to(m2)
corona_df2.apply(lambda x2:circle_maker2(x2),axis=1)

html_map2=m2._repr_html_()


## x-ray deployment

model=load_model('Covid_model.h5')

def model_predict(img_path,model):
    test_image=image.load_img(img_path,target_size=(224,224))
    test_image=image.img_to_array(test_image)
    test_image=test_image/255
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image)
    return result






app = Flask(__name__)
app.debug = True

@app.route("/home")
def home():
    return render_template("home.html",table=cdf, cmap=html_map,pairs=pairs)



@app.route('/gra')
def gra():
    return render_template("gra.html",table=cdf1, cmap=html_map1,pairs=pairs1)

@app.route('/graph')
def graph():
    return render_template("graph.html",table=cdf2, cmap=html_map2,pairs=pairs2)


@app.route('/dash')
def dash():
    return render_template("dash.html")


@app.route("/")
def index():
    return render_template("index.html")





@app.route("/signup")
def signup():
    
    
    name = request.args.get('username','')
    number = request.args.get('number','')
    email = request.args.get('email','')
    password = request.args.get('password','')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `detail` (`name`,`number`,`email`, `password`) VALUES (?, ?, ?, ?)",(name,number,email,password))
    con.commit()
    con.close()

    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `name`, `password` from detail where `name` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("home.html",table=cdf, cmap=html_map,pairs=pairs)

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html",table=cdf, cmap=html_map,pairs=pairs)
    else:
        return render_template("signin.html")

# @app.route("/home")
# def home():
#     return render_template("home.html",table=cdf, cmap=html_map,pairs=pairs)


@app.route('/signout')
def signout():
	return render_template('signin.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/about')
def about():
	return render_template('about.html')



@app.route('/newgra')
def newgra():
	return render_template('newgra.html')

@app.route('/1')
def one():
	return render_template('1.html')

@app.route('/2')
def two():
	return render_template('2.html')

@app.route('/4')
def four():
	return render_template('4.html')

@app.route('/5')
def five():
	return render_template('5.html')




@app.route('/ray',methods=['GET'])
def ray():
    return render_template('ray.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']

        basepath=os.path.dirname(os.path.realpath('__file__'))
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        result = model_predict(file_path,model)

        categories=['Covid','Normal']

        pred_class = result.argmax()
        output=categories[pred_class]
        if output=="Covid":
            return render_template("one.html")
        elif output=="Normal":
            return render_template("two.html")
        return output
    return None

    
if __name__ == "__main__":
    app.run(debug=False,port=7989)
