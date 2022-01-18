import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

app = Flask(__name__)

@app.route("/")
def index():
	return render_template('home.html')

@app.route("/result", methods=["POST","GET"])
def result():
	list_col=['item_weight', 'item_fat_content', 'item_visibility', 'item_type',
			  'item_mrp', 'outlet_establishment_year', 'outlet_size',
			  'outlet_location_type', 'outlet_type']

	item_weight=float(request.form['item_weight'])
	item_fat_content=str(request.form['item_fat_content'])
	item_visibility=float(request.form['item_visibility'])
	item_type=str(request.form['item_type'])
	item_mrp=float(request.form['item_mrp'])
	outlet_establishment_year=int(request.form['outlet_establishment_year'])
	outlet_size=str(request.form['outlet_size'])
	outlet_location_type=str(request.form['outlet_location_type'])
	outlet_type=str(request.form['outlet_type'])


   # print(item_fat_content)

	# Label Encoding

	le_item_fat_content=joblib.load(r'C:\wonder\flask\bigmartsales\models\item_fat_content.sav')
	le_item_type=joblib.load(r'C:\wonder\flask\bigmartsales\models\item_type.sav')
	le_outlet_size=joblib.load(r'C:\wonder\flask\bigmartsales\models\outlet_size.sav')
	le_outlet_location_type=joblib.load(r'C:\wonder\flask\bigmartsales\models\outlet_location_type.sav')
	le_outlet_type=joblib.load(r'C:\wonder\flask\bigmartsales\models\outlet_type.sav')
	# temp = list(le.classes_)

	item_fat_content=le_item_fat_content.transform([item_fat_content])
	item_type=le_item_type.transform([item_type])
	outlet_size=le_outlet_size.transform([outlet_size])
	outlet_location_type=le_outlet_location_type.transform([outlet_location_type])
	outlet_type=le_outlet_type.transform([outlet_type])

	inputs= np.array([item_weight,item_fat_content,item_visibility,item_type, item_mrp, outlet_establishment_year,
					  outlet_size,outlet_location_type, outlet_type]).reshape(1,-1)
	print("===================================================================inputs===================================================================")
	print(inputs)
	# print(temp)
	# print(outlet_type)


# Lets put all in the list



	# Lets apply Standard Scaler

	sc = joblib.load(r'C:\wonder\flask\bigmartsales\models\sc.sav')
	inputs_std= sc.transform(inputs)

	# Lets apply prediction

	model=joblib.load(r'C:\wonder\flask\bigmartsales\models\random_forest_grid.sav')
	

	prediction=model.predict(inputs_std)
	prediction=prediction.tolist()


	return jsonify({'prediction': prediction})



if __name__ == '__main__':
	app.run(debug=True,port=7890)
