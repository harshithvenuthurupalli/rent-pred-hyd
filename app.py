from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('localityId'):
        m = np.mean(subdf.rent_per_sqft)
        st = np.std(subdf.rent_per_sqft)
        reduced_df = subdf[(subdf.rent_per_sqft > (m - st)) & (subdf.rent_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

house_data = pd.read_csv(r"C:\Users\harsh\Downloads\hyd_v2.csv\hyd_v2.csv")
new_data = house_data.drop(['amenities', 'locality', 'balconies', 'lift', 'active', 'loanAvailable', 'location', 
                            'ownerName', 'parkingDesc', 'propertyTitle', 'propertyType', 'combineDescription', 
                            'completeStreetName', 'facing', 'facingDesc', 'furnishingDesc', 'gym', 'id', 
                            'isMaintenance', 'weight', 'waterSupply', 'swimmingPool', 'shortUrl', 
                            'sharedAccomodation', 'reactivationSource'], axis=1)
new_data.replace({'parking': {'BOTH': 0, 'TWO_WHEELER': 1, 'FOUR_WHEELER': 2, 'NONE': 3}}, inplace=True)
new_data.replace({'type_bhk': {'RK1': 0.5, 'BHK1': 1, 'BHK2': 2, 'BHK3': 3, 'BHK4': 4, 'BHK4PLUS': 5}}, inplace=True)
new_data.replace({'maintenanceAmount': {'None': int(0)}}, inplace=True)
new_data['maintenanceAmount'].fillna(new_data['maintenanceAmount'].mean(), inplace=True)
new_data['maintenanceAmount'] = new_data['maintenanceAmount'].astype(int)
hd = new_data.copy()
hd1 = hd.drop(['maintenanceAmount', 'deposit', 'property_age', 'totalFloor'], axis=1)
hd2 = hd1.copy()
hd2['rent_per_sqft'] = hd2['rent_amount'] / hd2['property_size']
location_stats = hd2.groupby('localityId')['localityId'].agg('count').sort_values(ascending=False)
location_less_than_10 = location_stats[location_stats <= 10]
hd2.localityId = hd2.localityId.apply(lambda x: 'NOT_FOUND' if x in location_less_than_10 else x)
hd3 = hd2[~(hd2.property_size / hd2.type_bhk < 300)]
hd4 = remove_pps_outliers(hd3)
hd5 = hd4[hd4.bathroom <= hd4.type_bhk + 1]
hd6 = hd5.drop('rent_per_sqft', axis=1)
dummies = pd.get_dummies(hd6.localityId)
print(dummies)
hd7 = pd.concat([hd6.drop('localityId', axis='columns'), dummies.drop('NOT_FOUND', axis='columns')], axis='columns')
print(hd7)
X = hd7.drop('rent_amount', axis='columns')
Y = hd7['rent_amount']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
lr = LinearRegression()
lr.fit(X_train, Y_train)

y_pred = lr.predict(X_test)
print('Mean Absolute Error:', mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(Y_test, y_pred))
print('R^2 Score:', r2_score(Y_test, y_pred))

def predict_price(localityId, bathroom, floor, parking, property_size, type_bhk, maintenance):
    loc_index = np.where(X.columns == localityId)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = bathroom
    x[1] = floor
    x[2] = parking
    x[3] = property_size
    x[4] = type_bhk
    x[5] = maintenance
    if loc_index >= 0:
        x[loc_index] = 1
    return lr.predict([x])[0]

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    localityId = data['localityId']
    bathroom = data['bathroom']
    floor = data['floor']
    parking = data['parking']
    property_size = data['property_size']
    type_bhk = data['type_bhk']
    maintenance = data['maintenance']
    price = predict_price(localityId, bathroom, floor, parking, property_size, type_bhk, maintenance)
    return jsonify({'price': price})

if __name__ == "__main__":
    app.run(debug=True)
