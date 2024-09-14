import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open('logistic_regression.pickle', 'rb') as f:
    logistic_regression = pickle.load(f)

with open('naive_bayes.pickle', 'rb') as f:
    naive_bayes = pickle.load(f)

with open('random_forest.pickle', 'rb') as f:
    random_forest = pickle.load(f)
    
with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f);

def main():
    # Set page configuration
    st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

    # Custom CSS to change background color and style input fields
    st.markdown("""
        <style>
        .st-emotion-cache-13k62yr, .st-emotion-cache-h4xjwg{
            background-color:rgb(7 101 163);
        }
        .reportview-container {
            background-color: #f0f8ff; /* Light blue background */
        }
        .css-1l02u2i {
            border-radius: 50%; /* Make input fields circular */
            padding: 10px;
            text-align: center;
        }
        .css-1l02u2i input {
            border-radius: 50%; /* Ensure the input itself is circular */
            text-align: center;
        }
        .st-b7,.st-emotion-cache-1hgxyac{
            background-color:#fff;
        }
        .st-emotion-cache-1hgxyac, .st-bb{
         color:#000   
        }
        
        .st-emotion-cache-1hgxyac:hover:enabled, .st-emotion-cache-1hgxyac:focus:enabled{
            background-color:#dbb300;
        }
        
        .st-emotion-cache-19cfm8f, .st-emotion-cache-19cfm8f.focused{
            border-color:#dbb300;
            border-width:2px;
            border-radius:0.2rem;
        }
        .st-emotion-cache-1hgxyac:last-of-type{
            border-top-right-radius:0rem;
            border-bottom-right-radius:0rem;
        }
        .st-au{
            border-bottom-left-radius:0rem;
            border-top-left-radius:0rem;
        }
        .st-br{
            caret-color:#000;
        }
        .st-emotion-cache-15hul6a{
            background-color:#dbb300;
        }
        .st-emotion-cache-15hul6a:hover{
            background-color:#dbb300;
            color:#fff;
            border-color:#dbb300;
        }
        .st-emotion-cache-15hul6a:focus:not(:active){
            background-color:#dbb300;
            color:#fff;
            border-color:#dbb300;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title('Bankruptcy Prediction')
    st.markdown("""
        This application predicts whether a company will go bankrupt or not based on the financial details provided
    """)

    # Create two columns for input fields
    col1, col2, col3, col4 = st.columns(4)

    best_features = [{'name': ' Fixed Assets Turnover Frequency', 'min': '0.0', 'max': '9990000000.0'}, {'name': ' Cash/Current Liability', 'min': '0.0', 'max': '9650000000.0'}, {'name': ' Research and development expense rate', 'min': '0.0', 'max': '9980000000.0'}, {'name': ' Cash Turnover Rate', 'min': '0.0', 'max': '10000000000.0'}, {'name': ' Total Asset Growth Rate', 'min': '0.0', 'max': '9990000000.0'}, {'name': ' Quick Asset Turnover Rate', 'min': '0.0', 'max': '10000000000.0'}, {'name': ' Fixed Assets to Assets', 'min': '0.0', 'max': '8320000000.0'}, {'name': ' Total assets to GNP price', 'min': '0.0', 'max': '9820000000.0'}, {'name': ' Operating Expense Rate', 'min': '0.0', 'max': '9990000000.0'}, {'name': ' Inventory Turnover Rate (times)', 'min': '0.0', 'max': '9990000000.0'}, {'name': ' Revenue per person', 'min': '0.0', 'max': '8810000000.0'}, {'name': ' Net Value Growth Rate', 'min': '0.0', 'max': '9330000000.0'}, {'name': ' Interest-bearing debt interest rate', 'min': '0.0', 'max': '990000000.0'}, {'name': ' Current Asset Turnover Rate', 'min': '0.0', 'max': '10000000000.0'}, {'name': ' Average Collection Days', 'min': '0.0', 'max': '9730000000.0'}, {'name': ' Long-term Liability to Current Assets', 'min': '0.0', 'max': '9540000000.0'}, {'name': ' Quick Ratio', 'min': '0.0', 'max': '9230000000.0'}, {'name': ' Accounts Receivable Turnover', 'min': '0.0', 'max': '9740000000.0'}, {'name': ' Quick Assets/Current Liability', 'min': '0.0', 'max': '8820000000.0'}, {'name': ' Total debt/Total net worth', 'min': '0.0', 'max': '9940000000.0'}, {'name': ' Inventory/Current Liability', 'min': '0.0', 'max': '9910000000.0'}, {'name': ' Revenue Per Share (Yuan ¥)', 'min': '0.0', 'max': '3020000000.0'}, {'name': ' Allocation rate per person', 'min': '0.0', 'max': '9570000000.0'}, {'name': ' Current Ratio', 'min': '0.0', 'max': '2750000000.0'}, {'name': ' Tax rate (A)', 'min': '0.0', 'max': '1.0'}, {'name': ' Cash/Total Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' Debt ratio %', 'min': '0.0', 'max': '1.0'}, {'name': ' Current Liability to Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' Quick Assets/Total Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' ROA(A) before interest and % after tax', 'min': '0.0', 'max': '1.0'}, {'name': ' Current Liability to Current Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' ROA(B) before interest and depreciation after tax', 'min': '0.0', 'max': '1.0'}, {'name': ' ROA(C) before interest and depreciation before interest', 'min': '0.0', 'max': '1.0'}, {'name': ' Equity to Liability', 'min': '0.0', 'max': '1.0'}, {'name': ' Total Asset Turnover', 'min': '0.0', 'max': '1.0'}, {'name': ' Total expense/Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' Per Share Net profit before tax (Yuan ¥)', 'min': '0.0', 'max': '1.0'}, {'name': ' Persistent EPS in the Last Four Seasons', 'min': '0.0', 'max': '1.0'}, {'name': ' Net profit before tax/Paid-in capital', 'min': '0.0', 'max': '1.0'}, {'name': ' Net worth/Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' Net Income to Total Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' Net Value Per Share (A)', 'min': '0.0', 'max': '1.0'}, {'name': ' Net Value Per Share (B)', 'min': '0.0', 'max': '1.0'}, {'name': ' Net Value Per Share (C)', 'min': '0.0', 'max': '1.0'}, {'name': ' Operating Profit Per Share (Yuan ¥)', 'min': '0.0', 'max': '1.0'}, {'name': ' Working Capital to Total Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' Operating profit/Paid-in capital', 'min': '0.0', 'max': '1.0'}, {'name': ' Current Assets/Total Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' CFO to Assets', 'min': '0.0', 'max': '1.0'}, {'name': ' Contingent liabilities/Net worth', 'min': '0.0', 'max': '1.0'}]


    # Define the number of columns
    num_cols = 3

    # Split the array into chunks based on the number of columns
    rows = [best_features[i:i + num_cols] for i in range(0, len(best_features), num_cols)]
    input_values = {}
    non_bankrupt = {
  " Fixed Assets Turnover Frequency": 0.0001452475676706,
  " Cash/Current Liability": 0.0010503117869961,
  " Research and development expense rate": 730000000.0,
  " Cash Turnover Rate": 2390000000.0,
  " Total Asset Growth Rate": 5720000000.0,
  " Quick Asset Turnover Rate": 9560000000.0,
  " Fixed Assets to Assets": 0.355853781517835,
  " Total assets to GNP price": 0.0183718717412392,
  " Operating Expense Rate": 0.0003984833580005,
  " Inventory Turnover Rate (times)": 0.0001620299292972,
  " Revenue per person": 0.0114595420295815,
  " Net Value Growth Rate": 0.0003517569706697,
  " Interest-bearing debt interest rate": 0.0008050805080508,
  " Current Asset Turnover Rate": 0.0001058010525107,
  " Average Collection Days": 0.0038050184230862,
  " Long-term Liability to Current Assets": 0.0093474263545427,
  " Quick Ratio": 0.0028690421678704,
  " Accounts Receivable Turnover": 0.0016617126393637,
  " Quick Assets/Current Liability": 0.0039738300376393,
  " Total debt/Total net worth": 0.0244412223346921,
  " Inventory/Current Liability": 0.0021593494146816,
  " Revenue Per Share (Yuan ¥)": 0.0309148932952191,
  " Allocation rate per person": 0.0097031414638638,
  " Current Ratio": 0.004672060305584,
  " Tax rate (A)": 0.0,
  " Cash/Total Assets": 0.0229885979786101,
  " Debt ratio %": 0.216101823019016,
  " Current Liability to Assets": 0.115920411679634,
  " Quick Assets/Total Assets": 0.255093170002626,
  " ROA(A) before interest and % after tax": 0.445704317488007,
  " Current Liability to Current Assets": 0.060765124792415,
  " ROA(B) before interest and depreciation after tax": 0.436158252583115,
  " ROA(C) before interest and depreciation before interest": 0.390922829425243,
  " Equity to Liability": 0.0156630746538442,
  " Total Asset Turnover": 0.100449775112444,
  " Total expense/Assets": 0.0928023065929925,
  " Per Share Net profit before tax (Yuan ¥)": 0.128944791745123,
  " Persistent EPS in the Last Four Seasons": 0.161482461945731,
  " Net profit before tax/Paid-in capital": 0.127939069451776,
  " Net worth/Assets": 0.783898176980984,
  " Net Income to Total Assets": 0.736619097683939,
  " Net Value Per Share (A)": 0.158821794277527,
  " Net Value Per Share (B)": 0.158821794277527,
  " Net Value Per Share (C)": 0.158821794277527,
  " Operating Profit Per Share (Yuan ¥)": 0.0999104307466819,
  " Working Capital to Total Assets": 0.729416462238773,
  " Operating profit/Paid-in capital": 0.0998664647352908,
  " Current Assets/Total Assets": 0.295221006636909,
  " CFO to Assets": 0.56065299962732,
  " Contingent liabilities/Net worth": 0.008043896850931,
}
    
    bankrupt = {
  " Fixed Assets Turnover Frequency": 2650000000.0,
  " Cash/Current Liability": 5340000000.0,
  " Research and development expense rate": 25500000.0,
  " Cash Turnover Rate": 761000000.0,
  " Total Asset Growth Rate": 7280000000.0,
  " Quick Asset Turnover Rate": 0.001022676471902,
  " Fixed Assets to Assets": 0.276179222234543,
  " Total assets to GNP price": 0.0400028528527523,
  " Operating Expense Rate": 0.0002361297205563,
  " Inventory Turnover Rate (times)": 65000000.0,
  " Revenue per person": 0.0289969595934374,
  " Net Value Growth Rate": 0.0003964253147034,
  " Interest-bearing debt interest rate": 0.0007900790079007,
  " Current Asset Turnover Rate": 0.0017910937001909,
  " Average Collection Days": 0.004226849460116,
  " Long-term Liability to Current Assets": 0.0037151156933692,
  " Quick Ratio": 0.0053475602224365,
  " Accounts Receivable Turnover": 0.0014953384801111,
  " Quick Assets/Current Liability": 0.006302481382547,
  " Total debt/Total net worth": 0.0212476860084444,
  " Inventory/Current Liability": 0.0138787857963224,
  " Revenue Per Share (Yuan ¥)": 0.0059440083488361,
  " Allocation rate per person": 0.141016311871331,
  " Current Ratio": 0.0115425536893801,
  " Tax rate (A)": 0.0,
  " Cash/Total Assets": 0.0009909444511248,
  " Debt ratio %": 0.207515796474892,
  " Current Liability to Assets": 0.0981620645437837,
  " Quick Assets/Total Assets": 0.340200878481779,
  " ROA(A) before interest and % after tax": 0.499018752725687,
  " Current Liability to Current Assets": 0.0253464891288078,
  " ROA(B) before interest and depreciation after tax": 0.472295090743616,
  " ROA(C) before interest and depreciation before interest": 0.426071271876371,
  " Equity to Liability": 0.0164741143272785,
  " Total Asset Turnover": 0.0149925037481259,
  " Total expense/Assets": 0.0213874282332057,
  " Per Share Net profit before tax (Yuan ¥)": 0.142803344128945,
  " Persistent EPS in the Last Four Seasons": 0.180580504869056,
  " Net profit before tax/Paid-in capital": 0.148035593092527,
  " Net worth/Assets": 0.792484203525108,
  " Net Income to Total Assets": 0.774669696989803,
  " Net Value Per Share (A)": 0.177910749652353,
  " Net Value Per Share (B)": 0.177910749652353,
  " Net Value Per Share (C)": 0.193712865028865,
  " Operating Profit Per Share (Yuan ¥)": 0.0923377575116033,
  " Working Capital to Total Assets": 0.829501914934622,
  " Operating profit/Paid-in capital": 0.0923184653215431,
  " Current Assets/Total Assets": 0.602805701668708,
  " CFO to Assets": 0.538490539586983,
  " Contingent liabilities/Net worth": 0.0065619820610314,
}
    
    # Display the elements in the input fields within columns
    for row in rows:
        cols = st.columns(num_cols)
        for col, element in zip(cols, row):
            input_values[element['name']] = col.number_input(f"{element['name']}",min_value=float(element['min']), max_value=float(element['max']), value=non_bankrupt[element['name']], step=0.01 ,format="%.16f")
   
    # Prediction button
    if st.button('Predict'):
        input_list = np.array(list(input_values.values())).reshape(1, -1)

        # Fit and transform the inputs
        scaled_data = scaler.transform(input_list)

        logistic_regression_prediction = logistic_regression.predict(input_list)
        naive_bayes_prediction = naive_bayes.predict(input_list)
        random_forest_prediction = random_forest.predict(input_list)

        # Display the prediction result
        classes = {0: 'not go bankrupt', 1: 'go bankrupt'}
        st.success(f'This company will {classes[logistic_regression_prediction[0]]} according to logistic regression.')
        st.success(f'This company will {classes[naive_bayes_prediction[0]]} according to Naive Bayes.')
        st.success(f'This company will {classes[random_forest_prediction[0]]} according to Random Forest.')
        
        
# Run the main function when the script is executed
if __name__ == '__main__':
    main()
