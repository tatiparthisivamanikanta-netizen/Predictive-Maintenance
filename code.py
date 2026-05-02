import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime as dt

#configuring the output of pandas dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

#First we will begin by creating columns for our DataSet 
eng_cycle_col=['engine', 'cycle']
setting_col=['setting1', 'setting2', 'setting3']
sensor_col = ['sensor1',
       'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6', 'sensor7',
       'sensor8', 'sensor9', 'sensor10', 'sensor11', 'sensor12', 'sensor13',
       'sensor14', 'sensor15', 'sensor16', 'sensor17', 'sensor18', 'sensor19',
       'sensor20', 'sensor21' ]
columns=eng_cycle_col+setting_col+sensor_col

#Then we will be importing our necessary Data
#NB: I will only work on the FD001 DataSet
train_data=pd.read_csv("../input/nasa-cmaps/CMaps/train_FD001.txt",sep='\s+',names=columns)
test_data=pd.read_csv("../input/nasa-cmaps/CMaps/test_FD001.txt",sep='\s+',names=columns)
true_rul=pd.read_csv("../input/nasa-cmaps/CMaps/RUL_FD001.txt",sep='\s+',names=['RUL'])
And let's print it out and its shape to check it
print('train data and shape:')
print(train_data)
print(train_data.shape)
print('test data and shape: ')
print(test_data)
print(test_data.shape)                                                                                                                                #now we will creat a function to describe our data
def data_desc(data): 
    print('Data description:')
    return data.describe().transpose()

print(data_desc(train_data))                                                                                                                      def check_missing_values(data):
    print('Verifing the existance of null data:')
    return data.isnull().sum()

print(check_missing_values(train_data))                                                                                                   def find_max_cycle(data):
    print('The max cycles of each engine: ')
    max_cycle = data[['engine', 'cycle']].groupby(['engine']).count().reset_index().rename(columns={'cycle': 'max_cycles'})
    return max_cycle

max_cycle=find_max_cycle(train_data)
print(max_cycle)

#i.e we can see that engine 1 failed after 192 cyclees and engin 2 failed after 287 cycles as it goes throw each engine
#these cyclce will serve us in wuhile plotting the data since we are going to use this late in the data plotting as an x axis                                                                                                                          So acutally let's plot this for better understanding
def barplt(data):
       plt.figure(figsize=(15,10))
       sns.barplot(x='engine', y='max_cycles', data=data,palette='magma')
       sns.set_context(font_scale=0.01)
       plt.title('Turbofan Engines LifeTime',fontweight='bold',size=20)
       plt.xlabel('engine',fontweight='bold',size=20)
       plt.ylabel('cycle',fontweight='bold',size=20)
       plt.xticks(rotation=90)
       plt.grid(True)
       plt.tight_layout()
barplt(max_cycle)             So acutally let's plot this for better understanding
def barplt(data):
       plt.figure(figsize=(15,10))
       sns.barplot(x='engine', y='max_cycles', data=data,palette='magma')
       sns.set_context(font_scale=0.01)
       plt.title('Turbofan Engines LifeTime',fontweight='bold',size=20)
       plt.xlabel('engine',fontweight='bold',size=20)
       plt.ylabel('cycle',fontweight='bold',size=20)
       plt.xticks(rotation=90)
       plt.grid(True)
       plt.tight_layout()
barplt(max_cycle)                                                                                                                                       def add_remaining_RUL(data):
    train_data_by_engine = data.groupby(by='engine')
    max_cycles = train_data_by_engine['cycle'].max()
    merged = data.merge(max_cycles.to_frame(name='max_cycles'), left_on='engine',right_index=True)
    merged["RUL"] = merged["max_cycles"] - merged['cycle']
    merged = merged.drop("max_cycles", axis=1)
    return merged
train_data=add_remaining_RUL(train_data)
print(train_data)                                                                                                                                         #and for more specific anaylsis this a function that shows plots the behavior of the sensors for a specific given engine number
#and this, along its remaining RUL
def info_plotting_per_engine(eng_num,data):
    engine_data = data[data['engine'] == eng_num]

    columns_to_plot = ['setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]

    num_columns = 6
    num_rows = (len(columns_to_plot) + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, num_rows * 3), sharex=True)

    for ax, column in zip(axes.flatten(), columns_to_plot):
        ax.plot(engine_data['cycle'], engine_data[column], label=column)
        ax.set_title(f'{column} over Cycles')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
#just select the desired engine number
info_plotting_per_engine(20,train_data)                                                                                                   #now let's really see the correlation between data
def corr_matrix(data):
    plt.figure(figsize=(15, 10))
    sns.set_context(font_scale=0.01)
    sns.heatmap(data.corr(), annot=True, cmap='RdYlGn')
    plt.grid(False)


corr_matrix(train_data)                                                                                                                               clean_train_data=train_data.drop(['setting1','setting2','sensor6','sensor5','sensor16','setting3','sensor1','sensor10','sensor18','sensor19'],axis=1)
clean_test_data=test_data.drop(['setting1','setting2','sensor6','sensor5','sensor16','setting3','sensor1','sensor10','sensor18','sensor19'],axis=1)
print('Data after our cleaning: ')
print(clean_train_data)                                                                                                                                                                  sens_names={
 'sensor2': '(LPC outlet temperature) (◦R)',
 'sensor3': '(HPC outlet temperature) (◦R)',
 'sensor4': '(LPT outlet temperature) (◦R)',
 'sensor7': '(HPC outlet pressure) (psia)',
 'sensor8': '(Physical fan speed) (rpm)',
 'sensor9': '(Physical core speed) (rpm)',
 'sensor11': '(HPC outlet Static pressure) (psia)',
 'sensor12': '(Ratio of fuel flow to Ps30) (pps/psia)',
 'sensor13': '(Corrected fan speed) (rpm)',
 'sensor14': '(Corrected core speed) (rpm)',
 'sensor15': '(Bypass Ratio) ',
 'sensor17': '(Bleed Enthalpy)',
 'sensor20': '(High-pressure turbines Cool air flow)',
 'sensor21': '(Low-pressure turbines Cool air flow)'}

def plot_sensor(sensor_name,sens_names,data):
    for S in sensor_name:

        if S in data.columns:
            plt.figure(figsize=(13, 5))
            for i in data['engine'].unique():

                if (i % 5 == 0):

                    plt.plot('RUL', S,
                             data=data[data['engine']==i].rolling(8).mean())


            plt.xlim(250, 0)
            plt.xticks(np.arange(0, 275, 25))
            plt.ylabel(sens_names[S])
            plt.xlabel('Remaining Usefull Life ')
            plt.grid(True)
            plt.show()
plot_sensor(sensor_col,sens_names,clean_train_data)                                                                                 scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(clean_train_data.drop(['engine','cycle','RUL'],axis=1))
scaled_data=pd.DataFrame(scaled_data, columns=clean_train_data.drop(['engine','cycle', 'RUL'], axis=1).columns)


#since our values in the dataset are contunious, numerical and it contains the targetted outcome we will be using supervised ML algorithms
# and one of  the simplest and most common models that we will be using at first is Linear Regression
#Linear regression predicts the relationship between two variables by assuming they have a straight-line connection.
# It finds the best line that minimizes the differences between predicted and actual values.
#preparin the data for ML models the X defines the data we are using for the training and Y is the desired prediction
X_train = clean_train_data
Y_train = clean_train_data.pop('RUL')
X_test = test_data.groupby('engine').last().reset_index().drop(['setting1','setting2','sensor6','sensor5','sensor16','setting3','sensor1','sensor10','sensor18','sensor19'], axis=1)
#Here wwe will scale the test data using the same method
scaled_test_data=scaler.transform(X_test.drop(['engine','cycle'],axis=1))
scaled_test_data=pd.DataFrame(scaled_test_data, columns=X_test.drop(['engine','cycle'], axis=1).columns)
print('Cheking the scaled data')
print(scaled_data)
print(scaled_test_data)
Y_test= true_rul
X_train_s=scaled_data
X_test_s=scaled_test_data

def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))                                                                        #TEST 1 linear regression
start_1=dt.now()
lm=LinearRegression()
lm.fit(X_train_s,Y_train)
Y_predict_train=lm.predict(X_train_s)
Y_predict_test = lm.predict(X_test_s)
print('Linear Regression evaluation score: ')
print('run time equals: '+str((dt.now() - start_1).seconds)+'s')
evaluate(Y_train, Y_predict_train, 'train')
evaluate(Y_test, Y_predict_test)
fig = plt.figure(figsize=(18,10))
plt.plot(Y_test,color='red', label='RUL')
plt.plot(Y_predict_test, label='Linear regression prediction')
plt.legend(loc='upper left')
plt.grid(True)

#Test 2 decision tree
start_2=dt.now()
rf = RandomForestRegressor(max_features="sqrt", random_state=42)
rf.fit(X_train_s,Y_train)
Y_predict_train_rf=rf.predict(X_train_s)
Y_predict_test_rf = rf.predict(X_test_s)
print('Random Forest Regressor evaluation: ')
print('run time equals: '+str((dt.now() - start_2).seconds)+'s')
evaluate(Y_train, Y_predict_train_rf, 'train')
evaluate(Y_test, Y_predict_test_rf)
plt.plot(Y_predict_test_rf,color='orange', label='Random forest prediction')
plt.legend(loc='upper left')
plt.grid(True)

#Test support vector machine (SVM)
start_3 = dt.now()
svm= SVR(kernel='linear')
svm.fit(X_train_s,Y_train)
svm_train_prediction=svm.predict(X_train_s)
svm_test_predict=svm.predict(X_test_s)
print('Support vector machine evaluation')
print('run time equals: '+str((dt.now() - start_3).seconds)+'s')
evaluate(Y_train,svm_train_prediction,'train')
evaluate(Y_test,svm_test_predict)
plt.plot(svm_test_predict,color='black',label='SVM prediction')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()            
