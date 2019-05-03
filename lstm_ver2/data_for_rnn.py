####################################################
####################################################
#numpy仕様のRC
#これは時系列データ予測
#行列がでかいとエラーあり20181203
#gfortran -shared -o rc_karman.so rc_karman.f90 -llapack -lblas -fPIC
#python3 kabu_rc.py
#head ./data/output.csvで読み込みの先頭確認
#tail ./data/output.csvで読み込みの末尾確認
####################################################
#import pandas_detareader.data as web
import pandas as pd
import numpy as np
import ctypes
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore


target_columns=1
DT    = 0.01
ALPHA =0.8
RC_NODE = 3000
A_RAN = 1/(RC_NODE)**0.5
AV_DEGREE = 0.3#この割合で行列Aに値を入れる。
G = 0.007#0.005がすげえ
GUSAI = 1.0
CI =  0.
BETA =1.
SIGMA = 1.
OUT_NODE=1
SPECTRAL_RADIUS = 1
ITRMAX=5000000
Traning_Ratio = 80
R0= 1.
GOD_STEP=3
SAMPLE_NUM =4
steps_of_history=2
steps_in_future=1



#過去にずらすかつデータセットの作る関数、一行目はオリジナル
#インプットが一つ。
class create_dataset:
    def __init__(self, num,node1,node2):
        self.sample_num=num+1
        self.in_node=node1
        self.out_node=node2
    
    def split_timestep_to_dataframe2(self,df):
        df=df.drop([0,1])
        len_index0=len(df.index)#tate
        len_column0 =len(df.columns)#yoko
        sample_step=(len_index0-steps_of_history)//self.sample_num
        print(sample_step)
        sample_step=sample_step+steps_of_history
        print(sample_step)
#        df=df.drop([0,1])
        df=df.rename(columns={0:"TIME"})
        df["TIME"] = pd.to_datetime(df["TIME"])
        df=df.set_index("TIME")
        date0=df.index.date
        df= df[df.index.hour < 11 ]
    #    df= df['11:30':'12:30']
    #    df.to_csv("employee.csv")
        df=df.replace(np.NaN, "NO")
    #    for i in range(1,len(df.columns)+1):
    #        df[i] = df[i].str.replace(',', '')
        df=df.replace("NO",np.NaN )
        df=df.dropna(how='all', axis=1)#全部欠損値ならその列削除
        df = df.dropna(how='all').dropna(how='all', axis=0)
#        data1=df.astype("float64").values
        print(sample_step)
        for j in range(0,self.sample_num):
            df_tmp=df.iloc[j*(sample_step-steps_of_history) :j*(sample_step-steps_of_history)+sample_step,:]
#            df_tmp=pd.Dataframe(data2)
            if j==0:
                data_tmp3 = call_create_dataset(df_tmp)
                print(data_tmp3.shape)
            else:
                data_tmp2=call_create_dataset(df_tmp)
                data_tmp3=np.dstack([data_tmp3,data_tmp2])
        return data_tmp3


def call_create_dataset(df):
#        df=df.drop([0,1])
#        df=df.rename(columns={0:"TIME"})
#        df["TIME"] = pd.to_datetime(df["TIME"])
#        df=df.set_index("TIME")
#        date0=df.index.date
#        df= df[df.index.hour < 11 ]
#    #    df= df['11:30':'12:30']
#    #    df.to_csv("employee.csv")
#        df=df.replace(np.NaN, "NO")
#    #    for i in range(1,len(df.columns)+1):
#    #        df[i] = df[i].str.replace(',', '')
#        df=df.replace("NO",np.NaN )
#        df=df.dropna(how='all', axis=1)#全部欠損値ならその列削除
#        df = df.dropna(how='all').dropna(how='all', axis=0)#全部欠損値ならその行削除
        target_number = df.columns.get_loc(target_columns)
        data=df.astype("float64").values
        print(df)
        len_index=len(df.index)
        len_column =len(df.columns)
        print(len(df.index))
        print(len(df.columns))
        X,ORIGINAL,FUTURE = [],[],[]
        X_tmp=np.empty((len_index-steps_of_history-GOD_STEP,len_column*steps_of_history))
        for i in range(0, len_index-steps_of_history-GOD_STEP):
            for xno in range(0,len_column):
                X_tmp[i,xno*steps_of_history:xno*steps_of_history+steps_of_history] = data[i:i+steps_of_history,xno]
        
        for i in range(0, len_index-steps_of_history-GOD_STEP):
            ORIGINAL.append(data[i+steps_of_history,0:len_column])
            FUTURE.append(data[i+GOD_STEP+steps_of_history,target_number])
        
        X=X_tmp
        print(np.array(X).shape)
        print(np.array(ORIGINAL).shape)
        print(np.array(FUTURE).shape)
        X = np.reshape(np.array(X), [len_index-steps_of_history-GOD_STEP,steps_of_history*len_column])
        ORIGINAL = np.reshape(np.array(ORIGINAL), [len_index-steps_of_history-GOD_STEP,len_column])
        FUTURE = np.reshape(np.array(FUTURE), [len_index-steps_of_history-GOD_STEP,OUT_NODE])
        print(X.shape)
        print(ORIGINAL.shape)
        print(FUTURE.shape)
        dataset=np.hstack((X,ORIGINAL,FUTURE))
        print(dataset.shape)
        print(dataset)
        np.savetxt('./data_out/dataset.npy' ,dataset, delimiter=',')
        return dataset


#DATA INPUT
#DataFrameで読み込んでいる
dataframe = pd.read_csv('./data/international-airline-passengers.csv',
        header = None,
        usecols=[0,1],
        engine='python',
        skiprows=0,
        skipfooter=0)
#dataframe = pd.read_csv('./data/karman.csv',
#        header = None,
#        usecols=[0,1],
#        engine='python',
#        skiprows=0,
#        skipfooter=0)
#dataframe=pd.read_csv("./data/gaku.csv",
#        header = None,
#        engine='python',
##        index_col =0,
#        skiprows=5,
#        skipfooter=4)
#dataframe=pd.read_csv("./data/nikkei.csv",
#        header = None,
#        engine='python',
##        index_col =0,
#        skiprows=0,
#        skipfooter=0)
#
#dataframe=pd.read_csv("./data/real-gdp-per-capita.csv",
#        engine='python',
#        skiprows=0,
#        skipfooter=2)
print(dataframe)
#DATASET = call_create_dataset2(dataframe)
cd=create_dataset(SAMPLE_NUM,1,1)
DATASET=cd.split_timestep_to_dataframe2(dataframe)

print(DATASET)
#===================================================
#===================================================
#+++++++++++++++++++++++++++++++++++++++++++++++++++
datalen = DATASET.shape[0] #時間長さ
datalen2 = DATASET.shape[1] #入力＋出力長さ
print(datalen2)
TRANING_STEP = int(datalen*Traning_Ratio/100)
#################
RC_STEP = datalen - TRANING_STEP #トレーニングとRCとを分ける
#RC_STEP = datalen                #RCの出力にトレーニング時間も含める。
#################
IN_NODE = datalen2 - OUT_NODE
E=np.eye(RC_NODE)
#W_IN= W_IN/(float(IN_NODE))**0.5
W_out = np.empty((RC_NODE,OUT_NODE))
r_befor = np.zeros((RC_NODE))
S_rc = np.zeros((RC_STEP,OUT_NODE))
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#===================================================
#===================================================

U_in  = DATASET[:,0:IN_NODE]
S_out = DATASET[:,IN_NODE:IN_NODE+OUT_NODE]
U_in = zscore(U_in,axis=0)
S_out = zscore(S_out,axis=0)

print(U_in.shape)
print(S_out.shape)
