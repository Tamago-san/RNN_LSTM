####################################################
####################################################
#numpy仕様のRC
#これは時系列データ予測
#行列がでかいとエラーあり20181203
#gfortran -shared -o rc_karman.so rc_karman.f90 -llapack -lblas -fPIC
#gfortran -shared -o rc_tanh.so rc_tanh.f90 -llapack -lblas -fPIC
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
ALPHA =0.5
RC_NODE = 10
RNN_NODE=RC_NODE
A_RAN = 1/(RC_NODE)**0.5
AV_DEGREE = 0.6#この割合で行列Aに値を入れる。
G = 0.1#0.005がすげえ
GUSAI = 1.0
CI =  0.
BETA =1.
SIGMA = 1.
OUT_NODE=1
SPECTRAL_RADIUS = 1
ITRMAX=5000000
Traning_Ratio = 90
R0= 1.
GOD_STEP=20
steps_of_history=1
steps_in_future=1
EPOCH=5000
SAMPLE_NUM =5
EPSILON=0.000001
#RNN_NODE=50




def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore
    
def nrmse(y_pred, y_true):
    return np.sqrt(np.mean( (y_true - y_pred) ** 2) )

#過去にずらすかつデータセットの作る関数、一行目はオリジナル
#インプットが一つ。
def call_create_dataset2(df):
    df=df.drop([0,1])
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
    df = df.dropna(how='all').dropna(how='all', axis=0)#全部欠損値ならその行削除
    target_number = df.columns.get_loc(target_columns)
    data=df.astype("float64").values
    print(df)
    len_index=len(df.index)
    len_column =len(df.columns)
#    print(len(data.index))
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

def call_create_dataset(df):
    df=df.drop([0,1])
    df=df.dropna(how='all', axis=1)#全部欠損値ならその列削除
    df = df.dropna(how='all').dropna(how='all', axis=0)#全部欠損値ならその行削除
    len_index=len(df.index)
    len_column =len(df.columns)
    dataset=np.array(df)
    print(dataset.shape)
    print(dataset)
    np.savetxt('./data_out/dataset.npy' ,dataset, delimiter=',')
    return dataset

def call_fortran_rc_traning_own(_in_node,_out_node,_rc_node,_traning_step,_rc_step,_gusai,_alpha,_g,
                        U_in,S_out,U_rc,S_rc,W_out):
    f = np.ctypeslib.load_library("rc_tanh.so", ".")
    f. rc_traning_own_fortran_.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ]
    f. rc_traning_own_fortran_.restype = ctypes.c_void_p

    f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
    f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
    f_rc_node = ctypes.byref(ctypes.c_int32(_rc_node))
    f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
    f_rc_step = ctypes.byref(ctypes.c_int32(_rc_step))
    f_gusai = ctypes.byref(ctypes.c_double(_gusai))
    f_alpha = ctypes.byref(ctypes.c_double(_alpha))
    f_g = ctypes.byref(ctypes.c_double(_g))
    f.rc_traning_own_fortran_(f_in_node,f_out_node,f_rc_node,f_traning_step,f_rc_step,f_gusai,f_alpha,f_g,
                            U_in,S_out,U_rc,S_rc,W_out)

def call_fortran_rc_karman(_in_node,_out_node,_rc_node,_traning_step,_rc_step,
                        U_in,S_out,U_rc,S_rc,W_out):
    f = np.ctypeslib.load_library("rc_karman.so", ".")
    f. rc_traning_own_karman_.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ]
    f. rc_traning_own_karman_.restype = ctypes.c_void_p

    f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
    f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
    f_rc_node = ctypes.byref(ctypes.c_int32(_rc_node))
    f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
    f_rc_step = ctypes.byref(ctypes.c_int32(_rc_step))

    f.rc_traning_own_karman_(f_in_node,f_out_node,f_rc_node,f_traning_step,f_rc_step,
                            U_in,S_out,U_rc,S_rc,W_out)

def call_fortran_rnn_traning_own(_in_node,_out_node,_rnn_node,_traning_step,_rnn_step,
                        _sample_num,_epoch,_epsilon,_g,
                        U_in,S_out,U_rc,S_rc,W_out,W_rnn,W_in,_Tre_CH):
    f = np.ctypeslib.load_library("rnn_tanh.so", ".")
    f. rnn_traning_own_fortran_.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ]
    f. rnn_traning_own_fortran_.restype = ctypes.c_void_p

    f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
    f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
    f_rnn_node = ctypes.byref(ctypes.c_int32(_rnn_node))
    f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
    f_rnn_step = ctypes.byref(ctypes.c_int32(_rnn_step))
    f_sample_num = ctypes.byref(ctypes.c_int32(_sample_num))
    f_epoch = ctypes.byref(ctypes.c_int32(_epoch))
    f_epsilon = ctypes.byref(ctypes.c_double(_epsilon))
    f_g = ctypes.byref(ctypes.c_double(_g))
    f.rnn_traning_own_fortran_(f_in_node,f_out_node,f_rnn_node,f_traning_step,f_rnn_step,
                            f_sample_num,f_epoch,f_epsilon,f_g,
                            U_in,S_out,U_rc,S_rc,W_out,W_rnn,W_in,Tre_CH)

def call_fortran_lstm_traning_own(_in_node,_out_node,_rnn_node,_traning_step,_rnn_step,
                        _sample_num,_epoch,_epsilon,_g,
                        U_in,S_out,U_rc,S_rc,W_out,W_rnn,W_in,_Tre_CH):
    f = np.ctypeslib.load_library("lstm_tanh.so", ".")
    f. lstm_traning_own_fortran_.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ]
    f. lstm_traning_own_fortran_.restype = ctypes.c_void_p

    f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
    f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
    f_rnn_node = ctypes.byref(ctypes.c_int32(_rnn_node))
    f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
    f_rnn_step = ctypes.byref(ctypes.c_int32(_rnn_step))
    f_sample_num = ctypes.byref(ctypes.c_int32(_sample_num))
    f_epoch = ctypes.byref(ctypes.c_int32(_epoch))
    f_epsilon = ctypes.byref(ctypes.c_double(_epsilon))
    f_g = ctypes.byref(ctypes.c_double(_g))
    f.lstm_traning_own_fortran_(f_in_node,f_out_node,f_rnn_node,f_traning_step,f_rnn_step,
                            f_sample_num,f_epoch,f_epsilon,f_g,
                            U_in,S_out,U_rc,S_rc,W_out,W_rnn,W_in,Tre_CH)

#DATA INPUT
#DataFrameで読み込んでいる
#dataframe = pd.read_csv('./data/international-airline-passengers.csv',
#        header = None,
#        usecols=[0,1],
#        engine='python',
#        skiprows=0,
#        skipfooter=0)
#dataframe = pd.read_csv('./data/karman.csv',
#        header = None,
#        usecols=[0,1],
#        engine='python',
#        skiprows=0,
#        skipfooter=0)

dataframe = pd.read_csv('./data/output_Runge_Lorenz.csv',
        header = None,
        usecols=[0,1],
        engine='python',
        skiprows=0,
        skipfooter=0)

print(dataframe)
#DATASET = call_create_dataset2(dataframe)
DATASET = call_create_dataset(dataframe)
print(DATASET)
#===================================================
#===================================================
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#===================================================
#===================================================
#+++++++++++++++++++++++++++++++++++++++++++++++++++
datalen0 = DATASET.shape[0] #時間長さ
datalen1 = DATASET.shape[1] #入力＋出力長さ
print(datalen1)
TRANING_STEP = int(datalen0*Traning_Ratio/100)
#################
SAMPLE_STEP=int(TRANING_STEP/SAMPLE_NUM)
TRANING_STEP=SAMPLE_STEP*SAMPLE_NUM
RC_STEP = datalen0 - TRANING_STEP #トレーニングとRCとを分ける
#RC_STEP = datalen0                #RCの出力にトレーニング時間も含める。
#################
IN_NODE = datalen1 - OUT_NODE
E=np.eye(RC_NODE)
#W_IN= W_IN/(float(IN_NODE))**0.5
W_in = np.empty((RNN_NODE,IN_NODE))
W_rnn = np.empty((RNN_NODE,RNN_NODE))
W_out1 = np.empty((OUT_NODE,RNN_NODE))
W_out = np.empty((RC_NODE,OUT_NODE))
Tre_CH = np.empty((EPOCH,3))
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

#call_fortran_rc_karman(IN_NODE,OUT_NODE,RC_NODE,TRANING_STEP,RC_STEP
#            ,U_in[0:TRANING_STEP,0:IN_NODE],S_out[0:TRANING_STEP,0:OUT_NODE]
#            ,U_in[TRANING_STEP:TRANING_STEP+RC_STEP,0:IN_NODE],S_rc[0:RC_STEP,0:OUT_NODE]
#            ,W_out)
call_fortran_rc_traning_own(IN_NODE,OUT_NODE,RC_NODE,TRANING_STEP,RC_STEP,GUSAI,ALPHA,G
            ,U_in[0:TRANING_STEP,0:IN_NODE],S_out[0:TRANING_STEP,0:OUT_NODE]
            ,U_in[TRANING_STEP:TRANING_STEP+RC_STEP,0:IN_NODE],S_rc[0:RC_STEP,0:OUT_NODE]
            ,W_out)

#W_out1=W_out.T
#call_fortran_rnn_traning_own(IN_NODE,OUT_NODE,RNN_NODE,SAMPLE_STEP,RC_STEP,
#            SAMPLE_NUM,EPOCH,EPSILON,G
#            ,U_in[0:TRANING_STEP,0:IN_NODE],S_out[0:TRANING_STEP,0:OUT_NODE]
#            ,U_in[TRANING_STEP:TRANING_STEP+RC_STEP,0:IN_NODE],S_rc[0:RC_STEP,0:OUT_NODE]
#            ,W_out1,W_rnn,W_in,Tre_CH)
#
#call_fortran_lstm_traning_own(IN_NODE,OUT_NODE,RNN_NODE,SAMPLE_STEP,RC_STEP,
#            SAMPLE_NUM,EPOCH,EPSILON,G
#            ,U_in[0:TRANING_STEP,0:IN_NODE],S_out[0:TRANING_STEP,0:OUT_NODE]
#            ,U_in[TRANING_STEP:TRANING_STEP+RC_STEP,0:IN_NODE],S_rc[0:RC_STEP,0:OUT_NODE]
#            ,W_out1,W_rnn,W_in,Tre_CH)
           
DATA_ori  = np.concatenate([U_in[TRANING_STEP:TRANING_STEP+RC_STEP,], S_out[TRANING_STEP:TRANING_STEP+RC_STEP,]], axis=1)
DATA_rc  = U_in[TRANING_STEP:TRANING_STEP+RC_STEP,]
DATA_rc = DATA_rc.reshape((RC_STEP,IN_NODE))

##RCの出力にトレーニング時間も含める。
#call_fortran_rc_own(IN_NODE,OUT_NODE,RC_NODE,GUSAI,ALPHA,RC_STEP,G
#                    ,U_in[0:RC_STEP,0:IN_NODE],
#                    S_rc[0:RC_STEP,0:OUT_NODE]
##                    S_out[TRANING_STEP:TRANING_STEP+RC_STEP,0:OUT_NODE]
#                    ,W_out,W_IN,A,r_befor[0:RC_NODE])
#DATA_ori  = np.concatenate([U_in[0:RC_STEP,], S_out[0:RC_STEP,]], axis=1)
#DATA_rc  = U_in[0:RC_STEP,]
#DATA_rc = DATA_rc.reshape((RC_STEP,IN_NODE))
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#===================================================
#===================================================

print(DATA_rc.shape)
DATA_rc = np.append(DATA_rc,S_rc,axis = 1)
print(DATA_rc.shape)
np.savetxt('./data_out/out_ori.npy' ,DATA_ori)
np.savetxt('./data_out/out_rc.npy' ,DATA_rc)
print(nrmse(DATA_ori[:,IN_NODE:IN_NODE+OUT_NODE],DATA_rc[:,IN_NODE:IN_NODE+OUT_NODE]))


#data = np.loadtxt("./data_out/tmp.csv",delimiter=",")
#plt.plot(data[1000:1500,1],"-" , label="rc")
#plt.legend(loc=2)
#plt.show()
#__________________________________
#2次元プロット
#plt.plot(DATA_ori[:,IN_NODE+OUT_NODE-1],"-" , label="ori")
#plt.plot(DATA_rc[:,IN_NODE+OUT_NODE-1],"-" , label="rc")
#plt.legend(loc=2)
#plt.show()
#__________________________________
#三次元プロット
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot(DATA_rc[:,0], DATA_rc[:,1], DATA_rc[:,2])
#plt.show()
#__________________________________
#hukusuu
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlim([800,1000])
ax1.plot(DATA_ori[:,IN_NODE+OUT_NODE-1],"-" , label="ori")
ax1.plot(DATA_rc[:,IN_NODE+OUT_NODE-1],"-" , label="rc")
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(Tre_CH[:,0],"-" , label="W_out")
ax2.plot(Tre_CH[:,1],"-" , label="W_rnn")
ax2.plot(Tre_CH[:,2],"-" , label="W_in")
plt.show()