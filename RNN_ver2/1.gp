#set multiplot layout 2,2
unset key
#set xrange[-1:7]
#set yrange[-2:2]

plot './data_out/tmp.csv' using 1:2 w l
replot './data_out/tmp.csv' using 1:3 w l

#pause -1

#unset multiplotls

#    datalen0 = dataset.shape[0] #時間長さ
#    datalen1 = dataset.shape[1] #入力＋出力長さ
#    traning_step = int(datalen0*Traning_Ratio/100)
#    sample_step=int(traning_step/SAMPLE_NUM)
#    traning_step=sample_step*SAMPLE_NUM
#    rc_step = datalen0 - traning_step
#    dataset_test=dataset[traning_step:traning_step+rc_step,0:datalen1]
#    dataset_rnn = dataset[0:traning_step,0:datalen1]