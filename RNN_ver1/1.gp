#set multiplot layout 2,2
unset key
#set xrange[-1:7]
#set yrange[-2:2]

plot './data_out/tmp.csv' using 1:2 w l
replot './data_out/tmp.csv' using 1:3 w l

#pause -1

#unset multiplotls
