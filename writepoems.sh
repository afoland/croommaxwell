sleep 30000

# th sample.lua cv/lm_lstm_epoch19.16_4.5845.t7 -temperature 0.8 > rfromF_458_t8.txt
# th sample.lua cv/lm_lstm_epoch19.16_4.5845.t7 -temperature 0.7 > rfromF_458_t7.txt
# th sample.lua cv/lm_lstm_epoch19.16_4.5845.t7 -temperature 0.6 > rfromF_458_t6.txt
# th sample.lua cv/lm_lstm_epoch19.16_4.5845.t7 -temperature 0.5 > rfromF_458_t5.txt

last_cp=`ls -rt -1 cv/* | tail -1`
th sample.lua ${last_cp} -temperature 0.8 > EPrfromF_last_t8.txt
th sample.lua ${last_cp} -temperature 0.7 > EPrfromF_last_t7.txt
th sample.lua ${last_cp} -temperature 0.6 > EPrfromF_last_t6.txt
th sample.lua ${last_cp} -temperature 0.5 > EPrfromF_last_t5.txt
th sample.lua ${last_cp} -temperature 0.4 > EPrfromF_last_t4.txt
