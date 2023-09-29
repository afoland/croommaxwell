date

mydir=data/Fpoem111
thold=25
epochs=1

timestamp=`date -u +%s`

#
# Get started, create vocab with Fromantics
#

cd ${mydir}
rm input.txt
ln -s Fromantics.lower.txt input.txt
mv data_w${thold}.t7 data_w${thold}.t7.bak
mv vocab_w${thold}.t7 vocab_w${thold}.t7.bak
mv glove_*.t7 oldgloves
cd ../..

echo "Starting 40/35"
th train.lua -gpuid 0 -num_layers 3 -rnn_size 256 -eval_val_every 20000 -print_every 100 -word_level 1 -glove 1 -data_dir ${mydir} -threshold ${thold} -batch_size 40 -seq_length 35 -learning_rate 5.0e-5 -max_epochs ${epochs} -dropout 0.25 -seed 77 -learning_rate_decay_after 1 -learning_rate_decay 0.97 >  tee_createvocab_${timestamp}.log &

pnum=`ps aux | grep luajit | grep -v grep | awk '{split($0,a," "); print a[2];}'`

echo "Will kill process " ${pnum}
sleep 900
kill -9 ${pnum}

#
# Train prose on bitsF
#
echo "Training on bitsF"
cd ${mydir}
rm input.txt
ln -s bitsF.lower.txt input.txt
mv data_w${thold}.t7 data_w${thold}.t7.Fromantics
touch vocab_w${thold}.t7 
touch glove_*.t7
cd ../..

epochs=3
# epochs=1

th train.lua -gpuid 0 -num_layers 3 -rnn_size 256 -eval_val_every 20000 -print_every 100 -word_level 1 -glove 1 -data_dir ${mydir} -threshold ${thold} -batch_size 40 -seq_length 35 -learning_rate 5.0e-5 -max_epochs ${epochs} -dropout 0.25 -seed 77 -learning_rate_decay_after 1 -learning_rate_decay 0.97 > tee_bitsF_${timestamp}.log

sleep 360
lastlm=`ls -rt cv/lm*.t7 | tail -1`

#
# Train poetry on romantics.mixed
#
echo "Training on romantics.mixed"
echo "Training from " ${lastlm}
cd ${mydir}
rm input.txt
ln -s romantics.mixed.lower.txt input.txt
mv data_w${thold}.t7 data_w${thold}.t7.bitsF
touch vocab_w${thold}.t7 
touch glove_*.t7
cd ../..

epochs=100
# epochs=1

th train.lua -init_from ${lastlm} -gpuid 0 -num_layers 3 -rnn_size 256 -eval_val_every 5000 -print_every 100 -word_level 1 -glove 1 -data_dir ${mydir} -threshold ${thold} -batch_size 40 -seq_length 75 -learning_rate 9.0e-5 -max_epochs ${epochs} -dropout 0.25 -seed 77 -learning_rate_decay_after 5 -learning_rate_decay 0.975 > tee_romantics1_${timestamp}.log

#
# Train poetry on romantics.mixed, slower
#
sleep 360
lastlm=`ls -rt cv/lm*.t7 | tail -1`

# epochs=1

epochs=50
echo "Training on romantics.mixed"
echo "Training from " ${lastlm}

th train.lua -init_from ${lastlm} -gpuid 0 -num_layers 3 -rnn_size 256 -eval_val_every 5000 -print_every 100 -word_level 1 -glove 1 -data_dir ${mydir} -threshold ${thold} -batch_size 40 -seq_length 75 -learning_rate 1.0e-5 -max_epochs ${epochs} -dropout 0.25 -seed 77 -learning_rate_decay_after 5 -learning_rate_decay 0.975 > tee_romantics2_${timestamp}.log

lastlm=`ls -rt cv/lm*.t7 | tail -1`

#
# Write poetry
#


temper=0.75

th sample.lua ${lastlm} -temperature ${temper} -length 35000 > write_${temper}_${timestamp}.log
