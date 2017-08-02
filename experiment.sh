#/usr/bin/env bash
for nenc in 3 4
do
    for ndec in 3 4
    do
        echo "Processing nenc = $nenc, ndec = $ndec"
        TRAIN_DIR="trainings/fconv_binaryWeightActivation$nenc$ndec"
        if [ -d $TRAIN_DIR ]
        then
            rm -rf $TRAIN_DIR
        fi
        mkdir $TRAIN_DIR
        fairseq train -sourcelang de -targetlang en -datadir data-bin/iwslt14.tokenized.de-en -model fconv -nenclayer $nenc -nlayer $ndec -dropout 0.2 -optim nag -lr 0.5 -lrshrink 2 -clip 0.1 -momentum 0.99 -timeavg -bptt 0 -savedir $TRAIN_DIR -log_interval 500 -binaryWeight -binaryActivation -log 2>&1 | tee "$TRAIN_DIR/stdout.txt"
    done
done
