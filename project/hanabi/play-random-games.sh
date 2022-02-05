#!/bin/sh

# set -x

for i in {0..100}
do
    echo "Running game n. ${i}"

    N_PLAYERS=$((2 + $RANDOM % 4))
    python server.py $N_PLAYERS 1>>/dev/null 2>> server-errors.log &
    SERVER_PID=$!

    sleep 1

    is_alive=$(ps -a | grep $SERVER_PID | wc -l)
    if [[ $is_alive -eq '0' ]]
    then
        echo "Error!"
        exit
    fi

    for (( n=0; n<$N_PLAYERS; n++ ))
    do 
        python client.py 127.0.0.1 1024 "RL${n}" 1>>/dev/null 2>> client-errors.log & 
    done

    wait


    sleep 1

done