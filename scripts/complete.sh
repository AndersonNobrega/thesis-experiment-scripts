#!/bin/bash
./train.sh --simple_eval
./eval.sh --simple_eval

./train.sh --hard_eval
./eval.sh --hard_eval