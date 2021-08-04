#!/usr/bin/env bash

#use this line to run the main.py file with a specified config file
if [[ "$1" == "test" ]];then
    python3 main.py config/example_exp_0_testing.json
else
    python3 main.py config/example_exp_0.json
fi