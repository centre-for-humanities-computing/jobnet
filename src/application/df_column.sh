#!/bin/bash

file_names=('lemmas'
            'nn_adj_lemmas');         

for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 tfidf_occ_area.py $my_file_name

done
