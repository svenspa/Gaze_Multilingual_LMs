#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
python3 ner_lime.py --test_only --dataset $DATA --use_mt_learning --n_gaze_feat 4 --use_data_parallel --langs $FROM_LAN --unlabeled_langs $TO_LAN --dev_langs $TO_LAN --model_save_file "save/$MODEL/" --fix_emb --lowercase_char --default_emb $EMB --private_hidden_size 200 --shared_hidden_size 200 --n_critic 1 

