#!/bin/bash

while getopts 'e:c:i:l:w:t:n:d:f:p:' OPT; do
    case $OPT in
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    i) identifier=$OPTARG;;
		    l) lr=$OPTARG;;
		    w) stu_w=$OPTARG;;
		    t) task=$OPTARG;;
		    n) num_epochs=$OPTARG;;
		    d) train_flag=$OPTARG;;
        f) full_path=$OPTARG;;
        p) patch_size=$OPTARG;;
    esac
done
echo "exp:" $exp
echo "cuda:" $cuda
echo "num_epochs:" $num_epochs
echo "train_flag:" $train_flag

CURRENT_TIME=$(date +"%Y%m%d-%s")

if [[ "${task}" == *"la"* ]];
  then
    labeled_data="split_train"
    eval_data="split_val"
    test_data="test"
    # test_data='split_val'
    folder="Exp_LA/"${CURRENT_TIME}"-"
    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier} --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -se ${eval_data} -t ${task}
    else
      python code/test_othermodels.py --full_path ${full_path} --model_name ${exp} -g ${cuda} --split ${test_data} -t ${task}
      python code/evaluate_la.py --full_path ${full_path} --folds 1 --split ${test_data} -t ${task}
    fi
fi

if [[ "${task}" == *"lv"* ]];
  then
    labeled_data="train"
    eval_data="eval"
    test_data="test"
    # test_data='split_val'
    folder="Exp_LV/"${CURRENT_TIME}"-"
    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier} --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -se ${eval_data} -t ${task}
    else
      python code/test_othermodels.py --full_path ${full_path} --model_name ${exp} -g ${cuda} --split ${test_data} -t ${task} --speed 3
      python code/evaluate.py --full_path ${full_path} --folds 1 --split ${test_data} -t ${task}
    fi
fi


if [[ "${task}" == *"oss"* ]];
  then
    labeled_data="train"
    # labeled_data="complete_DG"
    # labeled_data="tmp_test"
    eval_data="eval"
    # eval_data="complete_DG_eval"
    # eval_data="tmp_test"
    # test_data="test"
    test_data="all"
    # test_data="all"
    train_std="train_std"
    eval_std="eval_std"
    train_eval="train+eval"
    folder="Exp_OSS/"${CURRENT_TIME}"-"
    if [ ${train_flag} = "true" ]; then
       python code/train_othermodels_oss.py --exp ${folder}${exp}${identifier} --model_name ${exp} --seed 0 -g ${cuda} --base_lr ${lr} -ep ${num_epochs} -sl ${labeled_data} -se ${eval_data} -t ${task} --patch_size ${patch_size}
    else
      python code/test_othermodels_v2.py --full_path ${full_path} -g ${cuda} --split ${test_data} -t ${task} --patch_size ${patch_size} --model_name ${exp}
      # python code/test_othermodels_v2.py --full_path ${full_path} -g ${cuda} --split ${eval_data} -t ${task} --patch_size ${patch_size} --model_name ${exp}
    fi
fi
