#!/bin/bash

while getopts 'e:c:i:l:s:t:n:d:b:r:p:' OPT; do
    case $OPT in
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    i) identifier=$OPTARG;;
		    l) lr=$OPTARG;;
        s) timestep=$OPTARG;;
		    t) task=$OPTARG;;
		    n) num_epochs=$OPTARG;;
		    d) train_flag=$OPTARG;;
        b) basemodel_path=$OPTARG;;
        r) refiner_path=$OPTARG;;
        p) patch_size=$OPTARG;;
    esac
done
export CUDA_VISIBLE_DEVICES=${cuda}
# if [ -z "$refiner_path" ]; then
#   refiner_path="/media/HDD/fanlinqian/work_dirs_ssl/Exp_refiner_LA/"${folder}${exp}${identifier}
# fi
# echo "exp:" $exp
# echo "cuda:" $cuda
# echo "num_epochs:" $num_epochs
# echo "train_flag:" $train_flag

CURRENT_TIME=$(date +"%Y%m%d-%s")

if [[ "${task}" == *"la"* ]];
  then
    labeled_data="train_0.7"
    unlabeled_data="unlabeled"
    eval_data="val_0.1"
    test_data="test"
    folder="Exp_refiner_LA/"${CURRENT_TIME}"-"
    if [ ${train_flag} = "true" ]; then
      python code/train_${exp}.py --exp ${folder}${exp}${identifier} --seed 0 -g ${cuda} --base_lr ${lr} --timestep ${timestep} -ep ${num_epochs} -sl ${labeled_data} -se ${eval_data} -t ${task} --patch_size ${patch_size}
    else
      python code/test_diffusion_refiner.py --refiner_path ${refiner_path} --basemodel_path ${basemodel_path} -g ${cuda} --split ${test_data} -t ${task} --timestep ${timestep}
      python code/evaluate_la.py --full_path ${refiner_path} --folds 1 --split ${test_data} -t ${task} --folds 1 --split ${test_data} -t ${task}
    fi
fi

if [[ "${task}" == *"oss"* ]];
  then
    # labeled_data="all"
    # labeled_data="complete_DG"
    labeled_data="new_manual1"
    # labeled_data="data12_train"
    unlabeled_data="unlabeled"
    # eval_data="eval"
    eval_data="data12_eval"
    # test_data="manual_test"
    test_data="all"
    train_std="train_std"
    eval_std="eval_std"
    train_eval="train+eval"
    folder="Exp_refiner_OSS/"${CURRENT_TIME}"-"
    if [ ${train_flag} = "true" ]; then
       python code/train_${exp}.py --exp ${folder}${exp}${identifier} --seed 0 -g ${cuda} --base_lr ${lr} --timestep ${timestep} -ep ${num_epochs} -sl ${labeled_data} -se ${eval_data} -t ${task} --patch_size ${patch_size} --base_model ${basemodel_path}
    else
      # python code/test_diffusion_refiner_oss.py --refiner_path ${refiner_path} -g ${cuda} --split ${labeled_data} -t ${task} --timestep ${timestep} --patch_size ${patch_size}
      # python code/test_diffusion_refiner_oss.py --refiner_path ${refiner_path} -g ${cuda} --split ${eval_data} -t ${task} --timestep ${timestep} --patch_size ${patch_size} #--base_model ${basemodel_path}
      python code/test_${exp}.py --refiner_path ${refiner_path} -g ${cuda} --split ${test_data} -t ${task} --timestep ${timestep} --patch_size ${patch_size} #--base_model ${basemodel_path}
    # python code/evaluate_la.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} -t ${task}
    fi
fi

