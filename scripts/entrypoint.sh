#!/bin/bash

print_usage() {
  printf "Usage: \n"
  printf "m : insert_mode \n"
  printf "mode can be -> kc, ka, an, tr, te \n"
  printf "kc = kernel creation through KernelGAN \n"
  printf "ka = kernel application to images \n"
  printf "na = noise addition to images \n"
  printf "tr = train the model \n"
  printf "te = test the model \n"
}


while getopts ":m:" opt; do
    case $opt in
        m)
            mode=$OPTARG
            ;;
        \?)
            echo "Invalid option : "
            print_usage
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument."
            exit 1
            ;;
    esac
done


echo "starting... \n"
if [ "$mode" = "kc" ]
then
    echo "kernel creation"
    cd Real-SR/codes/preprocess/KernelGAN
    CUDA_VISIBLE_DEVICES=1,0 python3 train.py --X4 --input-dir ../../../../HC18/train_set
elif [ "$mode" = "ka" ] 
then
    echo "kernel application"
    cd Real-SR/codes
    python3 ./preprocess/create_kernel_dataset.py --dataset hc18 --artifacts clean --kernel_path ./preprocess/KernelGAN/results
    
elif [ "$mode" = "na" ] 
then
    echo "noise addition"
    cd Real-SR/codes
    python3 ./preprocess/collect_noise.py --dataset hc18 --artifacts clean
elif [ "$mode" = "tr" ] 
then 
    echo "training"
    cd Real-SR/codes
    CUDA_VISIBLE_DEVICES=0,1 python3 train.py -opt options/dped/train_kernel_noise.yml
elif [ "$mode" = "te" ] 
then
    echo "testing"
    cd Real-SR/codes
    CUDA_VISIBLE_DEVICES=0 python3 test.py -opt options/dped/test_dped.yml
else 
    echo "invalid option"
    print_usage
    exit 1
fi



