python train.py --config configs/new/mul-fusion/full-market1501.yaml --resume outputs/mul-fusion-market1501/20241007044338/save/checkpoint-epoch240.pth --test-only

python train.py --config configs/new/mul-fusion/full-msmt17.yaml --resume outputs/mul-fusion-msmt17/20241010165423/save/checkpoint-epoch240.pth --test-only

python train.py --config configs/new/concat-fusion/full-market1501.yaml --resume outputs/concat-fusion-market1501/20241008001030/save/checkpoint-epoch240.pth --test-only

python train.py --config configs/new/concat-fusion/full-msmt17.yaml --resume outputs/concat-fusion-msmt17/20241008082033/save/checkpoint-epoch240.pth --test-only
