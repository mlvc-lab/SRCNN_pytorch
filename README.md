# SRGAN
Variation of SRGAN

## Requirement
torch
torchvision
pyhton3-tk

## Training

```sh
# Just run
$ python3 main.py --upscale_factor 3 --cuda --batch_size 6 --test_batch_size 6 --epochs 50 --lr 0.0001
```

## Sample Usage

> Caution! : 현재 버그로 `run.py` 실행이 안됨 고쳐주실 분 모집 하는중...

```sh
python3 run.py --input_image test.jpg --scale_factor 3.0 --model model_epoch_100.pth --cuda --output_filename <output_filename>
```

> Caution! : sample에서 input image는 학습에 사용된 data가 아닌 인터넷에서 가져온 이미지 등 다른 이미지를 사용해야 정확한 성능을 확인할 수 있다.
