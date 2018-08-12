# SRCNN
PyTorch Implementation

## Requirement
torch
torchvision
python-tk (or pyhton3-tk)

## Training

	> python my_main.py --upscale_factor 3 --batch_size 10 --cuda --test_batch_size 10 --epochs 100 --lr 0.01 --gpuid 0

or

	> python3 my_main.py --upscale_factor 3 --batch_size 10 --cuda --test_batch_size 10 --epochs 100 --lr 0.01 --gpuid 0

## Sample Usage

	> python run.py --input_image test.jpg --scale_factor 3.0 --model model_epoch_200.pth --cuda --output_filename <output_filename>

or

	> python3 run.py --input_image test.jpg --scale_factor 3.0 --model model_epoch_200.pth --cuda --output_filename <output_filename>

# 주의
sample에서 input image는 학습에 사용된 BSDS300 data가 아닌 인터넷에서 가져온 이미지 등 다른 이미지를 사용해야 정확한 성능을 확인할 수 있다.
트레이닝할 때 gpuid 숫자를 바꾸어서 사용하지 않는 gpu를 사용하면 됩니다. (0 ~ 3 사용가능)
서버에 0 ~ 3까지 총 네개 GPU가 있습니다.
