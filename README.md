# SRGAN
Variation of SRGAN

## Requirement
torch
torchvision
pyhton3-tk

## Training

```sh
# Just training
$ python3 main.py --upscale_factor 3 --cuda --gpuids 0 --batch_size 6 --test_batch_size 6 --epochs 50 --lr 0.0001 
```

### options
| option | explain | example | default |
| ------ | ------- | ------- | ------- |
| --save_path        | set save path of trained model         | --save_path model   | 'model' |
| --upscale_factor   | set upscaling factor, must be Inteager | --upscale_factor 3  | 3       |
| --batch_size       | set training batch size                | --batch_size 10     | 32      |
| --test_batch_size  | set test batch size                    | --test_batch_size 6 | 4       |
| --epochs           | set epochs                             | --epochs 10         | 50      |
| --lr               | set learning rate                      | --lr 0.0025         | 0.0001  |
| --cuda             | set cuda                               | --cuda              | -       |
| --threads          | set data loader worker threads         | --threads 16        | 32      |
| --gpuids           | set gpuids, only to work with --cuda   | --gpuids 0 1 2 3    | 0       |
| --alpha            | set alpha of loss                      | --alpha 0.5         | 0.75    |

## Sample Usage


```sh
python3 run.py --input_image test.jpg --scale_factor 3.0 --model model_epoch_100.pth --cuda --output_filename <output_filename>
```

> Caution! : sample에서 input image는 학습에 사용된 data가 아닌 인터넷에서 가져온 이미지 등 다른 이미지를 사용해야 정확한 성능을 확인할 수 있다.

### options
| option | explain | example | default |
| ------ | ------- | ------- | ------- |
| --model            | set saved model path                   | --save_path model.pth     | -        |
| --input_image      | set input image path                   | --input_image test.jpg    | test.jpg |
| --output_filename  | set output file name of path           | --output_filename out.jpg | out.jpg  |
| --upscale_factor   | set upscaling factor, must be Inteager | --upscale_factor 3        | 3        |
| --cuda             | set cuda                               | --cuda                    | -        |
| --gpuids           | set gpuids, only to work with --cuda   | --gpuids 0 1 2 3          | 0        |

