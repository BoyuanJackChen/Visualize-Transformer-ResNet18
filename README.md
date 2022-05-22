# ECE7123 Deep Learning Project2 - Understanding Why ViT Doesn’t Perform Well on Small Datasets: An Intuitive Perspective

- Please first run "python3 ./data/downloader.py" to download CIFAR-10, CIFAR-100 and SVHN, into ./data folder. No need to add parameters to the call, as they are set by default.

- To reproduce figures in Section 4, run ./training/vis_vit.py and vis_resnet.py. Please make sure you open the files and set the args parser parameters correct. Below we provide the exact code to reproduce Figures in Section 4: <br />
```
python vis_vit.py --image_idx=2 --dataset="SVHN" --load_checkpoint="../checkpoint/vit_SHVN_e100_b10_lr0.0001.pt" 
```
```
python vis_vit.py --image_idx=10 --dataset="CIFAR-10" --load_checkpoint="../checkpoint/vit_CIFAR-10_e500_b100_lr0.0001.pt"
```
```
python vis_resnet.py --image_idx=2 --dataset="SVHN" --load_checkpoint="../checkpoint/res18_svhn-4-ckpt.t7"
```
```
python vis_resnet.py --image_idx=10 --dataset="CIFAR-10" --load_checkpoint="../checkpoint/res18_CIFAR-10_e500_b100_lr0.0001.pt"
```

- Trained torch model parameters are saved in checkpoints folder. They have the following keys <br />
    'epoch': epoch <br />
    'model_state_dict': model.state_dict() <br />
    'optimizer_state_dict': optimizer.state_dict() <br />
    'train_loss': train_loss_history <br />
    'test_loss': test_loss_history <br />
    'accuracy': test_accuracy_history

- Activation visualization and feature map visualization: See ./visualize

- The models are stored in ./training/models. main.py and utils.py are for training the models. 

- Representation Similarity: See ./torch_cka 

To reproduce the results:
```
python torch_cka/cka_compare.py --dataset cifar10
```
```
python torch_cka/cka_compare.py --dataset cifar100
```
```
python torch_cka/cka_compare.py --dataset svhn
```
