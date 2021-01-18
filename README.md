# Siamese network for one shot learning

A implementation of the paper : [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) using pytorch. In the model, somethings, such as learning rates or regression, may differ from the original paper.

You can run one shot learning step by step. Also, I posted the details of the code in ***Korean*** on my blog.

한글로 논문과 코드에 대해 작성한 글이 있으니 관심있으신 분은 확인해보세요!



### 🚀How to run

You can execute three action. *just run*, *download-data*, *train*, *test*.

1. #### Clone

   Clone this repository and go into the directory.

   ```bash
   git clone https://github.com/Rhcsky/siamese-one-shot-pytorch.git
   
   cd siamese-one-shot-pytorch
   ```

2. #### Run

   This commend automatically executes the entire process according to `config_maker`(download data + train + test).

   If you just want to try this network, I recommend this.

   ```bash
   python main.py run
   ```

3. #### Download-data

   The Omniglot data is downloaded and divided into 30 types of train data, 10 types of validation data, and 10 types of test data. All data is contained in `./data/processed/`.

   ```bash
   python main.py download-data
   ```

4. #### Train

   Only model learning is conducted. If you want to run 'train', you have to run 'download-data' first.

   ```bash
   python main.py train
   ```

5. #### Test

   Only test the model. Stored models and datasets must exist.

   ```bash
   python main.py test
   ```

All parameters are present in `config_maker`. If you want to adjust the parameters, modify them and run the code.



### Check Result

Train logs, saved model and configuration data were in `./result/[model_number]`.  Logs are made by `tensorboard`. So if you want to see more detail about train metrics, write commend on `./siamese_network/result/[model_number]`  like this.

```
tensorboard --logdir=logs
```



### 📌Reference

* [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

* [kevinzakka/one-shot-siamese](https://github.com/kevinzakka/one-shot-siamese)
* [fangpin/siamese-pytorch](https://github.com/fangpin/siamese-pytorch)

