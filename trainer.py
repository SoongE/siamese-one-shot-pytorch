import os
import shutil

import torch
import torch.optim as optim
from torch.nn.functional import sigmoid
from tqdm import tqdm

from data_loader import get_train_validation_loader, get_test_loader
from model import SiameseNet
from utils import AverageMeter


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for training
    the Siamese Network model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator.
        - layer_hyperparams: dict containing layer-wise hyperparameters
          such as the initial learning rate, the end momentum, and the l2
          regularization strength.
        """
        self.config = config

        # path params
        self.ckpt_dir = os.path.join(config.ckpt_dir, config.num_model)
        self.logs_dir = os.path.join(config.logs_dir, config.num_model)

    def train(self):
        # Dataloader
        train_loader, valid_loader = get_train_validation_loader(self.config.data_dir, self.config.batch_size,
                                                                 self.config.num_train,
                                                                 self.config.augment, self.config.way,
                                                                 self.config.valid_trials,
                                                                 self.config.shuffle, self.config.seed)

        # Model, Optimizer, criterion
        model = SiameseNet()
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=6e-5)
        criterion = torch.nn.BCEWithLogitsLoss()
        if self.config.use_gpu:
            model.cuda()

        # Load check point
        if self.config.resume:
            start_epoch, best_valid_acc, model_state, optim_state = self.load_checkpoint(best=False)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optim_state)
        else:
            start_epoch = 0
            best_valid_acc = 0

        # create train and validation log files
        train_file = open(os.path.join(self.logs_dir, 'train.csv'), 'w')
        valid_file = open(os.path.join(self.logs_dir, 'valid.csv'), 'w')

        counter = 0
        num_train = len(train_loader.dataset)
        num_valid = valid_loader.dataset.trials
        print(f"\n[*] Train on {num_train} sample pairs, validate on {num_valid} trials")

        # Train
        for epoch in tqdm(range(start_epoch, self.config.epochs), total=self.config.epochs):
            train_losses = AverageMeter()

            # TRAIN
            model.train()
            pbar = tqdm(enumerate(train_loader), total=num_train, desc="Train")
            for i, (x1, x2, y) in pbar:
                if self.config.use_gpu:
                    x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
                out = model(x1, x2)
                loss = criterion(out, y)

                # compute gradients and update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # store batch statistics
                batch_size = x1.shape[0]
                train_losses.update(loss.data[0], batch_size)

                # log loss
                train_file.write(f'{(epoch * len(train_loader)) + i},{train_losses.val}\n')
                pbar.set_postfix(f"loss: {train_losses.val:.3f}")

            # VALIDATION
            model.eval()
            with torch.no_grad:
                correct = 0
                pbar = tqdm(enumerate(valid_loader), total=num_valid, desc="Valid")
                for i, (x1, x2, y) in pbar:
                    if self.config.use_gpu:
                        x1, x2 = x1.cuda(), x2.cuda()

                    # compute log probabilities
                    out = model(x1, x2)
                    log_probas = sigmoid(out)

                    # get index of max log prob
                    pred = log_probas.data.max(0)[1][0]
                    if pred == 0:
                        correct += 1

                # compute acc and log
                valid_acc = (100. * correct) / num_valid
                valid_file.write(f'{epoch},{valid_acc}\n')
                pbar.set_postfix({"accuracy": correct / num_valid})

            # check for improvement
            if valid_acc > best_valid_acc:
                is_best = True
            else:
                is_best = False

            msg = "train loss: {:.3f} - valid acc: {:.3f}"
            if is_best:
                msg += " [*]"
                counter = 0
            print(msg.format(train_losses.avg, valid_acc))

            # checkpoint the model
            if not is_best:
                counter += 1
            if counter > self.config.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            best_valid_acc = max(valid_acc, best_valid_acc)
            self.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'best_valid_acc': best_valid_acc,
                }, is_best
            )
        # release resources
        train_file.close()
        valid_file.close()

    def test(self):

        # Load best model
        model = SiameseNet()
        start_epoch, best_valid_acc, model_state, _ = self.load_checkpoint(best=self.config.best)
        model.load_state_dict(model_state)
        if self.config.use_gpu:
            model.cuda()

        test_loader = get_test_loader(self.config.data_dir, self.config.way, self.config.test_trials,
                                      self.config.random_seed)
        num_test = test_loader.dataset.trials
        correct = 0

        pbar = tqdm(enumerate(test_loader), total=num_test, desc="Test")
        for i, (x1, x2) in pbar:
            if self.config.use_gpu:
                x1, x2 = x1.cuda(), x2.cuda()

            # compute log probabilities
            out = model(x1, x2)
            log_probas = sigmoid(out)

            # get index of max log prob
            pred = log_probas.data.max(0)[1][0]
            if pred == 0:
                correct += 1
            pbar.set_postfix({"accuracy": {correct / num_test}})

        test_acc = (100. * correct) / num_test
        print(f"Test Acc: {correct}/{num_test} ({test_acc:.2f}%)")

    def save_checkpoint(self, state, is_best):
        filename = 'model_ckpt.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = 'best_model_ckpt.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best):
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = 'model_ckpt.tar'
        if best:
            filename = 'best_model_ckpt.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        if best:
            print(
                f"[*] Loaded {filename} checkpoint @ epoch {ckpt['epoch']}with best valid acc of {ckpt['best_valid_acc']:.3f}")
        else:
            print(f"[*] Loaded {filename} checkpoint @ epoch {ckpt['epoch']}")

        return ckpt['epoch'], ckpt['best_valid_acc'], ckpt['model_state'], ckpt['optim_state']
