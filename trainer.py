import os
import shutil
from glob import glob

import torch
import torch.optim as optim
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
                                                                 self.config.shuffle, self.config.seed,
                                                                 self.config.num_workers, self.config.pin_memory)

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

        best_epoch = 0
        counter = 0
        num_train = len(train_loader)
        num_valid = len(valid_loader)
        print(f"[*] Train on {len(train_loader.dataset)} sample pairs, validate on {len(valid_loader.dataset)} trials")

        # Train & Validation
        main_pbar = tqdm(range(start_epoch, self.config.epochs), total=self.config.epochs, ncols=100, desc="Process")
        for epoch in main_pbar:
            train_losses = AverageMeter()

            # TRAIN
            model.train()
            train_pbar = tqdm(enumerate(train_loader), total=num_train, desc="Train", ncols=100)
            for i, (x1, x2, y) in train_pbar:
                if self.config.use_gpu:
                    x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
                out = model(x1, x2)
                loss = criterion(out, y.unsqueeze(1))

                # compute gradients and update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # store batch statistics
                batch_size = x1.shape[0]
                train_losses.update(loss.item(), batch_size)

                # log loss
                train_file.write(f'{(epoch * len(train_loader)) + i},{train_losses.val}\n')
                train_pbar.set_postfix_str(f"loss: {train_losses.val:0.3f}")

            main_pbar.refresh()

            # VALIDATION
            model.eval()
            correct = 0
            valid_acc = 0
            valid_pbar = tqdm(enumerate(valid_loader), total=num_valid, desc="Valid", ncols=100)
            for i, (x1, x2, y) in valid_pbar:
                if self.config.use_gpu:
                    x1, x2 = x1.cuda(), x2.cuda()

                with torch.no_grad():
                    # compute log probabilities
                    out = model(x1, x2)
                    log_probas = torch.sigmoid(out)

                # get index of max log prob
                pred = log_probas.data.max(0)[1][0]
                if pred == 0:
                    correct += 1

                # compute acc and log
                valid_acc = correct / num_valid
                valid_file.write(f'{epoch},{valid_acc}\n')
                valid_pbar.set_postfix_str(f"accuracy: {valid_acc:0.3f}")

            main_pbar.refresh()

            # check for improvement
            if valid_acc > best_valid_acc:
                is_best = True
                best_valid_acc = valid_acc
                best_epoch = epoch
                counter = 0
            else:
                is_best = False
                counter += 1

            # checkpoint the model
            if counter > self.config.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return

            if is_best or epoch % 5 == 0 or epoch == self.config.epochs:
                self.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict(),
                        'best_valid_acc': best_valid_acc,
                    }, is_best
                )

            main_pbar.set_postfix_str(
                f"train loss: {train_losses.avg:.3f} - valid acc: {valid_acc:.3f} - best: {best_epoch}")

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
                                      self.config.seed, self.config.num_workers, self.config.pin_memory)
        num_test = test_loader.dataset.trials
        correct = 0

        pbar = tqdm(enumerate(test_loader), total=num_test, desc="Test")
        for i, (x1, x2, y) in pbar:
            if self.config.use_gpu:
                x1, x2 = x1.cuda(), x2.cuda()

            # compute log probabilities
            out = model(x1, x2)
            log_probas = torch.sigmoid(out)

            # get index of max log prob
            pred = log_probas.data.max(0)[1][0]
            if pred == 0:
                correct += 1
            pbar.set_postfix_str(f"accuracy: {correct} / {num_test}")

        test_acc = (100. * correct) / num_test
        print(f"Test Acc: {correct}/{num_test} ({test_acc:.2f}%)")

    def save_checkpoint(self, state, is_best):
        filename = f'model_ckpt_{state["epoch"]}.tar'
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
        ckpt_path = sorted(glob(self.ckpt_dir + '/model_ckpt_*'))[0]

        if best:
            filename = 'best_model_ckpt.tar'
            ckpt_path = os.path.join(self.ckpt_dir, filename)

        ckpt = torch.load(ckpt_path)

        if best:
            print(
                f"[*] Loaded {filename} checkpoint @ epoch {ckpt['epoch']} with best valid acc of {ckpt['best_valid_acc']:.3f}")
        else:
            print(f"[*] Loaded {filename} checkpoint @ epoch {ckpt['epoch']}")

        return ckpt['epoch'], ckpt['best_valid_acc'], ckpt['model_state'], ckpt['optim_state']
