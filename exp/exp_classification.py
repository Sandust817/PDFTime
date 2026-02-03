from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy

import os
import time
import random
import warnings
import datetime
import pytz

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def normalize_per_series_torch(x, eps=1e-6):
    """
    x: Tensor [B, L, C]
    Per-sample, per-channel z-score over L.
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    std = torch.clamp(std, min=eps)
    return (x - mean) / std

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        exp_id = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y%m%d%H')
        self.code_save_root = f'/root/sxh/mymodel2/mymodel/checkpoints/_record{exp_id}'
        os.makedirs(self.code_save_root, exist_ok=True)

    def _build_model(self):
        train_data, _ = self._get_data(flag='TRAIN')
        test_data, _ = self._get_data(flag='TEST')

        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        losses, preds, trues = [], [], []

        with torch.no_grad():
            for batch_x, label, padding_mask in vali_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, label, None)
                loss = criterion(outputs, label.long().squeeze(-1))

                losses.append(loss.item())
                preds.append(outputs)
                trues.append(label)

        preds = torch.cat(preds)
        trues = torch.cat(trues).flatten()
        probs = F.softmax(preds, dim=1)
        acc = cal_accuracy(torch.argmax(probs, dim=1).cpu().numpy(), trues.cpu().numpy())

        self.model.train()
        return float(np.mean(losses)), acc

    def train(self, setting, pov=1):

        train_data, train_loader = self._get_data('TRAIN')
        vali_data, vali_loader = self._get_data('TEST')
        test_data, test_loader = self._get_data('TEST')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        scheduler = optim.lr_scheduler.StepLR(model_optim, step_size=max(1, self.args.patience // 2), gamma=0.5)
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            print(f"Epoch {epoch+1}/{self.args.train_epochs} | LR {model_optim.param_groups[0]['lr']:.6f}")

            self.model.train()
            losses = []
            start = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.args.data == 'UCR':
                    batch_x = normalize_per_series_torch(batch_x)

                outputs, aux_loss = self.model(batch_x, padding_mask, label, epoch)
                loss = aux_loss + criterion(outputs, label.long().squeeze(-1))

                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

                losses.append(loss.item())

            scheduler.step()
            train_loss = float(np.mean(losses))
            print(f"Epoch {epoch+1} | Train Loss {train_loss:.4f} | Time {time.time()-start:.4f}s")

            vali_loss, vali_acc = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc = self.vali(test_data, test_loader, criterion)

            print(f"Val Acc {vali_acc:.4f} | Test Acc {test_acc:.4f}")
            early_stopping(-vali_acc, self.model, path)
            if early_stopping.early_stop or vali_acc >= 1.0:
                break

        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('TEST')

        if test:
            self.model.load_state_dict(torch.load(self.args.ckpt_path))

        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch_x, label, padding_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, label, None)
                preds.append(outputs)
                trues.append(label)

        preds = torch.cat(preds)
        trues = torch.cat(trues).flatten()
        probs = F.softmax(preds, dim=1)
        acc = cal_accuracy(torch.argmax(probs, dim=1).cpu().numpy(), trues.cpu().numpy())

        print(f"Test Accuracy: {acc:.4f}")
        return acc
