import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_factory import data_provider
from exp_basic import Exp_Basic
import TCDformer
from tools import EarlyStopping, adjust_learning_rate, visual
from metrics import metric

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.best_threshold = None

    def _build_model(self):
        model_dict = {'TCDformer': TCDformer}
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                device = self.model.device
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :], device=device).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :].to(device), dec_inp], dim=1)

                # 模型推理
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 特征对齐
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:f_dim + 1]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:f_dim + 1].to(self.device)

                # 收集数据
                val_preds.append(outputs.detach().cpu().numpy())
                val_trues.append(batch_y.detach().cpu().numpy())
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)

        # 计算指标
        threshold = np.median(val_trues)
        metrics = metric(val_preds, val_trues, threshold, target_dim=0)
        metrics["val_trues"] = val_trues  # 新增关键行

        self.model.train()
        return np.average(total_loss), metrics

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            print(f"Batch {i} 维度检查:")
            print(f"batch_x shape: {batch_x.shape}")  # 应为 [batch_size, seq_len, 202]
            print(f"batch_y shape: {batch_y.shape}")  # 应为 [batch_size, pred_len, 1]

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.model.device)
                batch_y = batch_y.float().to(self.model.device)
                batch_x_mark = batch_x_mark.float().to(self.model.device)
                batch_y_mark = batch_y_mark.float().to(self.model.device)

                # 解码器输入构建
                device = self.model.device  # 确保使用模型所在的设备
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :], device=device).float()
                dec_inp = torch.cat([
                    batch_y[:, :self.args.label_len, :].to(device),
                    dec_inp
                ], dim=1)
                # 在训练循环中添加检查
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    print(f"Batch {i} 维度检查:")
                    print(f"batch_x shape: {batch_x.shape}")  # [32, 96, 202]
                    print(f"batch_y shape: {batch_y.shape}")  # [32, 96, 1]
                    print(f"dec_inp shape: {dec_inp.shape}")  # [32, 48+96=144, 1]
                    break

                # 模型前向传播
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:f_dim + 1].to(self.device)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:f_dim + 1].to(self.device)
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                # 反向传播
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # 验证阶段
            train_loss = np.average(train_loss)
            vali_loss, val_metrics = self.vali(vali_data, vali_loader, criterion)

            # 关键修复：从返回的metrics中获取真实值
            self.best_threshold = np.median(val_metrics["val_trues"])

            print(f"\nEpoch: {epoch + 1}/{self.args.train_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {vali_loss:.4f}")
            print(f"Best Threshold: {self.best_threshold:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

            # 早停机制
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载最佳模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            model_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(model_path))

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 解码器输入构建
                device = self.model.device  # 确保使用模型所在的设备
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :], device=device).float()
                dec_inp = torch.cat([
                    batch_y[:, :self.args.label_len, :].to(device),
                    dec_inp
                ], dim=1)

                # 模型推理
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 特征对齐
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:f_dim + 1].cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:f_dim + 1].cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)

        # 处理结果
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        threshold = self.best_threshold if self.best_threshold else np.median(trues)
        metrics = metric(preds, trues, threshold, target_dim=0)

        # 打印结果（修复点：添加Precision输出）
        print("\n========== 测试结果 ==========")
        print(f"MAE: {metrics['mae']:.4f} | MSE: {metrics['mse']:.4f}")
        print(
            f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | F1: {metrics['f1_score']:.4f}")
        print(f"使用阈值: {threshold:.4f}")

        # 保存结果
        result_dir = f'./results/{setting}/'
        os.makedirs(result_dir, exist_ok=True)
        np.savez(os.path.join(result_dir, 'results.npz'),
                 preds=preds, trues=trues, metrics=metrics)
        return metrics

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            model_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(model_path))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                device = self.model.device  # 确保使用模型所在的设备
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :], device=device).float()
                dec_inp = torch.cat([
                    batch_y[:, :self.args.label_len, :].to(device),
                    dec_inp
                ], dim=1)

                # 模型推理
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                preds.append(outputs.detach().cpu().numpy())

        # 保存结果
        preds = np.concatenate(preds)
        pred_dir = f'./results/{setting}/pred/'
        os.makedirs(pred_dir, exist_ok=True)
        np.save(os.path.join(pred_dir, 'predictions.npy'), preds)
        return preds