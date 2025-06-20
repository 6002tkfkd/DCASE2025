from typing import Dict, Optional
import torch
import torchinfo
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from model.backbones import _BaseBackbone
from util.lr_scheduler import exp_warmup_linear_down
from util import _SpecExtractor, ClassificationSummary, _DataAugmentation
import pandas as pd
from collections import defaultdict
import numpy as np
from util import unique_labels
from sklearn.metrics import log_loss
from model.shared import DeviceFilter



class LitAcousticSceneClassificationSystem(L.LightningModule):
    def __init__(self,
                 backbone: _BaseBackbone,
                 data_augmentation: Dict[str, Optional[_DataAugmentation]],
                 class_label: str = "scene",
                 domain_label: str = "device",
                 spec_extractor: _SpecExtractor = None,
                 device_list: list[str] = None,
                 device_unknown_prob: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'spec_extractor'])
        self.backbone = backbone
        self.data_aug = data_augmentation
        self.class_label = class_label
        self.domain_label = domain_label
        self.cla_summary = ClassificationSummary(class_label, domain_label)
        self.spec_extractor = spec_extractor
        self._test_step_outputs = {'emb': [], 'y': [], 'pred': [], 'd': [], 'logits': [], 'loss': []}
        self._test_input_size = None

        self.idx_to_label = {
            0: "airport",
            1: "bus",
            2: "metro",
            3: "metro_station",
            4: "park",
            5: "public_square",
            6: "shopping_mall",
            7: "street_pedestrian",
            8: "street_traffic",
            9: "tram"
        }

        self.device_unknown_prob = device_unknown_prob
        if device_list is not None:
            self.device_filter = DeviceFilter(device_list, input_channels=1)
        else:
            self.device_filter = None

    @staticmethod
    def accuracy(logits, labels):
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == labels).item() / len(labels)
        return acc, pred

    def apply_device_filter(self, x, device_names: list):
        if self.device_filter is None:
            return x

        # 문자열 → index 변환
        if isinstance(device_names[0], str):
            if self.training and self.device_unknown_prob > 0:
                device_idxs = [
                    self.device_filter.device_to_idx.get(name, self.device_filter.default_device_idx)
                    if torch.rand(1).item() > self.device_unknown_prob
                    else self.device_filter.default_device_idx
                    for name in device_names
                ]
            else:
                device_idxs = [
                    self.device_filter.device_to_idx.get(name, self.device_filter.default_device_idx)
                    for name in device_names
                ]
        else:
            # 이미 int index이면 그대로 사용
            device_idxs = device_names

        print(f"[DeviceFilter] input device_names: {device_names}")
        print(f"[DeviceFilter] mapped device_idxs: {device_idxs}")

        device_tensor = torch.tensor(device_idxs, dtype=torch.long, device=x.device)
        return self.device_filter(x, device_tensor)



    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # Load a batch of waveforms with size (N, X)
        x = batch[0]
        # Store label dices in a dict
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        # Choose class label
        y = labels[self.class_label]
        # Instantiate data augmentations
        dir_aug = self.data_aug.get('dir_aug', None) # self.data_aug['dir_aug']
        mix_style = self.data_aug.get('mix_style', None)
        spec_aug = self.data_aug.get('spec_aug', None)
        mix_up = self.data_aug.get('mix_up', None)

        filt_aug = self.data_aug.get('filt_aug', None)
        noise_aug = self.data_aug.get('add_noise', None)
        freq_mask_aug = self.data_aug.get('freq_mask', None)
        time_mask_aug = self.data_aug.get('time_mask', None)
        frame_shift_aug = self.data_aug.get('frame_shift', None)

        x = dir_aug(x, labels['device']) if dir_aug is not None else x # Apply dir augmentation on waveform
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1) # Extract spectrogram from waveform

        x = mix_style(x) if mix_style is not None else x
        x = spec_aug(x) if spec_aug is not None else x
        if mix_up is not None:
            x, y = mix_up(x, y)
        
        x = filt_aug(x) if filt_aug is not None else x
        x = noise_aug(x) if noise_aug is not None else x
        x = freq_mask_aug(x) if freq_mask_aug is not None else x
        x = time_mask_aug(x) if time_mask_aug is not None else x
        x = frame_shift_aug(x) if frame_shift_aug is not None else x

        x = self.apply_device_filter(x, labels['device'])
        y_hat = self(x)
        
        # Calculate the loss and accuracy
        if mix_up is not None:
            pred = torch.argmax(y_hat, dim=1)
            train_loss = mix_up.lam * F.cross_entropy(y_hat, y[0]) + (1 - mix_up.lam) * F.cross_entropy(
                y_hat, y[1])
            corrects = (mix_up.lam * torch.sum(pred == y[0]) + (1 - mix_up.lam) * torch.sum(
                pred == y[1]))
            train_acc = corrects.item() / len(x)
        else:
            train_loss = F.cross_entropy(y_hat, y)
            train_acc, _ = self.accuracy(y_hat, y)
        # Log for each epoch
        self.log_dict({'train_loss': train_loss, 'train_acc': train_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)
        x = self.apply_device_filter(x, labels['device'])
        
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc, _ = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': val_loss, 'val_acc': val_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("epoch", self.current_epoch, on_epoch=True, prog_bar=False, logger=True)
        # Lightning이 epoch을 자동으로 기록하지 않으므로 수동으로 기록, 즉 epoch이라는게 있는지도 모른다.
        # Lighting이 epoch라는 이름의 metric을 실제로 log로 남기고 있어야 하기 때문에
        # epoch이라는 값의 변화를 추적해서 저장여부를 판단하겠다는 의미 yaml의 monitor : epoch
        # epoch을 볼 수 있게 수치화 해버린 것 ex) loss, acc, epoch
        return val_acc




    def test_step(self, batch, batch_idx):
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]
        d = labels[self.domain_label]

        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)

        # device specific log_loss, acc
        # x = self.apply_device_filter(x, labels['device'])

        # general log_loss, acc
        devices_label = torch.full((x.size(0),), 9, dtype=torch.int64)  # 모든 데이터에 대해 device 필터링을 적용
        x = self.apply_device_filter(x, devices_label)  # 모든 데이터에 대해 device 필터링을 적용

        self._test_input_size = (1, 1, x.size(-2), x.size(-1))
        
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        test_acc, pred = self.accuracy(y_hat, y)
        
        self.log_dict({'test_loss': test_loss, 'test_acc': test_acc})

        self._test_step_outputs['y'] += y.cpu().numpy().tolist()
        self._test_step_outputs['pred'] += pred.cpu().numpy().tolist()
        self._test_step_outputs['d'] += d.cpu().numpy().tolist()
        self._test_step_outputs['logits'] += y_hat.detach().cpu().numpy().tolist()
        self._test_step_outputs['loss'] += [test_loss.item()]
        return test_acc



    def on_test_epoch_end(self):

        tensorboard = self.logger.experiment

        # --- 모델 프로파일 출력 ---
        print("\n Model Profile:")
        model_profile = torchinfo.summary(self.backbone, input_size=self._test_input_size)
        macc = model_profile.total_mult_adds
        params = model_profile.total_params
        print('MACC:\t \t %.6f' % (macc / 1e6), 'M')
        print('Params:\t \t %.3f' % (params / 1e3), 'K\n')

        model_summary = str(model_profile)
        model_summary += f'\n MACC:\t \t {macc / 1e6:.3f}M'
        model_summary += f'\n Params:\t \t {params / 1e3:.3f}K\n'
        model_summary = model_summary.replace('\n', '<br/>').replace(' ', '&nbsp;').replace('\t', '&emsp;')
        tensorboard.add_text('model_summary', model_summary)

        # --- 분류 리포트 및 confusion matrix ---
        tab_report = self.cla_summary.get_table_report(self._test_step_outputs)
        tensorboard.add_text('classification_report', tab_report)

        cm = self.cla_summary.get_confusion_matrix(self._test_step_outputs)
        tensorboard.add_figure('confusion_matrix', cm)

        # --- Accuracy / LogLoss 계산 ---
        pred = np.array(self._test_step_outputs['pred'])
        label = np.array(self._test_step_outputs['y'])
        device = np.array(self._test_step_outputs['d'])
        logits = np.array(self._test_step_outputs['logits'])
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        classes = list(range(probs.shape[1]))
        devices_list = np.unique(device)

        print("\nClass-wise Accuracy and LogLoss:")
        for class_idx in classes:
            idx = np.where(label == class_idx)[0]
            if len(idx) == 0: continue
            acc = np.mean((np.argmax(probs[idx], axis=1) == label[idx]))
            ll = log_loss(label[idx], probs[idx], labels=classes)
            class_name = unique_labels["scene"][class_idx]
            print(f"  {class_name:20s} : acc={acc:.4f}, logloss={ll:.4f}")

        print("\nDevice-wise Accuracy and LogLoss:")
        for d in devices_list:
            idx = np.where(device == d)[0]
            if len(idx) == 0: continue
            acc = np.mean((np.argmax(probs[idx], axis=1) == label[idx]))
            ll = log_loss(label[idx], probs[idx], labels=classes)
            device_name = unique_labels["device"][d]
            print(f"  {device_name:10s} : acc={acc:.4f}, logloss={ll:.4f}")

        # --- 평균 log loss ---
        losses = np.array(self._test_step_outputs['loss'])
        avg_logloss = losses.mean()
        print(f"\nAverage LogLoss: {avg_logloss:.4f}")
        self.log("test_logloss", avg_logloss)
        tensorboard.add_scalar("LogLoss/test", avg_logloss, self.current_epoch)

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        filenames = batch[1]  # 리스트: batch size 만큼의 파일 이름

        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)

        # ⛏️ 여기서 각 파일명에서 디바이스 추출
        devices = [fn.split('-')[-1].split('.')[0] for fn in filenames]  # ex) ['a', 'b', 'c', ...]
        x = self.apply_device_filter(x, devices)

        y_hat = self(x)  # [B, num_classes]
        probs = torch.softmax(y_hat, dim=1)  # 확률값으로 변환
        preds = torch.argmax(probs, dim=1)

        return [
            {
                'filename': filename,
                'scene_label': self.idx_to_label[pred.item()],
                'probs': prob.tolist()
            }
            for filename, pred, prob in zip(filenames, preds, probs)
        ]





class LitAscWithKnowledgeDistillation(LitAcousticSceneClassificationSystem):
    """
    ASC system with knowledge distillation using Top-2 margin-based teacher weighting.

    Args:
        temperature (float): A higher temperature indicates a softer distribution of pseudo-probabilities.
        kd_lambda (float): Weight to control the balance between kl loss and label loss.
        logits_index (int): Index of the logits in Dataset, as multiple logits may be used during training.
    """
    def __init__(self, temperature: float, kd_lambda: float, logits_index: int = -1, **kwargs):
        super(LitAscWithKnowledgeDistillation, self).__init__(**kwargs)
        self.temperature = temperature
        self.kd_lambda = kd_lambda
        self.logits_index = logits_index
        self.kl_div_loss = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')
        # self.kl_div_loss = torch.nn.KLDivLoss(log_target=True)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        labels = {'scene': batch[1], 'device': batch[2], 'city': batch[3]}
        y = labels[self.class_label]

        logits_list = batch[4]  # teachers가 예측한 logits list 각 원소는 [N, C] 형태

        # --- Margin-based weighting ---
        probs_list = [torch.softmax(logits, dim=1) for logits in logits_list]  # [T, N, C]
        margins = [torch.topk(p, k=2, dim=1).values[:, 0] - torch.topk(p, k=2, dim=1).values[:, 1] for p in probs_list]  # [T, N]
        score_stack = torch.stack(margins, dim=1)  # [N, T]
        norm_weights = score_stack / (score_stack.sum(dim=1, keepdim=True) + 1e-6)  # [N, T]

        logits_stack = torch.stack(logits_list, dim=1)  # [N, T, C]
        weights_expanded = norm_weights.unsqueeze(2)  # [N, T, 1]
        teacher_logits = (logits_stack * weights_expanded).sum(dim=1)  # [N, C]

        y_soft = F.log_softmax(teacher_logits / self.temperature, dim=-1)

        # --- Data augmentation ---
        aug = self.data_aug

        # dir_aug on waveform
        if aug.get('dir_aug') is not None:
            x = aug['dir_aug'](x, labels['device'])

        # Spectrogram
        x = self.spec_extractor(x).unsqueeze(1) if self.spec_extractor is not None else x.unsqueeze(1)

        # Other augmentations on spectrogram
        for key in ['mix_style', 'spec_aug', 'filt_aug', 'add_noise', 'freq_mask', 'time_mask', 'frame_shift']:
            if aug.get(key) is not None:
                x = aug[key](x)

        # MixUp
        if aug.get('mix_up') is not None:
            x, y, y_soft = aug['mix_up'](x, y, y_soft)

        # Forward
        y_hat = self(x)
        with torch.cuda.amp.autocast():
            y_hat_soft = F.log_softmax(y_hat / self.temperature, dim=-1)

        # Loss
        if aug.get('mix_up') is not None:
            label_loss = aug['mix_up'].lam * F.cross_entropy(y_hat, y[0]) + (1 - aug['mix_up'].lam) * F.cross_entropy(y_hat, y[1])
            kd_loss = aug['mix_up'].lam * self.kl_div_loss(y_hat_soft, y_soft[0]) + (1 - aug['mix_up'].lam) * self.kl_div_loss(y_hat_soft, y_soft[1])
        else:
            label_loss = F.cross_entropy(y_hat, y)
            kd_loss = self.kl_div_loss(y_hat_soft, y_soft)

        kd_loss = kd_loss * (self.temperature ** 2)
        loss = self.kd_lambda * label_loss + (1 - self.kd_lambda) * kd_loss

        # --- Logging ---
        avg_weights = norm_weights.mean(dim=0)  # [T]
        for i, avg_w in enumerate(avg_weights):
            self.log(f"t{i+1}_avg_w", avg_w.item(), on_step=False, on_epoch=True, prog_bar=True)
            self.logger.experiment.add_histogram(f"train/teacher{i+1}_weights", norm_weights[:, i], global_step=self.current_epoch)

        y_for_logging = y[0] if isinstance(y, tuple) else y
        num_classes = torch.max(y_for_logging).item() + 1
        for c in range(num_classes):
            mask = (y_for_logging == c)
            if mask.sum() > 0:
                for t in range(norm_weights.shape[1]):
                    avg_w_c = norm_weights[mask, t].mean().item()
                    self.logger.experiment.add_scalar(f"classwise_teacher{t+1}_weight/class_{c}", avg_w_c, self.current_epoch)
                    self.logger.experiment.add_scalar(f"class_{c}/teacher{t+1}_weight", avg_w_c, self.current_epoch)
                top_teacher_idx = torch.argmax(norm_weights[mask].mean(dim=0)).item()
                self.logger.experiment.add_scalar(f"classwise_top_teacher_idx/class_{c}", top_teacher_idx, self.current_epoch)

        self.log_dict({
            'loss': loss,
            'label_loss': label_loss,
            'kd_loss': kd_loss,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss



        
# No changes required for the scheduler subclasses
class LitAscWithWarmupLinearDownScheduler(LitAcousticSceneClassificationSystem):
    def __init__(self, optimizer: OptimizerCallable, warmup_len=4, down_len=26, min_lr=0.005, **kwargs):
        super(LitAscWithWarmupLinearDownScheduler, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.warmup_len = warmup_len
        self.down_len = down_len
        self.min_lr = min_lr

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        schedule_lambda = exp_warmup_linear_down(self.warmup_len, self.down_len, self.warmup_len, self.min_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitAscWithTwoSchedulers(LitAcousticSceneClassificationSystem):
    def __init__(self, optimizer: OptimizerCallable, scheduler1: LRSchedulerCallable, scheduler2: LRSchedulerCallable, milestones, **kwargs):
        super(LitAscWithTwoSchedulers, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.milestones = milestones

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler1 = self.scheduler1(optimizer)
        scheduler2 = self.scheduler2(optimizer)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], self.milestones)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitAscWithThreeSchedulers(LitAcousticSceneClassificationSystem):
    def __init__(self, optimizer: OptimizerCallable, scheduler1: LRSchedulerCallable, scheduler2: LRSchedulerCallable, scheduler3: LRSchedulerCallable, milestones, **kwargs):
        super(LitAscWithThreeSchedulers, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.scheduler3 = scheduler3
        self.milestones = milestones

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        scheduler1 = self.scheduler1(optimizer)
        scheduler2 = self.scheduler2(optimizer)
        scheduler3 = self.scheduler3(optimizer)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2, scheduler3], self.milestones)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}