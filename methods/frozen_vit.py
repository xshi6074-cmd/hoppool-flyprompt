import gc
import logging

import torch

from methods._trainer import _Trainer

logger = logging.getLogger()


class FrozenViTTrainer(_Trainer):
    """Online trainer for frozen-ViT models with a trainable FC head.

    Used directly by ``baseline``, ``shared_prompt`` and ``gate``. Evaluation
    uses the online FC, or the online + EMA ensemble when the model has
    ``use_ema``. Subclasses customise the hooks:

    - ``prepare_online_batch``: work shared by all ``online_iter`` updates.
    - ``clip_gradients``: called between backward and the optimizer step.
    - ``after_online_updates``: called once per stream batch after training.
    - ``evaluation_forward_kwargs``: extra model inputs at evaluation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = 0

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        _loss, _acc, _iter = 0.0, 0.0, 0

        prepared = self.prepare_online_batch(images)
        for _ in range(int(self.online_iter)):
            if prepared is None:
                loss, acc = self.online_train([images.clone(), labels.clone()])
            else:
                train_images, forward_kwargs = prepared
                loss, acc = self.online_train(
                    [train_images, labels.clone()],
                    inputs_transformed=True,
                    forward_kwargs=forward_kwargs,
                )
            _loss += loss
            _acc += acc
            _iter += 1

        self.after_online_updates(images, labels)

        # Advance the internal step schedule from seen samples only
        # (task-boundary-free).
        self._maybe_advance_internal_step(images.size(0) * self.world_size)

        del images, labels
        gc.collect()
        return _loss / _iter, _acc / _iter

    def prepare_online_batch(self, images):
        """Return ``(transformed_images, forward_kwargs)`` to reuse, or None."""
        return None

    def after_online_updates(self, images, labels):
        pass

    def clip_gradients(self):
        pass

    def evaluation_forward_kwargs(self, x):
        return {}

    def online_train(self, data, inputs_transformed=False, forward_kwargs=None):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)

        if not inputs_transformed:
            x = self.train_transform(x)

        self.optimizer.zero_grad()
        if not self.no_batchmask:
            logit, loss = self.model_forward(x, y, mask=logit_mask, forward_kwargs=forward_kwargs)
        else:
            logit, loss = self.model_forward(x, y, forward_kwargs=forward_kwargs)

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.clip_gradients()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        if getattr(self.model_without_ddp, "use_ema", False):
            self.model_without_ddp.update_ema_fc()

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct / total_num_data

    def model_forward(self, x, y, mask=None, forward_kwargs=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x, **(forward_kwargs or {}))
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        model = self.model_without_ddp
        model.update()

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                forward_kwargs = self.evaluation_forward_kwargs(x)
                if getattr(model, "use_ema", False):
                    logit_ls = model.forward_with_ema(x, **forward_kwargs)
                    logit_ls = [logit + self.mask for logit in logit_ls]
                    logit = self._ensemble_logits(logit_ls)
                else:
                    logit = model(x, **forward_kwargs) + self.mask

                loss = self.criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def _ensemble_logits(self, logit_ls):
        if not hasattr(self, 'ensemble_method'):
            self.ensemble_method = "softmax_max_prob"

        if "softmax" in self.ensemble_method:
            logit_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]

        logit_stack = torch.stack(logit_ls, dim=-1)  # Shape: [batch_size, n_classes, n_heads]

        if "mean" in self.ensemble_method:
            return logit_stack.mean(dim=-1)
        elif "max_prob" in self.ensemble_method:
            return logit_stack.max(dim=-1)[0]
        elif "min_entropy" in self.ensemble_method:
            entropies = -torch.sum(logit_stack * torch.log(logit_stack + 1e-8), dim=1)  # [batch_size, n_heads]
            min_entropy_indices = torch.argmin(entropies, dim=-1)  # [batch_size]
            batch_indices = torch.arange(logit_stack.size(0), device=logit_stack.device)
            return logit_stack[batch_indices, :, min_entropy_indices]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        """Hook called after each benchmark task.

        ``task_id`` is kept for logging/analysis only; the model's internal
        step state advances exclusively via ``_maybe_advance_internal_step``.
        """
        self.task_id += 1
