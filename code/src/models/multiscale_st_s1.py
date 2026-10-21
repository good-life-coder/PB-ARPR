"""P2-S1: existing features, unchanged official losses, one frozen auxiliary term."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from anomalib.models import EfficientAd
from anomalib.models.image.efficient_ad.torch_model import reduce_tensor_elems
from src.evaluation.intermediate_feature_audit import Capture

TARGET = (256, 1024)
S1 = 'conv1.post_relu'
WANTED = {'teacher': [S1], 'student': [S1]}


def input_gate(x):
    if tuple(x.shape) != (1, 3, *TARGET) or x.dtype != torch.float32:
        raise ValueError('FAIL_DIMENSION_ORDER: expected FP32 [1,3,256,1024]')


def discrepancy(teacher, student):
    if teacher.shape != student.shape:
        raise ValueError('FAIL_NATURAL_MATCH: no projection, slicing or interpolation allowed')
    return (teacher.detach() - student).square()


def hard_loss(distance):
    # Direct official helper reuse matters: S1 exceeds torch.quantile's 2**24 cap.
    values = reduce_tensor_elems(distance)
    threshold = torch.quantile(values, 0.999)
    return values[values >= threshold].mean()


def s1_features(model, x):
    input_gate(x)
    with Capture(model, WANTED) as capture:
        student, distance = model.model.compute_student_teacher_distance(x)
    t, s = capture.values['teacher', S1], capture.values['student', S1]
    if t.shape != (1, 128, 253, 1021) or s.shape != t.shape:
        raise ValueError('FAIL_S1_SHAPE')
    if distance.shape != (1, 384, 56, 248):
        raise ValueError('FAIL_FINAL_SHAPE')
    return t, s, student, distance


def s1_map(distance):
    return distance.mean(dim=1, keepdim=True)


def normalized_map(raw, qa, qb):
    if not bool(torch.isfinite(qa).all() and torch.isfinite(qb).all()) or not bool(qb > qa):
        raise ValueError('FAIL_S1_QUANTILE_STATE')
    return 0.1 * (raw - qa) / (qb - qa)


def fuse(s1_normalized, final_normalized, ae_normalized):
    resize = lambda x: F.interpolate(x, size=TARGET, mode='bilinear', align_corners=False)
    s1, final, ae = map(resize, (s1_normalized, final_normalized, ae_normalized))
    ms = 0.5 * s1 + 0.5 * final
    return {'NEW_MODEL_FINAL_ONLY': 0.5 * final + 0.5 * ae,
            'NEW_MODEL_MS_ST_ONLY': ms,
            'NEW_MODEL_MS_COMBINED': 0.5 * ms + 0.5 * ae}


@torch.no_grad()
def inference_outputs(model, x, qa_s1, qb_s1):
    if model.model.training:
        raise ValueError('FAIL_INFERENCE_MODE')
    t, s, student, distance = s1_features(model, x)
    final, ae = model.model.compute_maps(x, student, distance, normalize=True)
    shallow = normalized_map(s1_map(discrepancy(t, s)), qa_s1, qb_s1)
    return fuse(shallow, final, ae)


def loss_ratio(final_losses, s1_losses, *, split, labels):
    if split != 'train' or len(labels) != 32 or any(int(y) != 0 for y in labels):
        raise ValueError('FAIL_LAMBDA_SOURCE')
    if len(final_losses) != 32 or len(s1_losses) != 32:
        raise ValueError('FAIL_LAMBDA_SAMPLE_COUNT')
    final, shallow = float(np.median(final_losses)), float(np.median(s1_losses))
    value = final / (shallow + 1e-12)
    finite = bool(np.isfinite(final_losses).all() and np.isfinite(s1_losses).all() and math.isfinite(value))
    return dict(samples=32, source='train_normal_only', median_final=final, median_s1=shallow,
                lambda_s1=value, finite=finite, used_test_data=False,
                safety_gate='PASS' if finite and 0.05 <= value <= 20 else 'FAIL_LOSS_SCALE',
                formula='median_final / (median_s1 + 1e-12)', clipped=False)


class MultiScaleSTS1(EfficientAd):
    """No new modules/parameters; checkpoint-persistent scalar calibration buffers."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('lambda_s1', torch.tensor(float('nan'), dtype=torch.float64))
        self.register_buffer('lambda_frozen', torch.tensor(False))
        self.register_buffer('qa_s1', torch.tensor(float('nan')))
        self.register_buffer('qb_s1', torch.tensor(float('nan')))
        self.register_buffer('s1_quantiles_fitted', torch.tensor(False))
        self.last_loss_log = {}

    def freeze_lambda(self, value):
        if bool(self.lambda_frozen):
            if float(self.lambda_s1) != value:
                raise ValueError('FAIL_FROZEN_LAMBDA')
            return
        if not math.isfinite(value) or not 0.05 <= value <= 20:
            raise ValueError('FAIL_LOSS_SCALE')
        self.lambda_s1.fill_(value)
        self.lambda_frozen.fill_(True)

    def p2_loss(self, x, imagenette):
        input_gate(x)
        input_gate(imagenette)
        if not bool(self.lambda_frozen):
            raise ValueError('FAIL_UNFROZEN_LAMBDA')
        if not self.model.training:
            raise ValueError('FAIL_LOSS_MODE')
        penalty = {}; calls = 0
        def penalty_log(module, args, output):
            nonlocal calls
            calls += 1
            if calls == 2:
                with torch.no_grad():
                    penalty['value'] = float(output[:, :self.model.teacher_out_channels].square().mean())
        handle = self.model.student.register_forward_hook(penalty_log)
        try:
            with Capture(self, WANTED) as capture:
                # Includes original hard ST, exactly one ImageNette penalty,
                # original random augmentation, AE and STAE. Never rewritten.
                st, ae, stae = self.model(batch=x, batch_imagenet=imagenette)
            t, s = capture.values['teacher', S1], capture.values['student', S1]
            aux = hard_loss(discrepancy(t, s))
        finally:
            handle.remove()
        if calls != 3 or 'value' not in penalty:
            raise ValueError('FAIL_OFFICIAL_LOSS_API')
        official = st + ae + stae
        weighted = float(self.lambda_s1) * aux
        total = official + weighted
        values = [float(v.detach()) for v in (st, ae, stae, aux, official, weighted, total)]
        if not np.isfinite(values).all():
            raise ValueError('FAIL_NUMERICS')
        stv, aev, staev, auxv, offv, wv, tv = values
        return total, dict(official_st_with_penalty=stv, final_st=stv-penalty['value'],
                          ae=aev, stae=staev, penalty=penalty['value'], s1_raw=auxv,
                          official_total=offv, weighted_s1=wv, p2_total=tv,
                          aux_fraction=wv/tv if tv else 0., lambda_s1=float(self.lambda_s1),
                          student_forward_calls=calls)

    def training_step(self, batch, batch_idx, *args, **kwargs):
        try:
            imagenet_batch = next(self.imagenet_iterator)[0].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            imagenet_batch = next(self.imagenet_iterator)[0].to(self.device)
        loss, self.last_loss_log = self.p2_loss(batch.image, imagenet_batch)
        return {'loss': loss}

    @torch.no_grad()
    def fit_s1_maps(self, maps, labels, split):
        if split != 'validation' or len(maps) != 19 or len(labels) != 19 or any(int(y) != 0 for y in labels):
            raise ValueError('FAIL_VALIDATION_SOURCE')
        qa, qb = self._get_quantiles_of_maps(maps)
        if not bool(torch.isfinite(qa) and torch.isfinite(qb) and qb > qa):
            raise ValueError('FAIL_S1_QUANTILE_STATE')
        if bool(self.s1_quantiles_fitted) and (not torch.equal(self.qa_s1, qa) or not torch.equal(self.qb_s1, qb)):
            raise ValueError('FAIL_FROZEN_S1_QUANTILES')
        self.qa_s1.copy_(qa); self.qb_s1.copy_(qb)
        self.s1_quantiles_fitted.fill_(True)

    @torch.no_grad()
    def on_validation_start(self):
        # Final ST/AE quantile routine remains inherited and unchanged.
        super().on_validation_start()
        maps, labels = [], []
        for batch in self.trainer.datamodule.val_dataloader():
            if not bool((batch.gt_label == 0).all()):
                raise ValueError('FAIL_VALIDATION_SOURCE')
            x = batch.image.to(self.device)
            t, s, _, _ = s1_features(self, x)
            maps.append(s1_map(discrepancy(t, s)))
            labels.extend(batch.gt_label.tolist())
        self.fit_s1_maps(maps, labels, 'validation')

    @torch.no_grad()
    def candidate_outputs(self, x):
        if not bool(self.s1_quantiles_fitted) or not self.model.is_set(self.model.quantiles):
            raise ValueError('FAIL_QUANTILE_STATE')
        return inference_outputs(self, x, self.qa_s1, self.qb_s1)
