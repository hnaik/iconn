import logging
import math
import numpy as np
import torch

from datetime import datetime

from iconn import utils as ic_utils

ic_utils.init_logging()

logger = logging.getLogger(__name__)

n1 = 64
n2 = 21

tau = 0.3
alpha = 0.001
beta = 0.1


def l1_norm(x1, x2):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])


def l2_norm(x1, x2):
    return np.sqrt(math.pow(x1[0] - x2[0], 2) + math.pow(x1[1] - x2[1], 2))


def compute_dist(pos, center, n, func):
    dist = 1 - (beta * func(pos, center) / n)
    return tau * np.max(dist, -1)


def make_templates(n):
    template_count = n * n
    t_l1 = [torch.zeros([n, n]).to('cuda')] * template_count
    t_l2 = [torch.zeros([n, n]).to('cuda')] * template_count

    logger.info(f'making {template_count} of size {n}x{n}')

    # for t_idx in range(template_count):
    #     center = (t_idx // n, t_idx % n)
    #     for i in range(n):
    #         for j in range(n):
    #             t_l1[t_idx][i, j] = compute_dist((i, j), center, n, l1_norm)
    #             t_l2[t_idx][i, j] = compute_dist((i, j), center, n, l2_norm)

    t_neg_np = np.zeros([n, n])
    t_neg_np.fill(-tau)
    t_neg = torch.from_numpy(t_neg_np)

    logger.info('Finished making templates')

    return t_l1, t_l2, t_neg


_t_stage1_l1, _t_stage1_l2, _stage1_neg = make_templates(n1)
_t_stage2_l1, _t_stage2_l2, _stage2_neg = make_templates(n2)


def pick_template(stage, norm_type, idx):
    try:
        if norm_type == 'l1':
            if stage == 1:
                return _t_stage1_l1[idx]
            elif stage == 2:
                return _t_stage2_l1[idx]
            raise RuntimeError(f'stage not supported {stage}')
        elif norm_type == 'l2':
            if stage == 1:
                return _t_stage1_l2[idx]
            elif stage == 2:
                return _t_stage2_l2[idx]
            raise RuntimeError(f'stage not supported {stage}')

        raise RuntimeError(f'norm_type={norm_type} not supported')
    except IndexError as err:
        logger.error(
            f'{err} (stage={stage}, norm_type={norm_type}, '
            + f'idx={idx}, len_stage_1={len(_t_stage1_l1)}, '
            + f'len_stage_2={len(_t_stage2_l1)})'
        )


class FilterLossBase(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, stage, norm_type):
        data = X.clone()
        ctx.save_for_backward(X)
        ctx.template = torch.zeros(X.shape).to('cuda')
        for i, d_0 in enumerate(X):
            for j, d_1 in enumerate(d_0):
                max_idx = np.argmax(X[i][j].cpu().detach().numpy())
                template = pick_template(stage, norm_type, max_idx).to('cuda')
                data[i][j] *= template
                ctx.template[i][j] = template  # .to('cpu')
        return data

    @staticmethod
    def backward(ctx, grad_output):
        (X,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        n = X.shape[2]
        p_t = alpha / (n * n)

        masked = ctx.template * X

        p_xt = torch.zeros([X.shape[0], X.shape[1]]).to('cuda')
        trace = torch.zeros([X.shape[0], X.shape[1]]).to('cuda')
        p_x = torch.zeros([X.shape[0], X.shape[1]]).to('cuda')

        for i, d_0 in enumerate(X):
            for j, d_1 in enumerate(d_0):
                trace[i][j] = torch.trace(masked[i][j])
                p_xt[i][j] = torch.exp(trace[i][j])

        zt = p_xt.sum()
        p_x = (p_t * torch.mean(trace / zt)).sum()

        for i, d_0 in enumerate(X):
            for j, d_1 in enumerate(d_0):
                t = ctx.template[i][j]
                e_tr = p_xt[i][j]
                y = trace[i][j] - torch.log(zt) - p_x

                # Finally assembling equation 4 -- zhang et al.
                dl_x = p_t * e_tr * t * y / zt
                grad_input[i][j] += dl_x

        return grad_input


class Filter_Stage1_L1(FilterLossBase):
    @staticmethod
    def forward(ctx, X):
        return FilterLossBase.forward(ctx, X, stage=1, norm_type='l1')


class Filter_Stage1_L2(FilterLossBase):
    @staticmethod
    def forward(ctx, X):
        return FilterLossBase.forward(ctx, X, stage=1, norm_type='l2')


class Filter_Stage2_L1(FilterLossBase):
    @staticmethod
    def forward(ctx, X):
        return FilterLossBase.forward(ctx, X, stage=2, norm_type='l1')


class Filter_Stage2_L2(FilterLossBase):
    @staticmethod
    def forward(ctx, X):
        return FilterLossBase.forward(ctx, X, stage=2, norm_type='l2')


class IntermediateLoggerBase(FilterLossBase):
    @staticmethod
    def forward(ctx, X, stage, norm_type):
        stat_data = np.zeros([X.shape[0], X.shape[1]])
        for i, d_0 in enumerate(X):
            for j, d_1 in enumerate(d_0):
                stat_data[i, j] = np.argmax(X[i][j].cpu().detach().numpy())

        # ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        # filename = f'data_{stage}_{norm_type}_{ts}.csv'
        # np.savetxt(filename, stat_data, delimiter=',')

        return X


class IntermediateLogger_Stage1_L1(IntermediateLoggerBase):
    @staticmethod
    def forward(ctx, X):
        return IntermediateLoggerBase.forward(ctx, X, stage=1, norm_type='l1')


class IntermediateLogger_Stage1_L2(IntermediateLoggerBase):
    @staticmethod
    def forward(ctx, X):
        return IntermediateLoggerBase.forward(ctx, X, stage=1, norm_type='l2')


class IntermediateLogger_Stage1_Original(IntermediateLoggerBase):
    @staticmethod
    def forward(ctx, X):
        return IntermediateLoggerBase.forward(
            ctx, X, stage=1, norm_type='original'
        )


class IntermediateLogger_Stage2_L1(IntermediateLoggerBase):
    @staticmethod
    def forward(ctx, X):
        return IntermediateLoggerBase.forward(ctx, X, stage=2, norm_type='l1')


class IntermediateLogger_Stage2_L2(IntermediateLoggerBase):
    @staticmethod
    def forward(ctx, X):
        return IntermediateLoggerBase.forward(ctx, X, stage=2, norm_type='l2')


class IntermediateLogger_Stage2_Original(IntermediateLoggerBase):
    @staticmethod
    def forward(ctx, X):
        return IntermediateLoggerBase.forward(
            ctx, X, stage=2, norm_type='original'
        )
