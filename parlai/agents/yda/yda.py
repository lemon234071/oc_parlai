#!/usr/bin/env python3

# Copyright (c) 2019-present, Shaojie Jiang.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# This programme is modified on top of the Seq2Seq implementation of Facebook Inc.,
# please visit http://parl.ai/ for more details.
#
# Should you have any problems using this programme, please contact Shaojie Jiang
# via shaojiejiang.1991@gmail.com

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.core.utils import NEAR_INF, padded_tensor, round_sigfigs, warn_once
from parlai.core.torch_agent import TorchAgent, Output

import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import math
import numpy as np
from collections import Counter

from .modules import YdaModel


class YdaAgent(TorchGeneratorAgent):

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Face Arguments')
        agent.add_argument('--init-model', type=str, default=None,
                           help='load dict/model/opts from this path')
        agent.add_argument('-yda', '--yda', default=True,
                           help='Yda training strategy.')
        agent.add_argument('-soft', '--numsoftmax', default=1, type=int,
                           help='default 1, if greater then uses mixture of '
                                'softmax (see arxiv.org/abs/1711.03953).')
        agent.add_argument('-idr', '--input-dropout', type=float, default=0.0,
                           help='Probability of replacing tokens with UNK in training.')
        agent.add_argument('-ft', '--frequency-type', default='out',
                           choices=['out', 'gt', 'none'],
                           help='What to use for calculating token frequency.')
        agent.add_argument('-wt', '--weighing-time', default='pre',
                           choices=['pre', 'post', 'none'],
                           help='When to apply weight to losses.')
        agent.add_argument('-cp', '--confidence-penalty', default='none',
                           choices=['cp', 'cpf', 'cpfw', 'cpfwn', 'none'],
                           help='Which kind of confidence penalty to use: '
                                "'cp' is the confidence-penalty function reported in https://arxiv.org/abs/1809.01941. "
                                "'cpf' is the parameter-free version proposed in https://arxiv.org/abs/1902.09191. "
                                "'cpfw' means using the parameter-free version as the weight of FACE. "
                                "'cpfwn' is a new design that normalizes the weight to the range of [1, +inf], which is "
                                "more favorable as the weight of FACE.")
        agent.add_argument('-b', '--beta', type=float, default=2.5,
                           help='Penalty strength for type "cp".')
        # transformer
        agent.add_argument('-esz', '--embedding-size', type=int, default=512,
                           help='Size of all embedding layers')
        agent.add_argument('-nl', '--n-layers', type=int, default=6)
        agent.add_argument('-hid', '--ffn-size', type=int, default=512,
                           help='Hidden size of the FFN layers')
        agent.add_argument('--attention-dropout', type=float, default=0.1)
        agent.add_argument('--relu-dropout', type=float, default=0.1)
        agent.add_argument('--n-heads', type=int, default=8,
                           help='Number of multihead attention heads')
        agent.add_argument('--learn-positional-embeddings', type='bool', default=False)
        agent.add_argument('--embeddings-scale', type='bool', default=True)
        agent.add_argument('--n-positions', type=int, default=512, hidden=True,
                           help='Number of positional embeddings to learn. Defaults '
                                'to truncate or 1024 if not provided.')

        super(cls, YdaAgent).add_cmdline_args(argparser)
        YdaAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        return 2

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'FACE'
        if getattr(self, 'word_freq', None) is None:
            self.word_freq = np.zeros(len(self.dict))
        self.ft = opt['frequency_type']
        self.wt = opt['weighing_time']
        self.cp = opt['confidence_penalty']
        self.beta = opt['beta']
        self.masked_entropy = HLoss(ignore_index=self.NULL_IDX)
        self.ideal_entropy = math.log(1 / len(self.dict))

        self.yda = opt["yda"]
        self.metrics['cloze_correct_tokens'] = 0
        self.metrics['cloze_loss'] = 0

    def build_model(self, states=None):
        self.model = YdaModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model.cuda()
        return self.model

    def _init_cuda_buffer(self, batchsize, maxlen, force=False):
        """Pre-initialize CUDA buffer by doing fake forward pass."""
        if self.use_cuda and (force or not hasattr(self, 'buffer_initialized')):
            try:
                dummy_xs = torch.ones(batchsize, maxlen).long().cuda()
                dummy_ys = torch.ones(batchsize, 3).long().cuda()
                scores, _, _, label_scores = self.model(dummy_xs, dummy_ys)
                if self.yda:
                    label_scores = label_scores.view(-1, label_scores.size(-1))
                    label_loss = F.cross_entropy(label_scores, dummy_ys.view(-1),
                                                 ignore_index=self.criterion.ignore_index, reduction="sum")
                    loss = self.criterion(scores.view(-1, scores.size(-1)), dummy_ys.view(-1),
                                          label_scores, dummy_ys.size())
                    loss += label_loss
                else:
                    loss = self.criterion(
                        scores.view(-1, scores.size(-1)), dummy_ys.view(-1)
                    )
                loss.backward()
                self.buffer_initialized = True
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    m = ('CUDA OOM: Lower batch size (-bs) from {} or lower '
                         ' max sequence length (-tr) from {}'
                         ''.format(batchsize, maxlen))
                    raise RuntimeError(m)
                else:
                    raise e

    def train_step(self, batch):
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
        # helps with memory usage
        self._init_cuda_buffer(batchsize, self.truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            scores, preds, _, label_scores = self.model(batch.text_vec, batch.label_vec)
            score_view = scores.view(-1, scores.size(-1))
            preds_clean = self.clean_preds(preds)
            # Update token frequency, or not
            if self.ft == 'gt':
                self.update_frequency(self.clean_preds(batch.label_vec))
            elif self.ft == 'out':
                self.update_frequency(preds_clean)
            # calculate loss w/ or w/o pre-/post-weight
            if self.yda:
                label_scores_view = label_scores.view(-1, label_scores.size(-1))
                label_loss = F.cross_entropy(label_scores_view, batch.label_vec.view(-1),
                                             ignore_index=self.criterion.ignore_index, reduction="sum")
                loss = self.criterion(score_view, batch.label_vec.view(-1), label_scores_view, batch.label_vec.size())
            elif self.wt == 'pre':
                self.criterion.weight = self.loss_weight()
                loss = self.criterion(score_view, batch.label_vec.view(-1))
            elif self.wt == 'post':
                self.criterion.reduction = 'none'
                loss = self.criterion(score_view, batch.label_vec.view(-1))
                device = loss.device
                freq_pred = self.word_freq[preds.view(-1).cpu().numpy()]
                freq_pred = torch.FloatTensor(freq_pred).to(device)
                freq_GT = self.word_freq[batch.label_vec.view(-1).cpu().numpy()]
                freq_GT = torch.FloatTensor(freq_GT).to(device)
                total_freq = self.word_freq.sum()
                weight = 1 + F.relu(freq_pred - freq_GT) / total_freq
                loss = torch.matmul(loss, weight)
            else:
                loss = self.criterion(score_view, batch.label_vec.view(-1))

            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            # Use confidence penalty or not
            if self.cp != 'none':
                entropy = self.masked_entropy(score_view, batch.label_vec.view(-1))
                mean_entropy = entropy / target_tokens
                if self.cp == 'cp':
                    loss -= self.beta * mean_entropy
                elif self.cp == 'cpf':
                    loss += 1 / mean_entropy
                elif self.cp == 'cpfw':
                    # TODO: normalize weight to [1, ++]?
                    loss *= (1 + 1 / mean_entropy)
                elif self.cp == 'cpfwn':
                    loss *= (self.ideal_entropy / mean_entropy)
            # save loss to metrics
            correct = ((batch.label_vec == preds) * notnull).sum().item()
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens
            self.metrics['preds'].extend(preds_clean)
            if self.yda:
                cloze_preds = label_scores.max(-1)[1]
                cloze_correct = ((batch.label_vec == cloze_preds) * notnull).sum().item()
                self.metrics['cloze_correct_tokens'] += cloze_correct
                self.metrics['cloze_loss'] += label_loss.item()
                loss += label_loss
            loss = loss / target_tokens
            loss.backward()
            self.update_params()
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
            else:
                raise e

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        bsz = batch.text_vec.size(0)
        self.model.eval()
        cand_scores = None

        if self.skip_generation:
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning
            )
            scores, preds, _ = self.model(batch.text_vec, batch.label_vec)
        elif self.beam_size == 1:
            # greedy decode
            scores, preds, _, _ = self.model(batch.text_vec)
        elif self.beam_size > 1:
            out = self.beam_search(
                self.model,
                batch,
                self.beam_size,
                start=self.START_IDX,
                end=self.END_IDX,
                pad=self.NULL_IDX,
                min_length=self.beam_min_length,
                min_n_best=self.beam_min_n_best,
                block_ngram=self.beam_block_ngram
            )
            beam_preds_scores, _, beams = out
            preds, scores = zip(*beam_preds_scores)

            if self.beam_dot_log is True:
                self._write_beam_dots(batch.text_vec, beams)

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            f_scores, f_preds, _, f_label_scores = self.model(batch.text_vec, batch.label_vec)
            score_view = f_scores.view(-1, f_scores.size(-1))
            # save loss to metrics
            notnull = batch.label_vec.ne(self.NULL_IDX)
            target_tokens = notnull.long().sum().item()
            correct = ((batch.label_vec == f_preds) * notnull).sum().item()
            self.criterion.reduction = 'sum'
            if self.yda:
                cloze_preds = f_label_scores.max(-1)[1]
                cloze_correct = ((batch.label_vec == cloze_preds) * notnull).sum().item()
                f_label_scores = f_label_scores.view(-1, f_label_scores.size(-1))
                label_loss = F.cross_entropy(f_label_scores, batch.label_vec.view(-1),
                                             ignore_index=self.criterion.ignore_index, reduction="sum")
                loss = self.criterion(score_view, batch.label_vec.view(-1),
                                      f_label_scores, batch.label_vec.size())
                self.metrics['cloze_correct_tokens'] += cloze_correct
                self.metrics['cloze_loss'] += label_loss.item()
            else:
                loss = self.criterion(score_view, batch.label_vec.view(-1))
            self.metrics['correct_tokens'] += correct
            self.metrics['loss'] += loss.item()
            self.metrics['num_tokens'] += target_tokens

        cand_choices = None
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(batch.text_vec)
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = padded_tensor(
                    batch.candidate_vecs[i], self.NULL_IDX, self.use_cuda
                )
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        text = [self._v2t(p) for p in preds]
        self.metrics['preds'].extend(self.clean_preds(preds))
        return Output(text, cand_choices)

    def build_criterion(self):
        # set up criteria
        if self.opt.get("yda", None):
            self.criterion = YdaLoss(ignore_index=self.NULL_IDX, reduction="sum",
                                     sample_weight=self.opt.get("yda_weight", False))
        elif self.opt.get('numsoftmax', 1) > 1:
            self.criterion = nn.NLLLoss(
                ignore_index=self.NULL_IDX, reduction="sum")
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, reduction="sum")

        if self.use_cuda:
            self.criterion.cuda()

    def _v2t(self, vec, end_early=True):
        """Convert token indices to string of tokens."""
        new_vec = []
        if hasattr(vec, 'cpu'):
            vec = vec.cpu()
        for i in vec:
            if i == self.END_IDX and end_early:
                break
            if i == self.NULL_IDX:
                continue
            new_vec.append(i)
        return self.dict.vec2txt(new_vec)

    def _vectorize_text(self, text, add_start=False, add_end=False,
                        truncate=None, truncate_left=False):
        return super()._vectorize_text(text, add_start=add_start, add_end=add_end,
                                       truncate=truncate, truncate_left=False)

    def vectorize(self, *args, **kwargs):
        """Override vectorize for seq2seq."""
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        return super().vectorize(*args, **kwargs)

    def batchify(self, *args, **kwargs):
        """Override batchify options for seq2seq."""
        kwargs['sort'] = True  # need sorted for pack_padded
        return super().batchify(*args, **kwargs)

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            if hasattr(self.model, 'module'):
                model['model'] = self.model.module.state_dict()
                model['longest_label'] = self.model.module.longest_label
            else:
                model['model'] = self.model.state_dict()
                model['longest_label'] = self.model.longest_label
            model['optimizer'] = self.optimizer.state_dict()
            model['optimizer_type'] = self.opt['optimizer']
            model['word_freq'] = self.word_freq

            with open(path, 'wb') as write:
                torch.save(model, write)

            # save opt file
            with open(path + '.opt', 'w') as handle:
                # save version string
                self.opt['model_version'] = self.model_version()
                json.dump(self.opt, handle)

    def load(self, path):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        if 'word_freq' in states:
            self.word_freq = states['word_freq']
        # set loaded states if applicable
        self.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.model.longest_label = states['longest_label']
        return states

    def reset_metrics(self):
        """Reset metrics for reporting loss and perplexity."""
        super().reset_metrics()
        self.metrics['loss'] = 0.0
        self.metrics['num_tokens'] = 0
        self.metrics['correct_tokens'] = 0
        self.metrics['preds'] = []
        self.metrics['cloze_correct_tokens'] = 0
        self.metrics['cloze_loss'] = 0

    def clean_preds(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            if self.END_IDX in pred:
                ind = pred.index(self.END_IDX) + 1  # end_idx included
                pred = pred[:ind]
            if len(pred) == 0:
                continue
            if pred[0] == self.START_IDX:
                pred = pred[1:]
            res.append(pred)
        return res

    def calc_diversity(self, metrics):
        unigram = set()
        bigram = set()
        num_tok = 0
        for vec in self.metrics['preds']:
            v_len = len(vec)
            num_tok += v_len
            unigram.update(vec)
            bigram.update([tuple(vec[i:i + 2]) for i in range(v_len - 1)])
        metrics['d_1'] = round(len(unigram) / num_tok * 100, 2)
        metrics['d_2'] = round(len(bigram) / num_tok * 100, 2)
        if not self.model.training:
            metrics['num_d1'] = len(unigram)
            metrics['num_d2'] = len(bigram)
            metrics['num_tok'] = num_tok

    def report(self):
        """Report loss and perplexity from model's perspective.

        Note that this includes predicting __END__ and __UNK__ tokens and may
        differ from a truly independent measurement.
        """
        m = {}
        num_tok = self.metrics['num_tokens']
        if num_tok > 0:
            if self.metrics['correct_tokens'] > 0:
                m['token_acc'] = self.metrics['correct_tokens'] / num_tok
            if self.metrics['cloze_correct_tokens'] > 0:
                m['cloze_token_acc'] = self.metrics['cloze_correct_tokens'] / num_tok
            if self.yda:
                m['cloze_loss'] = self.metrics['cloze_loss'] / num_tok
                try:
                    m['cloze_ppl'] = math.exp(m['cloze_loss'])
                except OverflowError:
                    m['cloze_ppl'] = float('inf')
            m['loss'] = self.metrics['loss'] / num_tok
            try:
                m['ppl'] = math.exp(m['loss'])
            except OverflowError:
                m['ppl'] = float('inf')
        if self.metrics['total_skipped_batches'] > 0:
            m['total_skipped_batches'] = self.metrics['total_skipped_batches']
        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        if self.metrics['preds']:
            self.calc_diversity(m)
        m["lr"] = self.optimizer.state_dict()['param_groups'][0]['lr']
        return m

    def update_frequency(self, preds):
        curr = Counter()
        for pred in preds:
            curr.update(pred)

        # self.word_freq *= self.opt['decay_factor']
        for k, v in curr.items():
            if k == self.END_IDX:  # do not suppress END token
                continue
            self.word_freq[k] += v

    def loss_weight(self):
        RF = self.word_freq / self.word_freq.sum()  # relative frequency
        a = -1 / RF.max()
        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)  # normalization
        if self.use_cuda:
            return torch.FloatTensor(weight).cuda()
        else:
            return torch.FloatTensor(weight)


class HLoss(nn.Module):
    """
    Entropy loss used for entropy maximization.
    """

    def __init__(self, ignore_index=-1):
        super(HLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, x, labels):
        mask = (labels != self.ignore_index).float()
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * torch.matmul(mask, b.sum(dim=1))
        return b


class YdaLoss(nn.Module):
    """
    With yda label,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, ignore_index=-100, reduction='sum', sample_weight=False, eos_index=3,
                 yida_smoothing="diag"):
        self.ignore_index = ignore_index
        self.eos_index = eos_index
        super(YdaLoss, self).__init__()
        self.yida_smoothing = yida_smoothing
        self.reduction = reduction
        self.sample_weight = sample_weight
        self.step = 0

    def forward(self, output, target, label_scores, batch_shape):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        if self.yida_smoothing == "diag" or self.yida_smoothing == "BERT":
            v = self._v_diag(label_scores, target)
        elif self.yida_smoothing == "distill":
            v = self._v_distill(label_scores)
        else:
            raise Exception

        # epsilon = self._yida_vmax_epsilon(output, target, v)
        epsilon = self._yida_halfmax_epsilon(output, target)

        self.sample_weight = True
        if self.sample_weight and not isinstance(label_scores, list):
            self.step += 1
            if self.step > 0:
                # weights = self._B_weight(label_scores, target, tgt_batch)
                # weights = self._C_weight(temp_for_c.view(tgt_batch.size(0), tgt_batch.size(1)),
                #                          output.detach().clone().view(tgt_batch.size(0), tgt_batch.size(1), -1),
                #                          tgt_batch)
                # weights = self._CE_weight(output.detach().clone().view(tgt_batch.size(0), tgt_batch.size(1), -1),
                #                          tgt_batch)
                # weights = self._CE_weight(
                #     label_scores.detach().clone().log_softmax(-1).view(tgt_batch.size(0), tgt_batch.size(1), -1),
                #     tgt_batch)
                weights = self._A_weight(
                    output,
                    target,
                    batch_shape)
                epsilon = epsilon.view(batch_shape[0], batch_shape[1])
                epsilon = weights.unsqueeze(0) * epsilon
                epsilon = epsilon.view(-1)

        confidence = 1 - epsilon
        smoothing_penalty = epsilon.unsqueeze(-1) * v

        # model_prob = torch.zeros_like(epsilon, device=epsilon.device, dtype=torch.float)
        # model_prob = model_prob.unsqueeze(1).repeat(1, self.tgt_vocab_size)
        model_prob = torch.zeros_like(output, device=output.device, dtype=torch.float)
        if False:
            eos_mask = target.eq(self.eos_index)
            confidence[eos_mask] = 1
            smoothing_penalty.masked_fill_((target == self.eos_index).unsqueeze(1), 0)
        model_prob.scatter_(1, target.unsqueeze(1), confidence.unsqueeze(1))
        model_prob += smoothing_penalty
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        return F.kl_div(output.log_softmax(dim=-1), model_prob, reduction=self.reduction)

    def _yida_halfmax_epsilon(self, output, target):
        probs = output.detach().clone().softmax(dim=-1)
        prob_max = probs.max(dim=1)[0]
        prob_gtruth = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze()
        epsilon = 1 - prob_max
        mask = epsilon.gt(0.5)
        epsilon[mask] = 0.5
        epsilon = prob_gtruth / prob_max * epsilon
        return epsilon

    def _yida_vmax_epsilon(self, output, target, v):
        probs = output.detach().clone().softmax(dim=-1)
        prob_max = probs.max(dim=1)[0]
        prob_gtruth = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze()
        epsilon = 1 - prob_max
        maxv = v.max(dim=-1)[0]
        up_bond = 1 / (1 + maxv)
        mask = epsilon.gt(up_bond)
        epsilon[mask] = up_bond[mask]
        epsilon = prob_gtruth / prob_max * epsilon
        return epsilon

    def _v_diag(self, label_scores, target):
        # TODO(yida) grad thinking
        v = label_scores.detach().clone()
        v /= 1.5
        # temp_for_c = v.clone().softmax(dim=-1)[gt_mask]
        v.scatter_(1, target.unsqueeze(1), -float('inf'))
        # v[gt_mask] = -float('inf')
        v[:, self.ignore_index] = -float('inf')
        v = v.softmax(dim=-1)
        return v

    def _compute_entropy(self, output):
        entropy = -torch.sum(output.exp() * output, -1)
        return entropy

    def _B_weight(self, label_score, target, tgt_batch):
        log_prob = label_score.detach().clone().log_softmax(-1)
        entropy = self._compute_entropy(log_prob)
        non_special = (target != self.ignore_index)  # & (target != self.eos_index)
        non_special = non_special.view(tgt_batch.size(0), tgt_batch.size(1))
        entropy = entropy.view(tgt_batch.size(0), tgt_batch.size(1))
        entropy[~non_special] = 0
        entropy = entropy.sum(0) / non_special.float().sum(0)
        weight = entropy.softmax(-1)
        # weight = entropy.sigmoid()
        # weight = entropy /entropy.sum()
        # weight = weight * tgt_batch.size(1)
        weight = 1 - weight
        return weight

    def _C_weight(self, label_score, output, tgt_batch):
        probs = output.exp()
        prob_max = probs.max(dim=-1)[0]
        prob_max[prob_max.lt(0.5)] = 0.5
        ignore_mask = tgt_batch.eq(self.ignore_index)
        diff = (label_score - prob_max)
        diff[ignore_mask] = 0
        diff = diff.sum(0) / (~ignore_mask).sum(0)
        weight = 1 - diff
        return weight

    def _CE_weight(self, output, tgt_batch):
        probs = output.exp()
        non_pad = tgt_batch.ne(self.ignore_index)
        pred = probs.max(dim=-1)[1]
        correct = pred.eq(tgt_batch)
        correct[~non_pad] = False
        correct = correct.sum(0).float()
        acc = correct / non_pad.sum(0).float()
        prob_gtruth = probs.gather(dim=-1, index=tgt_batch.unsqueeze(-1)).squeeze()
        prob_gtruth[~non_pad] = 0
        prob_mean = prob_gtruth.sum(0) / non_pad.sum(0)
        weight = 1 + (acc - prob_mean)
        return weight

    def _A_weight(self, output, target, batch_shape):
        probs = output.detach().clone().view(batch_shape[0], batch_shape[1], -1).softmax(dim=-1)
        non_pad = target.view(batch_shape[0], batch_shape[1]).ne(self.ignore_index)
        pred = probs.max(dim=-1)[1]
        correct = pred.eq(target.view(batch_shape[0], batch_shape[1]))
        correct[~non_pad] = False
        correct = correct.float().sum(0)
        acc = correct / non_pad.float().sum(0)
        weight = acc
        return weight
