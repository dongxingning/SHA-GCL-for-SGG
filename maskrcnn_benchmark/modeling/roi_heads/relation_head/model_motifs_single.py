# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_motifs import FrequencyBias, DecoderRNN

class LSTMContext_visual(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(LSTMContext_visual, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        # TODO Kaihua Tang
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.obj_ctx_rnn = torch.nn.LSTM(
                input_size=self.obj_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_obj,
                dropout=self.dropout_rate if self.nl_obj > 1 else 0,
                bidirectional=True)
        self.decoder_rnn = DecoderRNN(self.cfg, self.obj_classes, embed_dim=self.embed_dim,
                inputs_dim=self.hidden_dim + self.obj_dim,
                hidden_dim=self.hidden_dim,
                rnn_drop=self.dropout_rate)
        self.edge_ctx_rnn = torch.nn.LSTM(
                input_size=self.hidden_dim + self.obj_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_edge,
                dropout=self.dropout_rate if self.nl_edge > 1 else 0,
                bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        
        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_dcd_feat", torch.zeros(self.hidden_dim + self.obj_dim))
            self.register_buffer("untreated_obj_feat", torch.zeros(self.obj_dim))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.obj_dim))

    def sort_rois(self, proposals):
        c_x = center_x(proposals)
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(proposals, scores)

    def obj_ctx(self, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep) # map to hidden_dim

        # untreated decoder input
        batch_size = encoder_rep.shape[0]
        
        if (not self.training) and self.effect_analysis and ctx_average:
            decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(batch_size, -1)
        else:
            decoder_inp = torch.cat((obj_inp_rep, encoder_rep), 1)

        if self.training and self.effect_analysis:
            self.untreated_dcd_feat = self.moving_average(self.untreated_dcd_feat, decoder_inp)
        
        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(decoder_inp, ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp, #obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
                )
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed

    def edge_ctx(self, inp_feats, perm, inv_perm, ls_transposed):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps) # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, proposals, rel_pair_idxs, logger=None, all_average=False, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        batch_size = x.shape[0]
        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:
            obj_pre_rep = x

        boxes_per_cls = None
        if self.mode == 'sgdet' and not self.training:
            boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0) # comes from post process of box_head

        # object level contextual feature
        obj_dists, obj_preds, obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(obj_pre_rep, proposals, obj_labels, boxes_per_cls, ctx_average=ctx_average)

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_ctx), dim=-1)
        else:
            obj_rel_rep = cat((x, obj_ctx), -1)
            
        edge_ctx = self.edge_ctx(obj_rel_rep, perm=perm, inv_perm=inv_perm, ls_transposed=ls_transposed)

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, x)

        return obj_dists, obj_preds, edge_ctx, None


class LSTMContext_semantic(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """

    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(LSTMContext_semantic, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        # TODO Kaihua Tang
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.obj_ctx_rnn = torch.nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0,
            bidirectional=True)
        self.decoder_rnn = DecoderRNN(self.cfg, self.obj_classes, embed_dim=self.embed_dim,
                                      inputs_dim=self.hidden_dim + self.embed_dim,
                                      hidden_dim=self.hidden_dim,
                                      rnn_drop=self.dropout_rate)
        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size=self.hidden_dim + self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
            bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_dcd_feat", torch.zeros(self.hidden_dim + self.embed_dim))
            self.register_buffer("untreated_obj_feat", torch.zeros(self.embed_dim))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.embed_dim))

    def sort_rois(self, proposals):
        c_x = center_x(proposals)
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(proposals, scores)

    def obj_ctx(self, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep)  # map to hidden_dim

        # untreated decoder input
        batch_size = encoder_rep.shape[0]

        if (not self.training) and self.effect_analysis and ctx_average:
            decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(batch_size, -1)
        else:
            decoder_inp = torch.cat((obj_inp_rep, encoder_rep), 1)

        if self.training and self.effect_analysis:
            self.untreated_dcd_feat = self.moving_average(self.untreated_dcd_feat, decoder_inp)

        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(decoder_inp, ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp,  # obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
            )
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed

    def edge_ctx(self, inp_feats, perm, inv_perm, ls_transposed):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps)  # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, proposals, rel_pair_idxs, logger=None, all_average=False, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        batch_size = x.shape[0]
        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:
            obj_pre_rep = obj_embed

        boxes_per_cls = None
        if self.mode == 'sgdet' and not self.training:
            boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0)  # comes from post process of box_head

        # object level contextual feature
        obj_dists, obj_preds, obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(obj_pre_rep, proposals, obj_labels,
                                                            boxes_per_cls, ctx_average=ctx_average)
        # edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_preds.long())

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_ctx), dim=-1)
        else:
            obj_rel_rep = cat((obj_embed2, obj_ctx), -1)

        edge_ctx = self.edge_ctx(obj_rel_rep, perm=perm, inv_perm=inv_perm, ls_transposed=ls_transposed)

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, obj_embed2)

        return obj_dists, obj_preds, edge_ctx, None


class LSTMContext_spatial(nn.Module):
    """
    Modified from neural-motifs to encode contexts for each objects
    """

    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(LSTMContext_spatial, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # object & relation context
        self.obj_dim = in_channels
        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nl_obj = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER
        self.nl_edge = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER
        assert self.nl_obj > 0 and self.nl_edge > 0

        # TODO Kaihua Tang
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.obj_ctx_rnn = torch.nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0,
            bidirectional=True)
        self.decoder_rnn = DecoderRNN(self.cfg, self.obj_classes, embed_dim=self.embed_dim,
                                      inputs_dim=self.hidden_dim + 128,
                                      hidden_dim=self.hidden_dim,
                                      rnn_drop=self.dropout_rate)
        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size=self.hidden_dim+128,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
            bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS

        if self.effect_analysis:
            self.register_buffer("untreated_dcd_feat", torch.zeros(self.hidden_dim + 128))
            self.register_buffer("untreated_obj_feat", torch.zeros(128))
            self.register_buffer("untreated_edg_feat", torch.zeros(128))

    def sort_rois(self, proposals):
        c_x = center_x(proposals)
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(proposals, scores)

    def obj_ctx(self, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(proposals)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep)  # map to hidden_dim

        # untreated decoder input
        batch_size = encoder_rep.shape[0]

        if (not self.training) and self.effect_analysis and ctx_average:
            decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(batch_size, -1)
        else:
            decoder_inp = torch.cat((obj_inp_rep, encoder_rep), 1)

        if self.training and self.effect_analysis:
            self.untreated_dcd_feat = self.moving_average(self.untreated_dcd_feat, decoder_inp)

        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(decoder_inp, ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp,  # obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None,
                boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
            )
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed

    def edge_ctx(self, inp_feats, perm, inv_perm, ls_transposed):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps)  # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, proposals, rel_pair_idxs, logger=None, all_average=False, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))

        batch_size = x.shape[0]
        if all_average and self.effect_analysis and (not self.training):
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:
            obj_pre_rep = pos_embed

        boxes_per_cls = None
        if self.mode == 'sgdet' and not self.training:
            boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0)  # comes from post process of box_head

        # object level contextual feature
        obj_dists, obj_preds, obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(obj_pre_rep, proposals, obj_labels,
                                                                        boxes_per_cls, ctx_average=ctx_average)

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):
            obj_rel_rep = cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_ctx), dim=-1)
        else:
            obj_rel_rep = cat((pos_embed, obj_ctx), -1)

        edge_ctx = self.edge_ctx(obj_rel_rep, perm=perm, inv_perm=inv_perm, ls_transposed=ls_transposed)

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, pos_embed)

        return obj_dists, obj_preds, edge_ctx, None
