import numpy as np
from cython.anchors import anchors_cython
from cython.cpu_nms import cpu_nms

def nms_wrapper(thresh):
    def _nms_c(dets):
        return cpu_nms(dets, thresh)
    def _nms(dets):
        return nms(dets, thresh)
    if cpu_nms is not None:
        return _nms_c
    else:
        return _nms

def anchors_plane(feat_h, feat_w, stride, base_anchor):
    return anchors_cython(feat_h, feat_w, stride, base_anchor)

def generate_anchors_fpn(cfg = None):
    anchors = []
    for k in cfg:
        scales = np.array([1, 2])
        scales = scales[:,np.newaxis]
        centre = np.array([8, 8])
        centre = centre[:,np.newaxis]
        # base_size / max(cfg) * k^2 / scale
        size = (16 / 32) * k * k / (scales) - 1
        anchor = np.hstack((centre, centre, size, size))
        anchors.append(anchor.astype(np.float32))

    return anchors

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

class RetinaFace_Utils:
    def __init__(self):
        self.nms_threshold = 0.4
        self.nocrop = False
        self.debug = False
        self.fpn_keys = []
        self.anchor_cfg = None
        self.preprocess = False

        self._feat_stride_fpn = [32, 16, 8]
        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s'%s)

        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(cfg=self._feat_stride_fpn)))

        self._num_anchors = 2 #dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        self.nms = nms_wrapper(self.nms_threshold)
        self.use_landmarks = True

    def detect(self, img, output, threshold=0.5, im_scale=1.0):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        
        for idx, s in enumerate(self._feat_stride_fpn):
            key = 'stride%s'%s
            stride = int(s)
            
            if self.use_landmarks:
                idx = idx*3
            else:
                idx = idx*2

            scores = output[idx].cpu().detach().numpy()
            bbox_deltas = output[idx+1].cpu().detach().numpy()
            # Pull out confidence values per scale
            scores = scores[: , 2:, :, :]

            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            A = self._num_anchors
            K = height * width
            anchors_fpn = self._anchors_fpn['stride%s'%s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_deltas = bbox_deltas.reshape((-1, 4))

            proposals = self.bbox_pred(anchors, bbox_deltas)

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel>=threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]
            if stride==4 and self.decay4<1.0:
                scores *= self.decay4

            proposals[:,0:4] /= im_scale

            proposals_list.append(proposals)
            scores_list.append(scores)

            if self.use_landmarks:
                landmark_deltas = output[idx+2].cpu().detach().numpy()
                landmark_pred_len = landmark_deltas.shape[1]//A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
                landmarks = self.landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]
                landmarks[:,:,0:2] /= im_scale
                landmarks_list.append(landmarks)

        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0]==0:
            if self.use_landmarks:
                landmarks = np.zeros( (0,5,2) )
            return np.zeros( (0,5) ), landmarks
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]
        if self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
        keep = self.nms(pre_det)
        det = np.hstack( (pre_det, proposals[:,4:]) )
        det = det[keep, :]
        if self.use_landmarks:
            landmarks = landmarks[keep]
        return det, landmarks

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        pred_ctr_x = box_deltas[:, 0] * boxes[:, 2] + boxes[:, 0]
        pred_ctr_y = box_deltas[:, 1] * boxes[:, 3] + boxes[:, 1]
        pred_w = np.exp(box_deltas[:, 2]) * boxes[:, 2]
        pred_h = np.exp(box_deltas[:, 3]) * boxes[:, 3]

        pred_boxes = np.zeros(box_deltas.shape)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

        if box_deltas.shape[1]>4:
            pred_boxes[:,4:] = box_deltas[:,4:]

        return pred_boxes

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
            return np.zeros((0, landmark_deltas.shape[1]))

        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:,i,0] = landmark_deltas[:,i,0]*(boxes[:, 2]+1.0) + boxes[:, 0]
            pred[:,i,1] = landmark_deltas[:,i,1]*(boxes[:, 3]+1.0) + boxes[:, 1]
        return pred
