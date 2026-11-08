# -*- coding: utf-8 -*-
"""消融实验与对比实验，统一训练和评估协议。
"""
import os
import sys
import json
import time
import argparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

from UNet_model import get_model
from SegNet import get_segnet
from DeepLabv3plus import get_deeplabv3plus
from ResUNet import get_resunet
from data_process import DEFAULT_BANDS, NUM_CLASSES, load_and_split

# 6 个原始波段 + 6 个光谱指数 + 标签
BANDS = DEFAULT_BANDS
CLASS_NUM = NUM_CLASSES
LABELS = np.arange(CLASS_NUM)
IN_CH = len(BANDS) - 1
SHORT = ['Bamboo', 'Non-bamboo', 'Shrub', 'Grass', 'Crop', 'Other']
NAMES = ['Bamboo forest', 'Non-bamboo forest', 'Shrubland', 'Grassland',
         'Cropland', 'Other land cover']

# 数据与输出目录：默认取仓库内相对路径，可用环境变量覆盖
DATA_DIR = os.environ.get('SSE_DATA_DIR', os.path.join(ROOT, 'data'))
PATCH_NPY = os.environ.get('SSE_PATCH_NPY', os.path.join(DATA_DIR, 'patches.npy'))
OUT_ROOT = os.environ.get('SSE_OUT_DIR', os.path.join(ROOT, 'results'))
MODEL_ROOT = os.environ.get('SSE_MODEL_DIR', os.path.join(ROOT, 'checkpoints'))

# ---------------- 模型定义 ----------------
# 四个 U-Net 系配置共享同一通道设置，只切换 DenseBlock / CBAM 开关
CH = [32, 64, 128, 256, 512, 512]
SPECS = {
    # 消融组
    'UNet':          dict(group='ablation', kind='unet', ifdense=[0, 0], ifcbam=[0, 0],
                          desc='U-Net baseline（无 DenseBlock、无 CBAM）'),
    'UNet_Dense':    dict(group='ablation', kind='unet', ifdense=[1, 1], ifcbam=[0, 0],
                          desc='U-Net + DenseBlock'),
    'UNet_CBAM':     dict(group='ablation', kind='unet', ifdense=[0, 0], ifcbam=[1, 1],
                          desc='U-Net + CBAM'),
    'SSENet':        dict(group='ablation', kind='unet', ifdense=[1, 1], ifcbam=[1, 1],
                          desc='SSE-Net 完整模型（DenseBlock + CBAM）'),
    # CBAM 位置消融（DenseBlock 固定为 encoder+decoder）
    'SSE_CBAM_en':   dict(group='ablation', kind='unet', ifdense=[1, 1], ifcbam=[1, 0],
                          desc='DenseBlock(both) + CBAM(encoder only)'),
    'SSE_CBAM_de':   dict(group='ablation', kind='unet', ifdense=[1, 1], ifcbam=[0, 1],
                          desc='DenseBlock(both) + CBAM(decoder only)'),
    # DenseBlock 位置消融（CBAM 固定为 encoder+decoder）
    'SSE_Dense_en':  dict(group='ablation', kind='unet', ifdense=[1, 0], ifcbam=[1, 1],
                          desc='DenseBlock(encoder only) + CBAM(both)'),
    'SSE_Dense_de':  dict(group='ablation', kind='unet', ifdense=[0, 1], ifcbam=[1, 1],
                          desc='DenseBlock(decoder only) + CBAM(both)'),
    # 对比组
    'SegNet':        dict(group='compare', kind='segnet',
                          desc='SegNet（VGG16 编码 + 最大池化反池化）'),
    'DeepLabv3p':    dict(group='compare', kind='deeplab',
                          desc='DeepLabv3+（ResNet50 编码 + ASPP）'),
    'ResUNet':       dict(group='compare', kind='resunet',
                          desc='ResU-Net（U-Net + 残差块，保留对称跳跃连接）'),
}


def build_model(name):
    s = SPECS[name]
    if s['kind'] == 'unet':
        m = get_model(BANDS, CLASS_NUM, 1, s['ifdense'], 3, s.get('ch', CH), s['ifcbam'])
    elif s['kind'] == 'segnet':
        m = get_segnet(IN_CH, CLASS_NUM)
    elif s['kind'] == 'deeplab':
        m = get_deeplabv3plus(IN_CH, CLASS_NUM)
    elif s['kind'] == 'resunet':
        m = get_resunet(IN_CH, CLASS_NUM)
    else:
        raise ValueError(name)
    return m


class WeightedCE(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('w', torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, prob, target):
        B, C, H, W = prob.shape
        t = target.reshape(B, H, W)
        valid = (t >= 0) & (t < C)
        if valid.sum() == 0:
            return prob.sum() * 0.0
        p = prob.permute(0, 2, 3, 1)[valid]
        y = t[valid]
        logp = torch.log(p.clamp(min=1e-7))
        w = self.w[y]
        ce = -logp.gather(1, y[:, None]).squeeze(1)
        return (w * ce).sum() / w.sum()


def stat(cm):
    tp = np.diag(cm).astype(float)
    rs = cm.sum(1).astype(float)
    cs = cm.sum(0).astype(float)
    T = cm.sum()
    rec = np.divide(tp, rs, out=np.zeros(CLASS_NUM), where=rs > 0)
    pre = np.divide(tp, cs, out=np.zeros(CLASS_NUM), where=cs > 0)
    f1 = np.divide(2 * pre * rec, pre + rec, out=np.zeros(CLASS_NUM), where=(pre + rec) > 0)
    oa = tp.sum() / T if T else 0.0
    pe = ((rs / T) * (cs / T)).sum() if T else 0.0
    kap = (oa - pe) / (1 - pe) if T and pe < 1 else 0.0
    return dict(OA=float(oa), Kappa=float(kap), mAcc=float(rec.mean()),
                F1_macro=float(f1.mean()), recall=rec, precision=pre, f1=f1,
                counts=rs.astype(int), colsums=cs.astype(int), n=int(T))


def load_data(val_size, test_size, seed):
    return load_and_split(PATCH_NPY, BANDS, val_size=val_size, test_size=test_size,
                          seed=seed,
                          part_glob=os.path.join(DATA_DIR, 'patches_*.npy'))


@torch.no_grad()
def predict(model, X, device, bs=16):
    model.eval()
    outs = []
    for i in range(0, X.shape[0], bs):
        bx = torch.from_numpy(X[i:i + bs]).permute(0, 3, 1, 2).to(device)
        outs.append(np.argmax(model(bx).cpu().numpy(), axis=1))
    return np.concatenate(outs, axis=0)


def quick_eval(model, Xev, Yev, device):
    yp = predict(model, Xev, device)
    v = Yev >= 0
    cm = confusion_matrix(Yev[v].ravel(), yp[v].ravel(), labels=LABELS)
    return stat(cm)


def evaluate_and_export(name, model, Xev, Yev, out_dir, extra):
    os.makedirs(out_dir, exist_ok=True)
    yp = predict(model, Xev, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    v = Yev >= 0
    cm = confusion_matrix(Yev[v].ravel(), yp[v].ravel(), labels=LABELS)
    s = stat(cm)
    rs, cs, T = cm.sum(1), cm.sum(0), int(cm.sum())

    np.savetxt(os.path.join(out_dir, 'confusion_full.csv'), cm, delimiter=',', fmt='%d')
    with open(os.path.join(out_dir, 'confusion_full_with_margins.csv'), 'w', encoding='utf-8') as f:
        f.write('True\\Pred,' + ','.join(SHORT) + ',RowTotal\n')
        for i in range(CLASS_NUM):
            f.write(SHORT[i] + ',' + ','.join(str(int(x)) for x in cm[i]) + ',%d\n' % rs[i])
        f.write('ColTotal,' + ','.join(str(int(x)) for x in cs) + ',%d\n' % T)

    met = dict(model=name, metrics='pixel-level metrics on the test set',
               epoch=extra.get('epoch'), desc=SPECS[name]['desc'],
               params_M=round(extra['params_M'], 3),
               n_patch=int(Xev.shape[0]), n_pixels=int(T),
               OA=s['OA'], Kappa=s['Kappa'], mAcc=s['mAcc'], F1_macro=s['F1_macro'],
               recall=[float(x) for x in s['recall']],
               precision=[float(x) for x in s['precision']],
               f1=[float(x) for x in s['f1']],
               counts=[int(x) for x in s['counts']],
               colsums=[int(x) for x in s['colsums']],
               train_seconds=extra.get('train_seconds'),
               CM=cm.tolist())
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(met, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 92)
    tag = 'ep%d' % extra['epoch'] if extra.get('epoch') else 'eval'
    print('[%s] %s | %s | %.3f M | 有效像素 %d'
          % (name, SPECS[name]['desc'], tag, extra['params_M'], T))
    print('OA %.2f%%   Kappa %.4f   mAcc %.2f%%   macroF1 %.2f%%'
          % (s['OA'] * 100, s['Kappa'], s['mAcc'] * 100, s['F1_macro'] * 100))
    print('%-12s %9s %11s %9s %11s' % ('Class', 'Recall', 'Precision', 'F1', 'Pixels'))
    for i in range(CLASS_NUM):
        print('%-12s %8.2f%% %10.2f%% %8.2f%% %11d'
              % (NAMES[i], s['recall'][i] * 100, s['precision'][i] * 100,
                 s['f1'][i] * 100, rs[i]))
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--weights', type=str, default='1,1,1,1,1,1')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val_ratio', type=float, default=0.3, help='验证集比例')
    ap.add_argument('--test_ratio', type=float, default=0.3, help='测试集比例')
    ap.add_argument('--t_max', type=int, default=200, help='余弦退火周期（默认 200）')
    ap.add_argument('--save_each', type=int, default=10)
    ap.add_argument('--eval_only', action='store_true', help='不训练，直接评估已有权重')
    ap.add_argument('--ckpt', type=str, default='', help='eval_only 时指定权重路径')
    ap.add_argument('--tag', type=str, default='', help='输出目录后缀')
    ap.add_argument('--eval_each', type=int, default=10,
                    help='每 N 轮在验证集上评估一次，按 --select 指标选档并保存 best.pth')
    ap.add_argument('--select', type=str, default='oa', choices=['oa', 'macc', 'f1'],
                    help='按验证集的哪个指标选档')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.list:
        for k, v in SPECS.items():
            print('%-12s %-9s %s' % (k, v['group'], v['desc']))
        return
    if args.model not in SPECS:
        raise SystemExit('未知模型 %s，可用: %s' % (args.model, list(SPECS)))

    name = args.model
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = os.path.join(OUT_ROOT, name + ('_' + args.tag if args.tag else ''))
    model_dir = os.path.join(MODEL_ROOT, name + ('_' + args.tag if args.tag else ''))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model = build_model(name).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print('=' * 92)
    print('%s | %s' % (name, SPECS[name]['desc']))
    print('device %s | 参数量 %.3f M | epochs %d | bs %d | lr %g | weights %s | T_max %d'
          % (device, n_par, args.epochs, args.batch_size, args.lr, args.weights, args.t_max))

    Xtr, Ytr, Xva, Yva, Xev, Yev = load_data(args.val_ratio, args.test_ratio, args.seed)
    v = Yev[Yev >= 0]
    print('训练 %d patch | 验证 %d patch | 评估 %d patch | 评估有效像素 %d | 各类 %s'
          % (Xtr.shape[0], Xva.shape[0], Xev.shape[0], v.size,
             np.bincount(v, minlength=6).tolist()))
    if args.eval_only:
        if not args.ckpt:
            raise SystemExit('--eval_only 需要用 --ckpt 指定权重文件')
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print('已加载权重:', args.ckpt)
        evaluate_and_export(name, model, Xev, Yev, out_dir,
                            dict(epoch=None, params_M=n_par, train_seconds=None))
        return

    cls_w = [float(x) for x in args.weights.split(',')]
    x_tr = torch.from_numpy(Xtr).permute(0, 3, 1, 2).contiguous()
    y_tr = torch.from_numpy(Ytr)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    criterion = WeightedCE(cls_w).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=1e-6)

    hist = {'loss': [], 'acc': []}
    curve = []
    best_va, best_ep = -1.0, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tl, ta, nb = 0.0, 0.0, 0
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tl += loss.item()
            with torch.no_grad():
                pred = torch.argmax(out, dim=1)
                m = (by >= 0) & (by < CLASS_NUM)
                if m.sum() > 0:
                    ta += (pred[m] == by[m]).float().mean().item()
            nb += 1
        tl /= nb
        ta /= nb
        scheduler.step()
        hist['loss'].append(tl)
        hist['acc'].append(ta)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print('Ep %3d/%d | Loss %.4f | TrainAcc %.4f | lr %.1e | %.0fs'
                  % (epoch + 1, args.epochs, tl, ta, optimizer.param_groups[0]['lr'],
                     time.time() - t0), flush=True)
        if args.save_each and (epoch + 1) % args.save_each == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, 'ep%03d.pth' % (epoch + 1)))
        # 只在验证集上选档
        if args.eval_each and (epoch + 1) % args.eval_each == 0:
            s_va = quick_eval(model, Xva, Yva, device)
            curve.append([epoch + 1, s_va['OA'], s_va['Kappa'], s_va['mAcc'],
                          s_va['F1_macro'], s_va['f1'][0]])
            cur = {'oa': s_va['OA'], 'macc': s_va['mAcc'], 'f1': s_va['F1_macro']}[args.select]
            star = ''
            if cur > best_va:
                best_va, best_ep = cur, epoch + 1
                torch.save(model.state_dict(), os.path.join(model_dir, 'best.pth'))
                star = '  <- best'
            print('   [ep%3d] val OA %.2f K %.4f mAcc %.2f F1 %.2f BambooF1 %.2f%s'
                  % (epoch + 1, s_va['OA'] * 100, s_va['Kappa'], s_va['mAcc'] * 100,
                     s_va['F1_macro'] * 100, s_va['f1'][0] * 100, star), flush=True)
            model.train()

    if curve:
        np.savetxt(os.path.join(out_dir, 'curve.csv'), np.array(curve), delimiter=',',
                   header='epoch,val_OA,val_Kappa,val_mAcc,val_F1,val_F1_Bamboo', comments='')

    train_sec = time.time() - t0
    torch.save(model.state_dict(), os.path.join(model_dir, 'final.pth'))
    np.savetxt(os.path.join(out_dir, 'history.csv'),
               np.array([hist['loss'], hist['acc']]), delimiter=',')

    if best_ep:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'best.pth'),
                                         map_location=device))
        print('\n按验证集 %s 选择最佳档位: ep%d（val %s %.4f），用于报告 test'
              % (args.select, best_ep, args.select, best_va))
    met = evaluate_and_export(name, model, Xev, Yev, out_dir,
                              dict(epoch=best_ep or args.epochs, params_M=n_par,
                                   train_seconds=round(train_sec, 1)))
    met['selected_by'] = 'val ' + args.select
    met['best_epoch'] = best_ep
    met['val_OA_at_best'] = best_va
    met['train_patches'] = int(Xtr.shape[0])
    met['val_patches'] = int(Xva.shape[0])
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(met, f, indent=2, ensure_ascii=False)
    print('训练耗时 %.1f 分钟 | 产物: %s' % (train_sec / 60, out_dir))


if __name__ == '__main__':
    main()
