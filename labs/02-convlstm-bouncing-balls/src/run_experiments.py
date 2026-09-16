"""Reproduce controlled ConvLSTM bouncing-ball experiments."""
import argparse, csv, json, random, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def balls(n, t=10, size=32, radius=2, seed=42):
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[:size, :size]
    out = np.zeros((n, t, size, size), np.float32)
    for s in range(n):
        x, y = rng.uniform(4, size - 5, 2)
        vx, vy = rng.choice([-1, 1], 2) * rng.uniform(.8, 1.6, 2)
        for i in range(t):
            x += vx; y += vy
            if x < radius or x > size - radius: vx = -vx; x = np.clip(x, radius, size-radius)
            if y < radius or y > size - radius: vy = -vy; y = np.clip(y, radius, size-radius)
            out[s, i] = (xx-x)**2 + (yy-y)**2 <= radius**2
    return out[:, :, None]


class Cell(nn.Module):
    def __init__(self, cin, hidden, kernel):
        super().__init__(); self.hidden = hidden
        self.conv = nn.Conv2d(cin + hidden, 4 * hidden, kernel, padding=kernel//2)
    def forward(self, x, state):
        h, c = state; i, f, g, o = self.conv(torch.cat((x, h), 1)).chunk(4, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f*c + i*torch.tanh(g); return o*torch.tanh(c), c


class ConvLSTM(nn.Module):
    def __init__(self, hidden=32, kernel=3, layers=1):
        super().__init__(); channels = [1] + [hidden]*layers
        self.cells = nn.ModuleList(Cell(channels[i], channels[i+1], kernel) for i in range(layers))
        self.out = nn.Conv2d(hidden, 1, 3, padding=1)
    def forward(self, x):
        b, _, _, h, w = x.shape
        states = [(torch.zeros(b, c.hidden, h, w, device=x.device), torch.zeros(b, c.hidden, h, w, device=x.device)) for c in self.cells]
        for t in range(x.shape[1]):
            z = x[:, t]
            for i, cell in enumerate(self.cells): states[i] = cell(z, states[i]); z = states[i][0]
        return self.out(z).squeeze(1)


class FlattenLSTM(nn.Module):
    def __init__(self):
        super().__init__(); self.lstm = nn.LSTM(1024, 256, batch_first=True); self.out = nn.Linear(256, 1024)
    def forward(self, x):
        b, t, _, h, w = x.shape; z, _ = self.lstm(x.reshape(b, t, h*w))
        return self.out(z[:, -1]).reshape(b, h, w)


def xy(data, frames):
    return torch.from_numpy(data[:, :frames]), torch.from_numpy(data[:, frames, 0])


def metrics(model, loader, device):
    model.eval(); se = ae = count = 0; pred, target = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)); y = y.to(device); se += (p-y).square().sum().item(); ae += (p-y).abs().sum().item(); count += y.numel()
            pred.append(p.cpu()); target.append(y.cpu())
    return se/count, ae/count, torch.cat(pred), torch.cat(target)


def curve(hist, path, name):
    fig, ax = plt.subplots(figsize=(7,4)); ep = range(1, len(hist['train'])+1)
    ax.plot(ep, hist['train'], marker='o', label='train loss'); ax.plot(ep, hist['mse'], marker='s', label='test MSE')
    ax.set(xlabel='epoch', ylabel='loss / MSE', title=name); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def predict_plot(x, y, p, frames, path):
    fig, axes = plt.subplots(3, frames+2, figsize=(1.7*(frames+2), 5.2)); ids = (0,1,2)
    for row, i in enumerate(ids):
        for t in range(frames): axes[row,t].imshow(x[i,t,0], cmap='gray', vmin=0, vmax=1); axes[row,t].set_title(f'in {t+1}')
        axes[row,-2].imshow(y[i], cmap='gray', vmin=0, vmax=1); axes[row,-2].set_title('target')
        mse = (p[i]-y[i]).square().mean().item(); axes[row,-1].imshow(p[i], cmap='gray', vmin=0, vmax=1); axes[row,-1].set_title(f'prediction\nMSE={mse:.5f}')
        for a in axes[row]: a.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)


def run(cfg, train, test, output, epochs, device, run_seed=42):
    seed_all(run_seed); frames = cfg.get('frames', 4); tx, ty = xy(train, frames); vx, vy = xy(test, frames)
    loader = DataLoader(TensorDataset(tx,ty), batch_size=64, shuffle=True); valid = DataLoader(TensorDataset(vx,vy), batch_size=64)
    model = FlattenLSTM() if cfg.get('flat') else ConvLSTM(cfg.get('hidden',32), cfg.get('kernel',3), cfg.get('layers',1))
    model.to(device); loss_fn = nn.L1Loss() if cfg.get('loss') == 'l1' else nn.MSELoss(); opt = torch.optim.Adam(model.parameters(), lr=.001); hist={'train':[],'mse':[]}; start=time.perf_counter()
    for _ in range(epochs):
        model.train(); total=0
        for x,y in loader:
            opt.zero_grad(); loss=loss_fn(model(x.to(device)),y.to(device)); loss.backward(); opt.step(); total += loss.item()*x.size(0)
        mse, _, _, _ = metrics(model, valid, device); hist['train'].append(total/len(loader.dataset)); hist['mse'].append(mse)
    mse, mae, pred, target = metrics(model, valid, device); elapsed=time.perf_counter()-start
    result={'id':cfg['id'],'name':cfg['name'],'seed':run_seed,'test_mse':mse,'test_mae':mae,'seconds':elapsed,'parameters':sum(p.numel() for p in model.parameters()),'config':cfg}
    folder=output/cfg['id']/f'seed_{run_seed}'; folder.mkdir(parents=True,exist_ok=True); curve(hist,folder/'loss_curve.png',cfg['name']); predict_plot(vx,target,pred,frames,folder/'prediction_examples.png')
    (folder/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); return result, vx, target, pred


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=Path('results/reproduced')); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    ntrain, ntest, epochs = (800,100,3) if args.quick else (2000,200,5); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); args.output.mkdir(parents=True,exist_ok=True)
    train, test = balls(ntrain,seed=42), balls(ntest,seed=43)
    experiments=[
      {'id':'baseline','name':'baseline','layers':1,'hidden':32,'kernel':3,'frames':4},
      {'id':'exp1','name':'two layers','layers':2,'hidden':32,'kernel':3,'frames':4},
      {'id':'exp2','name':'64 channels','layers':1,'hidden':64,'kernel':3,'frames':4},
      {'id':'exp3','name':'5x5 kernel','layers':1,'hidden':32,'kernel':5,'frames':4},
      {'id':'exp4','name':'eight frames','layers':1,'hidden':32,'kernel':3,'frames':8},
      {'id':'exp5','name':'flattened LSTM','flat':True,'frames':4},
      {'id':'exp6','name':'L1 loss','layers':1,'hidden':32,'kernel':3,'frames':4,'loss':'l1'}]
    results=[]
    for cfg in experiments: results.append(run(cfg,train,test,args.output,epochs,device)[0])
    with (args.output/'experiment_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','name','seed','test_mse','test_mae','seconds','parameters','config']); w.writeheader(); [w.writerow({**r,'config':json.dumps(r['config'])}) for r in results]
    byid={r['id']:r for r in results}; best={'id':'best','name':'best combination','layers':2,'hidden':64,'kernel':5,'frames':8}; repeats=[run(best,train,test,args.output,epochs,device,s)[0] for s in (42,43,44)]
    avg={k:float(np.mean([r[k] for r in repeats])) for k in ('test_mse','test_mae','seconds','parameters')}
    with (args.output/'best_repeat_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['seed','test_mse','test_mae','seconds','parameters']); w.writeheader(); [w.writerow({k:r[k] for k in w.fieldnames}) for r in repeats]; w.writerow({'seed':'mean',**avg})
    _,x,y,p=run(best,train,test,args.output,epochs,device,42); predict_plot(x,y,p,8,args.output/'result_pred.png')
    print(json.dumps({'device':str(device),'baseline':byid['baseline'],'best_mean':avg},indent=2))

if __name__ == '__main__': main()
