import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel as P
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm, trange
from ignite.metrics import PSNR
from ignite.metrics import SSIM
from utils.util import get_luminance, calc_psnr, quantize
from utils.loss import CharbonnierLoss
import os
import copy
import torch_pruning as tp

class ForwardManager():
    def __init__(self, model, training, args, use_max_size=False):
        self.scale = args.scale
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.device = torch.device('cuda')
        self.n_GPUs = torch.cuda.device_count()
        self.training = training

        self.model = model
        if  self.precision == 'half': self.model.half()

        if self.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(self.n_GPUs))

        self.max_inference_size = self._find_max_inference_size()
        self.use_max_size = use_max_size

    def _find_max_inference_size(self):
        # found = False
        # w, h, size = 4096, 2160, 4096 * 2160

        # while not found:
        #     test_tensor = torch.rand(1, 3, w, h)

        #     try: 
        #         _ = self.model(test_tensor)
        #         found = True
        #     except: 
        #         w, h = w//2, h//2
        #         size = w*h        

        return 128
    

    def forward(self, x):
        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        else:
            return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def forward_chop(self, x, shave=None, min_size=160000):
        scale = self.scale
        if not shave: shave = scale
        if self.use_max_size:  min_size = self.max_inference_size
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

    def __call__(self, x) :
        self.forward(x)

def trainFor(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: torch.nn.Module, 
    device: torch.device,
    run_folder: str,
    epochs: int,
    args,
    pruner = None,
    optimizer = None,
    scheduler = None,
): 
    # paths
    ckpt_path = os.path.join(run_folder, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    
    # create the optmizer
    if not optimizer: optimizer = Adam(model.parameters(), lr=args.lr)
    if not scheduler: scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.5, verbose=True)
    
    model = model.to(device) 

    # Set Up Metric Trackers
    psnr_train = PSNR(data_range=args.data_range, output_transform=get_luminance, device=device)
    psnr_val = PSNR(data_range=args.data_range, output_transform=get_luminance, device=device)
    ssim_train = SSIM(data_range=args.data_range, output_transform=get_luminance, device=device)
    ssim_val = SSIM(data_range=args.data_range, output_transform=get_luminance, device=device)
    
    log = pd.DataFrame()
    best_validation_mse = None
    start_epoch = 1
    best_model = model
    for epoch in trange(start_epoch, epochs + 1):
        #Reset metrics calculator
        psnr_train.reset()
        psnr_val.reset()
        ssim_train.reset()
        ssim_val.reset()
        metrics = {'epoch': epoch}

        cur_loss = trainEval(
            loader=train_dataloader, 
            model=model, 
            optimizer=scheduler.optimizer,
            device=device,
            args=args, 
            bTrain = True, 
            psnr=psnr_train, 
            ssim=ssim_train,
            pruner = None,
            data_range=args.data_range
        )
        
        val_loss = trainEval(
            loader=val_dataloader, 
            model=model, 
            optimizer=scheduler.optimizer,
            args=args,
            device=device,
            bTrain = False, 
            psnr=psnr_val, 
            ssim=ssim_val, 
            data_range=args.data_range
        )


        # Log Metrics to CSV File
        metrics['mse_train'] = cur_loss
        metrics['mse_val'] = val_loss
        metrics['psnr_train'] = float(psnr_train.compute())
        metrics['psnr_val'] = float(psnr_val.compute())
        metrics['ssim_train'] = float(ssim_train.compute())
        metrics['ssim_val'] = float(ssim_val.compute())
        
        if best_validation_mse is None or (val_loss < best_validation_mse):
            best_validation_mse = val_loss
            best_model = copy.deepcopy(model)
            ckpt = os.path.join(ckpt_path, 'ckpt_e{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'mse_train': cur_loss,
                'mse_val': val_loss,
                'model': model,
                'optimizer': scheduler.optimizer,
            }, ckpt)
            

        log = log.append(metrics, ignore_index=True)
        log.to_csv(os.path.join(run_folder, f'train_log.csv'), index=False)
          
         
        scheduler.step(val_loss)
        
    ## Save last epoch
    ckpt_path = os.path.join(ckpt_path, 'final_ckpt_e{}.pth'.format(epoch))
    if not os.path.exists(ckpt_path):
        torch.save({
            'epoch': epoch,
            'mse_train': cur_loss,
            'mse_val': val_loss,
            'model': model,
            'optimizer': scheduler.optimizer,
        }, ckpt_path)

    
    last_epoch_model = model
    last_epoch_validation_mse = best_validation_mse
    return (best_model, best_validation_mse, last_epoch_model, last_epoch_validation_mse, scheduler.optimizer, log)
        
#training for a single epoch
def trainEval(loader, model, optimizer, device: torch.device, args, bTrain = True, psnr=None, ssim=None, data_range=1.0, pruner=None, training_iter=0):
    forward = ForwardManager(model, bTrain, args)
    local_psnr = PSNR(data_range=data_range, device=device)
    local_ssim = SSIM(data_range=data_range, device=device)

    if bTrain:
        model.train()
    else:
        model.eval()
        
    loss_function = CharbonnierLoss(args.loss_epsylon)

    total_loss = 0.0
    counter = 0
    progress = tqdm(loader)
    for input, target in progress:
        if bTrain:#train
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            
            model_out = forward.forward(input)
            
        else: #eval
            with torch.no_grad():
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()
        
                model_out = forward.forward(input)

        # quantize the out for SSIM and PSNR calc
        q_out = quantize(model_out, args.data_range)
        local_psnr.reset()
        local_ssim.reset()
        for b in range(q_out.shape[0]):
            local_psnr.update((q_out[b:b+1], target[b:b+1]))
            local_ssim.update((q_out[b:b+1], target[b:b+1]))

            if psnr: psnr.update((q_out[b:b+1], target[b:b+1]))
            if ssim: ssim.update((q_out[b:b+1], target[b:b+1]))
        loss = loss_function(model_out, target)
        
        if bTrain:
            optimizer.zero_grad()
            loss.backward()
            if pruner is not None: pruner.regularize(model) # for sparsity learning
            optimizer.step()
        
        total_loss += loss.item()
        counter += 1
        
        progress.set_postfix({
            'avg_loss': total_loss / counter,
            'loss_iteration': loss.item(),
            'psnr_iteration': float(local_psnr.compute()),
            'ssim_iteration': float(local_ssim.compute())
        })

        training_iter += 1
        if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner) and training_iter % args.update_reg_interval == 0:
            pruner.update_reg() # increase the strength of regularization
            #print(pruner.group_reg[pruner._groups[0]])
            
    return total_loss / counter


#training for a single epoch
def sparsityLearning(loader, model,pruner, args, training_iter=0):
    
    pruner.update_regularizor() # Regrenerate the regularizator needed to handle pruned models
    model.train()   
    loss_function = CharbonnierLoss(args.loss_epsylon)
    optimizer = Adam(model.parameters(), lr=args.lr)

    total_loss = 0.0
    counter = 0
    stop_condition_satisfied = False 
    while (not stop_condition_satisfied):
        for input, target in loader:
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                
            model_out = model(input)   

            # quantize the out for SSIM and PSNR calc
            loss = loss_function(model_out, target)
            optimizer.zero_grad()
            loss.backward()
            pruner.regularize(model) # for sparsity learning
            optimizer.step()
            
            total_loss += loss.item()
            counter += 1
            
            training_iter += 1
            if pruner is not None and isinstance(pruner, tp.pruner.GrowingRegPruner) and training_iter % args.update_reg_interval == 0:
                pruner.update_reg() # increase the strength of regularization
                stop_condition_satisfied = True
                for i, group in enumerate(pruner._groups):
                    gamma = pruner.group_reg[group]
                    stop_condition_satisfied = torch.min(gamma) < args.target_regularization
        
                
        # Generic case is to stop after a full train epoch for sparsity learning               
        if pruner is not None and not isinstance(pruner, tp.pruner.GrowingRegPruner):
            stop_condition_satisfied=True 
                          
    return model, pruner

def forward_one_batch(    
    train_dataloader: DataLoader,
    model: torch.nn.Module, 
    device: torch.device,
    run_folder: str,
    args,
    batch_idx =0,
    optimizer = None,
    scheduler = None,
    local_psnr = None,
    local_ssim = None,
):
    # paths
    ckpt_path = os.path.join(run_folder, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    
    # create the optmizer
    if not optimizer: optimizer = Adam(model.parameters(), lr=args.lr)
    if not scheduler: scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.5, verbose=True)
    
    model = model.to(device) 
    forward = ForwardManager(model, True, args)

    loss_function = CharbonnierLoss(args.loss_epsylon)
    
    model.train()
    for input, target in train_dataloader[batch_idx]:
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        
        model_out = forward.forward(input)

        # quantize the out for SSIM and PSNR calc
        q_out = quantize(model_out, args.data_range)
        if local_psnr: local_psnr.reset()
        if local_ssim: local_ssim.reset()
        for b in range(q_out.shape[0]):
            if local_psnr: local_psnr.update((q_out[b:b+1], target[b:b+1]))
            if local_ssim: local_ssim.update((q_out[b:b+1], target[b:b+1]))
        loss = loss_function(model_out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), scheduler, optimizer, batch_idx +1
    
def eval(loader, model, device: torch.device, args, data_range=1.0):
    forward = ForwardManager(model, True, args)
    local_psnr = PSNR(data_range=data_range, device=device)
    local_ssim = SSIM(data_range=data_range, device=device)

    model.train()

    loss_function = CharbonnierLoss(args.loss_epsylon)
    
    total_loss = 0.0
    counter = 0
    progress = tqdm(loader)
    for input, target in progress:
        with torch.no_grad():
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
    
            model_out = forward.forward(input)

        # quantize the out for SSIM and PSNR calc
        q_out = quantize(model_out, args.data_range)
        for b in range(q_out.shape[0]):
            local_psnr.update((q_out[b:b+1], target[b:b+1]))
            local_ssim.update((q_out[b:b+1], target[b:b+1]))

        loss = loss_function(model_out, target)
        
        total_loss += loss.item()
        counter += 1
        
        progress.set_postfix({
            'avg_loss': total_loss / counter,
            'loss_iteration': loss.item(),
            'psnr_iteration': float(local_psnr.compute()),
            'ssim_iteration': float(local_ssim.compute())
        })

    return total_loss / counter, local_psnr, local_ssim

def testModel(loader, model, psnr, ssim, args, device: torch.device, data_range=1.0, metricTable:pd.DataFrame=None, out= None):
    forward = ForwardManager(model, False, args)
    if not metricTable: metricTable = pd.DataFrame(columns=['sequence','frame','loss','psnr','ssim'])
    local_ssim = SSIM(data_range=data_range, output_transform=get_luminance, device=device)
    custom_psnr = []

    model.eval()
    forward.self_ensemble = True
    
    loss_function = CharbonnierLoss(args.loss_epsylon)
    total_loss = 0.0
    counter = 0
    progress = tqdm(loader)
    for input, target, info in progress:
        with torch.no_grad():
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
    
            model_out = forward.forward(input)
            
        q_out = quantize(model_out, args.data_range)
        local_ssim.reset()
        local_ssim.update((q_out, target))
        psnr_score = calc_psnr(q_out, target, args.scale, args.data_range)
        custom_psnr.append(psnr_score)

       
        if psnr: psnr.update((q_out, target))
        if ssim: ssim.update((q_out, target))
        loss = loss_function(model_out, target)

     
        new_row = pd.DataFrame({
            'sequence': info['sequence_name'],  
            'frame': info['frame'], 
            'loss': loss.item(),
            'psnr': float(psnr_score), 
            'ssim': float(local_ssim.compute())}, 
            index=[0]
        )
        metricTable = pd.concat([new_row, metricTable.loc[:]]).reset_index(drop=True)

        if out:
            (B, C, H, W) = q_out.shape
            model_out_img = q_out.cpu().numpy() # allready clamped between 0 and 255
            for b in range(B):
                model_out_img = model_out_img[b].transpose((1, 2, 0)) 
                if not loader.dataset.useBGR:
                    model_out_img = cv2.cvtColor(model_out_img, cv2.COLOR_RGB2BGR) # CV save in BGR
                s = info['sequence_name']
                n = info['frame']
                cv2.imwrite(f'{out}/{s}_{n}.png', model_out_img)

        total_loss += loss.item()
        counter += 1
        
        progress.set_postfix({
            'loss': total_loss / counter,
            'psnr_iteration': sum(custom_psnr)/len(custom_psnr),
            'ssim_iteration': float(local_ssim.compute())
        })


    return total_loss / counter, metricTable
