import json
import os
import torch
import pandas as pd

from WLN.data_loading import Candidate_DataLoader
from WLN.models import WLNCandidateRanker
from WLN.metrics import top_100_acc, top_10_acc, top_20_acc


def flatten_inputs(inputs):
    inputs = list(inputs)
    B = inputs[0].shape[0]
    N = 151
    C = 76
    max_nb = 10 
    def pad_or_crop(t, target_size, dim):
        if t.shape[dim] > target_size:
            slices = [slice(None)] * t.dim()
            slices[dim] = slice(0, target_size)
            return t[tuple(slices)]
        elif t.shape[dim] < target_size:
            # pad
            pad_sizes = [(0, 0)] * t.dim()
            pad_sizes[dim] = (0, target_size - t.shape[dim])
            return torch.nn.functional.pad(t, [p for pair in reversed(pad_sizes) for p in pair])
        else:
            return t

    inputs[0] = pad_or_crop(inputs[0], C, 2)
    B_, N_, C_, H = inputs[0].shape
    inputs[0] = inputs[0].permute(0, 2, 1, 3).reshape(B_ * C_, N_, H)

    inputs[1] = pad_or_crop(inputs[1], C, 2)
    B_, N_, C_, F = inputs[1].shape
    inputs[1] = inputs[1].permute(0, 2, 1, 3).reshape(B_ * C_, N_, F)

    for i in [2, 3]:
        inputs[i] = pad_or_crop(inputs[i], C, 2)
        inputs[i] = pad_or_crop(inputs[i], max_nb, 3)
        B_, N_, C_, M, D = inputs[i].shape
        inputs[i] = inputs[i].permute(0, 2, 1, 3, 4).reshape(B_ * C_, N_, M, D)
        inputs[i] = inputs[i].long()

    inputs[4] = pad_or_crop(inputs[4], C, 2)
    B_, N_, C_ = inputs[4].shape
    inputs[4] = inputs[4].permute(0, 2, 1).reshape(B_ * C_, N_)

    B_, N_ = inputs[5].shape
    inputs[5] = inputs[5].reshape(B_ * 1, N_)

    return tuple(inputs)


import wandb

def wln_diffnet(train=None, valid=None, test=None, batch_size=1, hidden=500,
                depth=3, epochs=10, learning_rate=0.001, early_stopping=0,
                clipnorm=5.0, model_name='wln_diffnet', model_dir='models',
                use_multiprocessing=True, workers=1, device='cpu'):
    assert train and valid, 'Please specify a training set and valid set'

    wandb.init(project='wln-torch-test', config={
        'batch_size': batch_size,
        'hidden': hidden,
        'depth': depth,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'clipnorm': clipnorm,
        'model_name': model_name
    })

    train_detailed = f'{model_dir}/train_{model_name}.cbond_detailed.txt'
    valid_detailed = f'{model_dir}/valid_{model_name}.cbond_detailed.txt'

    train_gen = Candidate_DataLoader(train_detailed, batch_size)
    val_gen = Candidate_DataLoader(valid_detailed, batch_size)

    assert len(train_gen) > 0, f'Training set has {len(train_gen)} examples'
    assert len(val_gen) > 0, f'Validation set has {len(val_gen)} examples'

    model = WLNCandidateRanker(hidden, depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience = 0
    history = {'loss': [], 'val_loss': [], 'top_10_acc': [], 'val_top_10_acc': [], }

    os.makedirs(model_dir, exist_ok=True)

    model_output = f'{model_dir}/{model_name}_diffnet-weights.pt'
    history_output = f'{model_dir}/{model_name}_diffnet-history.json'
    params_output = f'{model_dir}/{model_name}_diffnet-params.txt'

    for epoch in range(epochs):
        model.train()
        running_loss, running_top10, n_batches = 0.0, 0.0, 0

        for batch_idx, batch in enumerate(train_gen):
            inputs, targets = batch[:2]
            inputs = flatten_inputs(inputs)
            inputs = tuple(i.to(device) for i in inputs)
            targets = targets.to(device)
            if targets.ndim > 1:
                targets = torch.argmax(targets, dim=1)

            outputs = model(inputs)
            targets = targets.view(-1).long()
            outputs = outputs.view(targets.shape[0], -1)

            loss = loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            k = min(10, outputs.size(1))
            top20 = top_20_acc(targets, outputs)
            top10 = top_10_acc(targets, outputs)
            top100 = top_100_acc(targets, outputs)
            running_top10 += top10
            n_batches += 1
            wandb.log({
                'train_loss': loss.item(),
                'train_top_10_acc': top10,
                'train_top_20_acc': top20,
                'train_top_100_acc': top100,
                'epoch': epoch + 1,
                'batch': batch_idx + 1
            })


        avg_loss = running_loss / n_batches
        avg_top10 = running_top10 / n_batches

        model.eval()
        val_loss, val_top10, val_batches = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_gen:
                inputs, targets = batch[:2]
                inputs = flatten_inputs(inputs)
                inputs = tuple(i.to(device) for i in inputs)
                targets = targets.to(device)
                if targets.ndim > 1:
                    targets = torch.argmax(targets, dim=1)

                outputs = model(inputs)
                targets = targets.view(-1).long()
                outputs = outputs.view(targets.shape[0], -1)

                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                k = min(10, outputs.size(1))
            top20 = top_20_acc(targets, outputs)
            top10 = top_10_acc(targets, outputs)
            top100 = top_100_acc(targets, outputs)
            val_top10 += top10
            val_top20 += top20
            val_top100 += top100
            val_batches += 1
                

        avg_val_loss = val_loss / val_batches
        avg_val_top10 = val_top10 / val_batches

        history['loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)
        history['top_10_acc'].append(avg_top10)
        history['val_top_10_acc'].append(avg_val_top10)

        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - val_loss: {avg_val_loss:.4f} - top_10_acc: {avg_top10:.4f} - val_top_10_acc: {avg_val_top10:.4f}")

        wandb.log({
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'train_top_10_acc': avg_top10,
            'val_top_10_acc': avg_val_top10,
            'epoch': epoch + 1
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), model_output)
        else:
            patience += 1
            if early_stopping and patience >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break

    with open(params_output, 'w') as f:
        json.dump({'batch_size': batch_size, 'hidden': hidden, 'depth': depth,
                   'epochs': epochs, 'learning_rate': learning_rate, 'clipnorm': clipnorm}, f)

    pd.DataFrame(history).to_json(history_output)
    wandb.finish()
    return model



if __name__ == '__main__':
    wln_diffnet(train='/work/data/train_trunc.txt.proc',
                valid='/work/data/valid_trunc.txt.proc',
                test='/work/data/test_trunc.txt.proc',
                hidden=100, epochs=5, workers=6)
