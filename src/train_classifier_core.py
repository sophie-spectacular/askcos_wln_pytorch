import json
import os

import torch
import pandas as pd
import wandb

from WLN.data_loading import Graph_DataLoader
from WLN.metrics import wln_loss, top_10_acc, top_20_acc, top_100_acc
from WLN.models import WLNPairwiseAtomClassifier
from graph_utils.ioutils_direct import nbos


def train_wln_core(train=None, valid=None, reagents=False, batch_size=10, hidden=300,
                   depth=3, epochs=10, output_dim=nbos, learning_rate=0.001,
                   clipnorm=5.0, model_name='wln-core', model_dir='models',
                   early_stopping=0, use_multiprocessing=True, workers=1, device='cpu'):

    assert train and valid, 'Please specify a training set and valid set'

    # Init wandb
    wandb.init(project="wln-torch-test", name=model_name, config={
        "batch_size": batch_size,
        "hidden": hidden,
        "depth": depth,
        "epochs": epochs,
        "output_dim": output_dim,
        "learning_rate": learning_rate,
        "clipnorm": clipnorm,
        "early_stopping": early_stopping
    })

    if reagents:
        print("USING REAGENTS DURING TRAINING")

    train_gen = Graph_DataLoader(train, batch_size, reagents=reagents)
    val_gen = Graph_DataLoader(valid, batch_size, reagents=reagents)

    assert len(train_gen) > 0, f'Training set has {len(train_gen)} examples, has to be greater than 0'
    assert len(val_gen) > 0, f'Validation set has {len(val_gen)} examples, has to be greater than 0'

    model = WLNPairwiseAtomClassifier(hidden, depth, output_dim=output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = wln_loss(batch_size)

    best_val_loss = float('inf')
    patience = 0
    history = {
        'loss': [], 'val_loss': [],
        'top_10_acc': [], 'top_20_acc': [], 'top_100_acc': [],
        'val_top_10_acc': [], 'val_top_20_acc': [], 'val_top_100_acc': []
    }

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_name += '_core'
    model_output = f'{model_dir}/{model_name}-weights.pt'
    history_output = f'{model_dir}/{model_name}-history.json'
    params_output = f'{model_dir}/{model_name}-params.txt'

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_top10 = 0.0
        running_top20 = 0.0
        running_top100 = 0.0
        n_batches = 0

        for batch in train_gen:
            graph, bond_labels = batch[:2]
            if isinstance(graph, list):
                graph = [g.to(device) for g in graph]
            else:
                graph = graph.to(device)
            bond_labels = bond_labels.to(device)

            optimizer.zero_grad()
            scores = model(graph)
            loss = loss_fn(bond_labels, scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            optimizer.step()
            top_10_ac = top_10_acc(bond_labels, scores).item()
            top_20_ac = top_20_acc(bond_labels, scores).item()
            top_100_ac = top_100_acc(bond_labels, scores).item()

            running_loss += loss.item()
            running_top10 += top_10_ac
            running_top20 += top_20_ac
            running_top100 += top_100_ac
            n_batches += 1
            wandb.log({
                'train/loss': loss.item(),
                'train/top_10_acc': top_10_ac,
                'train/top_20_acc': top_20_ac,
                'train/top_100_acc': top_100_ac,
            })

        avg_loss = running_loss / n_batches
        avg_top10 = running_top10 / n_batches
        avg_top20 = running_top20 / n_batches
        avg_top100 = running_top100 / n_batches

        model.eval()
        val_loss = 0.0
        val_top10 = 0.0
        val_top20 = 0.0
        val_top100 = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_gen:
                graph, bond_labels = batch[:2]
                if isinstance(graph, list):
                    graph = [g.to(device) for g in graph]
                else:
                    graph = graph.to(device)
                bond_labels = bond_labels.to(device)
                scores = model(graph)
                loss = loss_fn(bond_labels, scores)
                val_loss += loss.item()
                val_top10 += top_10_acc(bond_labels, scores).item()
                val_top20 += top_20_acc(bond_labels, scores).item()
                val_top100 += top_100_acc(bond_labels, scores).item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_val_top10 = val_top10 / val_batches
        avg_val_top20 = val_top20 / val_batches
        avg_val_top100 = val_top100 / val_batches

        # Log to wandb
        wandb.log({
            'val/loss': avg_val_loss,
            'val/top_10_acc': avg_val_top10,
            'val/top_20_acc': avg_val_top20,
            'val/top_100_acc': avg_val_top100,
            'epoch': epoch
        })

        history['loss'].append(avg_loss)
        history['val_loss'].append(avg_val_loss)
        history['top_10_acc'].append(avg_top10)
        history['top_20_acc'].append(avg_top20)
        history['top_100_acc'].append(avg_top100)
        history['val_top_10_acc'].append(avg_val_top10)
        history['val_top_20_acc'].append(avg_val_top20)
        history['val_top_100_acc'].append(avg_val_top100)

        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - val_loss: {avg_val_loss:.4f} - top_10_acc: {avg_top10:.4f} - val_top_10_acc: {avg_val_top10:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), model_output)
        else:
            patience += 1
            if early_stopping and patience >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break

    pd.DataFrame(history).to_json(history_output)

    with open(params_output, 'w') as f:
        f.write(json.dumps({
            'batch_size': batch_size,
            'hidden': hidden,
            'depth': depth,
            'epochs': epochs,
            'output_dim': output_dim,
            'learning_rate': learning_rate,
            'clipnorm': clipnorm
        }))

    wandb.finish()
    return model


if __name__ == '__main__':
    model = train_wln_core(
        train='/work/data/train_trunc.txt.proc',
        valid='/work/data/valid_trunc.txt.proc',
        hidden=100,
        epochs=2,
        workers=4
    )