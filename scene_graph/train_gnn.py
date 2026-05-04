import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

import configs.scan3r.define as scan3rdefine
from scene_graph.graph_net import SceneGraphEdgeNet
from scene_graph.scan3r_dataset import Scan3RDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cosine_rel_loss_batched(
    pred_edge_feat,
    edge_feat_gt_dict_batched,
    mask_batched,
    num_edges_per_graph=None,
):
    """
    Average cosine loss over all edges that have relationship embeddings.
    """
    device = pred_edge_feat.device
    loss_sum = 0.0
    count = 0

    if num_edges_per_graph is None:
        num_edges_per_graph = [m.shape[0] for m in mask_batched]
    edge_offsets = [0] + list(torch.cumsum(torch.tensor(num_edges_per_graph), dim=0).tolist())

    for b, (edge_gt_dict, mask) in enumerate(zip(edge_feat_gt_dict_batched, mask_batched)):
        start, end = edge_offsets[b], edge_offsets[b + 1]
        pred_edge_feat_local = pred_edge_feat[start:end]
        mask_local = mask.bool().to(device)

        if mask_local.sum() == 0:
            continue

        pred_edge_feat_local = F.normalize(pred_edge_feat_local[mask_local], dim=-1)
        gt_indices = list(edge_gt_dict.keys())
        for i, e_idx in enumerate(gt_indices):
            if e_idx >= len(pred_edge_feat_local):
                continue

            gt_feats = edge_gt_dict[e_idx].to(device)
            pred_feat = pred_edge_feat_local[i:i + 1]
            cos_sim = (pred_feat @ gt_feats.t()).squeeze(0).mean()
            loss_sum += (1 - cos_sim)
            count += 1

    if count > 0:
        return loss_sum / count
    return torch.tensor(0.0, device=device, requires_grad=True)


def contrastive_rel_loss_batched(
    pred_edge_feat,
    edge_feat_gt_dict_batched,
    mask_batched,
    relationship_jina_feature,
    relationship_class_names,
    num_edges_per_graph=None,
    tau=0.07,
    neg_scale=1.0,
    num_neg_samples=None,
):
    """
    Multi-positive InfoNCE over relationship text embeddings.
    """
    device = pred_edge_feat.device
    rel_embeds = F.normalize(relationship_jina_feature.to(device), dim=-1)
    num_rels = rel_embeds.shape[0]

    if num_edges_per_graph is None:
        num_edges_per_graph = [m.numel() for m in mask_batched]
    edge_offsets = [0]
    for edge_count in num_edges_per_graph:
        edge_offsets.append(edge_offsets[-1] + edge_count)

    total_loss, count = 0.0, 0
    all_indices = list(range(num_rels))

    for b, gt_dict in enumerate(edge_feat_gt_dict_batched):
        start, end = edge_offsets[b], edge_offsets[b + 1]
        pred_local = F.normalize(pred_edge_feat[start:end], dim=-1)

        for e_idx, gt_feats in gt_dict.items():
            if not (0 <= e_idx < pred_local.shape[0]):
                continue

            gt_feats = F.normalize(gt_feats.to(device), dim=-1)
            pred = pred_local[e_idx:e_idx + 1]
            pos_sim = torch.exp((pred @ gt_feats.T) / tau).sum()

            if gt_feats.shape[0] > 0:
                sims_to_vocab = gt_feats @ rel_embeds.T
                gt_indices = torch.topk(sims_to_vocab, 1, dim=-1).indices.flatten().tolist()
            else:
                gt_indices = []

            neg_indices = list(set(all_indices) - set(gt_indices))
            if num_neg_samples is not None and len(neg_indices) > num_neg_samples:
                neg_indices = random.sample(neg_indices, num_neg_samples)
            neg_emb = rel_embeds[neg_indices]
            neg_sim = torch.exp((pred @ neg_emb.T) / tau).sum() * neg_scale

            total_loss += -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / count


def save_checkpoint(path, epoch, model, optimizer, scheduler):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def train(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    dataset = Scan3RDataset(
        root=args.dataset_root,
        split=args.train_split,
        augment=True,
        device="cpu",
    )
    val_dataset = Scan3RDataset(root=args.dataset_root, split=args.val_split, device="cpu")
    test_dataset = Scan3RDataset(root=args.dataset_root, split=args.test_split, device="cpu")

    print("Dataset size:", len(dataset))
    print("Validation Dataset size:", len(val_dataset))
    print("Test Dataset size:", len(test_dataset))

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=(device == "cuda"),
    )

    relationship_class_names = dataset.relationship_classes
    model = SceneGraphEdgeNet(L=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    last_epoch = 0
    for epoch in range(args.epochs):
        last_epoch = epoch + 1
        model.train()
        running = 0.0
        n_batches = 0

        with tqdm.tqdm(dataloader, desc=f"Train {last_epoch}/{args.epochs}", unit="batch") as pbar:
            for batch in pbar:
                optimizer.zero_grad()

                edge_index = batch["edge_index"].to(device, non_blocking=True).contiguous()
                node_geom_feat = batch["node_geo_feat"].float().to(device, non_blocking=True)
                node_clip_feat = batch["node_clip_feat"].float().to(device, non_blocking=True)
                edge_geom_feat = batch["edge_geo_feat"].float().to(device, non_blocking=True)
                edge_init_feat = batch["edge_init_feat"].float().to(device, non_blocking=True)

                edge_feat = model(
                    edge_index,
                    edge_init_feat,
                    edge_geom_feat,
                    node_clip_feat,
                    node_geom_feat,
                )

                if epoch < args.contrastive_start_epoch:
                    loss = cosine_rel_loss_batched(
                        edge_feat,
                        batch["rel_gt_jina_feat_dict"],
                        batch["rel_gt_edge_index_mask"],
                        num_edges_per_graph=batch["num_edges_per_graph"],
                    )
                else:
                    loss = contrastive_rel_loss_batched(
                        edge_feat,
                        batch["rel_gt_jina_feat_dict"],
                        batch["rel_gt_edge_index_mask"],
                        dataset.relationship_jina_feature,
                        relationship_class_names,
                        num_edges_per_graph=batch["num_edges_per_graph"],
                        tau=args.tau,
                        neg_scale=args.neg_scale,
                        num_neg_samples=args.num_neg_samples,
                    )

                loss.backward()
                optimizer.step()

                loss_val = float(loss.detach().cpu())
                running += loss_val
                n_batches += 1
                pbar.set_postfix(
                    {
                        "loss": f"{loss_val:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

        epoch_loss = running / max(n_batches, 1)
        print(f"Epoch {last_epoch}: mean train loss = {epoch_loss:.4f}")

        if last_epoch >= args.save_start_epoch and last_epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"graph_transformer_512_{last_epoch}.pth")
            save_checkpoint(ckpt_path, last_epoch, model, optimizer, scheduler)

        scheduler.step()

    final_path = os.path.join(args.output_dir, f"graph_transformer_512_{last_epoch}.pth")
    save_checkpoint(final_path, last_epoch, model, optimizer, scheduler)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 3RScan scene graph GNN")
    parser.add_argument("--dataset_root", default=scan3rdefine.SCAN3R_ROOT_PATH)
    parser.add_argument("--output_dir", default="checkpoints/gnn")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="validation")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--layers", default=1, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--scheduler_step_size", default=100, type=int)
    parser.add_argument("--scheduler_gamma", default=0.1, type=float)
    parser.add_argument("--contrastive_start_epoch", default=20, type=int)
    parser.add_argument("--tau", default=0.07, type=float)
    parser.add_argument("--neg_scale", default=1.0, type=float)
    parser.add_argument("--num_neg_samples", default=50, type=int)
    parser.add_argument("--save_start_epoch", default=30, type=int)
    parser.add_argument("--save_every", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
