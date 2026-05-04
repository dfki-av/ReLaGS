import torch


def _get_hierarchical_candidates(
    query_label,
    snag,
    nag_indices,
    vlm,
    topk,
    filter_small,
    apply_leaf_size_filter,
):
    """Return root and leaf candidates for a text query."""
    vlm.encode_text(query_label)
    similarities = [vlm.compute_similarity(f) for f in snag.feat[1:]]

    root_sim, root_ids = torch.topk(
        similarities[-1], min(topk, similarities[-1].shape[0])
    )

    root_candidates = []
    for rid, rsim in zip(root_ids.tolist(), root_sim.tolist()):
        if ((snag.labels[-1] == rid).sum() > filter_small).item():
            root_candidates.append((rid, float(rsim)))

    leaf_candidates = []
    for rid, rsim in root_candidates:
        child_ids = nag_indices[-2][nag_indices[-1] == rid].unique()
        if child_ids.numel() == 0:
            continue

        if apply_leaf_size_filter:
            leaf_sizes = [
                (snag.labels[-2] == leaf_id).sum().item() for leaf_id in child_ids
            ]
            total_pts = max(sum(leaf_sizes), 1)
            keep = [pt_num >= (total_pts * 0.01) for pt_num in leaf_sizes]
            child_ids = child_ids[keep]
            if child_ids.numel() == 0:
                continue

        child_sims = similarities[-2][child_ids]
        better_mask = child_sims > rsim
        if torch.any(better_mask):
            kept_ids = child_ids[better_mask].tolist()
            kept_sims = child_sims[better_mask].tolist()
            leaf_candidates.extend(
                (cid, float(csim)) for cid, csim in zip(kept_ids, kept_sims)
            )

    return root_candidates, leaf_candidates


def _collect_found_edges(
    root_subject_candidates,
    leaf_subject_candidates,
    root_object_candidates,
    leaf_object_candidates,
    jina_edges,
):
    """Collect candidate edges from root (level 3) and leaf (level 2)."""
    found_edges = []

    for subject_id, subject_sim in root_subject_candidates:
        for object_id, object_sim in root_object_candidates:
            spair = (subject_id, object_id)
            if spair in jina_edges[-1]:
                found_edges.append(
                    {
                        "spair": spair,
                        "jina_rel": jina_edges[-1][spair],
                        "subject_sim": subject_sim,
                        "object_sim": object_sim,
                        "candidate_subject_id": subject_id,
                        "candidate_object_id": object_id,
                        "level": 3,
                    }
                )

    for subject_id, subject_sim in leaf_subject_candidates:
        for object_id, object_sim in leaf_object_candidates:
            spair = (subject_id, object_id)
            if spair in jina_edges[-2]:
                found_edges.append(
                    {
                        "spair": spair,
                        "jina_rel": jina_edges[-2][spair],
                        "subject_sim": subject_sim,
                        "object_sim": object_sim,
                        "candidate_subject_id": subject_id,
                        "candidate_object_id": object_id,
                        "level": 2,
                    }
                )

    return found_edges


def _rank_edges(found_edges, relation_label, jina_encode_fn):
    """Rank edges by (subject sim, object sim, relation sim)."""
    rel_feat_jina = torch.from_numpy(jina_encode_fn(relation_label)).cuda().unsqueeze(0)
    rel_feat_jina = rel_feat_jina / rel_feat_jina.norm(dim=-1, keepdim=True)

    ranked_edges = []
    for edge in found_edges:
        jina_rel = edge["jina_rel"].cuda().unsqueeze(0)
        jina_rel = jina_rel / jina_rel.norm(dim=-1, keepdim=True)
        rel_sim = torch.matmul(rel_feat_jina, jina_rel.transpose(0, 1)).item()
        sims = (edge["subject_sim"], rel_sim, edge["object_sim"])
        ranked_edges.append((edge, sims))

    ranked_edges.sort(key=lambda x: (x[1][0], x[1][2], x[1][1]), reverse=True)
    return ranked_edges


def _select_subject_candidates(
    ranked_edges,
    nag_indices,
    sim_thresholds,
    max_subjects,
):
    """Select subject candidates from ranked edges."""
    sbj_thr, rel_thr, obj_thr = sim_thresholds
    selected = []
    selected_keys = set()

    for edge, sims in ranked_edges:
        if len(selected) >= max_subjects:
            break

        key = (edge["candidate_subject_id"], edge["level"])
        if key in selected_keys:
            continue

        if edge["level"] == 2:
            # Avoid selecting level-2 subject if corresponding parent is already selected.
            object_id = edge["candidate_object_id"]
            parent_candidates = nag_indices[-1][nag_indices[-2] == object_id]
            if parent_candidates.numel() > 0:
                parent_key = (parent_candidates[0].item(), 3)
                if parent_key in selected_keys:
                    continue

        if sims[0] >= sbj_thr and sims[1] >= rel_thr and sims[2] >= obj_thr:
            if (not selected) or (selected[0][1][0] - sims[0] < 0.02):
                selected.append((key, sims))
                selected_keys.add(key)

    if not selected and ranked_edges:
        edge, sims = ranked_edges[0]
        key = (edge["candidate_subject_id"], edge["level"])
        selected.append((key, sims))

    return selected


def search_triplet_subjects(
    subject_label,
    object_label,
    relation_label,
    snag,
    nag_indices,
    jina_edges,
    vlm,
    jina_encode_fn,
    topk=10,
    filter_small=300,
    sim_thresholds=(0.55, 0.4, 0.55),
    max_subjects=3,
):
    """Triplet-based subject search over hierarchical candidates and predicted edges."""
    root_subject, leaf_subject = _get_hierarchical_candidates(
        subject_label,
        snag,
        nag_indices,
        vlm,
        topk,
        filter_small,
        apply_leaf_size_filter=True,
    )
    root_object, leaf_object = _get_hierarchical_candidates(
        object_label,
        snag,
        nag_indices,
        vlm,
        topk,
        filter_small,
        apply_leaf_size_filter=False,
    )

    found_edges = _collect_found_edges(
        root_subject,
        leaf_subject,
        root_object,
        leaf_object,
        jina_edges,
    )
    if not found_edges:
        return []

    ranked_edges = _rank_edges(found_edges, relation_label, jina_encode_fn)
    return _select_subject_candidates(
        ranked_edges,
        nag_indices,
        sim_thresholds,
        max_subjects,
    )
