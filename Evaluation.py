@torch.no_grad()
def collect(model, loader, seed):
    """One pass at T=MAX_T. Returns logits [N,T,C], labels [N], per-image pixel-variance [N]."""
    model.eval()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    L, Y, V = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        V.append(x.flatten(1).var(dim=1).cpu())
        L.append(model(x).float().cpu()); Y.append(y)
    return torch.cat(L).numpy(), torch.cat(Y).numpy(), torch.cat(V).numpy()

# load best checkpoint explicitly (works even if Cell 2 was skipped/resumed)
src = find_file(os.path.basename(CKPT))
assert src, "Best checkpoint not found."
model = build_model()
unwrap(model).load_state_dict(torch.load(src, map_location=device))
model.eval()
print("Loaded best checkpoint from:", src)

EVAL_SEEDS = [0, 1, 2, 3, 4]
VAL_SEEDS  = [100, 101]

t0 = time.time()
test_data = [collect(model, test_loader, s)[:2] for s in EVAL_SEEDS]
val_data  = [collect(model, val_loader, s)[:2]  for s in VAL_SEEDS]
print(f"Collected logits in {time.time()-t0:.0f}s")
print(f"Test T=6 accuracy: {np.mean([(L[:,-1].argmax(-1)==Y).mean() for L,Y in test_data]):.4f}")
print(f"Val  T=6 accuracy: {np.mean([(L[:,-1].argmax(-1)==Y).mean() for L,Y in val_data]):.4f}")

np.savez_compressed(f"{cfg.OUT_DIR}/logits_{cfg.LOSS_MODE}.npz",
    **{f"testL{i}": d[0] for i, d in enumerate(test_data)},
    **{f"testY{i}": d[1] for i, d in enumerate(test_data)},
    **{f"valL{i}": d[0] for i, d in enumerate(val_data)},
    **{f"valY{i}": d[1] for i, d in enumerate(val_data)})
print("Logits cached to disk.")
