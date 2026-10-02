def find_file(name):
    """Look in /kaggle/working first, then any added input dataset."""
    p = f"{cfg.OUT_DIR}/{name}"
    if os.path.exists(p): return p
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None

def atomic_save(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = torch.zeros(cfg.MAX_T, device=device); n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        correct += (out.argmax(-1) == y[:, None]).float().sum(0); n += y.size(0)
    return (correct / n).cpu().numpy()

def loss_fn(out, y):
    B, Tn, C = out.shape
    if cfg.LOSS_MODE == "tet":
        return F.cross_entropy(out.reshape(-1, C), y.repeat_interleave(Tn))
    return F.cross_entropy(out[:, -1], y)

def train():
    for p in (LAST, CKPT):
        src = find_file(os.path.basename(p))
        if src and not os.path.exists(p): shutil.copy(src, p)

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == n_params
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.AMP)
    start_ep, best, best_ep, bad, history = 1, -1, 0, 0, []

    if os.path.exists(LAST):
        ck = torch.load(LAST, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_ep, best, best_ep, bad, history = ck["epoch"] + 1, ck["best"], ck["best_ep"], ck["bad"], ck["history"]
        if ck["done"]:
            print(f"Training already finished (best {best:.4f} @ epoch {best_ep}). Skipping.")
            return model
        print(f"Resuming from epoch {start_ep} (best so far {best:.4f} @ epoch {best_ep})")
    else:
        print(f"Starting fresh. Params: {n_params:,} | loss={cfg.LOSS_MODE} | encoding={cfg.ENCODING}")

    t_start = time.time()
    for ep in range(start_ep, cfg.EPOCHS + 1):
        model.train(); t0 = time.time()
        tot_loss = torch.zeros((), device=device); tot_corr = torch.zeros((), device=device); n = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(x); loss = loss_fn(out, y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            tot_loss += loss.detach(); tot_corr += (out[:, -1].argmax(-1) == y).sum(); n += y.size(0)
        sched.step()

        msg = (f"Ep {ep:3d}/{cfg.EPOCHS} | {time.time()-t0:5.1f}s | loss {tot_loss.item()/len(train_loader):.3f} "
               f"| train {tot_corr.item()/n:.4f} | lr {opt.param_groups[0]['lr']:.5f}")
        stop = False
        if ep % cfg.EVAL_EVERY == 0 or ep == cfg.EPOCHS:
            acc_t = evaluate(model, val_loader)
            score = float(acc_t.mean() if cfg.LOSS_MODE == "tet" else acc_t[-1])
            history.append((ep, [float(a) for a in acc_t]))
            msg += " | val " + " ".join(f"{a:.3f}" for a in acc_t)
            if score > best:
                best, best_ep, bad = score, ep, 0
                atomic_save(unwrap(model).state_dict(), CKPT); msg += "  *best saved*"
            else:
                bad += 1; stop = bad >= cfg.PATIENCE
        print(msg)

        done = stop or ep == cfg.EPOCHS
        atomic_save({"model": unwrap(model).state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": ep, "best": best, "best_ep": best_ep, "bad": bad,
                     "history": history, "done": done}, LAST)
        if stop:
            print("Early stopping."); break

    print(f"\nBest val score {best:.4f} @ epoch {best_ep} | session time {(time.time()-t_start)/3600:.2f} h")
    return model

model = train()
