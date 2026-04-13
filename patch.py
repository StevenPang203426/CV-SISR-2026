with open("train.py", "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("import torch.nn as nn\n", "import torch.nn as nn\nimport wandb\n")

old_init = "    # 学习率调度器\n    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, min_lr=1e-6)\n\n    # 创建保存目录\n"
new_init = "    # 学习率调度器\n    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, min_lr=1e-6)\n\n    # 初始化 wandb\n    wandb.init(project=\"SISR\", config=vars(args), name=f\"{args.model}_x{args.scale}\")\n\n    # 创建保存目录\n"
code = code.replace(old_init, new_init)

old_log = "            dt = time.time() - t0\n            log_line(fp, f\"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | train_psnr={train_psnr:.2f} dB | val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f} dB | time={dt:.1f}s\")\n            \n            # 保存最佳模型\n"
new_log = "            dt = time.time() - t0\n            log_line(fp, f\"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | train_psnr={train_psnr:.2f} dB | val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f} dB | time={dt:.1f}s\")\n            \n            # 使用 wandb 记录日志\n            wandb.log({\n                \"epoch\": epoch,\n                \"train_loss\": train_loss,\n                \"train_psnr\": train_psnr,\n                \"val_loss\": val_loss,\n                \"val_psnr\": val_psnr,\n                \"lr\": opt.param_groups[0]['lr'] if type(opt) != dict else list(opt.values())[0].param_groups[0]['lr']\n            })\n\n            # 保存最佳模型\n"
code = code.replace(old_log, new_log)

old_finish = "        plot_metrics(args, train_loss_history, train_psnr_history, val_loss_history, val_psnr_history, args.save_dir)\n        log_line(fp, f\"Training complete. Metrics plot saved to {args.save_dir}/metrics.png\")\n"
new_finish = "        plot_metrics(args, train_loss_history, train_psnr_history, val_loss_history, val_psnr_history, args.save_dir)\n        log_line(fp, f\"Training complete. Metrics plot saved to {args.save_dir}/metrics.png\")\n        wandb.finish()\n"
code = code.replace(old_finish, new_finish)

with open("train.py", "w", encoding="utf-8") as f:
    f.write(code)
print("patched")
