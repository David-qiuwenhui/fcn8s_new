import os

import torch
from tqdm import tqdm

from nets.fcn_training import CE_Loss, Dice_loss, Focal_Loss
from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(
    model_train,
    model,
    aux_branch,
    loss_history,
    eval_callback,
    optimizer,
    epoch,
    epoch_step,
    epoch_step_val,
    gen,
    gen_val,
    Epoch,
    cuda,
    dice_loss,
    focal_loss,
    cls_weights,
    num_classes,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank=0,
):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    # ---------- 模型的训练过程 --------------------------------------------------------------------
    if local_rank == 0:
        print("---------- Start Train ----------")
        pbar = tqdm(
            total=epoch_step,
            desc=f"🚀Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.1,
        )

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            if aux_branch:
                # ---------- 前向传播 带辅助分类器----------
                main_outputs, aux_outputs = model_train(imgs)
                if focal_loss:
                    main_loss = Focal_Loss(
                        inputs=main_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                    aux_loss = Focal_Loss(
                        inputs=aux_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                else:
                    main_loss = CE_Loss(
                        inputs=main_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                    aux_loss = CE_Loss(
                        inputs=aux_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                    loss = main_loss + aux_loss * 0.4
                    if dice_loss:
                        main_dice = Dice_loss(inputs=main_outputs, target=labels)
                        aux_dice = Dice_loss(inputs=aux_outputs, target=labels)
                        loss = main_dice + 0.4 * aux_dice + loss
            else:
                # ---------- 前向传播 不带辅助分类器 ----------
                main_outputs = model_train(imgs)
                # 计算损失
                if focal_loss:
                    loss = Focal_Loss(
                        inputs=main_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                else:
                    loss = CE_Loss(
                        inputs=main_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )

                if dice_loss:
                    main_dice = Dice_loss(inputs=main_outputs, target=labels)
                    loss = loss + main_dice

            with torch.no_grad():
                # 计算f_score
                _f_score = f_score(inputs=main_outputs, target=labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast

            with autocast():
                if aux_branch:
                    # ---------- 前向传播 带辅助分类器----------
                    main_outputs, aux_outputs = model_train(imgs)
                    if focal_loss:
                        main_loss = Focal_Loss(
                            inputs=main_outputs,
                            target=pngs,
                            cls_weights=weights,
                            num_classes=num_classes,
                        )
                        aux_loss = Focal_Loss(
                            inputs=aux_outputs,
                            target=pngs,
                            cls_weights=weights,
                            num_classes=num_classes,
                        )
                    else:
                        main_loss = CE_Loss(
                            inputs=main_outputs,
                            target=pngs,
                            cls_weights=weights,
                            num_classes=num_classes,
                        )
                        aux_loss = CE_Loss(
                            inputs=aux_outputs,
                            target=pngs,
                            cls_weights=weights,
                            num_classes=num_classes,
                        )
                    loss = main_loss + aux_loss * 0.4
                    if dice_loss:
                        main_dice = Dice_loss(inputs=main_outputs, target=labels)
                        aux_dice = Dice_loss(inputs=aux_outputs, target=labels)
                        loss = main_dice + 0.4 * aux_dice + loss
                else:
                    # ---------- 前向传播 不带辅助分类器 ----------
                    main_outputs = model_train(imgs)
                    # 计算损失
                    if focal_loss:
                        loss = Focal_Loss(
                            inputs=main_outputs,
                            target=pngs,
                            cls_weights=weights,
                            num_classes=num_classes,
                        )
                    else:
                        loss = CE_Loss(
                            inputs=main_outputs,
                            target=pngs,
                            cls_weights=weights,
                            num_classes=num_classes,
                        )

                    if dice_loss:
                        main_dice = Dice_loss(inputs=main_outputs, target=labels)
                        loss = loss + main_dice

                with torch.no_grad():
                    # 计算f_score
                    _f_score = f_score(main_outputs, labels)

            # ---------- 反向传播 ----------
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # 累加单个epoch的训练总损失值和score
        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{
                    "📝️train_total_loss": total_loss / (iteration + 1),
                    "📒f_score": total_f_score / (iteration + 1),
                    "📖lr": get_lr(optimizer),
                }
            )
            pbar.update(1)
    # ----------------------------------------------------------------------------------------------

    # ---------- 模型的验证过程 ----------------------------------------------------------------------
    if local_rank == 0:
        pbar.close()
        print("--------- Finish Train! ----------")
        print("********** Start Validation **********")
        pbar = tqdm(
            total=epoch_step_val,
            desc=f"💡Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.1,
        )

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            if aux_branch:
                # ---------- 前向传播 带辅助分类器----------
                main_outputs, aux_outputs = model_train(imgs)
                # 损失计算
                if focal_loss:
                    main_loss = Focal_Loss(
                        inputs=main_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                    aux_loss = Focal_Loss(
                        inputs=aux_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                else:
                    main_loss = CE_Loss(
                        inputs=main_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                    aux_loss = CE_Loss(
                        inputs=aux_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                loss = main_loss + 0.4 * aux_loss

                if dice_loss:
                    main_dice = Dice_loss(main_outputs, labels)
                    aux_dice = Dice_loss(aux_outputs, labels)
                    loss = main_dice + 0.4 * aux_dice + loss
            else:
                # ---------- 前向传播 不带辅助分类器----------
                main_outputs = model_train(imgs)
                # 损失计算
                if focal_loss:
                    loss = Focal_Loss(
                        inputs=main_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )
                else:
                    loss = CE_Loss(
                        inputs=main_outputs,
                        target=pngs,
                        cls_weights=weights,
                        num_classes=num_classes,
                    )

                if dice_loss:
                    main_dice = Dice_loss(main_outputs, labels)
                    loss = loss + main_dice
            # 计算f_score
            _f_score = f_score(main_outputs, labels)

            # 累加单个epoch的验证总损失值和score
            val_loss += loss.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{
                    "📝️val_loss": val_loss / (iteration + 1),
                    "📒f_score": val_f_score / (iteration + 1),
                    "📖lr": get_lr(optimizer),
                }
            )
            pbar.update(1)
    # ----------------------------------------------------------------------------------------------

    # -------------------- 保存本次epoch的训练和验证结果 ------------------------
    if local_rank == 0:
        pbar.close()
        print("********** Finish Validation **********")
        loss_history.append_loss(
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val
        )
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print("Epoch:" + str(epoch + 1) + "/" + str(Epoch))
        print(
            "Total Loss: %.3f || Val Loss: %.3f "
            % (total_loss / epoch_step, val_loss / epoch_step_val)
        )

        # 周期保存epoch的权重参数
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_dir,
                    "ep%03d-loss%.3f-val_loss%.3f.pth"
                    % ((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val),
                ),
            )
        # 保存当前最好的epoch的权重参数
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(
            loss_history.val_loss
        ):
            print("Save best model to best_epoch_weights.pth")
            torch.save(
                model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth")
            )
        # 保存最后一个epoch的权重参数
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
    # -------------------------------------------------------------------------
