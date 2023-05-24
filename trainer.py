import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data_handling.logger import print_and_log


def run_testing(args, device, model, save_subfolder, test_loader, logger):
    if args.verbose:
        print('Testing model')
    model.load_state_dict(torch.load(os.path.join(save_subfolder, 'model_final.pth'))['state_dict'])
    test_loss, test_acc = evaluate(args, model, device, test_loader, is_test_set=True)
    logger.log_no_step("test_loss", test_loss)
    logger.log_no_step("test_acc", test_acc)

def run_eval(args, device, model, save_subfolder, test_loader, logger):
    if args.verbose:
        print('Testing model')
    #model.load_state_dict(torch.load(os.path.join(save_subfolder, 'model_final.pth'))['state_dict'])
    test_loss, test_acc = evaluate(args, model, device, test_loader, is_test_set=True)
    logger.log_no_step("eval_loss", test_loss)
    logger.log_no_step("eval_acc", test_acc)


def run_training(args, best_acc, device, lr_scheduler, mask, model, optimizer, save_subfolder, train_loader,
                 valid_loader, logger):
    if mask is not None:
        save_checkpoint({
            'epoch': 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "masks": mask.masks},
            filename=os.path.join("./save", 'model_start.pth'))
    for epoch in range(1, args.epochs * args.multiplier + 1):
        t0 = time.time()
        train_loss, train_acc, stopped = train(args, model, device, train_loader, optimizer, epoch, mask, args.manual_stop)
        lr_scheduler.step()

        metrics = {"val_loss": np.nan,
                   "val_acc": np.nan,
                   "train_loss": train_loss,
                   "train_acc": train_acc,
                   }

        if args.valid_split > 0.0:
            val_loss, val_acc = evaluate(args, model, device, valid_loader)
            metrics["val_loss"] = val_loss
            metrics["val_acc"] = val_acc

        if metrics["val_acc"] != np.nan and (metrics["val_acc"] > best_acc):
            if args.verbose:
                print('Saving model')
            best_acc = metrics["val_acc"]
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(save_subfolder, 'model_final.pth'))

        if mask is not None:
            metrics["running_death_rate"] = mask.death_rate
            if mask.running_layer_density is not None:
                metrics.update(mask.running_layer_density)

        logger.log(metrics)

        if args.verbose:
            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(
                optimizer.param_groups[0]['lr'], time.time() - t0))

        if args.manual_stop and stopped:
            break

    if not args.manual_stop:
        if mask is None:
            save_checkpoint({
                'epoch': args.epochs * args.multiplier,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename=os.path.join(save_subfolder, 'model_last.pth'))
        else:
            save_checkpoint({
                'epoch': args.epochs * args.multiplier,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'masks': mask.masks,
            }, filename=os.path.join(save_subfolder, 'model_last.pth'))
    else:
        save_checkpoint({
            'epoch': args.epochs * args.multiplier,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "masks": mask.masks,
            "grads": mask.grads_at_update,
            "weights": mask.weights_at_update,
            "scores": mask.scores_at_update,
            "masks_before": mask.masks_at_update
        }, filename=os.path.join("./save", 'model_last.pth'))


def resume(args, device, model, optimizer, test_loader):
    if os.path.isfile(args.resume):
        if args.verbose:
            print_and_log("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.verbose:
            print_and_log("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.resume, checkpoint['epoch']))
            print_and_log('Testing...')
        evaluate(args, model, device, test_loader)
        model.feats = []
        model.densities = []
    else:
        if args.verbose:
            print_and_log("=> no checkpoint found at '{}'".format(args.resume))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("SAVING")
    torch.save(state, filename)


def train(args, model, device, train_loader, optimizer, epoch, mask=None, manual_stop=False):
    model.train()
    train_loss = 0
    correct = 0
    n = 0
    stopped = False
    # global gradient_norm
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        if args.data == 'higgs':
            loss = F.binary_cross_entropy_with_logits(output, target.unsqueeze(1))
            pred = torch.round(torch.sigmoid(output))
        else:
            loss = F.cross_entropy(output, target) if model.last == "logits" else F.nll_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        train_loss += loss.item()
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                epoch, batch_idx * len(data), len(train_loader) * args.batch_size,
                       100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))

        if mask is not None and mask.after_at_least_one_prune and manual_stop:
            stopped = True
            break

    # training summary
    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary',
        train_loss / batch_idx, correct, n, 100. * correct / float(n)))

    return train_loss / batch_idx, 100. * correct / float(n), stopped


def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            # target = target.to(torch.int64)
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            if args.data == 'higgs':
                test_loss_p = F.binary_cross_entropy_with_logits(output, target.unsqueeze(1), reduction='sum')
                pred = torch.round(torch.sigmoid(output))
            else:
                test_loss_p = F.cross_entropy(output, target, reduction='sum') if model.last == "logits" else F.nll_loss(output, target, reduction='sum')
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_loss += test_loss_p.item()  # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return test_loss, 100. * correct / float(n)
