# hours wasted: 2

import torch
import optuna
import gc
import collections
from torchinfo import summary
import nndl.models.CNN as cnn
import nndl.models.CNNLSTM as cnnlstm
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision
from torchvision.transforms import v2
from tqdm.notebook import tqdm

# ==============================================================================
# START OF train_model()
# ==============================================================================
def train_model(model,
                criterion,
                optimizer,
                scheduler,
                train_loader=None,
                val_loader=None,
                test_loader=None,
                num_epochs=100,
                learning=False,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                trial=None):
    # we return these
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')

        # ======================================================================
        # START OF TRAINING
        # ======================================================================
        model.train()
        train_count = 0
        train_correct_count = 0

        # minibatch
        for batch_idx, (train_x, train_y) in enumerate(tqdm(train_loader)):
            train_x = train_x.float().to(device)
            train_y = train_y.long().to(device)

            logits = model(train_x)
            loss = criterion(logits, train_y)

            optimizer.zero_grad()   # no gradient accumulation between batches
            loss.backward()         # backprop
            optimizer.step()        # gradient step

            # training accuracy
            with torch.no_grad():
                y_hat = torch.argmax(logits, dim=-1)
                train_correct_count += torch.sum(y_hat == train_y, axis=-1)
                train_count += train_x.size(0)

        train_acc = train_correct_count / train_count
        train_accuracies.append(train_acc.item())
        # ======================================================================
        # END OF TRAINING
        # ======================================================================

        # ======================================================================
        # START OF VALIDATION
        # ======================================================================
        model.eval()
        val_count = 0
        val_correct_count = 0
        val_loss = 0

        # validation accuracy
        with torch.no_grad():
            for idx, (val_x, val_y) in enumerate(val_loader):
                val_x = val_x.float().to(device)
                val_y = val_y.long().to(device)

                logits = model(val_x).detach()
                y_hat = torch.argmax(logits, dim=-1)

                val_correct_count += torch.sum(y_hat == val_y, axis=-1)
                val_count += val_y.size(0)

                # for the learning rate scheduler
                val_loss = criterion(logits, val_y)

        val_acc = val_correct_count / val_count
        val_accuracies.append(val_acc.item())
        scheduler.step(val_loss)
        # ======================================================================
        # END OF VALIDATION
        # ======================================================================

        # performance info
        print('Train acc: {:.3f}, Val acc: {:.3f}, Val loss: {:.3f}'.format(train_acc,
                                                                            val_acc,
                                                                            val_loss))

        if learning:
            # ======================================================================
            # START OF TRIAL PRUNING
            # ======================================================================
            trial.report(val_acc.item(), epoch)
            # so my gpu doesn't shit itself, also gets rid of shit trials
            if trial.should_prune():
                # garbage collection so my gpu doesn't shit itself
                del model
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()
            # ======================================================================
            # START OF TRIAL PRUNING
            # ======================================================================

    if learning:
        # garbage collection so my gpu doesn't shit itself
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return train_accuracies, val_accuracies
# ==============================================================================
# END OF train_model()
# ==============================================================================


# ==============================================================================
# START OF test_model()
# ==============================================================================
def test_model(model, test_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    with torch.no_grad():
        test_count = 0
        test_correct_count = 0

        for _, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.float().to(device)
            test_y = test_y.long().to(device)

            logits = model(test_x).detach()
            y_hat = torch.argmax(logits, dim=-1)

            test_correct_count += torch.sum(y_hat == test_y, axis=-1)
            test_count += test_x.size(0)

        test_acc = test_correct_count / test_count
        print('Test acc: {:.3f}'.format(test_acc))
# ==============================================================================
# END OF test_model()
# ==============================================================================


# =============================================================================
# START OF info_dump()
# =============================================================================
def info_dump(model_name,
              batch_size,
              optimizer_name,
              learning_rate,
              weight_decay,
              momentum,        # this is only for sgd and rmsprop
              dropout,
              kernel1,
              kernel2,
              kernel3,
              kernel4,
              pool_kernel,
              depth,
              scale,
              hidden_dims=0,     # this is for the cnn-lstm
              training=True):
    if training:
        print(f'Training with the following hyperparameters:')
        print('---------------------------------------------')
    else:
        print(f'    Learned Hyperparameters')
        print('    ------------------------')

    print(f'    Batch Size:                         {batch_size}')
    if model_name == 'CNNLSTM':
        print(f'    Hidden Dimensions:                  {hidden_dims}')
    print(f'    Optimizer:                          {optimizer_name}')
    print(f'        Learning Rate:                  {learning_rate}')
    print(f'        Weight Decay:                   {weight_decay}')
    if optimizer_name == 'RMSprop' or optimizer_name == 'SGD':
        print(f'        Momentum:                       {momentum}')
    print(f'    Model:                              {model_name}')
    print(f'        Dropout:                        {dropout}')
    print(f'        (Block 1) Conv1d Kernel Size:   {kernel1}')
    print(f'        (Block 1) Conv2d Kernel Size:   {kernel2}')
    print(f'        (Block 2) Conv1d Kernel Size:   {kernel3}')
    print(f'        (Block 3) Conv1d Kernel Size:   {kernel4}')
    print(f'        Pool Kernel Size:               {pool_kernel}')
    print(f'        Depth:                          {depth}')
    print(f'        Scale:                          {scale}')
# =============================================================================
# END OF info_dump()
# =============================================================================


# =============================================================================
# START OF objective()
# =============================================================================
def objective(trial,
              X_train,
              y_train,
              X_valid,
              y_valid,
              model_name='CNN',
              num_epochs=10,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # =========================================================================
    # START OF HYPERPARAMETER INITIALIZATION
    # =========================================================================
    batch_sizes = [64, 128, 256]

    optimizers = ['Adamax', 'NAdam', 'Adam', 'RMSprop', 'SGD']
    lr_min = 1e-4 # learning rate
    lr_max = 1e-2 # learning rate
    wd_min = 1e-6 # weight decay
    wd_max = 1e-2 # weight decay
    mu_min = 0.8  # momentum
    mu_max = 0.99 # momentum

    # data hyperparameters
    batch_size = trial.suggest_categorical('batch_size', batch_sizes)

    # optimizer hyperparameters
    optimizer_name = trial.suggest_categorical('optimizer_name', optimizers)
    learning_rate = trial.suggest_float('learning_rate', lr_min, lr_max, log=True)
    weight_decay = trial.suggest_float('weight_decay', wd_min, wd_max, log=True)
    momentum = trial.suggest_float('momentum', mu_min, mu_max, log=True)  # only used for RMSprop and SGD

    do_min = 0.4
    do_max = 0.8
    ks2 = ['(1, 17)', '(1, 22)','(1, 25)']
    ks_min = 2
    ks_max = 10
    ps_min = 2
    ps_max = 5
    offset = 3
    depths = [25, 50, 100, 150, 200] # initial out_channels
    scale_min = 2
    scale_max = 5
    hidden_dimss = [16, 32, 64, 128, 256]

    # model hyperparameters
    dropout = trial.suggest_float('dropout', do_min, do_max)
    kernel1 = trial.suggest_int('kernel1', ks_min, ks_max)
    kernel2 = eval(trial.suggest_categorical('kernel2', ks2))
    kernel3 = trial.suggest_int('kernel3', ks_min, ks_max - offset)
    kernel4 = trial.suggest_int('kernel4', ks_min, ks_max - offset)
    pool_kernel = trial.suggest_int('pool_kernel', ps_min, ps_max)
    depth = trial.suggest_categorical('depth', depths)
    scale = trial.suggest_int('scale', scale_min, scale_max)
    hidden_dims = trial.suggest_categorical('hidden_dims', hidden_dimss)
    # =========================================================================
    # END OF HYPERPARAMETER INITIALIZATION
    # =========================================================================

    # =========================================================================
    # START OF MODEL INITIALIZATION
    # =========================================================================
    if model_name == 'CNN':
        model = cnn.CNN(num_classes=4,
                        dropout=dropout,
                        kernel1=kernel1,
                        kernel2=kernel2,
                        kernel3=kernel3,
                        kernel4=kernel4,
                        pool_kernel=pool_kernel,
                        time_bins=400,
                        channels=22,
                        depth=depth,
                        scale=scale).to(device)
    elif model_name == 'CNNLSTM':
        # TODO manually adjusting for now. need to change
        depth = 22
        scale = 2
        model = cnnlstm.CNNLSTM(num_classes=4,
                                hidden_dims=hidden_dims,
                                dropout=dropout,
                                kernel1=kernel1,
                                kernel2=kernel2,
                                kernel3=kernel3,
                                kernel4=kernel4,
                                pool_kernel=pool_kernel,
                                time_bins=400,
                                channels=22,
                                depth=depth,
                                scale=scale).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # set optimizer. note: only rmsprop and sgd use momentum. i'm pretty sure
    # i need to adjust the adam family to have extra parameters.
    # TODO: check if (n)adam(ax) needs extra params
    if optimizer_name == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(),
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
    elif optimizer_name == 'NAdam':
        optimizer = torch.optim.NAdam(model.parameters(),
                                      lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay,
                                        momentum=momentum)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum)
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # =========================================================================
    # END OF MODEL INITIALIZATION
    # =========================================================================

    # im not entirely sure i need this, but it's here since it works
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False)

    # info dump
    info_dump(model_name=model_name,
              batch_size=batch_size,
              optimizer_name=optimizer_name,
              learning_rate=learning_rate,
              weight_decay=weight_decay,
              momentum=momentum,
              dropout=dropout,
              kernel1=kernel1,
              kernel2=kernel2,
              kernel3=kernel3,
              kernel4=kernel4,
              pool_kernel=pool_kernel,
              depth=depth,
              scale=scale,
              hidden_dims=hidden_dims,
              training=True)

    _, val_accuracies = train_model(model=model,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    scheduler=scheduler,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    num_epochs=num_epochs,
                                    learning=True,
                                    trial=trial)

    return max(val_accuracies)
# ==============================================================================
# END OF objective()
# ==============================================================================


# ==============================================================================
# START OF learn_hyperparameters()
# ==============================================================================
def learn_hyperparameters(X_train,
                          y_train,
                          X_valid,
                          y_valid,
                          model_name='CNN',
                          num_epochs=10,
                          trials=100):
    if model_name == 'CNN':
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.ThresholdPruner(lower=0.28, n_warmup_steps=3)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(),
                                pruner=pruner)

    study.optimize(lambda trial: objective(trial,
                                           X_train=X_train,
                                           y_train=y_train,
                                           X_valid=X_valid,
                                           y_valid=y_valid,
                                           model_name=model_name,
                                           num_epochs=num_epochs),
                   n_trials=trials)

    params = study.best_trial.params
    print(f'Number of finished trials: {len(study.trials)}')
    print('Best trial:')
    print(f'    Validation Accuracy: {study.best_trial.value}')
    info_dump(model_name=model_name,
              batch_size=params.get('batch_size'),
              optimizer_name=params.get('optimizer_name'),
              learning_rate=params.get('learning_rate'),
              weight_decay=params.get('weight_decay'),
              momentum=params.get('momentum'),
              dropout=params.get('dropout'),
              kernel1=params.get('kernel1'),
              kernel2=params.get('kernel2'),
              kernel3=params.get('kernel3'),
              kernel4=params.get('kernel4'),
              pool_kernel=params.get('pool_kernel'),
              depth=params.get('depth'),
              scale=params.get('scale'),
              hidden_dims=params.get('hidden_dims'),
              training=False)

    return params
# ==============================================================================
# END OF learn_hyperparameters()
# ==============================================================================
