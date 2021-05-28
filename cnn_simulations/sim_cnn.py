# Header
import sys
import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils import model_zoo
import network.conv0 as conv0
import network.myvgg as myvgg
import numpy as np
import errno
import glob
import re
import pandas
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc

def get_colors():
    return np.array([
        [0.568, 0.721, 0.741],  # economist mid-green
        [0.831, 0.866, 0.866],  # economist light-gray
        [0.290, 0.290, 0.290],  # economist dark gray
        [0.890, 0.070, 0.043],  # economist red
        [0.541, 0.733, 0.815],  # economist mid-blue
        [0.985, 0.726, 0],      # yellow
        [100/255, 100/255, 100/255],# economist dark gray
        [0, 0, 0],              # black
        [1/255, 82/255, 109/255],   # economist dark blue
        # [41/255, 176/255, 14/255]   # economist green
        [70/255, 140/255, 140/255]   # economist green
    ])

def color_bars(axes, colors):
    # Iterate over each subplot
    for ax in axes:

        # Pull out the dark and light colors for
        # the current subplot
        dark_color = colors[0]
        light_color = colors[3]
        line_color = colors[6]
        blue_color = colors[4]
        yellow_color = colors[5]
        darkgray_color = colors[6]
        darkblue_color = colors[8]
        green_color = colors[9]

        # These are the patches (matplotlib's terminology
        # for the rectangles corresponding to the bars)
        p1, p2, p3, p4 = ax.patches

        # The first bar gets the dark color
        p1.set_color(green_color)
        p1.set_edgecolor(line_color)
        p1.set_hatch('---')
        
        # The second bar gets the light color, plus
        # hatch marks int he dark color
        p2.set_color(light_color)
        p2.set_edgecolor(line_color)
        p2.set_hatch('////')

        p3.set_color(dark_color)
        p3.set_edgecolor(line_color)

        p4.set_color(darkgray_color)
        p4.set_edgecolor(line_color)
        # p4.set_hatch('----')

def get_color_palette(colors):

    dark_color = colors[0]
    light_color = colors[3]
    line_color = colors[6]
    blue_color = colors[4]
    yellow_color = colors[5]
    darkgray_color = colors[6]
    darkblue_color = colors[8]
    green_color = colors[9]
    
    palette=sns.color_palette([green_color, light_color, dark_color, darkgray_color])

    return palette

def set_plot_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    # plt.style.use(['fivethirtyeight'])
    matplotlib.rc('font', family='Times')
    # sns.set_context('paper')
    sns.set(font='serif')
    sns.set_style('white', {
        'font.family': 'serif',
        'font.serif': ['Times', 'Palatino', 'serif']
    })


def plot_multpile_exp(results, valid_value='p0', ncat=5, columns=[0, 1, 2, 3],\
                      labels=['Exp 1', 'Exp 2', 'Exp 3', 'Exp 4'], model_name='resnet', learnconv=True, continual_testing=False):
    ### Plot Same and Diff at different stops
    # sns.set(style='whitegrid')
    set_plot_style()

    ### Create barplots in a FacetGrid
    g = sns.FacetGrid(results, col='exid', col_order=columns, sharex=False)
    if continual_testing == False:
        g.map(sns.barplot, 'exid', 'avg_accu', 'condition', hue_order=['same', 'diff', 'nodiag', 'inv'])
        axes = np.array(g.axes.flat)
        color_bars(axes, get_colors()) # Set colors
        for i,ax in enumerate(axes):
            ax.set_xticks([-0.3, -0.1, 0.1, 0.3])
            ax.set_xticklabels(['Same', 'Swap', 'Shape', 'Non-shape'], rotation=90, position=[0, 0, 0, 0])
    else:
        g.map(sns.lineplot, 'epoch_id', 'avg_accu', 'condition', hue_order=['same', 'diff', 'nodiag', 'inv'],
              palette=get_color_palette(get_colors()))

    ### Set axis labels
    # labels = [str(updates[0]) + ' trials', str(updates[1]) + ' trials', str(updates[2]) + ' trials']
    axes = np.array(g.axes.flat)
    for i,ax in enumerate(axes):
        ax.hlines((100/ncat), ax.get_xlim()[0], ax.get_xlim()[1], linestyle='--', linewidth=1, colors='k') # chance line
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(labels[i])
        if i != 0:
            sns.despine(ax=axes[i], left=True)
    axes.flat[0].set_ylabel('Accuracy\n(DeepCNN)')

    ### set the figure title
    fig = plt.gcf()
    if continual_testing == False:
        fig.set_size_inches(8, 3)
    else:
        fig.set_size_inches(8, 2.3)
    fig.tight_layout()

    if learnconv == True:
        if continual_testing == False:
            out_file = os.path.join('analysis', model_name + '_' + valid_value + '_all_exp.png')
        else:
            out_file = os.path.join('analysis', model_name + '_' + valid_value + '_dynamics_all_exp.png')
    else:
        if continual_testing == False:
            out_file = os.path.join('analysis', model_name + '_frozen_' + valid_value + '_all_exp.png')
        else:
            out_file = os.path.join('analysis', model_name + '_frozen_' + valid_value + '_dynamics_all_exp.png')
    fig.savefig(out_file)


def plot_exp5(results, ncat=5, labels=['Patch', 'Segment', 'Size', 'Colour']):
    ### Plot Same and Diff at different stops
    plt.figure()
    set_plot_style()

    colors = get_colors()
    ax = sns.barplot(data=results, x='exid', y='avg_accu', color=colors[1])
    ax.set_xticklabels(labels, rotation=90, position=[0, 0, 0, 0])
    ax.hlines((100/ncat), ax.get_xlim()[0], ax.get_xlim()[1], linestyle='--', linewidth=1, colors='k') # chance line
    ax.set_xlabel("")
    ax.set_ylabel('Accuracy\n(DeepCNN)')
    sns.despine(ax=ax, left=False)

    ### set the figure title
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    fig.tight_layout()

    out_file = os.path.join('analysis', model_name + '_exp5.png')
    fig.savefig(out_file)


def plot_bars(results, results_file, ncat=5, hue=None, hue_order=None, title=None):
    ### Display stats using seaborn
    sns.set_context("paper")
    sns.set(style="white")
    sns.set(font='serif')
    sns.set(font_scale=2.5)
    sns.set_style("white",
                      {"font.family": "serif",
                       "font.serif": ["Times", "Palatino", "serif"]})
    ax = sns.barplot(x="condition", y="avg_accu", capsize=0.2, palette=("Greys"), edgecolor=".1",
                         order=["same", "diff", "nodiag", "inv"], hue=hue, hue_order=hue_order, data=results)
    # ax.hlines(10, -0.5, 2.5, linestyle='--', linewidth=3, colors='#e74c3c')
    ax.hlines((1/ncat)*100, -0.5, 3.5, linestyle='--', linewidth=3, colors='r')
    ax.set_xticklabels(["Same", "Swapped", "No Diag", "No Shape"], rotation=40, ha='right')
    # ax.set_xticklabels(["Same Loc+Col", "Same Loc", "Swapped Loc+Col", "No overlap"], rotation=40, ha="right")
    plt.ylim(0, 105)
    ax.set_ylabel("Test Accuracy")
    ax.set_xlabel("")
    ax.set_title(title)
    if hue is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:])
        art = []
        if hue == 'mask_type':
            lgd = plt.legend(loc=9, bbox_to_anchor=(0.95, 1.00), frameon=True, fancybox=True, framealpha=1, shadow=True, fontsize=20)
        else:
            lgd = plt.legend(loc=9, bbox_to_anchor=(1.15, 1.05), frameon=True, fancybox=True, framealpha=1, shadow=True, fontsize=20)
        art.append(lgd)
    else:
        art = []
    sns.despine()
    plt.gcf().set_size_inches(8,6)
    plt.tight_layout()
    plt.savefig(results_file, additional_artists=art, bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_points(results, results_file, ncat=5, xvar='pinv', hue='condition', title=None):
    ### Display stats using seaborn
    sns.set_context("paper")
    sns.set(style="white")
    sns.set(font='serif')
    sns.set(font_scale=2)
    sns.set_style("white",
                      {"font.family": "serif",
                       "font.serif": ["Times", "Palatino", "serif"]})
    results = results[results['condition'].isin(['same','diff','nodiag', 'inv'])]  # only consider these conds
    ax = sns.lineplot(x=xvar, y="avg_accu",
                          color="black", markers=True, lw=3, hue_order=["same", "diff", "nodiag", "inv"],
                          dashes=True, hue=hue, style=hue, data=results)
    ax.hlines((1/ncat)*100, ax.get_xlim()[0], ax.get_xlim()[1], linestyle='--', linewidth=3, colors='k')
    if xvar == 'pinv':
        ax.set_xlabel("$p(invalid)$ during training")
    elif xvar == 'epoch_id':
        ax.set_xlabel("Training epoch")
    plt.ylim(0, 100)
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].set_visible(False)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)

    if hue is not None:
        ### Remove title of legend
        art = []
        handles, labels = ax.get_legend_handles_labels()
        lgd = plt.legend(frameon=True, fancybox=True, framealpha=0.8,
                         handles=handles[1:], labels=labels[1:], fontsize=20)
        art.append(lgd)
    else:
        art = []
    sns.despine()
    plt.gcf().set_size_inches(7,6)
    # plt.tight_layout()
    plt.savefig(results_file, additional_artists=art, bbox_inches="tight")
    plt.clf()
    plt.close()

def load_sin_model(model_name):

    model_urls = {
            'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
            'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
    }

    if "resnet50" in model_name:
        print("Using the ResNet50 architecture.")
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name])
    elif "vgg16" in model_name:
        print("Using the VGG-16 architecture.")
       
        # download model from URL manually and save to desired location
        filepath = "./vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar"

        assert os.path.exists(filepath), "Please download the VGG model yourself from the following link and save it locally: https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be downloaded automatically like the other models)"

        model = torchvision.models.vgg16(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = torch.load(filepath)


    elif "alexnet" in model_name:
        print("Using the AlexNet architecture.")
        model = torchvision.models.alexnet(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name])
    else:
        raise ValueError("unknown model architecture.")

    model.load_state_dict(checkpoint["state_dict"])
    return model


def learn_model(model_name, model_file, results_dir, num_epochs, usetrained, learnconv, dataset, train_loader, cv_loader, seed, retrain=False, stim_start=0, stim_end=999999):
    if retrain == True:
        regime = 'retrain'
    else:
        regime = 'train'
    BASE_PATH = dataset
    path_train = os.path.join(BASE_PATH, 'train')
    path_test_cv = os.path.join(BASE_PATH, 'test_cv')

    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    ### Specify the model
    if regime == 'retrain':
        model = torch.load(model_file)
    else:
        nclasses = len(sorted(os.listdir(path_train)))
        if model_name == 'vgg':
            model = torchvision.models.vgg16(pretrained=usetrained)
        elif model_name == 'vgg_bn':
            model = torchvision.models.vgg16_bn(pretrained=usetrained)
        elif model_name == 'resnet':
            model = torchvision.models.resnet50(pretrained=usetrained)
        elif model_name == 'resnet50_trained_on_SIN':
            model = load_sin_model(model_name)
        elif model_name == 'alexnet':
            model = torchvision.models.alexnet(pretrained=usetrained)
        if usetrained == True and learnconv == False:
            if model_name == 'resnet':
                for param in model.parameters():
                    param.requires_grad = False
            elif model_name == 'resnet50_trained_on_SIN':
                for param in model.module.parameters():
                    param.requires_grad = False
        if model_name == 'vgg' or model_name == 'vgg_bn':
            # model.classifier = nn.Sequential(
            #         nn.Linear(512*7*7, 4096),
            #         nn.ReLU(True),
            #         nn.Dropout(),
            #         nn.Linear(4096, 4096),
            #         nn.ReLU(True),
            #         nn.Dropout(),
            #         nn.Linear(4096, nclasses))
            num_feat = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_feat, nclasses)
        elif model_name == 'resnet':
            num_feat = model.fc.in_features
            model.fc = nn.Linear(num_feat, nclasses)
        elif model_name == 'resnet50_trained_on_SIN':
            num_feat = model.module.fc.in_features
            model.module.fc = nn.Linear(num_feat, nclasses)
        elif model_name == 'alexnet':
            num_feat = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_feat, nclasses)

    ### Specify hyperparameters and train
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-5, weight_decay=1e-3)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    print('\nTraining...\n\tModel:{0}\n\tDataset:{1}'.format(model_file, dataset))
    conv0.train(model, train_loader, cv_loader, criterion, optimizer, scheduler, num_epochs, stim_start, stim_end)
    # conv0.train(model, train_loader, cv_loader, criterion, optimizer, num_epochs)
    torch.save(model, model_file)
    del scheduler, optimizer
    del model
    torch.cuda.empty_cache()
    # print('Trained {0} on dataset {1}'.format(model_file, dataset))


def test_model(model_file, model_name, results_dir, results_file, my_trans, seed, pinv, dataset, epoch_id=0):
    test_paths = glob.glob(os.path.join(dataset, 'test_*')) # Get all test sets based on test dirs
    results = pandas.DataFrame() ### Initialize DataFrame

    ### Load model, CUDA
    model = torch.load(model_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # send to GPU

    print('\nTesting...')
    print('\tModel:{0}'.format(model_file))
    for path_ii in test_paths:
        test_data = torchvision.datasets.ImageFolder(root=path_ii, transform=my_trans)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)

        ### Run the test
        print('\tDataset:{0}'.format(path_ii))
        [results_ii, confusion_ii] = conv0.test(model, test_loader, path_ii, device, True)

        ### Infer condition (same/diff) based on path
        condition = re.search('test_(.+)', path_ii).group(1)

        if condition == 'diff':
            print('Confusion matrix:')
            print("  {:>6} {:>6} {:>6} {:>6} {:>6}".format("S1", "S2", "S3", "S4", "S5"))
            with np.printoptions(formatter={'float': '{:6.0f}'.format}):
                print(confusion_ii)

        ### Concatenate results to DataFrame
        results_ii['condition'] = condition
        results = results.append(results_ii, ignore_index=True)

    results['pinv'] = pinv
    results['seed'] = seed
    results['model_name'] = model_name
    results['epoch_id'] = epoch_id
    if epoch_id > 1: # 0 indicates testing is not continual
        old_results = pandas.read_pickle(results_file)
        results = pandas.concat([results, old_results])
    results.to_pickle(results_file)
    del test_data, test_loader
    del model
    torch.cuda.empty_cache()


def analyse_test_set(experiments, test_set, base_dir, model_name, valid_value, learnconv, analysis_dir, num_seeds, continual_testing, exp5):
    results = pandas.DataFrame() ### Initialize DataFrame
    for exid in test_set:
        for seed in range(num_seeds):
            dataset = os.path.join(base_dir, experiments[exid][0], 'seed_' + str(seed), valid_value)
            if learnconv is True or usetrained is False:  # if not pretrained, then obviously learnall
                results_dir = os.path.join(dataset, 'results', 'learnall')
            else:
                results_dir = os.path.join(dataset, 'results', 'onlyclass')

            model_file = os.path.join(results_dir, model_name + '_seed_' + str(seed) + '.pt')
            results_file = os.path.join(results_dir, model_name + '_seed_' + str(seed) + '.df')

            results_ff = pandas.read_pickle(results_file)
            results_ff['exid'] = exid
            results = results.append(results_ff, ignore_index=True, sort=True)
    
    exp_names = [experiments[xx][2] for xx in test_set]
    if exp5 == True:
        plot_exp5(results, ncat=5, labels=['Patch', 'Segment', 'Size', 'Colour'])
    else:
        plot_multpile_exp(results, valid_value, columns=test_set, model_name=model_name, learnconv=learnconv, labels=exp_names, continual_testing=continual_testing)


def analyse_results(experiment, model_name, results_files, analysis_dir, num_seeds, nvalid, dataset,
                    continual_testing=False):
    results = pandas.DataFrame() ### Initialize DataFrame

    for fileix, file in enumerate(results_files):
        results_ff = pandas.read_pickle(results_files[fileix])
        results = results.append(results_ff, ignore_index=True, sort=True)

    fig_file = os.path.join(analysis_dir, model_name + '_' + experiment[1])
    nclasses = len(os.listdir(os.path.join(dataset, 'train')))

    if continual_testing == True:
        plot_points(results, fig_file, ncat=nclasses, xvar='epoch_id', hue='condition', title=experiment[2])
    else:
        if nvalid == 1:
            plot_bars(results, fig_file, ncat=nclasses, hue=None, title=experiment[2])
        else:
            plot_points(results, fig_file, ncat=nclasses, xvar='pinv', hue='condition', title=experiment[2])


### Global parameters
run_sims = True # if False, then only analyse (previously collected) results
exp5 = False # in Exp5, test only on Same condition as trained on only one (non-shape) cue
model_name = 'resnet' # 'vgg'; 'vgg_bn'; 'resnet'; 'resnet50_trained_on_SIN'; 'alexnet'
num_epochs = 2 # 20 number of training epochs
num_seeds = 1 #10 # number of training seeds
usetrained = True # False # whether or not to use the pretrained model
learnconv = True # whether or not convolutional layers are frozen
continual_testing = False # Track test-performance after each epoch
num_block = 20 # number of batches in one training block before testing during continuous_testing (must be < num_stims)
total_blocks = 300 
base_dir = '../datasets' #'./data/manuscript_sims'
experiments = {
    1: ['stim_dpatch_invalid_jit3', 'dpatch_invalid_jit3.png', 'Patch'],
    2: ['stim_dcol_invalid_jit3', 'dcol_invalid_jit3.png', 'Segment'],
    3: ['stim_dcell_size_invalid_jit3', 'dcell_size_invalid_jit3.png', 'Size'],
    4: ['stim_dunicol_jit3', 'dunicol_jit3.png', 'Colour'],
}
test_set = [1, 2, 3, 4]  # experiments to test
my_trans = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


if run_sims:
    for exid in test_set:
        all_results_files = []
        for seed in range(num_seeds):
            valid_values = sorted(os.listdir(os.path.join(base_dir, experiments[exid][0], 'seed_' + str(seed))))
            nvalid = len(valid_values)
            for vv in valid_values:
                dataset = os.path.join(base_dir, experiments[exid][0], 'seed_' + str(seed), vv)
                pinv = vv

                ### Load the dataset
                ### Loading here instead of inside learn_model for continutal_testing
                ### Each epoch then goes through part of test set, without shuffling each time
                path_train = os.path.join(dataset, 'train')
                path_test_cv = os.path.join(dataset, 'test_cv')
                train_data = torchvision.datasets.ImageFolder(root=path_train, transform=my_trans)
                train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
                cv_data = torchvision.datasets.ImageFolder(root=path_test_cv, transform=my_trans)
                cv_loader = torch.utils.data.DataLoader(cv_data, batch_size=32, shuffle=True, num_workers=2)

                ### define model_file and results_dir
                if learnconv is True or usetrained is False:  # if not pretrained, then obviously learnall
                    results_dir = os.path.join(dataset, 'results', 'learnall')
                else:
                    results_dir = os.path.join(dataset, 'results', 'onlyclass')

                model_file = os.path.join(results_dir, model_name + '_seed_' + str(seed) + '.pt')
                results_file = os.path.join(results_dir, model_name + '_seed_' + str(seed) + '.df')
                all_results_files.append(results_file)

                if continual_testing == False:
                    learn_model(model_name=model_name, model_file=model_file, results_dir=results_dir,
                                num_epochs=num_epochs, usetrained=usetrained, learnconv=learnconv,
                                dataset=dataset, train_loader=train_loader, cv_loader=cv_loader, seed=seed, retrain=False)
                    gc.collect() # force garbage collection to avoid "out-of-memory" errors
                    test_model(model_file=model_file, model_name=model_name, results_file=results_file,
                            results_dir=results_dir, my_trans=my_trans, seed=seed, pinv=pinv,
                            dataset=dataset)
                else:
                    # Learn 1 epoch on a new model and test
                    learn_model(model_name=model_name, model_file=model_file, results_dir=results_dir,
                                num_epochs=1, usetrained=usetrained, learnconv=learnconv,
                                dataset=dataset, train_loader=train_loader, cv_loader=cv_loader, seed=seed, retrain=False,
                                stim_start=0, stim_end=num_block)
                    gc.collect()  # force garbage collection to avoid "out-of-memory" errors
                    test_model(model_file=model_file, model_name=model_name, results_file=results_file,
                            results_dir=results_dir, my_trans=my_trans, seed=seed, pinv=pinv,
                            dataset=dataset, epoch_id=1)
                    # Learn remaining epochs on existing model and test after each epoch
                    start_count = 0
                    for epoch_id in range(num_epochs-1):
                        start_count += num_block
                        start_count = start_count % total_blocks # reset to 0 for multiple epochs
                        learn_model(model_name=model_name, model_file=model_file, results_dir=results_dir,
                                    num_epochs=1, usetrained=usetrained, learnconv=learnconv,
                                    dataset=dataset, train_loader=train_loader, cv_loader=cv_loader, seed=seed, retrain=True,
                                    stim_start=start_count, stim_end=start_count+num_block)
                        gc.collect()  # force garbage collection to avoid "out-of-memory" errors
                        test_model(model_file=model_file, model_name=model_name, results_file=results_file,
                                results_dir=results_dir, my_trans=my_trans, seed=seed, pinv=pinv,
                                dataset=dataset, epoch_id=epoch_id+2)
                del train_data, train_loader, cv_data, cv_loader
        # analyse_results(experiment=experiments[exid], model_name=model_name, results_files=all_results_files,
        #                 analysis_dir=analysis_dir, num_seeds=num_seeds, nvalid=nvalid, dataset=dataset,
        #                 continual_testing=continual_testing)
        gc.collect()

analysis_dir = os.path.join('.', 'analysis')
if exp5 == True:
    valid_value = 'p100'
else:
    valid_value = 'p20'
analyse_test_set(experiments, test_set, base_dir, model_name, valid_value, learnconv, analysis_dir, num_seeds, continual_testing, exp5)