import matplotlib.pyplot as plt
import seaborn as sns
import random 
import matplotlib.patches as mpatches
def plot_vanilla(data_list, min_len):

    sns.set_style("whitegrid", {'axes.grid' : True,
                                'axes.edgecolor':'black'

                                })
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    colors = ["green", ]
    labels = ["DQN", ]
    
    color_patch = []
    for color, label, data in zip(colors, labels, data_list):
        sns.lineplot( data=data, color=color, ci=95)
        color_patch.append(mpatches.Patch(color=color, label=label))
    print(min_len)
    plt.xlim([0, min_len])
    plt.xlabel('Training Episodes $(\\times10^6)$', fontsize=22)
    plt.ylabel('Average return', fontsize=22)
    lgd=plt.legend(
    frameon=True, fancybox=True, \
    prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    plt.title('Title', fontsize=14)
    ax = plt.gca()
    ax.set_xticks([100, 200, 300, 400, 500,600,700,800])
    # ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.show()
data_list = [ [x+random.randint(0,40) for x in range(1000)]]
plot_vanilla(data_list,1000)