import argparse
import torch
from torch.autograd import Variable
from torch import nn
from seq2seq.util.checkpoint import Checkpoint
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import scipy
from mpl_toolkits.axes_grid1 import AxesGrid
import os
import matplotlib.cm as cm

class Model(object):
    def __init__(self, model, title):
        self.model = model
        params = {}
        self.title = title
        for name, param in self.model.named_parameters():
            # extract cell internal weights

            #TODO: replace with isinstanceof
            if self.title == 'GRU':
                if 'weight_ih_l0' in name:
                    w_ir, w_iz, w_in = param.chunk(3, 0)
                    params[name.split('_')[0] +'_' +'w_ir']=pd.DataFrame(w_ir.data.numpy())
                    params[name.split('_')[0] +'_' +'w_iz']=pd.DataFrame(w_iz.data.numpy())
                    params[name.split('_')[0] +'_' +'w_in']=pd.DataFrame(w_in.data.numpy())

                elif 'weight_hh_l0' in name:
                    w_hr, w_hz, w_hn = param.chunk(3, 0)
                    params[name.split('_')[0] +'_' +'w_hr']=pd.DataFrame(w_hr.data.numpy())
                    params[name.split('_')[0] +'_' +'w_hz']=pd.DataFrame(w_hz.data.numpy())
                    params[name.split('_')[0] +'_' +'w_hn']=pd.DataFrame(w_hn.data.numpy())
                else:
                    param = param.data.numpy()
                    params[name]=pd.DataFrame(param)
            elif self.title == 'LSTM':
                if 'weight_ih_l0' in name:
                    w_iz, w_if, w_ic, w_io = param.chunk(4, 0)
                    params[name.split('_')[0] +'_' +'w_iz']=pd.DataFrame(w_iz.data.numpy())
                    params[name.split('_')[0] +'_' +'w_if']=pd.DataFrame(w_if.data.numpy())
                    params[name.split('_')[0] +'_' +'w_ic']=pd.DataFrame(w_ic.data.numpy())
                    params[name.split('_')[0] +'_' +'w_io']=pd.DataFrame(w_io.data.numpy())

                elif 'weight_hh_l0' in name:
                    w_hz, w_hf, w_hc, w_ho = param.chunk(4, 0)
                    params[name.split('_')[0] +'_' +'w']=pd.DataFrame(w_hz.data.numpy())
                    params[name.split('_')[0] +'_' +'w_hf']=pd.DataFrame(w_hf.data.numpy())
                    params[name.split('_')[0] +'_' +'w_hc']=pd.DataFrame(w_hc.data.numpy())
                    params[name.split('_')[0] +'_' +'w_ho']=pd.DataFrame(w_ho.data.numpy())
                else:
                    param = param.data.numpy()
                    params[name]=pd.DataFrame(param)

        self.params = params

    def get_param_names(self):
        return [name for name, _ in self.model.named_parameters()]

    def get_modules(self):
        return [mod for mod in self.model.modules()]

    def get_param_by_name(self, name):
        return pd.DataFrame(self.params[name])

    def heatmap(self):
        return {k: sns.heatmap(v) for k, v in self.params.items()}

    def apply_mean(self):
        return {k: np.ravel(v).mean() if v.shape != (1,1) else np.NaN for k, v in self.params.items()}

    def apply_std(self):
        return {k: np.ravel(v).std() if v.shape != (1,1) else np.NaN for k, v in self.params.items()}

    def apply_min(self):
        return {k: np.ravel(v).min() for k, v in self.params.items()}

    def apply_max(self):
        return {k: np.ravel(v).max() for k, v in self.params.items()}

    def apply_norm(self):
        return {k: np.linalg.norm(np.ravel(v)) if v.shape != (1,1) else np.NaN for k, v in self.params.items()}

    def param_to_dist(self,name):
        data = pd.DataFrame(np.ravel(self.params[name]))
        #data[data >]
        # data.append(np.ravel(data).quantile(0.8))
#         # best fit of data
#         (mu, sigma) = norm.fit(data)

#         # the histogram of the data
#         n, bins, patches = plt.hist(data, 20, normed=1)

#         # add a 'best fit' line
#         y = mlab.normpdf( bins, mu, sigma)
#         l = plt.plot(bins, y, 'r--', linewidth=2)
#        return scipy.stats.norm(mu, sigma)
        return np.histogram(data, bins=50, range=[-1, 1], density=True)



class Models(object):
    def mean_of(self, data):
        one_key = list(data.keys())[0]
        return {param: np.mean([data[name][param] for name in data.keys()]) for param in data[one_key].keys()}

    def load_models(self):
        models = {}
        files = os.listdir(self.model_path)
        for file in files:
            if not file.startswith('.'):
                print('loading: ', self.model_path + '/' + file)
                checkpoint = Checkpoint.load(self.model_path + '/' + file)
                seq2seq = checkpoint.model
                title = self.title.split('_')[1]
                models[file] = Model(seq2seq, title)
        return models

    def __init__(self, model_path,title):

        self.image_folder = 'images/'
        self.title = title
        self.model_path = model_path

        self.models = self.load_models()

        ## calculate mean, std, norm
        self.means = {name: model.apply_mean() for name,model in self.models.items()}
        self.stds = {name : model.apply_std() for name,model in self.models.items()}
        self.norms = {name: model.apply_norm() for name, model in self.models.items()}
        self.mins = {name: model.apply_min() for name, model in self.models.items()}
        self.maxs = {name: model.apply_max() for name, model in self.models.items()}

        ## caluclate mean of means, stds, norms
        self.mean_of_means = self.mean_of(self.means)
        self.mean_of_stds = self.mean_of(self.stds)
        self.mean_of_norms = self.mean_of(self.norms)
        self.maxs = self.mean_of(self.maxs)
        self.mins = self.mean_of(self.mins)

        # fill data into df
        df = pd.DataFrame.from_dict(self.mean_of_means,  orient='index')
        df = df.rename(columns={0: 'mean of means'})
        df['mean of stds'] = self.mean_of_stds.values()
        df['mean of norms'] = self.mean_of_norms.values()
        df['mean of maxs'] = self.maxs.values()
        df['mean of mins'] = self.mins.values()
        self.stats = df


    def apply_heatmap(self):
        for model_name, model in self.models.items():
            for param_name in self.models[model_name].params.keys():
                plt.figure()
                sns.heatmap(model.params[param_name], vmin=-1, vmax=1, cmap=cm.seismic)
                plt.xlabel('Column')
                plt.ylabel('Row')
                plt.title(self.title +' model ' + str(model_name) + ' '+param_name)
                plt.tight_layout()
                plt.savefig(self.image_folder +'heatmap/'+ self.title + '_' + model_name + '_' + param_name + '.png', dpi=300)
                plt.show()

    def apply_heatmap_by_name(self,param_name):
        for model_name, model in self.models.items():
                plt.figure()
                sns.heatmap(model.params[param_name], vmin=-1, vmax=1, cmap=cm.seismic)
                plt.xlabel('Column')
                plt.ylabel('Row')
                plt.title(self.title +' model ' + str(model_name) + ' '+param_name)
                plt.tight_layout()
                plt.savefig(self.image_folder +'heatmap/'+ self.title + '_' + model_name + '_' + param_name + '.png', dpi=300)
                plt.show()




class Analysis(object):

    def intersection(self, hist_1, hist_2):
        minima = np.minimum(hist_1, hist_2)
        intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
        return intersection

    def bhattacharyya(self, h1, h2):
      '''Calculates the Byattacharyya distance of two histograms.'''
      def normalize(h):
        return h / np.sum(h)
      return 1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2))))

    def KL(self,dist_1, dist_2):
        x = np.linspace(-1, 1, 100)
        return scipy.stats.entropy(dist_1.pdf(x),dist_2.pdf(x))

    def compare_distributions(self):
        dist = {}
        for model_name_base in self.models_baseline.keys():
            for model_name_guided in self.models_guided.keys():
                per_model = {}
                for param in self.models_baseline[list(self.models_baseline.keys())[0]].params.keys():
                    if not self.models_baseline[model_name_base].params[param].shape == (1,1):
                        dist_1, _ = self.models_baseline[model_name_base].param_to_dist(param)
                        dist_2, _ = self.models_guided[model_name_guided].param_to_dist(param)
                        per_model[param] = self.bhattacharyya(dist_1, dist_2)
                key = model_name_base + '_' + model_name_guided
                dist[key] = per_model
        return pd.DataFrame.from_dict(dist, orient='index')

    def compare_dist_within(self, models):
        dist = {}
        for param in models[list(models.keys())[0]].params.keys():
            within_model = {}
            for model_name_base in models.keys():
                for model_name_guided in models.keys():
                    if (model_name_base != model_name_guided) and (not models[model_name_base].params[param].shape == (1,1)):
                        dist_1,_ = models[model_name_base].param_to_dist(param)
                        dist_2,_ = models[model_name_guided].param_to_dist(param)
                        key = model_name_base + '_' + model_name_guided
                        within_model[key] = self.bhattacharyya(dist_1, dist_2)
                dist[param] = within_model
        return pd.DataFrame.from_dict(dist, orient='index')

    def __init__(self, models_baseline, models_guided):
        self.models_baseline = models_baseline.models
        self.models_guided = models_guided.models
        self.title_baseline = models_baseline.title
        self.title_guided = models_guided.title
        self.image_path = 'images/'
        self.dist = self.compare_distributions()
        self.dist_within_baseline = self.compare_dist_within(self.models_baseline)
        self.dist_within_guided = self.compare_dist_within(self.models_guided)

    def compare_dist_by_name(self,param):
        dist = {}
        for model_name in self.models_baseline.keys():
            assert(self.models_baseline[model_name].params[param].shape != (1,1))
            dist_1 = self.models_baseline[model_name].param_to_dist(param)
            dist_2 = self.models_guided[model_name].param_to_dist(param)
            dist[param] = self.bhattacharyya(dist_1, dist_2)
        return dist

    def plot_dist(self,param):
        for i in range(1,len(self.models_guided.keys())+1):
            plt.figure()
            plt.title(param + ' model ' +  str(i))
            plt.ylabel('Frequency')
            plt.xlabel('Weight value')
            plt.hist(np.ravel(self.models_baseline[str(i)].params[param]), bins=50,range=[-1, 1], density=True, alpha=0.4, label=self.title_baseline)
            plt.hist(np.ravel(self.models_guided[str(i)].params[param]), bins=50,range=[-1, 1], density=True, alpha=0.4,label=self.title_guided)
            plt.legend()
            plt.savefig(self.image_path+ 'distribution/hist_' + self.title_baseline + '_'+ param + '_' + str(i)+'.png', dpi=300 )
            plt.show()
