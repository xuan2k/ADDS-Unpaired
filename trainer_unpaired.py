from __future__ import absolute_import, division, print_function

import numpy as np
import time
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
import random

from torchsummary import summary

STEREO_SCALE_FACTOR = 5.4

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class OrthoLoss(nn.Module):

    def __init__(self):
        super(OrthoLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2 = input1
        input2_l2 = input2
        
        ortho_loss = 0
        dim = input1.shape[1]
        for i in range(input1.shape[0]):
            ortho_loss += torch.mean(((input1_l2[i:i+1,:].mm(input2_l2[i:i+1,:].t())).pow(2))/dim)

        ortho_loss = ortho_loss / input1.shape[0]

        return ortho_loss

class L_exp_z(nn.Module):
    def __init__(self, patch_size):
        super(L_exp_z, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, mean_val):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class TrainerUnpaired:
    def __init__(self, options):
        self.setup_seed(20)
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.result_path = os.path.join(self.log_path, 'result.txt')

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.only_depth_encoder:
            self.opt.frame_ids = [0]

        if not self.opt.shared_encoder:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
        else:
            self.models["encoder"] = networks.ResnetEncoderShared(
                self.opt.num_layers, self.opt.weights_init == "pretrained")

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.opt.pseudo_model:
            self.pretrained_models = {}
            self.pretrained_models["encoder"] = networks.PretrainedResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.pretrained_models["depth"] = networks.PretrainedDepthDecoder(
                self.pretrained_models["encoder"].num_ch_enc, self.opt.scales)

            self.opt.pseudo_model = os.path.expanduser(self.opt.pseudo_model)

            assert os.path.isdir(self.opt.pseudo_model), \
                "Cannot find folder {}".format(self.opt.pseudo_model)
            print("Loading depth model from: {}".format(self.opt.pseudo_model))

            encoder_path = os.path.join(self.opt.pseudo_model, "encoder.pth")
            decoder_path = os.path.join(self.opt.pseudo_model, "depth.pth")

            encoder_dict = torch.load(encoder_path)
            model_dict = self.pretrained_models["encoder"].state_dict()
            self.pretrained_models["encoder"].encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            self.pretrained_models["depth"].load_state_dict(torch.load(decoder_path))

            for m in self.pretrained_models.values():
                m.eval()


        if not self.opt.only_depth_encoder:
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
            self.models["depth"].to(self.device)
            self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net and not self.opt.only_depth_encoder:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoderPose(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
        
        if self.opt.light_enhance:
            self.lightnet = networks.LightNet()
            self.lightnet.train()
            self.lightnet.to(self.device)
            self.parameters_to_train += list(self.lightnet.parameters())

        self.discriminator = {}

        if self.opt.feature_disc:
            if self.opt.num_discriminator == 0:
                self.discriminator["domain_classifier"] = networks.FeatureClassifier(batch_size=self.opt.batch_size)
                self.discriminator["domain_classifier"].train()
                self.discriminator["domain_classifier"].to(self.device)
                self.parameters_to_train_D = list(self.discriminator["domain_classifier"].parameters())
            else:
                num_ch_enc = np.flip(self.models["encoder"].num_ch_enc)
                self.parameters_to_train_D = []
                for i_layer in range(self.opt.num_discriminator):
                    self.discriminator["domain_classifier_{}".format(i_layer)] = \
                        networks.NLayerDiscriminator(num_ch_enc[i_layer])
                    self.discriminator["domain_classifier_{}".format(i_layer)].train()
                    self.discriminator["domain_classifier_{}".format(i_layer)].to(self.device)
                    self.parameters_to_train_D += list(self.discriminator["domain_classifier_{}".format(i_layer)].parameters())
        else:
            self.discriminator["domain_classifier"] = networks.FCDiscriminator(num_classes=1)
            self.discriminator["domain_classifier"].train()
            self.discriminator["domain_classifier"].to(self.device)
            self.parameters_to_train_D = list(self.discriminator["domain_classifier"].parameters())


        if self.opt.predictive_mask and not self.opt.only_depth_encoder:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, weight_decay=0.0005)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        
        self.model_optimizer_D = optim.Adam(self.parameters_to_train_D, self.opt.learning_rate, weight_decay=0.0005)
        self.model_lr_scheduler_D = optim.lr_scheduler.StepLR(
            self.model_optimizer_D, self.opt.scheduler_step_size, 0.1)
       
        if self.opt.load_weights_folder is not None:
            self.load_model()
        
        if self.opt.load_depth_weights is not None:
            self.load_depth_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDatasetUnpaired}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_day_filenames = readlines(fpath.format("train_day"))
        train_night_filenames = readlines(fpath.format("train_night"))
        val_day_filenames = readlines(fpath.format("val_day"))
        val_night_filenames = readlines(fpath.format("val_night"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_day_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        train_day_dataset = self.dataset(self.opt,
            self.opt.data_path, train_day_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_day_loader = DataLoader(
            train_day_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        train_night_dataset = self.dataset(self.opt,
            self.opt.data_path, train_night_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_night_loader = DataLoader(
            train_night_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_day_dataset = self.dataset(self.opt,
            self.opt.data_path, val_day_filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext='.png')
        self.val_day_loader = DataLoader(
            val_day_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        self.val_iter_day = iter(self.val_day_loader)

        val_night_dataset = self.dataset(self.opt,
            self.opt.data_path, val_night_filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext='.png')
        self.val_night_loader = DataLoader(
            val_night_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        self.val_iter_night = iter(self.val_night_loader)

        self.writers = {}
        for mode in ["train", "val_day", "val_night"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} all validation items\n".format(
            len(train_day_dataset) + len(train_night_dataset), len(val_day_dataset) + len(val_night_dataset)))

        self.save_opts()

        self.loss_ortho = OrthoLoss().cuda()
        self.loss_recon1 = MSE().cuda()
        self.loss_recon2 = SIMSE().cuda()
        self.loss_exp_z = L_exp_z(32)
        self.loss_TV = L_TV()
        self.domain_depth_loss = torch.nn.MSELoss().cuda()
        self.domain_feat_loss = torch.nn.MSELoss().cuda()
        self.target_label = 1
        self.source_label = 0

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        best_absrel = best_sqrel = best_rmse = best_rmse_log = np.inf
        best_a1 = best_a2 = best_a3 = 0
        best_epoch = 0
        
        with open(self.result_path, 'a') as f: 
            f.write("abs_rel \t sq_rel \t rmse \t rmse_log \t a1 \t a2 \t a3") 
            f.write("\n")        
        f.close()
        
        for self.epoch in range(self.opt.num_epochs):
            with open(self.result_path, 'a') as f: 
                f.write('epoch: ' +str(self.epoch)) 
                f.write("\n")        
            f.close()
            if not self.opt.only_depth_encoder:
                mean_errors_day, mean_errors_night, mean_errors_all = self.run_epoch()
                if (self.epoch + 1) % self.opt.save_frequency == 0:
                    self.save_model()
                    if self.opt.light_enhance:
                        torch.save(self.lightnet.state_dict(), os.path.join(self.opt.log_path, 'lightnet_' +str(self.epoch) + '.pth'))

                mean_errors = []
                if best_rmse > mean_errors_all[2]:
                    best_epoch = self.epoch
                    best_absrel = mean_errors_all[0]
                    best_sqrel = mean_errors_all[1]
                    best_rmse = mean_errors_all[2]
                    best_rmse_log = mean_errors_all[3]
                    best_a1 = mean_errors_all[4]
                    best_a2 = mean_errors_all[5]
                    best_a3 = mean_errors_all[6]
                mean_errors.append(best_absrel)
                mean_errors.append(best_sqrel)
                mean_errors.append(best_rmse)
                mean_errors.append(best_rmse_log)
                mean_errors.append(best_a1)
                mean_errors.append(best_a2)
                mean_errors.append(best_a3)
                print('best results is %d epoch:' % best_epoch)
                print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                print(("&{: 8.3f}  " * 7).format(*mean_errors) + "\\\\")
                print("\n-> Done!")

                with open(self.result_path, 'a') as f:
                    f.write('best results is %d epoch:' % best_epoch)
                    for i in range(len(mean_errors)):
                        f.write(str(mean_errors[i]))  #
                        f.write('\t')
                    f.write("\n")

                f.close()
            else:
                output_day, output_night = self.run_epoch()
                if (self.epoch + 1) % self.opt.save_frequency == 0:
                    self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        # self.model_lr_scheduler.step()
        # self.model_lr_scheduler_D.step()
        print("Training")
        self.set_train()
        day_iterator = iter(self.train_day_loader)
        for batch_idx, night_inputs in enumerate(self.train_night_loader):
            try:
                day_inputs = next(day_iterator)
            except StopIteration:
                day_iterator = iter(self.train_day_loader)
                day_inputs = next(day_iterator)

            before_op_time = time.time()

            outputs_day, outputs_night, losses, log_losses, losses_day, losses_night, domain_loss_D, domain_loss_G = self.process_batch(day_inputs, night_inputs)

            self.model_optimizer.zero_grad()
            self.model_optimizer_D.zero_grad()

            if self.opt.only_depth_encoder:
                losses.backward()
            else:
                loss_G = losses + losses_day["loss"] + losses_night["loss"] + domain_loss_G
                if self.opt.feature_disc and self.opt.num_discriminator > 0:
                    for i_layer in range(self.opt.num_discriminator):
                        for param in self.discriminator["domain_classifier_{}".format(i_layer)].parameters():
                            param.requires_grad = False
                else:
                    for param in self.discriminator["domain_classifier"].parameters():
                        param.requires_grad = False
                loss_G.backward()
                
                if self.opt.feature_disc and self.opt.num_discriminator > 0:
                    for i_layer in range(self.opt.num_discriminator):
                        for param in self.discriminator["domain_classifier_{}".format(i_layer)].parameters():
                            param.requires_grad = True
                else:
                    for param in self.discriminator["domain_classifier"].parameters():
                        param.requires_grad = True

                domain_loss_D.backward()

            self.model_optimizer.step()
            self.model_optimizer_D.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            # late_phase = self.step % 2000 == 0

            # if early_phase or late_phase:
            if batch_idx % 100 == 0 and not self.opt.only_depth_encoder:
                self.log_time(batch_idx, duration, losses_day["loss"].cpu().data, \
                              losses_night["loss"].cpu().data, \
                              domain_loss_D, \
                              domain_loss_G, \
                              losses.cpu().data)  
                # self.log_time(batch_idx, duration, losses.cpu().data)

                if "depth_gt" in day_inputs:
                    self.compute_depth_losses(day_inputs, outputs_day, losses_day)
                if "depth_gt" in night_inputs:
                    self.compute_depth_losses(night_inputs, outputs_night, losses_night)

            if batch_idx % 100 == 0 and self.opt.only_depth_encoder:
                print("\n  " + ("{:>8} | " * 8).format("diff_day", "diff_night", "recon_day1", "recon_day2",
                                                       "recon_night1", "recon_night2", "similarity", "loss_all"))
                outputs_day.append(losses)
                print(("&{: 8.3f}  " * 8).format(*outputs_day) + "\\\\")

            # break
                    
        if not self.opt.only_depth_encoder:
            self.log("train", log_losses)
            self.step += 1

            mean_errors_day = self.evaluate('day')
            mean_errors_night = self.evaluate('night')
            mean_errors_all = (mean_errors_day + mean_errors_night) /2
            print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_errors_all.tolist()) + "\\\\")
            print("\n-> Done!")
        
            with open(self.result_path, 'a') as f:
                for i in range(len(mean_errors_all)):
                    f.write(str(mean_errors_all[i])) #
                    f.write('\t')
                f.write("\n")
  
            f.close()

            return mean_errors_day, mean_errors_night, mean_errors_all
        else:
            outputs1 = self.val_only_encoder(self.val_day_loader)
            print("\n  " + ("{:>8} | " * 8).format("diff_day", "diff_night", "recon_day1", "recon_day2", "recon_night1", "recon_night2", "similarity","loss_all"))
            print(outputs1)
            outputs2 = self.val_only_encoder(self.val_night_loader)
            print(outputs2)

            with open(self.result_path, 'a') as f:
                f.write(str(outputs1))  #
                f.write('\t')
                f.write("\n")
                f.write(str(outputs2))  #
                f.write('\t')
                f.write("\n")
            f.close()

            return outputs1, outputs2

    def process_batch(self, day_inputs, night_inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in day_inputs.items():
            day_inputs[key] = ipt.to(self.device)
        
        for key, ipt in night_inputs.items():
            night_inputs[key] = ipt.to(self.device)

        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        if self.opt.light_enhance:
            loss_enhance = 0
            day_img = day_inputs["color_aug", 0, 0].to(self.device)
            night_img = night_inputs["color_aug", 0, 0].to(self.device)
            mean_light = night_img.mean()
            r = self.lightnet(day_img)
            day_enhance = day_img + r
            loss_enhance_day = 10 * self.loss_TV(r) + torch.mean(self.ssim(day_enhance, day_img)) \
                            + torch.mean(self.loss_exp_z(day_enhance, mean_light))
            r = self.lightnet(night_img)
            night_enhance = night_img + r
            loss_enhance_night = 10 * self.loss_TV(r) + torch.mean(self.ssim(night_enhance, night_img)) \
                            + torch.mean(self.loss_exp_z(night_enhance, mean_light))
            loss_enhance = loss_enhance_day + loss_enhance_night
            day_features, result_day = self.models["encoder"](day_enhance, 'day', 'train')
            night_features, result_night = self.models["encoder"](night_enhance, 'night', 'train')
        else:
            day_features, result_day = self.models["encoder"](day_inputs["color_aug", 0, 0], 'day', 'train')
            night_features, result_night = self.models["encoder"](night_inputs["color_aug", 0, 0], 'night', 'train')

        if not self.opt.only_depth_encoder:
            day_outputs = self.models["depth"](day_features)
            night_outputs = self.models["depth"](night_features)

        if self.opt.predictive_mask and not self.opt.only_depth_encoder:
            day_outputs["predictive_mask"] = self.models["predictive_mask"](day_features)
            night_outputs["predictive_mask"] = self.models["predictive_mask"](night_features)

        if self.use_pose_net and not self.opt.only_depth_encoder:
            domain_loss_D = 0
            domain_loss_G = 0
            if self.opt.feature_disc:
                if self.opt.num_discriminator == 0:
                    D_loss, G_loss = 0, 0
                    day_pred = day_features[-1]
                    night_pred = night_features[-1]
                    predict_day = self.discriminator["domain_classifier"](day_pred)
                    predict_night = self.discriminator["domain_classifier"](night_pred)

                    # day = 1, night = 0
                    label_source = torch.FloatTensor(predict_day.data.size()).fill_(self.source_label).to(self.device)
                    label_target = torch.FloatTensor(predict_night.data.size()).fill_(self.target_label).to(self.device)
                    G_loss = self.domain_feat_loss(predict_day, label_source)
                    G_loss += self.domain_feat_loss(predict_night, label_target)

                    day_pred = day_features[-1].detach()
                    night_pred = night_features[-1].detach()
                    predict_day = self.discriminator["domain_classifier"](day_pred)
                    predict_night = self.discriminator["domain_classifier"](night_pred)

                    D_loss = self.domain_feat_loss(predict_day, label_target)
                    D_loss += self.domain_feat_loss(predict_night, label_source)

                    domain_loss_D += D_loss
                    domain_loss_G += G_loss
                else:
                    for i_layer in range(self.opt.num_discriminator):
                        D_loss, G_loss = 0, 0
                        day_pred = night_features[-(i_layer + 1)]
                        night_pred = day_features[-(i_layer + 1)]
                        predict_day = self.discriminator["domain_classifier_{}".format(i_layer)](day_pred)
                        predict_night = self.discriminator["domain_classifier_{}".format(i_layer)](night_pred)
                        
                        # day = 1, night = 0
                        label_source = torch.FloatTensor(predict_day.data.size()).fill_(self.source_label).to(self.device)
                        label_target = torch.FloatTensor(predict_night.data.size()).fill_(self.target_label).to(self.device)
                        G_loss = self.domain_feat_loss(predict_day, label_source)
                        G_loss += self.domain_feat_loss(predict_night, label_target)

                        day_pred = day_features[-(i_layer + 1)].detach()
                        night_pred = night_features[-(i_layer + 1)].detach()
                        predict_day = self.discriminator["domain_classifier_{}".format(i_layer)](day_pred)
                        predict_night = self.discriminator["domain_classifier_{}".format(i_layer)](night_pred)

                        D_loss = self.domain_feat_loss(predict_day, label_target)
                        D_loss += self.domain_feat_loss(predict_night, label_source)

                        domain_loss_D += (self.opt.num_discriminator - i_layer) * D_loss
                        domain_loss_G += G_loss

                    domain_loss_D /= sum(range(0, self.opt.num_discriminator))
            
            day_outputs.update(self.predict_poses(day_inputs, day_features))
            night_outputs.update(self.predict_poses(night_inputs, night_features))

            self.generate_images_pred(day_inputs, day_outputs)
            self.generate_images_pred(night_inputs, night_outputs)

            if not self.opt.feature_disc:
                if self.opt.pseudo_model:
                    day_feats = self.pretrained_models["encoder"](day_inputs["color_aug", 0, 0])
                    day_outs = self.pretrained_models["depth"](day_feats)
                    day_pred = day_outs[('disp', 0)].detach()
                else:
                    day_pred = day_outputs[('disp', 0)].detach()
                    
                night_pred = night_outputs[('disp', 0)].detach()
                predict_day = self.discriminator["domain_classifier"](day_pred)
                predict_night = self.discriminator["domain_classifier"](night_pred)

                D_loss, G_loss = 0, 0
                # day = 1, night = 0
                label_source = torch.FloatTensor(predict_day.data.size()).fill_(self.source_label).to(self.device)
                label_target = torch.FloatTensor(predict_night.data.size()).fill_(self.target_label).to(self.device)
                G_loss = self.domain_depth_loss(predict_day, label_source)
                G_loss += self.domain_depth_loss(predict_night, label_target)

                if self.opt.pseudo_model:
                    day_feats = self.pretrained_models["encoder"](day_inputs["color_aug", 0, 0])
                    day_outs = self.pretrained_models["depth"](day_feats)
                    day_pred = day_outs[('disp', 0)].detach()
                else:
                    day_pred = day_outputs[('disp', 0)].detach()

                night_pred = night_outputs[('disp', 0)].detach()
                predict_day = self.discriminator["domain_classifier"](day_pred)
                predict_night = self.discriminator["domain_classifier"](night_pred)

                D_loss = self.domain_depth_loss(predict_day, label_target)
                D_loss += self.domain_depth_loss(predict_night, label_source)

                domain_loss_D += D_loss
                domain_loss_G += 0.1 * G_loss
            
            losses_day = self.compute_losses(day_inputs, day_outputs)
            losses_night = self.compute_losses(night_inputs, night_outputs)
                
        loss = 0
        losses = {}
        losses["ortho"] = 0
        losses["day_feat"] = result_day[0].sum()
        losses["night_feat"] = result_night[0].sum()
        # ortho
        target_ortho1 = 0.5 * self.loss_ortho(result_day[0], result_day[2])  # 10 when batchsize=1
        target_ortho2 = 0.5 * self.loss_ortho(result_night[0], result_night[2])
        losses["ortho"] += target_ortho1
        losses["ortho"] += target_ortho2
        loss += target_ortho1
        loss += target_ortho2
        
        target_ortho3 = 1 * self.loss_ortho(result_day[1], result_day[3])  # 10 when batchsize=1
        target_ortho4 = 1 * self.loss_ortho(result_night[1], result_night[3])
        losses["ortho"] += target_ortho3
        losses["ortho"] += target_ortho4
        loss += target_ortho3
        loss += target_ortho4

        losses["recons_day"] = 0
        losses["recons_night"] = 0
        # recon
        target_mse = 0.1 * self.loss_recon1(result_day[5], day_inputs["color_aug", 0, 0])
        loss += target_mse
        target_simse = 0.1 * self.loss_recon2(result_day[5], day_inputs["color_aug", 0, 0])
        loss += target_simse
        losses["recons_day"] += target_mse
        losses["recons_day"] += target_simse
        target_mse_night = 0.1 * self.loss_recon1(result_night[5], night_inputs["color_aug", 0, 0])
        loss += target_mse_night
        target_simse_night = 0.1 * self.loss_recon2(result_night[5], night_inputs["color_aug", 0, 0])
        loss += target_simse_night
        losses["recons_night"] += target_mse_night
        losses["recons_night"] += target_simse_night

        losses["day_depth"] = losses_day["loss"]
        losses["night_depth"] = losses_night["loss"]

        if self.opt.pseudo_model:
            losses["pretrained_sim"] = self.loss_recon1(day_outs[('disp', 0)], day_outputs[('disp', 0)])
            loss += losses["pretrained_sim"]

        if self.opt.light_enhance:
            loss += loss_enhance

        if self.opt.only_depth_encoder:
            return losses, loss
        else:
            return day_outputs, night_outputs, loss, losses, losses_day, losses_night, domain_loss_D, domain_loss_G

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


    def val_only_encoder(self, val_loader):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        with torch.no_grad():
            loss_avg = 0
            batch = 0
            for batch_idx, inputs in enumerate(val_loader):
                outputs, losses = self.process_batch(inputs)

                if "depth_gt" in inputs and not self.opt.only_depth_encoder:
                    self.compute_depth_losses(inputs, outputs, losses)
                loss_avg += losses
                batch += 1
        self.set_train()
        return loss_avg / batch

    def evaluate(self, split='day'):
        """Evaluates a pretrained model using a specified test set
        """
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        self.set_eval()

        assert sum((self.opt.eval_mono, self.opt.eval_stereo)) == 1, \
            "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"


        pred_disps = []
        gt = []
        print("-> Computing predictions with size {}x{}".format(
            self.opt.width, self.opt.height))

        if split=='day':
            dataloader = self.val_day_loader
            val_split = 'val_day'
        elif split =='night':
            dataloader = self.val_night_loader
            val_split = 'val_night'

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if self.opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                features = self.models["encoder"](input_color, split, 'val')
                output = self.models["depth"](features)

                pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if self.opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = self.batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                gt.append(np.squeeze(data['depth_gt'].cpu().numpy()))

        pred_disps = np.concatenate(pred_disps)
        if gt[-1].ndim==2:
            gt[-1] = gt[-1][np.newaxis,:]
        gt = np.concatenate(gt)


        if self.opt.save_pred_disps:
            output_path = os.path.join(
                self.opt.load_weights_folder, "disps_{}_split.npy".format(self.opt.eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, pred_disps)

        if self.opt.no_eval:
            print("-> Evaluation disabled. Done.")
            quit()

        elif self.opt.eval_split == 'benchmark':

            save_dir = os.path.join(self.opt.load_weights_folder, "benchmark_predictions")
            print("-> Saving out benchmark predictions to {}".format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx in range(len(pred_disps)):
                disp_resized = cv2.resize(pred_disps[idx], (1280, 640))
                depth = STEREO_SCALE_FACTOR / disp_resized
                depth = np.clip(depth, 0, 80)
                depth = np.uint16(depth * 256)
                save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
                cv2.imwrite(save_path, depth)

            print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
            quit()

        # gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

        print("-> Evaluating")

        if self.opt.eval_stereo:
            print("   Stereo evaluation - "
                  "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
            self.opt.disable_median_scaling = True
            self.opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
        else:
            print("   Mono evaluation - using median scaling")

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):

            gt_depth = gt[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            if self.opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            # # range 60m
            mask2 = gt_depth<=60
            pred_depth = pred_depth[mask2]
            gt_depth = gt_depth[mask2]

            errors.append(self.compute_errors(gt_depth, pred_depth))

        if not self.opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")
        
        with open(self.result_path, 'a') as f: 
            for i in range(len(mean_errors)):
                f.write(str(mean_errors[i])) #
                f.write('\t')
            f.write("\n")        
  
        f.close()  

#         self.log_val(val_split, data, output)
        
        self.set_train()
        return mean_errors


    def compute_errors(self,gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def batch_post_process_disparity(self,l_disp, r_disp):
        """Apply the disparity post-processing method as introduced in Monodepthv1
        """
        _, h, w = l_disp.shape
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
        r_mask = l_mask[:, :, ::-1]
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [640,1280], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        # crop_mask = torch.zeros_like(mask)
        # crop_mask[:, :, 153:371, 44:1197] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss_day, loss_night, loss_D, loss_G, other_loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss_day: {:.5f} | loss_night: {:.5f} | loss_D: {:.5f} | loss_G: {:.5f} | loss_other: {:.5f}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss_day, loss_night, loss_D, loss_G, other_loss))

    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
        #     for s in self.opt.scales:
        #         for frame_id in self.opt.frame_ids:
        #             writer.add_image(
        #                 "color_{}_{}/{}".format(frame_id, s, j),
        #                 inputs[("color", frame_id, s)][j].data, self.step)
        #             if s == 0 and frame_id != 0:
        #                 writer.add_image(
        #                     "color_pred_{}_{}/{}".format(frame_id, s, j),
        #                     outputs[("color", frame_id, s)][j].data, self.step)

        #         writer.add_image(
        #             "disp_{}/{}".format(s, j),
        #             normalize_image(outputs[("disp", s)][j]), self.step)

        #         if self.opt.predictive_mask:
        #             for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
        #                 writer.add_image(
        #                     "predictive_mask_{}_{}/{}".format(frame_id, s, j),
        #                     outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
        #                     self.step)

        #         elif not self.opt.disable_automasking:
        #             writer.add_image(
        #                 "automask_{}/{}".format(s, j),
        #                 outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
    
    def log_val(self, mode, inputs, outputs):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
    
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def load_depth_model(self):
        self.opt.load_depth_weights = os.path.expanduser(self.opt.load_depth_weights)

        assert os.path.isdir(self.opt.load_depth_weights), \
            "Cannot find folder {}".format(self.opt.load_depth_weights)
        print("Loading depth model from: {}".format(self.opt.load_depth_weights))

        encoder_path = os.path.join(self.opt.load_depth_weights, "encoder.pth")
        decoder_path = os.path.join(self.opt.load_depth_weights, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        model_dict = self.models["encoder"].state_dict()
        self.models["encoder"].encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in encoder_dict.items() if k in model_dict})
        self.models["depth"].load_state_dict(torch.load(decoder_path))