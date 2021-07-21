import numpy as np
import torch
import os
import sys
sys.path.append("..")
sys.path.append(".")
from .patchnce import PatchNCELoss
from .base_model import BaseModel
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from . import networks


class I2VGAN(BaseModel):
    def name(self):
        return 'I2V-GANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.tmp_opt = opt
        self.first = True
        nb = opt.batchSize
        size = opt.fineSize
        self.input_A0 = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A1 = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A2 = self.Tensor(nb, opt.input_nc, size, size)

        self.input_B0 = self.Tensor(nb, opt.output_nc, size, size)
        self.input_B1 = self.Tensor(nb, opt.output_nc, size, size)
        self.input_B2 = self.Tensor(nb, opt.output_nc, size, size)

        self.netF_A = networks.define_F(opt.input_nc, opt.netF, opt.norm, not opt.no_dropout, opt.init_type, 0.02, opt.no_antialias, self.gpu_ids, opt)
        self.netF_B = networks.define_F(opt.input_nc, opt.netF, opt.norm, not opt.no_dropout, opt.init_type, 0.02, opt.no_antialias, self.gpu_ids, opt)
        
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        self.which_model_netP = opt.which_model_netP
        if opt.which_model_netP == 'prediction':
            self.netP_A = networks.define_G(opt.input_nc, opt.input_nc,
						opt.npf, opt.which_model_netP, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
            self.netP_B = networks.define_G(opt.output_nc, opt.output_nc,
						opt.npf, opt.which_model_netP, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        else:
            self.netP_A = networks.define_G(2*opt.input_nc, opt.input_nc, opt.ngf, opt.which_model_netP, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
            self.netP_B = networks.define_G(2*opt.output_nc, opt.output_nc, opt.ngf, opt.which_model_netP, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.criterionNCE = []
            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).cuda())


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            self.load_network(self.netP_A, 'P_A', which_epoch)
            self.load_network(self.netP_B, 'P_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            self.criterionGAN = networks.GANLoss('lsgan').cuda()
            self.criterionCycle = networks.VGGLoss()
            self.criterionIdt = torch.nn.L1Loss()
            

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(),
								self.netP_A.parameters(), self.netP_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # print('---------- Networks initialized -------------')
        # networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        # networks.print_network(self.netP_A)
        # networks.print_network(self.netP_B)
        # if self.isTrain:
        #     networks.print_network(self.netD_A)
        #     networks.print_network(self.netD_B)
        # print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A0 = input['A0']
        input_A1 = input['A1']
        input_A2 = input['A2']

        input_B0 = input['B0']
        input_B1 = input['B1']
        input_B2 = input['B2']

        self.input_A0.resize_(input_A0.size()).copy_(input_A0)
        self.input_A1.resize_(input_A1.size()).copy_(input_A1)
        self.input_A2.resize_(input_A2.size()).copy_(input_A2)

        self.input_B0.resize_(input_B0.size()).copy_(input_B0)
        self.input_B1.resize_(input_B1.size()).copy_(input_B1)
        self.input_B2.resize_(input_B2.size()).copy_(input_B2)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']




    def calculate_NCE_loss(self, src, tgt, G, netF):
        n_layers = len(self.nce_layers)
        feat_q = G(tgt, self.nce_layers, encode_only=True)

        feat_k = G(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def forward(self):
        self.real_A0 = Variable(self.input_A0)
        self.real_A1 = Variable(self.input_A1)
        self.real_A2 = Variable(self.input_A2)

        self.real_B0 = Variable(self.input_B0)
        self.real_B1 = Variable(self.input_B1)
        self.real_B2 = Variable(self.input_B2)

    def test(self):
        real_A0 = Variable(self.input_A0, volatile=True)
        real_A1 = Variable(self.input_A1, volatile=True)
        real_A2 = Variable(self.input_A2, volatile=True)

        fake_B0 = self.netG_B(real_A0)
        fake_B1 = self.netG_B(real_A1)
        self.fake_g_B2 = self.netG_B(real_A2)

        if self.which_model_netP == 'prediction':
            fake_B2 = self.netP_B(fake_B0, fake_B1)
        else:
            fake_B2 = self.netP_B(torch.cat((fake_B0, fake_B1),1))

        self.rec_A = self.netG_A(fake_B2).data
        self.fake_B0 = fake_B0.data
        self.fake_B1 = fake_B1.data
        self.fake_B2 = fake_B2.data
        self.c_fake_B2 = (fake_B2 / 2.0 + self.fake_g_B2 / 2.0).data
        self.fake_g_B2 = self.fake_g_B2.data

        real_B0 = Variable(self.input_B0, volatile=True)
        real_B1 = Variable(self.input_B1, volatile=True)
        real_B2 = Variable(self.input_B2, volatile=True)

        fake_A0 = self.netG_A(real_B0)
        fake_A1 = self.netG_A(real_B1)
        self.fake_g_A2 = self.netG_A(real_B2)
        if self.which_model_netP == 'prediction':
            fake_A2 = self.netP_A(fake_A0, fake_A1)
        else:
            fake_A2 = self.netP_A(torch.cat((fake_A0, fake_A1),1))

        self.rec_B = self.netG_B(fake_A2).data
        self.fake_A0 = fake_A0.data
        self.fake_A1 = fake_A1.data
        self.fake_A2 = fake_A2.data
        self.c_fake_A2 = (fake_A2 / 2.0 + self.fake_g_A2 / 2.0).data
        self.fake_g_A2 = self.fake_g_A2.data

        if self.which_model_netP == 'prediction':
            pred_A2 = self.netP_A(real_A0, real_A1)
        else:
            pred_A2 = self.netP_A(torch.cat((real_A0, real_A1),1))

        self.pred_A2 = pred_A2.data

        if self.which_model_netP == 'prediction':
            pred_B2 = self.netP_B(real_B0, real_B1)
        else:
            pred_B2 = self.netP_B(torch.cat((real_B0, real_B1),1))

        self.pred_B2 = pred_B2.data


    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):

        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        loss_D.backward()
        return loss_D

    def backward_D_B(self):
        fake_B0 = self.fake_B_pool.query(self.fake_B0)
        loss_D_B0 = self.backward_D_basic(self.netD_B, self.real_B0, fake_B0)

        fake_B1 = self.fake_B_pool.query(self.fake_B1)
        loss_D_B1 = self.backward_D_basic(self.netD_B, self.real_B1, fake_B1)

        fake_B2 = self.fake_B_pool.query(self.fake_B2)
        loss_D_B2 = self.backward_D_basic(self.netD_B, self.real_B2, fake_B2)

        pred_B = self.fake_B_pool.query(self.pred_B2)
        loss_D_B3 = self.backward_D_basic(self.netD_B, self.real_B2, pred_B)

        self.loss_D_B = loss_D_B0.data + loss_D_B1.data + loss_D_B2.data + loss_D_B3.data


    def backward_D_A(self):
        fake_A0 = self.fake_A_pool.query(self.fake_A0)
        loss_D_A0 = self.backward_D_basic(self.netD_A, self.real_A0, fake_A0)

        fake_A1 = self.fake_A_pool.query(self.fake_A1)
        loss_D_A1 = self.backward_D_basic(self.netD_A, self.real_A1, fake_A1)

        fake_A2 = self.fake_A_pool.query(self.fake_A2)
        loss_D_A2 = self.backward_D_basic(self.netD_A, self.real_A2, fake_A2)

        pred_A = self.fake_A_pool.query(self.pred_A2)
        loss_D_A3 = self.backward_D_basic(self.netD_A, self.real_A2, pred_A)

        self.loss_D_A = loss_D_A0.data + loss_D_A1.data + loss_D_A2.data + loss_D_A3.data

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        

        feat_A0 = self.netG_B(self.real_A0, self.nce_layers, encode_only=True)
        feat_A0_pool, sample_ids_A = self.netF_A(feat_A0, self.opt.num_patches, None)

        feat_A1 = self.netG_B(self.real_A1, self.nce_layers, encode_only=True)
        feat_A1_pool, _ = self.netF_A(feat_A1, self.opt.num_patches, sample_ids_A)

        feat_A2 = self.netG_B(self.real_A2, self.nce_layers, encode_only=True)
        feat_A2_pool, _ = self.netF_A(feat_A2, self.opt.num_patches, sample_ids_A)

        NCE_A0_A1 = 0.0
        NCE_A1_A2 = 0.0

        NCE_vector_A0_A1 = None
        for f_q, f_k, crit, nce_layer in zip(feat_A0_pool, feat_A1_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) 
            if NCE_vector_A0_A1 == None:
                NCE_vector_A0_A1 = loss.mean().unsqueeze(0)
            else:
                NCE_vector_A0_A1 = torch.cat((NCE_vector_A0_A1, loss.mean().unsqueeze(0)), dim=0)

        NCE_vector_A1_A2 = None
        for f_q, f_k, crit, nce_layer in zip(feat_A1_pool, feat_A2_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) 
            if NCE_vector_A1_A2 == None:
                NCE_vector_A1_A2 = loss.mean().unsqueeze(0)
            else:
                NCE_vector_A1_A2 = torch.cat((NCE_vector_A1_A2, loss.mean().unsqueeze(0)), dim=0)

        weight_inS_vector_real_A = (NCE_vector_A1_A2 / NCE_vector_A0_A1).clone().detach()



        feat_B0 = self.netG_A(self.real_B0, self.nce_layers, encode_only=True)
        feat_B0_pool, sample_ids_B = self.netF_B(feat_B0, self.opt.num_patches, None)

        feat_B1 = self.netG_A(self.real_B1, self.nce_layers, encode_only=True)
        feat_B1_pool, _ = self.netF_B(feat_B1, self.opt.num_patches, sample_ids_B)

        feat_B2 = self.netG_A(self.real_B2, self.nce_layers, encode_only=True)
        feat_B2_pool, _ = self.netF_B(feat_B2, self.opt.num_patches, sample_ids_B)

        NCE_B0_B1 = 0.0
        NCE_B1_B2 = 0.0


        NCE_vector_B0_B1 = None
        for f_q, f_k, crit, nce_layer in zip(feat_B0_pool, feat_B1_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) 
            if NCE_vector_B0_B1 == None:
                NCE_vector_B0_B1 = loss.mean().unsqueeze(0)
            else:
                NCE_vector_B0_B1 = torch.cat((NCE_vector_B0_B1, loss.mean().unsqueeze(0)), dim=0)


        NCE_vector_B1_B2 = None
        for f_q, f_k, crit, nce_layer in zip(feat_B1_pool, feat_B2_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) 
            if NCE_vector_B1_B2 == None:
                NCE_vector_B1_B2 = loss.mean().unsqueeze(0)
            else:
                NCE_vector_B1_B2 = torch.cat((NCE_vector_B1_B2, loss.mean().unsqueeze(0)), dim=0)

        weight_inS_vector_real_B = (NCE_vector_B1_B2 / NCE_vector_B0_B1).clone().detach()



        if lambda_idt > 0:

            idt_B0 = self.netG_B(self.real_B0)
            idt_B1 = self.netG_B(self.real_B1)
            loss_idt_B = (self.criterionIdt(idt_B0, self.real_B0) + self.criterionIdt(idt_B1, self.real_B1) )* lambda_A * lambda_idt

            idt_A0 = self.netG_A(self.real_A0)
            idt_A1 = self.netG_A(self.real_A1)
            loss_idt_A = (self.criterionIdt(idt_A0, self.real_A0) + self.criterionIdt(idt_A1, self.real_A1)) * lambda_B * lambda_idt

            self.idt_A = idt_A0.data
            self.idt_B = idt_B0.data
            self.loss_idt_A = loss_idt_A.data
            self.loss_idt_B = loss_idt_B.data

        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0


        fake_B0 = self.netG_B(self.real_A0)
        pred_fake = self.netD_B(fake_B0)

        loss_G_B0 = self.criterionGAN(pred_fake, True)

        fake_B1 = self.netG_B(self.real_A1)
        pred_fake = self.netD_B(fake_B1)
        loss_G_B1 = self.criterionGAN(pred_fake, True)


        if self.which_model_netP == 'prediction':
            fake_B2 = self.netP_B(fake_B0,fake_B1)
        else:
            fake_B2 = self.netP_B(torch.cat((fake_B0,fake_B1),1))


        feat_B0 = self.netG_B(fake_B0, self.nce_layers, encode_only=True)
        feat_B0_pool, _ = self.netF_A(feat_B0, self.opt.num_patches, sample_ids_A)

        feat_B1 = self.netG_B(fake_B1, self.nce_layers, encode_only=True)
        feat_B1_pool, _ = self.netF_A(feat_B1, self.opt.num_patches, sample_ids_A)

        feat_B2 = self.netG_B(fake_B2, self.nce_layers, encode_only=True)
        feat_B2_pool, _ = self.netF_A(feat_B2, self.opt.num_patches, sample_ids_A)

        NCE_B0_B1 = 0.0
        NCE_B0_B2 = 0.0
        NCE_B1_B2 = 0.0
        NCE_vector_B1_B2 = None
        for f_q, f_k, crit, nce_layer in zip(feat_B1_pool, feat_B2_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) 
            if NCE_vector_B1_B2 == None:
                NCE_vector_B1_B2 = loss.mean().unsqueeze(0)
            else:
                NCE_vector_B1_B2 = torch.cat((NCE_vector_B1_B2, loss.mean().unsqueeze(0)), dim=0)


        NCE_vector_B0_B1 = None
        for f_q, f_k, crit, nce_layer in zip(feat_B0_pool, feat_B1_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) 
            if NCE_vector_B0_B1 == None:
                NCE_vector_B0_B1 = loss.mean().unsqueeze(0)
            else:
                NCE_vector_B0_B1 = torch.cat((NCE_vector_B0_B1, loss.mean().unsqueeze(0)), dim=0)


        weight_inS_vector_fake_B = NCE_vector_B1_B2 / NCE_vector_B0_B1


        loss_inter_similarity_B = 1 - torch.cosine_similarity(weight_inS_vector_real_A, weight_inS_vector_fake_B, dim=0)

        loss_inter_similarity_B = loss_inter_similarity_B * self.opt.lam_INS



        loss_exter_similarity_B = self.calculate_NCE_loss(self.real_A0, fake_B0, self.netG_B, self.netF_A) + self.calculate_NCE_loss(self.real_A1, fake_B1, self.netG_B, self.netF_A) + self.calculate_NCE_loss(self.real_A2, fake_B2, self.netG_B, self.netF_A)
        loss_exter_similarity_B *= self.opt.lam_EXS



        pred_fake = self.netD_B(fake_B2)
        loss_G_B2 = self.criterionGAN(pred_fake, True)



        fake_A0 = self.netG_A(self.real_B0)
        pred_fake = self.netD_A(fake_A0)
        loss_G_A0 = self.criterionGAN(pred_fake, True)

        fake_A1 = self.netG_A(self.real_B1)
        pred_fake = self.netD_A(fake_A1)
        loss_G_A1 = self.criterionGAN(pred_fake, True)

        if self.which_model_netP == 'prediction':
            fake_A2 = self.netP_A(fake_A0,fake_A1)
        else:
            fake_A2 = self.netP_A(torch.cat((fake_A0,fake_A1),1))



        feat_A0 = self.netG_A(fake_A0, self.nce_layers, encode_only=True)
        feat_A0_pool, _ = self.netF_B(feat_A0, self.opt.num_patches, sample_ids_B)

        feat_A1 = self.netG_A(fake_A1, self.nce_layers, encode_only=True)
        feat_A1_pool, _ = self.netF_B(feat_A1, self.opt.num_patches, sample_ids_B)

        feat_A2 = self.netG_A(fake_A2, self.nce_layers, encode_only=True)
        feat_A2_pool, _ = self.netF_B(feat_A2, self.opt.num_patches, sample_ids_B)

        NCE_A0_A1 = 0.0
        NCE_A0_A2 = 0.0
        NCE_A1_A2 = 0.0

        NCE_vector_A1_A2 = None
        for f_q, f_k, crit, nce_layer in zip(feat_A1_pool, feat_A2_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            if NCE_vector_A1_A2 == None:
                NCE_vector_A1_A2 = loss.mean().unsqueeze(0)
            else:
                NCE_vector_A1_A2 = torch.cat((NCE_vector_A1_A2, loss.mean().unsqueeze(0)), dim=0)


        NCE_vector_A0_A1 = None
        for f_q, f_k, crit, nce_layer in zip(feat_A0_pool, feat_A1_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) 
            if NCE_vector_A0_A1 == None:
                NCE_vector_A0_A1 = loss.mean().unsqueeze(0)
            else:
                NCE_vector_A0_A1 = torch.cat((NCE_vector_A0_A1, loss.mean().unsqueeze(0)), dim=0)
            # NCE_A0_A1 += loss.mean()

        weight_inS_vector_fake_A = NCE_vector_A1_A2 / NCE_vector_A0_A1

        loss_inter_similarity_A = 1 - torch.cosine_similarity(weight_inS_vector_real_B, weight_inS_vector_fake_A, dim=0)
        loss_inter_similarity_A = loss_inter_similarity_A * self.opt.lam_INS

        loss_exter_similarity_A = self.calculate_NCE_loss(self.real_B0, fake_A0, self.netG_A, self.netF_B) + self.calculate_NCE_loss(self.real_B1, fake_A1, self.netG_A,self.netF_B) + self.calculate_NCE_loss(self.real_B2, fake_A2, self.netG_A, self.netF_B)
        loss_exter_similarity_A *= self.opt.lam_EXS
        

        pred_fake = self.netD_A(fake_A2)
        loss_G_A2 = self.criterionGAN(pred_fake, True)

        if self.which_model_netP == 'prediction':
            pred_A2 = self.netP_A(self.real_A0, self.real_A1)
        else:
            pred_A2 = self.netP_A(torch.cat((self.real_A0, self.real_A1),1))

        loss_pred_A = self.criterionCycle(pred_A2, self.real_A2) * lambda_A


        if self.which_model_netP == 'prediction':
            pred_B2 = self.netP_B(self.real_B0, self.real_B1)
        else:
            pred_B2 = self.netP_B(torch.cat((self.real_B0, self.real_B1),1))

        loss_pred_B = self.criterionCycle(pred_B2, self.real_B2) * lambda_B


        rec_A = self.netG_A(fake_B2)
        loss_recycle_A = self.criterionCycle(rec_A, self.real_A2) * lambda_A


        rec_B = self.netG_B(fake_A2)
        loss_recycle_B = self.criterionCycle(rec_B, self.real_B2) * lambda_B

        rec_A0 = self.netG_A(fake_B0)
        loss_cycle_A0 = self.criterionCycle(rec_A0, self.real_A0) * lambda_A

        rec_A1 = self.netG_A(fake_B1)
        loss_cycle_A1 = self.criterionCycle(rec_A1, self.real_A1) * lambda_A

        rec_B0 = self.netG_B(fake_A0)
        loss_cycle_B0 = self.criterionCycle(rec_B0, self.real_B0) * lambda_B

        rec_B1 = self.netG_B(fake_A1)
        loss_cycle_B1 = self.criterionCycle(rec_B1, self.real_B1) * lambda_B


        loss_G = loss_G_A0 + loss_G_A1 + loss_G_A2 + loss_G_B0 + loss_G_B1 + loss_G_B2 + loss_recycle_A + loss_recycle_B + loss_pred_A + loss_pred_B + loss_idt_A + loss_idt_B + loss_cycle_A0 + loss_cycle_A1 + loss_cycle_B0 + loss_cycle_B1 + \
                    loss_inter_similarity_A + loss_inter_similarity_B + loss_exter_similarity_A + loss_exter_similarity_B
        loss_G.backward()

        self.fake_B0 = fake_B0.data
        self.fake_B1 = fake_B1.data
        self.fake_B2 = fake_B2.data
        self.pred_B2 = pred_B2.data

        self.fake_A0 = fake_A0.data
        self.fake_A1 = fake_A1.data
        self.fake_A2 = fake_A2.data
        self.pred_A2 = pred_A2.data

        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A0.data + loss_G_A1.data + loss_G_A2.data
        self.loss_G_B = loss_G_B0.data + loss_G_B1.data + loss_G_B2.data
        self.loss_recycle_A = loss_recycle_A.data
        self.loss_recycle_B = loss_recycle_B.data
        self.loss_pred_A = loss_pred_A.data
        self.loss_pred_B = loss_pred_B.data
        self.loss_inter_similarity_A = loss_inter_similarity_A
        self.loss_inter_similarity_B = loss_inter_similarity_B

        self.loss_cycle_A = loss_cycle_A0.data + loss_cycle_A1.data
        self.loss_cycle_B = loss_cycle_B0.data + loss_cycle_B1.data

        self.loss_exter_similarity_A = loss_exter_similarity_A.data
        self.loss_exter_similarity_B = loss_exter_similarity_B.data

    def optimize_parameters(self):
        try:

            self.forward()
            self.optimizer_G.zero_grad()
            self.backward_G()
            if self.first:
                self.first = False
                self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF_A.parameters(), self.netF_B.parameters()),lr=self.tmp_opt.lr, betas=(self.tmp_opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_F)
                self.schedulers.append(networks.get_scheduler(self.optimizer_F, self.tmp_opt))
                if self.opt.continue_train:
                    self.load_network(self.netF_A, 'F_A', self.opt.which_epoch)
                    self.load_network(self.netF_B, 'F_B', self.opt.which_epoch)
            self.optimizer_F.zero_grad()
            
            self.optimizer_G.step()
            self.optimizer_F.step()

            self.optimizer_D_A.zero_grad()
            self.backward_D_A()
            self.optimizer_D_A.step()

            self.optimizer_D_B.zero_grad()
            self.backward_D_B()
            self.optimizer_D_B.step()
        except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, retrying batch',sys.stdout)
                    sys.stdout.flush()
                    for p in self.netG_A.parameters():
                        if p.grad is not None:
                            del p.grad  
                    torch.cuda.empty_cache()
                    y= self.forward()
                else:
                    raise e

    def get_current_errors(self):

        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Recyc_A', self.loss_recycle_A), ('Pred_A', self.loss_pred_A), ('Cyc_A', self.loss_cycle_A), ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Recyc_B',  self.loss_recycle_B), ('Pred_B', self.loss_pred_B), ('Cyc_B', self.loss_cycle_B), ('INS_A',self.loss_inter_similarity_A), ('INS_B', self.loss_inter_similarity_B), ('EXS_A', self.loss_exter_similarity_A), ('EXS_B', self.loss_exter_similarity_B)])

        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A0 = util.tensor2im(self.input_A0)
        real_A1 = util.tensor2im(self.input_A1)
        real_A2 = util.tensor2im(self.input_A2)

        fake_B0 = util.tensor2im(self.fake_B0)
        fake_B1 = util.tensor2im(self.fake_B1)
        fake_B2 = util.tensor2im(self.fake_B2)
        
        

        rec_A = util.tensor2im(self.rec_A)

        real_B0 = util.tensor2im(self.input_B0)
        real_B1 = util.tensor2im(self.input_B1)
        real_B2 = util.tensor2im(self.input_B2)

        fake_A0 = util.tensor2im(self.fake_A0)
        fake_A1 = util.tensor2im(self.fake_A1)
        fake_A2 = util.tensor2im(self.fake_A2)

        rec_B = util.tensor2im(self.rec_B)

        pred_A2 = util.tensor2im(self.pred_A2)
        pred_B2 = util.tensor2im(self.pred_B2)

        ret_visuals = OrderedDict([('real_A0', real_A0), ('fake_B0', fake_B0),
				   ('real_A1', real_A1), ('fake_B1', fake_B1),
				   ('fake_B2', fake_B2), ('rec_A', rec_A), ('real_A2', real_A2),
                                   ('real_B0', real_B0), ('fake_A0', fake_A0),
			           ('real_B1', real_B1), ('fake_A1', fake_A1),
				   ('fake_A2', fake_A2), ('rec_B', rec_B), ('real_B2', real_B2),
				   ('real_A2', real_A2), ('pred_A2', pred_A2),
				   ('real_B2', real_B2), ('pred_B2', pred_B2)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netP_A, 'P_A', label, self.gpu_ids)
        self.save_network(self.netP_B, 'P_B', label, self.gpu_ids)
        self.save_network(self.netF_A, 'F_A', label, self.gpu_ids)
        self.save_network(self.netF_B, 'F_B', label, self.gpu_ids)
