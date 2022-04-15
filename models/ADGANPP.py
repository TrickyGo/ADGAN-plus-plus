import torch
import models.networks as networks
import util.util as util


class Model(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)


    def forward(self, data, mode):

        if mode == 'transfer':
            input_semantics1, real_image1 = self.preprocess_input(data[0])
            input_semantics2, real_image2 = self.preprocess_input(data[1])
            input_semantics = (input_semantics1, input_semantics2,'transfer')
            real_image = (real_image1, real_image2,'transfer')
            
            with torch.no_grad():
                fake_image = self.netG(input_semantics, real_image)
            return fake_image
        else:
            input_semantics, real_image = self.preprocess_input(data)

        if mode == 'stage1':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image)
            return g_loss, generated
        elif mode == 'stage1_inference':
            with torch.no_grad():
                fake_image = self.netG(input_semantics, real_image)
            return fake_image

        elif mode == 'stage2':
            real_image, stage1_image = data['image'].cuda(), data['stage1'].cuda()
            label_map = data['label'].long()
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0).cuda()

            g_loss, generated = self.compute_generator_loss_revise(
                input_semantics, real_image, stage1_image)
            return g_loss, generated

        elif mode == 'stage2_inference':
            stage1_image = data['stage1'].cuda()
            label_map = data['label'].long().cuda()
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0).cuda()

            fake_image = self.netG(input_semantics, stage1_image)
            return fake_image

        elif mode == 'discriminator':
            if self.opt.stage == "2":
                label_map = data['label'].long()
                bs, _, h, w = label_map.size()
                nc = self.opt.label_nc
                input_label = self.FloatTensor(bs, nc, h, w).zero_()
                input_semantics = input_label.scatter_(1, label_map, 1.0).cuda()
            else:
                input_semantics = data['label'].cuda()

            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image)
            return d_loss

        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G_', epoch, self.opt)
        util.save_network(self.netD, 'D_', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G_', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D_', opt.which_epoch, opt)

        return netG, netD

    def preprocess_input(self, data):
        return data['label'].cuda(), data['image'].cuda()

    def compute_generator_loss_revise(self, input_semantics, real_image, stage1_image):
        G_losses = {}

        fake_image = self.generate_fake(
            input_semantics, stage1_image, compute_kld_loss=self.opt.use_vae)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D): 
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs): 
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            if real_image.shape[1] == 4:
                real_image = real_image[:,:-1,:,:]
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg
            
        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image = self.generate_fake(input_semantics, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        self.lambda_D = 1
        D_losses['D_Fake'] = self.lambda_D * self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.lambda_D * self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        fake_image  = self.netG(input_semantics, real_image)
        return fake_image

    def discriminate(self, input_semantics, fake_image, real_image):
        if real_image.shape[1] == 4:
            real_image = real_image[:,:-1,:,:]
        if input_semantics.shape[1] == 2:
            fake_concat = torch.cat([input_semantics[:,:-1,:,:], fake_image], dim=1)
            real_concat = torch.cat([input_semantics[:,:-1,:,:], real_image], dim=1)
        else:
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0