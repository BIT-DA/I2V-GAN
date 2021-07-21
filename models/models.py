
def create_model(opt):
    model = None
    # print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'i2vgan':
        assert(opt.dataset_mode == 'unaligned_triplet')
        from .i2v_gan_model import I2VGAN
        model = I2VGAN()
    
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
