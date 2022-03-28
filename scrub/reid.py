import torchreid
import pdb

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='msmt17',
    targets='msmt17',
    height=256,
    width=128,
    batch_size_train=128,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop'],
    combineall=True
)


model = torchreid.models.build_model(
    name='mobilenetv2_x1_4',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    start_epoch=40,
    save_dir='log/msmt17_mobilenetv2_x1_4',
    max_epoch=44,
    eval_freq=20,
    print_freq=1,
    test_only=False
)

print("Done")