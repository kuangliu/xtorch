dofile('./xtorch-dataset/listdatset.lua')

traindata = ListDataset({
    directory = '../dataset/cifar-list/train/',
    list = '../dataset/cifar-list/train.txt',
	imsize = 32,
	transform = { standardize = true }
})

mean,std = traindata:calcMeanStd()

testdata = ListDataset({
    directory = '../dataset/cifar-list/test/',
    list = '../dataset/cifar-list/test.txt',
	imsize = 32,
    mean = mean,
    std = std,
	transform = { standardize = true }
})

paths.mkdir('cache')
torch.save('./cache/traindata.t7',traindata)
torch.save('./cache/testdata.t7',testdata)
