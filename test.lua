require './xtorch-dataset/classdataset.lua'

traindata = ClassDataset({
    directory = '/search/ssd/liukuang/train',
	imsize = 32,
	transform = {
		standardize = true
	}
})
mean,std = traindata:calcMeanStd()

testdata = ClassDataset({
    directory = '/search/ssd/liukuang/test',
	imsize = 32,
    mean = mean,
    std = std,
	transform = {
		standardize = true
	}
})

torch.save('./cache/traindata.t7',traindata)
torch.save('./cache/testdata.t7',testdata)
