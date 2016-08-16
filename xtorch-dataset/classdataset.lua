--------------------------------------------------------------------------------
-- ClassDataset loads training/test data from disk. But unlike listClassDataset,
-- there is no index list, the images are organized in subfolders, the subfolder
-- name is the class name.
--
-- Directory is like:
-- -- train
--      |_ class 1
--      |_ class 2
--      |_ ...
-- -- test
--      |_ class 1
--      |_ class 2
--      |_ ...
--------------------------------------------------------------------------------

dofile('./xtorch-dataset/dataloader/classdataloader.lua')

local ClassDataset = torch.class 'ClassDataset'
pathcat = paths.concat

---------------------------------------------------------------
-- ClassDataset takes params:
--  - directory: directory containing the images
--  - imsize: image target size
--
--  other optional params:
--  - scale: input scale
--  - mean: explicit data mean
--  - std: explicit data std
--  - transform: table representing image processing functions, like
--     - standardize: perform zero mean and std normalization
--     - ...
--
function ClassDataset:__init(opt)
    -- parse args
    for k,v in pairs(opt) do
        self[k] = v
    end
    -- init data loader
    self.dataloader = ClassDataLoader(self.directory, self.imsize)
    self.N = self.dataloader.N
end

---------------------------------------------------------------
-- perform stardard transform including scale, zero mean
-- and std normalization
--
function ClassDataset:__standardize(inputs)
    -- scale
    if self.scale then inputs:mul(self.scale) end

    -- zero mean & std normalization
    if not self.mean or not self.std then
        self:calcMeanStd()
    end

    for i = 1,3 do
        inputs[{ {},{i},{},{} }]:add(-self.mean[i]):div(self.std[i])
    end
    return inputs
end

---------------------------------------------------------------
-- calculate mean & std
--
function ClassDataset:calcMeanStd()
    -- zero mean & std normalization
    if not self.mean or not self.std then
        print('==> computing mean & std..')
        local N = math.min(10000, self.N)
        self.mean = torch.zeros(3)
        self.std = torch.zeros(3)
        for i = 1,N do
            xlua.progress(i,N)
            local im = self.dataloader:sample(1)[1]
            for j = 1,3 do
                self.mean[j] = self.mean[j] + im[j]:mean()
                self.std[j] = self.std[j] + im[j]:std()
            end
        end
        self.mean:div(N)
        self.std:div(N)
    end
    return self.mean, self.std
end

---------------------------------------------------------------
-- perform a series of image processing functions on inputs
--
function ClassDataset:__imfunc(inputs)
    -- no transform needed, return the raw inputs
    if not self.transform then return inputs end

    if self.transform.standardize then
        inputs = self:__standardize(inputs)
    end
    return inputs
end

---------------------------------------------------------------
-- load training batch sample
--
function ClassDataset:sample(quantity)
    local inputs, targets = self.dataloader:sample(quantity)
    inputs = self:__imfunc(inputs)
    return inputs, targets
end

---------------------------------------------------------------
-- load test batch sample
--
function ClassDataset:get(i1,i2)
    local inputs, targets = self.dataloader:get(i1,i2)
    inputs = self:__imfunc(inputs)
    return inputs, targets
end
