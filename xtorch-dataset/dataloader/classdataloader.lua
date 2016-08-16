require 'os';
require 'sys';
require 'xlua';
require 'image';
require 'torch';
require 'paths';
ffi = require 'ffi';
dir = require 'pl.dir'

torch.setdefaulttensortype('torch.FloatTensor')

local ClassDataLoader = torch.class 'ClassDataLoader'
pathcat = paths.concat

---------------------------------------------------------------------------
-- ClassDataLoader takes:
--  - directory: a folder containing the images organized in subfolders.
--              the subfolder name is the class name.
--  - imsize: image load size
--
function ClassDataLoader:__init(directory, imsize)
    assert(paths.dirp(directory), directory..' not exist!')
    self.directory = directory
    self.imsize = imsize
    self:__getClasses()
    self:__getFileNames()
end

------------------------------------------------------------------------
-- get subfolder/class names
--
function ClassDataLoader:__getClasses()
    self.classes = {}
    local dirs = dir.getdirectories(self.directory)
    for k,path in pairs(dirs) do
        local class = paths.basename(path)
        self.classes[k] = class
    end
    print('classes:')
    print(self.classes)
end

------------------------------------------------------------------------
-- loop all subfolders to get file names
--
function ClassDataLoader:__getFileNames()
    self.names = {}
    for k,v in pairs(self.classes) do
        print('parsing '..v..'..')
        local classPath = pathcat(self.directory, v)
        self.names[k] = self:__loopfolder(classPath)
    end
end

------------------------------------------------------------------------
-- loop through a folder, return all file names in it as a 2D tensor.
--
function ClassDataLoader:__loopfolder(path)
    local N = tonumber(sys.fexecute('ls '..path..' | wc -l'))
    self.N = (self.N or 0) + N

    local constLength = 50           -- assume the length of all file names < constLength
    local maxNameLength = -1         -- max file name length

    local names = torch.CharTensor(N,constLength):fill(0)
    local name_data = names:data()
    -- local name_data = names:data()
    local i = 0
    for name in paths.iterfiles(path) do
        i = i + 1
        xlua.progress(i,N)
        ffi.copy(name_data, name)
        name_data = name_data + constLength
        maxNameLength = math.max(maxNameLength, #name)
    end

    return names[{ {},{1,maxNameLength} }]  -- trim
end

------------------------------------------------------------------------
-- sample a batch
-- we first randomly sample class indices, then randomly
-- sample names from that class.
--
function ClassDataLoader:sample(quantity)
    assert(quantity, '[ERROR] => No sample quantity specified!')
    local samples = torch.Tensor(quantity, 3, self.imsize, self.imsize)
    local targets = torch.Tensor(quantity)
    for i = 1,quantity do
        local class = torch.random(1, #self.classes)
        local im = self:__loadSampleByClass(class)
        samples[i] = image.scale(im, self.imsize, self.imsize)
        targets[i] = class
    end
    return samples, targets
end

------------------------------------------------------------------------
-- randomly load a sample by class
--
function ClassDataLoader:__loadSampleByClass(class)
    local names = self.names[class]
    local index = torch.random(1, names:size(1))
    local name = ffi.string(names[index]:data())
    local im = image.load(pathcat(self.directory, self.classes[class], name))
    return im
end

------------------------------------------------------------------------
-- load samples in range [i1, i2]
-- this function is a little bit confusing. as the sample names are
-- organized separately, we first need to get the class idx of the i-th
-- sample, and then get the sample idx within that class.
--
function ClassDataLoader:get(i1,i2)
    local quantity = i2 - i1 + 1
    local samples = torch.Tensor(quantity, 3, self.imsize, self.imsize)
    local targets = torch.Tensor(quantity)

    -- accumulated nb of samples for each class
    -- e.g. the # of 3 classes are 2,2,3
    -- then nbs = {2,4,7}
    if not nbs then
        nbs = {}
        for i = 1,#self.classes do
            nbs[i] = self.names[i]:size(1)
            if i > 1 then nbs[i] = nbs[i] + nbs[i-1] end
        end
    end

    for i = i1,i2 do
        local classidx
        for j = 1,#nbs do
            if nbs[j] >= i then
                classidx = j
                break
            end
        end

        local sampleidx = classidx==1 and i or i - nbs[classidx-1]
        local names = self.names[classidx]
        local name = ffi.string(names[sampleidx]:data())
        local im = image.load(pathcat(self.directory, self.classes[classidx], name))
        im = image.scale(im, self.imsize, self.imsize)
        samples[i-i1+1] = im
        targets[i-i1+1] = classidx
    end
    return samples, targets
end
