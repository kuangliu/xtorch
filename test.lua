local utils = dofile('utils.lua')

local TEST = torch.class 'TEST'
function TEST:__init(opt)
    utils.parse(self, opt)
end
