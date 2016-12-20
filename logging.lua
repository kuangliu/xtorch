----------------------------------------
-- Basic logging module for xtorch.
----------------------------------------
require 'os'
require 'paths'


local Logger = torch.class('Logger')

function Logger:__init(logpath)
    paths.mkdir('log')
    local logpath = logpath or 'log/'..os.date("%Y-%m-%d %X")
    self.logfile = io.open(logpath, 'a+')
end

----------------------------------------
-- Log content to file.
-- Args:
--   content: (table) logging content.
--
function Logger:log(content)
    local entry = { os.date('%x %X') }  -- e.g. '12/20/16 15:46:52'
    for _,v in pairs(content) do
        entry[#entry+1] = v
    end
    local entry_line = table.concat(entry, '  ')
    self.logfile:write(entry_line..'\n')
    self.logfile:flush()
end
