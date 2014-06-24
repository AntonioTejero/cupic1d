###------------------ GEMS ------------------###
require 'gnuplot'

###---------- SCRIPT CONFIGURATION ----------###

# data file parameters
MAGNITUDE = "field"
AVERAGED = true
BOT = 999
TOP = 3436000
STEP = 1000
# plot parameters 
param = {:title => "#{MAGNITUDE} averaged over each #{STEP} iteration (t = KEY)", #KEY will be changed by the value of t
         :xlabel => "node (ds = 10% debye lenght)",
         :ylabel => "#{MAGNITUDE} (simulation units)"}

###-------------- SCRIPT START --------------###

# format of the file names (KEY will be changed by the value of iter or counter respectively)
if AVERAGED
  IFNAME = "avg_#{MAGNITUDE}_t_KEY.dat"
  OFNAME = "avg_#{MAGNITUDE}_KEY.jpg"
else
  IFNAME = "#{MAGNITUDE}_t_KEY.dat"
  OFNAME = "#{MAGNITUDE}_KEY.jpg"
end

#------------------------------------------------------

$counter = 0
Gnuplot.open do |gp|
    (BOT..TOP).step(STEP) do |iter|
        Gnuplot::Plot.new(gp) do |plot|
            plot.terminal "jpeg size 1280,720" 
            plot.nokey
            plot.grid
            plot.ylabel param[:ylabel]
            plot.xlabel param[:xlabel]
            plot.title param[:title].gsub("KEY", iter.to_s)
            plot.output OFNAME.gsub("KEY", $counter.to_s)
            plot.arbitrary_lines << "plot \"#{IFNAME.gsub("KEY",iter.to_s)}\""
            $counter += 1
        end
    end
end
`avconv -f image2 -i #{OFNAME.gsub("KEY", "%d")} -b 32000k #{MAGNITUDE+"_movie.mov"}`
#`find . -name '*.jpg' -type f -print -delete`
Dir.glob("*.jpg").each {|f| `rm #{f}`}

