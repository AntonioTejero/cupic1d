###------------------ GEMS ------------------###
require 'gnuplot'

###---------- SCRIPT CONFIGURATION ----------###

# data file parameters
PARTICLE_TYPE = "ions"
BOT = 0
TOP = 4140000
STEP = 1000
# plot parameters 
param = {:title => "#{PARTICLE_TYPE} phase diagram }iteration (t = KEY)", #KEY will be changed by the value of t
         :xlabel => "position",
         :ylabel => "velocity"}

###-------------- SCRIPT START --------------###

# format of the file names (KEY will be changed by the value of iter or counter respectively)
IFNAME = "#{PARTICLE_TYPE}_t_KEY.dat"
OFNAME = "#{PARTICLE_TYPE}_pd_KEY.jpg"
MOVIENAME = "#{PARTICLE_TYPE}_phase_diagram.mov"

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
`avconv -f image2 -i #{OFNAME.gsub("KEY", "%d")} -b 32000k #{MOVIENAME}`
Dir.glob("*.jpg").each {|f| `rm #{f}`}
