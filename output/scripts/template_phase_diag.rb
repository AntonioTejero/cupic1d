###------------------ GEMS ------------------###
require 'gnuplot'

###---------- SCRIPT CONFIGURATION ----------###

# data file parameters
PARTICLE_TYPE = ["electrons", "ions"]
BOT = 100 
TOP = 500000
STEP = 100
DT = 1.0e-2
L = 100.0

###-------------- SCRIPT START --------------###

PARTICLE_TYPE.each do |ptype|
  
  # plot parameters 
  param = {:title => "#{ptype} phase diagram (t = KEY)"+'\n (everything measured in simulation units)', #KEY will be changed by the value of t
           :xlabel => "position",
           :ylabel => "velocity"}

  # format of the file names (KEY will be changed by the value of iter or counter respectively)
  iFNAME = "#{ptype}_t_KEY.dat"
  oFNAME = "#{ptype}_pd_KEY.jpg"
  mOVIENAME = "#{ptype}_phase_diagram.mov"

  # generate fotograms
  $counter = 0
  Gnuplot.open do |gp|
      (BOT..TOP).step(STEP) do |iter|
          Gnuplot::Plot.new(gp) do |plot|
              plot.terminal "jpeg size 1280,720" 
              plot.nokey
              plot.grid
              plot.xrange "[0:#{L}]"
              plot.ylabel param[:ylabel]
              plot.xlabel param[:xlabel]
              plot.title param[:title].gsub("KEY", (DT*iter).to_s)
              plot.output oFNAME.gsub("KEY", $counter.to_s)
              plot.arbitrary_lines << "plot \"#{iFNAME.gsub("KEY",iter.to_s)}\""
              $counter += 1
          end
      end
  end

  # generate movie
  `avconv -f image2 -i #{oFNAME.gsub("KEY", "%d")} -b 32000k #{mOVIENAME}`

  # remove fotograms
  Dir.glob("*.jpg").each {|f| `rm #{f}`}
end
