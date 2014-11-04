###------------------ GEMS ------------------###
require 'gnuplot'

###---------- SCRIPT CONFIGURATION ----------###

# data file parameters
INPUT_DATA_FILE = "../input_data"
PARTICLE_TYPE = ["electrons", "ions"]
BOT = 100 
TOP = 500
STEP = 100
DT = 1.0e-2
L = 100.0

###-------------- SCRIPT START --------------###

# read yrange limits
count = 0 
PARTICLE_TYPE.each do |ptype|
  count += 1
end
velocity_limits = Array.new(count){Array.new(2,0.0)}
f1 = File.open(INPUT_DATA_FILE, mode="r")
a = f1.readlines
(0..count-1).step do |index|
  velocity_limits[index][0] = a[22+2*index].split[2].to_f
  velocity_limits[index][1] = a[22+2*index+1].split[2].to_f
end
f1.close

puts velocity_limits[0][0]
puts velocity_limits[0][1]
puts velocity_limits[1][0]
puts velocity_limits[1][1]
 
# generate movie of each particle type
count = 0
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
              plot.yrange "[#{velocity_limits[count][1]}:#{velocity_limits[count][0]}]"
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

  # next particle type counter
  count += 1
end
