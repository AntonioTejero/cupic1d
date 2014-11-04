###------------------ GEMS ------------------###
require 'gnuplot'

###---------- SCRIPT CONFIGURATION ----------###

# data file parameters
MAGNITUDE = "field"
BOT = 100
TOP = 500000
STEP = 100
NODES = 2000
DS = 1.0e-1
DT = 1.0e-2

# plot parameters 
param = {:title => "#{MAGNITUDE} averaged over each #{STEP} iterations (t = KEY)"+'\n (everything measured in simulation units)', #KEY will be changed by the value of t
         :xlabel => "position",
         :ylabel => "#{MAGNITUDE}"}

###-------------- SCRIPT START --------------###

# format of the file names (KEY will be changed by the value of iter or counter respectively)
IFNAME = "avg_#{MAGNITUDE}_t_KEY.dat"
OFNAME = "avg_#{MAGNITUDE}_KEY.jpg"

#------------------------------------------------------

#---- Read mesh data

$counter = 0
Gnuplot.open do |gp|
  (BOT..TOP).step(STEP) do |iter|
    position_mesh = Array.new(NODES+1, 0.0)
    data_mesh = Array.new(NODES+1, 0.0)
    f1 = File.open(IFNAME.gsub("KEY",iter.to_s), mode="r")
    a = f1.readlines
    (0..NODES).step do |index|
      position_mesh[index] = a[index].split[0].to_f*DS
      data_mesh[index] = a[index].split[1].to_f
    end
    f1.close
    Gnuplot::Plot.new(gp) do |plot|
      plot.terminal "jpeg size 1280,720" 
      plot.nokey
      plot.grid
      plot.ylabel param[:ylabel]
      plot.xlabel param[:xlabel]
      plot.title param[:title].gsub("KEY", (DT*iter).to_s)
      plot.output OFNAME.gsub("KEY", $counter.to_s)
      plot.data = [
        Gnuplot::DataSet.new( [position_mesh, data_mesh] ) { |ds|
          ds.with = "lines"
          ds.title = "PIC Simulations"
        }
      ]
      $counter += 1
    end
  end
end

`avconv -f image2 -i #{OFNAME.gsub("KEY", "%d")} -b 32000k #{MAGNITUDE+"_movie.mov"}`
Dir.glob("*.jpg").each {|f| `rm #{f}`}

