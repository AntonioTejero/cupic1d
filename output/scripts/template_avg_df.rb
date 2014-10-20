###------------------ GEMS ------------------###
require 'gnuplot'

###---------- SCRIPT CONFIGURATION ----------###

# data file parameters
DATA_DIR = "../particles"
AVERAGED = true
BOT = 3000000
TOP = 5000000
STEP = 1000
BINS = 100
H = 0.1
PI = Math::PI

# plot parameters 
param_ddf = {:title => "Density distribution functions (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
             :xlabel => "Position (measured in simulation units)",
             :ylabel => "Counts"}
param_meanv_e = {:title => "Mean velocity of electrons (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
                 :xlabel => "Position (measured in simulation units)",
                 :ylabel => "Velocity (measured in simulation units)"}
param_meanv_i = {:title => "Mean velocity of ions (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
                 :xlabel => "Position (measured in simulation units)",
                 :ylabel => "Velocity (measured in simulation units)"}
param_vdf_e = {:title => "Electron velocity distribution functions (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
               :xlabel => "Velocity (measured in simulation units)",
               :ylabel => "Counts"}
param_vdf_i = {:title => "Ion velocity distribution functions (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
               :xlabel => "Velocity (measured in simulation units)",
               :ylabel => "Counts"}
param_flux_e = {:title => "Electron flux distribution (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
                :xlabel => "Position (measured in simulation units)",
                :ylabel => "Flux (measured in simulation units)"}
param_flux_i = {:title => "Ion flux distribution (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
                :xlabel => "Position (measured in simulation units)",
                :ylabel => "Flux (measured in simulation units)"}

###-------------- SCRIPT START --------------###

# format of the file names (KEY will be changed by the value of iter or counter respectively)
IFNAME_VDF_E = "#{DATA_DIR}/electrons_vdf_t_KEY.dat"
IFNAME_DDF_E = "#{DATA_DIR}/electrons_ddf_t_KEY.dat"
IFNAME_VDF_I = "#{DATA_DIR}/ions_vdf_t_KEY.dat"
IFNAME_DDF_I = "#{DATA_DIR}/ions_ddf_t_KEY.dat"
OFNAME_VDF = "./vdf_from_#{BOT}_to_#{TOP}.dat" 
OFNAME_DDF = "./ddf_from_#{BOT}_to_#{TOP}.dat"
OFNAME_FLUX = "./flux_from_#{BOT}_to_#{TOP}.dat"
OFNAME_VDF_GRAPH = "./vdf_from_#{BOT}_to_#{TOP}.tex"
OFNAME_DDF_GRAPH = "./ddf_from_#{BOT}_to_#{TOP}.tex"
OFNAME_MEAN_VEL_E_GRAPH = "./mean_vel_e_from_#{BOT}_to_#{TOP}.tex"
OFNAME_MEAN_VEL_I_GRAPH = "./mean_vel_i_from_#{BOT}_to_#{TOP}.tex"
OFNAME_FLUX_I_GRAPH = "./flux_i_from_#{BOT}_to_#{TOP}.tex"
OFNAME_FLUX_E_GRAPH = "./flux_e_from_#{BOT}_to_#{TOP}.tex"

#------------------------------------------------------

#---- Read, average and store df data
vdf_e_data = Array.new(BINS){Array.new(BINS, 0.0)}
ddf_e_data = Array.new(BINS, 0.0)
velocity_e = Array.new(BINS, 0.0)
vdf_i_data = Array.new(BINS){Array.new(BINS, 0.0)}
ddf_i_data = Array.new(BINS, 0.0)
velocity_i = Array.new(BINS, 0.0)
position = Array.new(BINS, 0.0)

puts"Reading simulation files: \n"
(BOT..TOP).step(STEP) do |iter|
  puts"\t Reading instant #{iter}...\n"
  f1 = File.open("#{IFNAME_VDF_E.gsub("KEY", iter.to_s)}", mode="r")
  f2 = File.open("#{IFNAME_VDF_I.gsub("KEY", iter.to_s)}", mode="r")
  f3 = File.open("#{IFNAME_DDF_E.gsub("KEY", iter.to_s)}", mode="r")
  f4 = File.open("#{IFNAME_DDF_I.gsub("KEY", iter.to_s)}", mode="r")
  a = f1.readlines
  b = f2.readlines
  c = f3.readlines
  d = f4.readlines
  (0..BINS-1).step do |binp|
    position[binp] = c[binp].split[0].to_f
    ddf_e_data[binp] += c[binp].split[1].to_f
    ddf_i_data[binp] += d[binp].split[1].to_f
    (0..BINS-1).step do |binv|
      vdf_e_data[binp][binv] += a[binp*(BINS+1)+binv].split[2].to_f
      velocity_e[binv] = a[binp*(BINS+1)+binv].split[1].to_f
      vdf_i_data[binp][binv] += b[binp*(BINS+1)+binv].split[2].to_f
      velocity_i[binv] = b[binp*(BINS+1)+binv].split[1].to_f
    end
  end
  f1.close
  f2.close
  f3.close
  f4.close
end

puts"Saving averaged distribution functions:\n"
f1 = File.open(OFNAME_DDF, mode="w")
f2 = File.open(OFNAME_VDF, mode="w")
(0..BINS-1).step do |binp|
  ddf_e_data[binp] *= 18513.0/(TOP-BOT+STEP).to_f
  ddf_i_data[binp] *= 16824.0/(TOP-BOT+STEP).to_f
  f1.write("#{position[binp]} #{ddf_e_data[binp]} #{ddf_i_data[binp]} \n")
  (0..BINS-1).step do |binv|
    vdf_e_data[binp][binv] /= (TOP-BOT+STEP).to_f
    vdf_i_data[binp][binv] /= (TOP-BOT+STEP).to_f
    f2.write("#{position[binp]} #{velocity_e[binv]} #{vdf_e_data[binp][binv]} #{velocity_i[binv]} #{vdf_i_data[binp][binv]} \n")
  end
  f2.write("\n")
end
f1.close
f2.close

#---- Evaluate particle flux
mean_velocity_e = Array.new(BINS, 0.0)
mean_velocity_i = Array.new(BINS, 0.0)
flux_e = Array.new(BINS, 0.0)
flux_i = Array.new(BINS, 0.0)

puts"Evaluating particle flux:\n"
f1 = File.open(OFNAME_FLUX, mode="w")
(0..BINS-1).step do |binp|
  norm_e = 0.0
  norm_i = 0.0
  (0..BINS-1).step do |binv|
    mean_velocity_e[binp] += velocity_e[binv]*vdf_e_data[binp][binv] 
    norm_e += vdf_e_data[binp][binv] 
    mean_velocity_i[binp] += velocity_i[binv]*vdf_i_data[binp][binv] 
    norm_i += vdf_i_data[binp][binv] 
  end
  mean_velocity_e[binp] /= norm_e
  mean_velocity_i[binp] /= norm_i
  flux_e[binp] = mean_velocity_e[binp]*ddf_e_data[binp]
  flux_i[binp] = mean_velocity_i[binp]*ddf_i_data[binp]
  f1.write("#{position[binp]} #{mean_velocity_e[binp]} #{flux_e[binp]} #{mean_velocity_i[binp]} #{flux_i[binp]} \n")
end
f1.close

#---- Plot results
puts"Ploting results:\n"
Gnuplot.open do |gp|
  # Density distribution plot
  puts "\t Ploting density distributions...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_ddf[:title]
    plot.xlabel param_ddf[:xlabel]
    plot.ylabel param_ddf[:ylabel]
    plot.output OFNAME_DDF_GRAPH
    plot.data = [
      Gnuplot::DataSet.new( [position, ddf_e_data] ) { |ds|
        ds.with = "boxes lt 1"
        ds.title = "Electrons"
        ds.linewidth = 4
        ds.linecolor = 'rgb "red"'
      }, 
      Gnuplot::DataSet.new( [position, ddf_i_data] ) { |ds|
        ds.with = "boxes lt 1"
        ds.title = "Ions"
        ds.linewidth = 4
        ds.linecolor = 'rgb "blue"'
      }
    ]
  end

  # Mean velocitys of electrons
  puts "\t Ploting mean velocity of electrons...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_meanv_e[:title]
    plot.xlabel param_meanv_e[:xlabel]
    plot.ylabel param_meanv_e[:ylabel]
    plot.output OFNAME_MEAN_VEL_E_GRAPH 
    plot.data = [
      Gnuplot::DataSet.new( [position, mean_velocity_e] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Electrons"
        ds.linewidth = 4
        ds.linecolor = 'rgb "red"'
      }
    ]
  end

  # Mean velocitys of ions
  puts "\t Ploting mean velocity of ions...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_meanv_i[:title]
    plot.xlabel param_meanv_i[:xlabel]
    plot.ylabel param_meanv_i[:ylabel]
    plot.output OFNAME_MEAN_VEL_I_GRAPH 
    plot.data = [
      Gnuplot::DataSet.new( [position, mean_velocity_i] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Ions"
        ds.linewidth = 4
        ds.linecolor = 'rgb "blue"'
      }
    ]
  end

  # Flux of electrons 
  puts "\t Ploting electrons's flux...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_flux_e[:title]
    plot.xlabel param_flux_e[:xlabel]
    plot.ylabel param_flux_e[:ylabel]
    plot.output OFNAME_FLUX_E_GRAPH 
    plot.data = [
      Gnuplot::DataSet.new( [position, flux_e] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Electrons"
        ds.linewidth = 4
        ds.linecolor = 'rgb "red"'
      }
    ]
  end

  # Flux of ions
  puts "\t Ploting ions's flux...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_flux_i[:title]
    plot.xlabel param_flux_i[:xlabel]
    plot.ylabel param_flux_i[:ylabel]
    plot.output OFNAME_FLUX_I_GRAPH 
    plot.data = [
      Gnuplot::DataSet.new( [position, flux_i] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Ions"
        ds.linewidth = 4
        ds.linecolor = 'rgb "blue"'
      }
    ]
  end
 
end

puts "Converting outputs to pdf..."
`pdflatex #{OFNAME_DDF_GRAPH}`
`pdflatex #{OFNAME_MEAN_VEL_E_GRAPH}`
`pdflatex #{OFNAME_MEAN_VEL_I_GRAPH}`
`pdflatex #{OFNAME_FLUX_E_GRAPH}`
`pdflatex #{OFNAME_FLUX_I_GRAPH}`
`rm *.aux *.log *.tex *.eps *converted-to.pdf`
